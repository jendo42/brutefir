/*
 * GStreamer Callback I/O Module for BruteFIR
 * Uses appsrc/appsink and GstAdapter for exact period_size alignment.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <pthread.h>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/base/gstadapter.h>

#include "compat.h"
#define IS_BFIO_MODULE
#include "bfmod.h"
#include "inout.h"

#define DEFAULT_GST_THREAD_PRIORITY 83

struct gst_state {
    GstElement *pipeline;
    GstElement *app_elem; // appsink for IN, appsrc for OUT
    int channels;
    char *pipeline_str;
};

static struct {
    int expected_priority;
    int sample_format_size;
    int period_size;
    int frame_size;
    bool debug;
    
    struct gst_state *handles[2][BF_MAXCHANNELS];
    void **states[2];
    int n_handles[2];
    
    GstAdapter *in_adapter;
    pthread_mutex_t process_mutex;
    
    int (*process_cb)(void **_states[2],
                      int state_count[2],
                      void **bufs[2],
                      int count,
                      int event);
    void *zerobuf;
} glob = {
    .expected_priority = DEFAULT_GST_THREAD_PRIORITY,
    .sample_format_size = 0,
    .period_size = 0,
    .frame_size = 0,
    .debug = false,
    .handles = {{NULL}},
    .states = {NULL, NULL},
    .n_handles = {0, 0},
    .in_adapter = NULL,
    .process_cb = NULL,
    .zerobuf = NULL
};

static const char *
bf_sample_format_to_gst(int sample_format)
{
    switch (sample_format) {
    case BF_SAMPLE_FORMAT_S16_LE: return "S16LE";
    case BF_SAMPLE_FORMAT_S24_LE: return "S24LE"; 
    case BF_SAMPLE_FORMAT_S32_LE: return "S32LE";
    case BF_SAMPLE_FORMAT_FLOAT_LE: return "F32LE";
    case BF_SAMPLE_FORMAT_FLOAT64_LE: return "F64LE";
    default: return NULL;
    }
}

// -------------------------------------------------------------------
// Core Callback Logic
// -------------------------------------------------------------------

static void
trigger_brutefir_processing()
{
    // We only process if we have a full period_size of bytes in the adapter
    int required_bytes = glob.period_size * glob.frame_size;

    while (gst_adapter_available(glob.in_adapter) >= required_bytes) {
        
        // 1. Extract exactly period_size frames from adapter
        gpointer in_data = gst_adapter_take(glob.in_adapter, required_bytes);
        
        void *in_bufs[BF_MAXCHANNELS] = {NULL};
        void *out_bufs[BF_MAXCHANNELS] = {NULL};
        void **iobufs[2] = {NULL, NULL};

        if (glob.n_handles[IN] > 0) {
            in_bufs[0] = in_data; // Interleaved data goes in index 0
            iobufs[IN] = in_bufs;
        }
        
        // 2. Allocate output buffers
        if (glob.n_handles[OUT] > 0) {
            // Allocate memory for the output chunk
            out_bufs[0] = g_malloc0(required_bytes); 
            iobufs[OUT] = out_bufs;
        }

        // 3. Call BruteFIR
        glob.process_cb(glob.states, glob.n_handles, iobufs, glob.period_size, BF_CALLBACK_EVENT_NORMAL);

        // 4. Push processed data to GStreamer appsrcs
        for (int n = 0; n < glob.n_handles[OUT]; n++) {
            struct gst_state *out_state = glob.handles[OUT][n];
            if (out_state->app_elem != NULL) {
                // Wrap the processed memory in a GstBuffer
                gpointer mem_copy = g_memdup2(out_bufs[0], required_bytes);
                GstBuffer *out_gst_buf = gst_buffer_new_wrapped(mem_copy, required_bytes);
                
                GstFlowReturn ret;
                g_signal_emit_by_name(out_state->app_elem, "push-buffer", out_gst_buf, &ret);
                gst_buffer_unref(out_gst_buf);
            }
        }

        g_free(in_data);
        if (out_bufs[0]) {
            g_free(out_bufs[0]);
        }
    }
}

static GstFlowReturn
new_sample_callback(GstAppSink *sink, gpointer user_data)
{
    GstSample *sample = gst_app_sink_pull_sample(sink);
    if (!sample) return GST_FLOW_ERROR;

    GstBuffer *buffer = gst_sample_get_buffer(sample);
    
    pthread_mutex_lock(&glob.process_mutex);
    
    // Push upstream data into our alignment adapter
    gst_adapter_push(glob.in_adapter, gst_buffer_ref(buffer));
    
    // Process as many period_size chunks as possible
    trigger_brutefir_processing();
    
    pthread_mutex_unlock(&glob.process_mutex);

    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

// -------------------------------------------------------------------
// BruteFIR API Implementation
// -------------------------------------------------------------------

int bfio_iscallback(void) {
    return true; // Force BruteFIR to use the callback architecture
}

#define GET_TOKEN(token, errstr)                                        \
    if (get_config_token(&lexval) != token) {                           \
        fprintf(stderr, "GST-CB I/O: Parse error: " errstr);            \
        return NULL;                                                    \
    }

void *
bfio_preinit(int *version_major, int *version_minor,
             int (*get_config_token)(union bflexval *lexval),
             int io, int *sample_format, int sample_rate,
             int open_channels, int *uses_sample_clock,
             int *callback_sched_policy, struct sched_param *cb_sched_param,
             int debug)
{
    *version_major = BF_VERSION_MAJOR;
    *version_minor = BF_VERSION_MINOR;
    glob.debug = !!debug;

    if (!gst_is_initialized()) {
        gst_init(NULL, NULL);
        glob.in_adapter = gst_adapter_new();
        pthread_mutex_init(&glob.process_mutex, NULL);
    }

    glob.sample_format_size = bf_sampleformat_size(*sample_format);
    glob.frame_size = glob.sample_format_size * open_channels;

    struct gst_state *state = calloc(1, sizeof(struct gst_state));
    state->channels = open_channels;

    union bflexval lexval;
    int token;
    while ((token = get_config_token(&lexval)) > 0) {
        if (token != BF_LEXVAL_FIELD) return NULL;
        
        if (strcmp(lexval.field, "pipeline") == 0) {
            GET_TOKEN(BF_LEXVAL_STRING, "expected string.\n");
            state->pipeline_str = strdup(lexval.string);
        } else if (strcmp(lexval.field, "priority") == 0) {
            GET_TOKEN(BF_LEXVAL_REAL, "expected integer.\n");
            glob.expected_priority = (int)lexval.real;
        }
        GET_TOKEN(BF_LEX_EOS, "expected end of statement (;).\n");
    }

    cb_sched_param->sched_priority = glob.expected_priority;
    *callback_sched_policy = SCHED_FIFO;
    *uses_sample_clock = 1;

    return (void *)state;
}

int
bfio_init(void *params, int io, int sample_format, int sample_rate,
          int open_channels, int used_channels, const int channel_selection[],
          int period_size, int *device_period_size, int *isinterleaved,
          void *callback_state,
          int (*bf_process_callback)(void **cb_states[2], int cb_state_count[2],
                                     void **buffers[2], int frame_count, int event))
{
    static void *callback_states_[2][BF_MAXCHANNELS];
    
    glob.process_cb = bf_process_callback;
    glob.period_size = period_size;
    
    // Force BruteFIR to handle interleaved memory (one buffer array pointer)
    *isinterleaved = true; 
    *device_period_size = period_size;

    struct gst_state *state = (struct gst_state *)params;
    const char *gst_fmt = bf_sample_format_to_gst(sample_format);
    char *full_pipeline;

    if (io == BF_IN) {
        full_pipeline = g_strdup_printf(
            "%s ! audioconvert ! audioresample ! "
            "audio/x-raw,format=%s,rate=%d,channels=%d,layout=interleaved ! "
            "appsink name=bf_sink emit-signals=true sync=false", 
            state->pipeline_str, gst_fmt, sample_rate, open_channels);
    } else {
        full_pipeline = g_strdup_printf(
            "appsrc name=bf_src format=time is-live=true ! "
            "audio/x-raw,format=%s,rate=%d,channels=%d,layout=interleaved ! "
            "audioconvert ! %s", 
            gst_fmt, sample_rate, open_channels, state->pipeline_str);
    }

    GError *err = NULL;
    state->pipeline = gst_parse_launch(full_pipeline, &err);
    if (err) {
        fprintf(stderr, "GST-CB I/O: Pipeline error: %s\n", err->message);
        return -1;
    }
    g_free(full_pipeline);

    if (io == BF_IN) {
        state->app_elem = gst_bin_get_by_name(GST_BIN(state->pipeline), "bf_sink");
        // Attach the callback that will drive the BruteFIR engine
        g_signal_connect(state->app_elem, "new-sample", G_CALLBACK(new_sample_callback), NULL);
    } else {
        state->app_elem = gst_bin_get_by_name(GST_BIN(state->pipeline), "bf_src");
    }

    callback_states_[io][glob.n_handles[io]] = callback_state;
    glob.handles[io][glob.n_handles[io]] = state;
    glob.n_handles[io]++;

    glob.states[IN] = glob.n_handles[IN] > 0 ? callback_states_[IN] : NULL;
    glob.states[OUT] = glob.n_handles[OUT] > 0 ? callback_states_[OUT] : NULL;

    return 0;
}

int
bfio_synch_start(void)
{
    // Set all pipelines to PLAYING
    FOR_IN_AND_OUT {
        for (int n = 0; n < glob.n_handles[IO]; n++) {
            struct gst_state *state = glob.handles[IO][n];
            gst_element_set_state(state->pipeline, GST_STATE_PLAYING);
        }
    }
    return 0;
}

void
bfio_synch_stop(void)
{
    FOR_IN_AND_OUT {
        for (int n = 0; n < glob.n_handles[IO]; n++) {
            struct gst_state *state = glob.handles[IO][n];
            gst_element_set_state(state->pipeline, GST_STATE_NULL);
            gst_object_unref(state->pipeline);
        }
    }
}
