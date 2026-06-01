/*
 * (c) Copyright 2013 -- Anders Torger
 *
 * This program is open source. For license terms, see the LICENSE file.
 *
 * ARM NEON port of the SSE/SSE2 partitioned-convolution complex MAC core.
 * Function signatures and the in-memory data layout are preserved bit-for-bit,
 * so this is a drop-in replacement for convolver_xmm.c on AArch64
 * (Cortex-A53 / i.MX8M Mini etc.). The double-precision path is AArch64-only,
 * since AArch32 NEON has no 64-bit float SIMD.
 */
#include "asmprot.h"

#include <arm_neon.h>

/*
 * CONV_USE_FMA:
 *   1 (default) -- use fused multiply-add (vfmaq/vfmsq). Fewer instructions
 *                  and a single rounding step per product, so results differ
 *                  very slightly from the separately-rounded SSE arithmetic
 *                  (the FMA result is in fact closer to the true value).
 *   0           -- use separate vmul + vadd/vsub, matching the SSE operation
 *                  order exactly (bit-identical to convolver_xmm.c).
 *   The DC/Nyquist scalars are computed in plain C either way, so those bins
 *   are bit-identical to the original regardless of this setting.
 */
#ifndef CONV_USE_FMA
#define CONV_USE_FMA 1
#endif

/*
 * CONV_PF_AHEAD:
 *   Software-prefetch distance in loop iterations. Cortex-A53 has a weak
 *   hardware prefetcher and limited out-of-order capability, so explicit
 *   prefetch is the main lever for overlapping DDR latency with compute.
 *   4-16 is a sane range; tune on the target. Prefetching past the end of
 *   the buffer is harmless (a prefetch to an unmapped address is ignored,
 *   never faults), so no bounds check is needed.
 */
#ifndef CONV_PF_AHEAD
#define CONV_PF_AHEAD 16
#endif

void
convolver_neon32_convolve_add(void *input_cbuf,
                           void *coeffs,
                           void *output_cbuf,
                           int loop_counter)
{
    const float32x4_t *b = (const float32x4_t *)input_cbuf;
    const float32x4_t *c = (const float32x4_t *)coeffs;
    float32x4_t *d = (float32x4_t *)output_cbuf;
    float d1s, d2s;
    int i;

    /* DC (bin 0) and Nyquist (bin N/2) are packed real-only values at lane 0
       of the first real vector and the first imag vector respectively. Capture
       the correct real-only MAC using the original (pre-loop) d, then restore
       after the loop -- identical to the original SSE code. */
    d1s = ((float *)d)[0] + ((float *)b)[0] * ((float *)c)[0];
    d2s = ((float *)d)[4] + ((float *)b)[4] * ((float *)c)[4];

    for (i = 0; i < loop_counter; i++) {
        int n = i << 1;
        int pf = (i + CONV_PF_AHEAD) << 1;
        float32x4_t b_re, b_im, c_re, c_im, d_re, d_im;

        /* Streaming prefetch of the read operands (locality 0, PLDL1STRM) so
           they do not evict the d accumulator, which is reused across the
           per-partition calls. Prefetch d for write (rw=1, keep). */
        __builtin_prefetch(b + pf, 0, 0);
        __builtin_prefetch(c + pf, 0, 0);
        __builtin_prefetch(d + pf, 1, 3);

        /* b[n+0] = 4 real parts, b[n+1] = 4 imag parts; same for c and d. */
        b_re = vld1q_f32((const float *)(b + n + 0));
        b_im = vld1q_f32((const float *)(b + n + 1));
        c_re = vld1q_f32((const float *)(c + n + 0));
        c_im = vld1q_f32((const float *)(c + n + 1));
        d_re = vld1q_f32((const float *)(d + n + 0));
        d_im = vld1q_f32((const float *)(d + n + 1));

#if CONV_USE_FMA
        /* real: d_re += b_re*c_re - b_im*c_im */
        d_re = vfmaq_f32(d_re, b_re, c_re);
        d_re = vfmsq_f32(d_re, b_im, c_im);
        /* imag: d_im += b_re*c_im + b_im*c_re */
        d_im = vfmaq_f32(d_im, b_re, c_im);
        d_im = vfmaq_f32(d_im, b_im, c_re);
#else
        d_re = vaddq_f32(d_re,
                         vsubq_f32(vmulq_f32(b_re, c_re),
                                   vmulq_f32(b_im, c_im)));
        d_im = vaddq_f32(d_im,
                         vaddq_f32(vmulq_f32(b_re, c_im),
                                   vmulq_f32(b_im, c_re)));
#endif

        vst1q_f32((float *)(d + n + 0), d_re);
        vst1q_f32((float *)(d + n + 1), d_im);
    }

    ((float *)d)[0] = d1s;
    ((float *)d)[4] = d2s;
}

#if defined(__aarch64__)

void
convolver_neon64_convolve_add(void *input_cbuf,
                            void *coeffs,
                            void *output_cbuf,
                            int loop_counter)
{
    const float64x2_t *b = (const float64x2_t *)input_cbuf;
    const float64x2_t *c = (const float64x2_t *)coeffs;
    float64x2_t *d = (float64x2_t *)output_cbuf;
    double d1s, d2s;
    int i;

    /* DC at lane 0 of the first real vector, Nyquist at lane 0 of the first
       imag vector (which is d[2] in the double layout). Real-only MAC. */
    d1s = ((double *)d)[0] + ((double *)b)[0] * ((double *)c)[0];
    d2s = ((double *)d)[4] + ((double *)b)[4] * ((double *)c)[4];

    for (i = 0; i < loop_counter; i++) {
        int n = i << 2;
        int pf = (i + CONV_PF_AHEAD) << 2;
        float64x2_t b0, b1, b2, b3, c0, c1, c2, c3, d0, d1, d2, d3;

        __builtin_prefetch(b + pf, 0, 0);
        __builtin_prefetch(c + pf, 0, 0);
        __builtin_prefetch(d + pf, 1, 3);

        /* reals in b0,b1 (4 values); imags in b2,b3; same for c and d. */
        b0 = vld1q_f64((const double *)(b + n + 0));
        b1 = vld1q_f64((const double *)(b + n + 1));
        b2 = vld1q_f64((const double *)(b + n + 2));
        b3 = vld1q_f64((const double *)(b + n + 3));
        c0 = vld1q_f64((const double *)(c + n + 0));
        c1 = vld1q_f64((const double *)(c + n + 1));
        c2 = vld1q_f64((const double *)(c + n + 2));
        c3 = vld1q_f64((const double *)(c + n + 3));
        d0 = vld1q_f64((const double *)(d + n + 0));
        d1 = vld1q_f64((const double *)(d + n + 1));
        d2 = vld1q_f64((const double *)(d + n + 2));
        d3 = vld1q_f64((const double *)(d + n + 3));

#if CONV_USE_FMA
        /* real */
        d0 = vfmaq_f64(d0, b0, c0); d0 = vfmsq_f64(d0, b2, c2);
        d1 = vfmaq_f64(d1, b1, c1); d1 = vfmsq_f64(d1, b3, c3);
        /* imag */
        d2 = vfmaq_f64(d2, b0, c2); d2 = vfmaq_f64(d2, b2, c0);
        d3 = vfmaq_f64(d3, b1, c3); d3 = vfmaq_f64(d3, b3, c1);
#else
        d0 = vaddq_f64(d0, vsubq_f64(vmulq_f64(b0, c0), vmulq_f64(b2, c2)));
        d1 = vaddq_f64(d1, vsubq_f64(vmulq_f64(b1, c1), vmulq_f64(b3, c3)));
        d2 = vaddq_f64(d2, vaddq_f64(vmulq_f64(b0, c2), vmulq_f64(b2, c0)));
        d3 = vaddq_f64(d3, vaddq_f64(vmulq_f64(b1, c3), vmulq_f64(b3, c1)));
#endif

        vst1q_f64((double *)(d + n + 0), d0);
        vst1q_f64((double *)(d + n + 1), d1);
        vst1q_f64((double *)(d + n + 2), d2);
        vst1q_f64((double *)(d + n + 3), d3);
    }

    ((double *)d)[0] = d1s;
    ((double *)d)[4] = d2s;
}

#endif /* __aarch64__ */
