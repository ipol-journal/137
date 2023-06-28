/**
 * @author Nicolas Limare <nicolas.limare@cmla.ens-cachan.fr>
 * All rights reserved.
 */

/**
 * Matrix operations requiring substantial computation time for the NN
 * denoising code. Operations are grouped here for easy timing,
 * comparison and development.
 */

#include "global.hpp"

#include <cblas.h>

//#define USE_GETTIMEOFDAY
#include "prng.hpp" // random number generators
#include "tanh.hpp" // tanh() approximation

#include <omp.h>

#include "matops.hpp" // self consistency

/**
 * matrix-matrix product plus replicated column
 */
void mmprc(float * out, size_t rows, size_t cols,
           const float * A, const float * B, size_t inner, const float * C)
{
    TIMING_TOGGLE(TIMER_MMPRC);
    TIMING_TOGGLE(TIMER_GEMM);
    // out = A * B
    cblas_sgemm(CblasColMajor,
                CblasNoTrans, CblasNoTrans,
                rows, cols, inner,
                1.f, A, rows,
                B, inner,
                0.f, out, rows);
    TIMING_TOGGLE(TIMER_GEMM);

    // out += CCCC...
    DEBUG("out += C on array size %lu and %lu", cols * rows, rows);
    #pragma omp parallel for schedule(static)
    for (size_t c = 0; c < cols; c++) {
        // loop on columns
        float * out_c = out + c * rows;
        for (size_t r = 0; r < rows; r++)
            out_c[r] += C[r];
    }
    TIMING_TOGGLE(TIMER_MMPRC);
}


/**
 * matrix-matrixT product
 */
void mmT(float * out, size_t rows, size_t cols,
         const float * A, const float * B, size_t inner)
{
    TIMING_TOGGLE(TIMER_MMT);
    TIMING_TOGGLE(TIMER_GEMM);
    // out = A * Bt
    cblas_sgemm(CblasColMajor,
                CblasNoTrans, CblasTrans,
                rows, cols, inner,
                1.f, A, rows,
                B, cols,
                0.f, out, rows);
    TIMING_TOGGLE(TIMER_GEMM);
    TIMING_TOGGLE(TIMER_MMT);
}

/**
 * matrixT-matrix product, multiplied element-wise
 */
void mTma(float * out, size_t rows, size_t cols,
          const float *  A, const float * B, size_t inner,
          const float * C)
{
    TIMING_TOGGLE(TIMER_MTMA);
    TIMING_TOGGLE(TIMER_GEMM);
    // out = At * B
    cblas_sgemm(CblasColMajor,
                CblasTrans, CblasNoTrans,
                rows, cols, inner,
                1.f, A, inner,
                B, inner,
                0.f, out, rows);
    TIMING_TOGGLE(TIMER_GEMM);

    // out *= C
    size_t size = rows * cols;
    DEBUG("out *= C on array size %lu", size);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++)
        out[i] *= C[i];
    TIMING_TOGGLE(TIMER_MTMA);
}


void mTm(float * out, size_t rows, size_t cols,
         const float *  A, const float * B, size_t inner)
{
    TIMING_TOGGLE(TIMER_MTM);
    TIMING_TOGGLE(TIMER_GEMM);
    // out = At * B
    cblas_sgemm(CblasColMajor,
                CblasTrans, CblasNoTrans,
                rows, cols, inner,
                1.f, A, inner,
                B, inner,
                0.f, out, rows);
    TIMING_TOGGLE(TIMER_GEMM);
    TIMING_TOGGLE(TIMER_MTM);
}

/**
 * one minus element-wise square
 */
void omsq(float * out, const float * A, size_t size)
{
    TIMING_TOGGLE(TIMER_OMSQ);
    DEBUG("omsq() on array size %lu", size);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++)
        out[i] = 1.f - A[i] * A[i];
    TIMING_TOGGLE(TIMER_OMSQ);
}

/**
 * row-wise sums
 */
void rsum(float * out, const float * A, size_t rows, size_t size)
{
    TIMING_TOGGLE(TIMER_SUM);
    DEBUG("rsum() on array size %lu", size);
    #pragma omp parallel for schedule(static)
    for (size_t r = 0; r < rows; r++) {
        out[r] = 0;
        for (size_t i = r; i < size; i += rows)
            out[r] += A[i];
    }
    TIMING_TOGGLE(TIMER_SUM);
}

/**
 * element-wise tanh()
 */
void mtanh(float * A, size_t size)
{
    TIMING_TOGGLE(TIMER_TANH);
    // Initializing tanh_inter only once before main() makes
    // mmprc() slower. A matter of memory layout or cache miss?
    tanh_inter_init();
    DEBUG("tanh() on array size %lu", size);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++)
        A[i] = tanh_inter(A[i]);
    TIMING_TOGGLE(TIMER_TANH);
}

/**
 * element-wise affine scaling
 */
void axpb(float * X, size_t size, float a, float b)
{
    TIMING_TOGGLE(TIMER_AXPB);
    DEBUG("axpb() on array size %lu", size);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) {
        X[i] *= a;
        X[i] += b;
    }
    TIMING_TOGGLE(TIMER_AXPB);
}

