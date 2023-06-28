/**
 * @author Nicolas Limare <nicolas.limare@cmla.ens-cachan.fr>
 * All rights reserved.
 */

/**
 * tanh() interpolation and the reference libc implementation.
 */

#ifndef _TANH_HPP_
#define  _TANH_HPP_

#include <cmath>

/**
 * tanh() interpolation
 *
 * The interpolation table is stored as a static variable, out of the
 * function (or of a class) to avoid the cost of guard variables (test
 * of the static initialization at every function call).
 *
 * Interpolation bounds are set so that out of these bounds,
 * tanh(x) = +-1 at the precision used. These bounds are
 *   12.5 ln(2) =  8.66433 97569 99316 36772 in single precision
 *   27.5 ln(2) = 19.06154 74653 98496 00897 in double precision
 * (from Nelson H. F. Beebe,
 *   http://www.math.utah.edu/~beebe/software/ieee/tanh.pdf
 *   http://www.math.utah.edu/~beebe/software/ieee/dtanh.f)
 *
 * Althought we could restrict the interpolation of tanh(x) to x>=0
 * and use the symmetry of tanh(), this would require us to compute
 * sign(x), more expensive than using a larger computation table and
 * an offset (on CPUs with large cache, etc.)
 */

/** interpolation boundaries */
static const float tanh_xmax = 12.5 * logf(2);
static const float tanh_xmin = -tanh_xmax;

/**
 * Interpolation error e is bounded by
 *     e <= M/2 * d^2/4
 * where
 *     M = 4 / (3 sqrt(3)) = 0.7698
 * is the maximum of the second tanh derivative and d is the
 * interpolation step (Taylor-Young for polynomial interpolation first
 * order). Conversely, for a given max error e, the interpolation step
 * must be
 *   d = sqrt( 8 e / M)
 *     = sqrt( e * 6 sqrt(3))
 *     = 1.0194               for e = 10^-1
 *     = 3.2237 10^-1         for e = 10^-2
 *     = 1.0194 10^-1         for e = 10^-3
 *     = 3.2237 10^-2         for e = 10^-4
 *     ...
 */

/** precision */
static const float tanh_err = 1e-8f;
/** interpolation step */
static const float tanh_step = sqrtf(tanh_err * 6 * sqrtf(3.));
static const float tanh_istep = 1.f / tanh_step;
/** interpolation table size */
static const size_t tanh_size = (size_t) ((tanh_xmax - tanh_xmin) / tanh_step) + 2;
static float * tanh_table = NULL;

/**
 * table initialization
 */
static void tanh_inter_init()
{
    if (NULL == tanh_table) {
        tanh_table = new float[2 * tanh_size];
        // tanh() samples
        for (size_t i=0; i<tanh_size; i++)
            tanh_table[2*i] = tanhf(tanh_xmin + i * tanh_step);
        // precomputed sample differences
        // results are interlaced to minimize cache miss
        for (size_t i=0; i<tanh_size-1; i++)
            tanh_table[2*i+1] = tanh_table[2*i+2] - tanh_table[2*i];
    }
}

/**
 * interpolated tanh() approximation
 *
 */
_UNUSED inline
float tanh_inter(float x)
{
    // (float) position of x in the table
    float fpos = (x - tanh_xmin) * tanh_istep;

    // (integer) position of x in the interpolation table
    long int ipos = (long int) fpos;

    // t in [0..1] is the continuous position of x
    // between step ipos and step ipos+1
    float t = fpos - (float) ipos;

    // Check that x is within [xmin, xmax].
    // fabsf(), test, and branch cost 5% CPU time. Many alternatives
    // tested (non-branching sum, sign function, integer comparisons,
    // alternative sign and abs() code, two tests without abs(),
    // in-place ops...), none faster (yet).
    if (fabsf(x) > tanh_xmax)
        // out of bounds, tanh(x) = +-1 at this precision
        return (x > 0.f ? 1.f : -1.f);
    else
        // table[2*ipos] is the i-th sample
        // table[2*ipos+1] is the difference between i-th sample and the next
        return tanh_table[2*ipos] + t * tanh_table[2*ipos+1];
}

/**
 * tanh() from libc
 */
_UNUSED inline
float tanh_libc(float x)
{ return tanhf(x); }

#endif  /* !_TANH_HPP_ */
