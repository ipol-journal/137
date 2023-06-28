/**
 * @author Nicolas Limare <nicolas.limare@cmla.ens-cachan.fr>
 * All rights reserved.
 */

/**
 * Pseudo-random number generators.
 *
 * All these functions are reentrant (thread-safe) if a state
 * parameter is provided.
 */

#ifndef _PRNG_HPP_
#define  _PRNG_HPP_

/* selectively allow unused functions and variables */

#ifndef _UNUSED
#if defined(__GNUC__)
/* from http://sourceforge.net/p/predef/wiki/Compilers/ */
#define _UNUSED __attribute__((__unused__))
#else
#define _UNUSED
#endif
#endif

#include <ctime>
#if defined(USE_GETTIMEOFDAY)
#include <sys/time.h>
#endif

#include <cmath>

/**
 * Internal state of the generators
 */
struct prng_state_s {
    unsigned long int z;
    unsigned long int w;
    unsigned long int jsr;
    unsigned long int jcong;
    bool normal_has_spare;
    float normal_spare;
    bool initialized;
};

/**
 * Create a new state structure
 */
static
prng_state_s prng_new_state()
{
    prng_state_s state = { // cf. http://stackoverflow.com/q/6181715
        /*.z=*/ 362436069,
        /*.w=*/ 521288629,
        /*.jsr=*/ 123456789,
        /*.jcong=*/ 380116160,
        /*.normal_has_spare=*/ false,
        /*.normal_spare=*/ 0,
        /*.initialized=*/ false,
    };
    return state;
};

/**
 * Global state, suitable for non-threaded code
 */
static
prng_state_s prng_state = prng_new_state();

/**
 * Marsaglia's multiply-with-carry generator.
 *
 * Period 2^60, from Usenet
 *   http://www.cse.yorku.ca/~oz/marsaglia-rng.html
 */
inline
unsigned long int prng_mwc(prng_state_s & state = prng_state)
{
    state.z = 36969 * (state.z & 65535) + (state.z >> 16);
    state.w = 18000 * (state.w & 65535) + (state.w >> 16);
    return (state.z << 16) + state.w;
}

/**
 * Marsaglia's 3-shift register generator.
 *
 * Period 2^32-1, version with 13/17 fix, from the Ziggurat paper
 *   http://www.jstatsoft.org/v05/i08
 * Function is reentrant if provided with its own state.
 */
inline
unsigned long int prng_shr3(prng_state_s & state = prng_state)
{
    unsigned long int tmp = state.jsr;
    state.jsr ^= (state.jsr << 13);
    state.jsr ^= (state.jsr >> 17);
    state.jsr ^= (state.jsr << 5);
    return tmp + state.jsr;
}

/**
 * Marsaglia's congruential generator with multiplier 69069.
 *
 * Period 2^32, from Usenet
 *   http://www.cse.yorku.ca/~oz/marsaglia-rng.html
 * Function is reentrant if provided with its own state.
 */
inline
unsigned long int prng_cong(prng_state_s & state = prng_state)
{
    state.jcong = 69069 * state.jcong + 1234567;
    return state.jcong;
}

/**
 * Marsaglia's kiss generator.
 *
 * Period 2^123, from Usenet
 *   http://www.cse.yorku.ca/~oz/marsaglia-rng.html
 * Function is reentrant if provided with its own state.
 */
inline
unsigned long int prng_kiss(prng_state_s & state = prng_state)
{
    return (prng_mwc(state) ^ prng_cong(state)) + prng_shr3(state);
}

/**
 * 32bit integer uniform random number.
 *
 * Backend is KISS, this is just a neutral name.
 * Function is reentrant if provided with its own state.
 */
inline
unsigned long int prng_uint32(prng_state_s & state = prng_state)
{
    return prng_kiss(state);
}

/**
 * Single precision floating point uniform random number on (0,1)
 *
 * Since it is derived from a 32bits random int, this float
 * has the maximum posible randomness.
 * Function is reentrant if provided with its own state.
 */
_UNUSED inline
float prng_unif(prng_state_s & state = prng_state)
{
    return .5f + (signed) prng_uint32(state) * .2328306e-9f;
}

/**
 * Standard normal distributed random number using the Marsaglia polar
 * transform. Two normal numbers can be produced for every pair of
 * uniform numbers, so one is saved for the next call.
 */
#pragma GCC diagnostic ignored "-Wfloat-equal"
_UNUSED inline
float prng_normal(prng_state_s & state = prng_state)
{
    if (state.normal_has_spare) {
        state.normal_has_spare = false;
        return (float) state.normal_spare;
    }

    float u, v, s;
    do {
        u = prng_unif(state) * 2.f - 1.f;
        v = prng_unif(state) * 2.f - 1.f;
        s = u * u + v * v;
    }
    while (s >= 1.f || s + 1.f == 1.f); // s+1: test for denormal, avoid div by 0

    s = sqrtf(-2.f * logf(s) / s);
    state.normal_spare = (float) v * s;
    state.normal_has_spare = true;
    return u * s;
}
#pragma GCC diagnostic warning "-Wfloat-equal"

/**
 * PRNG seeding
 */
static void prng_init(unsigned long int s1, unsigned long int s2=0,
                      unsigned long int s3=0, unsigned long int s4=0,
                      prng_state_s & state = prng_state)
{
    // add the state address, to differentiate multiple states in
    // similar (automatic) initializations
    s1 += (unsigned long int) &state;
    state.z ^= s1;

    // handle missing seeds by adding s1 and so on
    s2 += s1;
    state.w ^= s2;

    s3 += s2;
    state.jsr ^= s3;

    s4 += s3;
    state.jcong ^= s4;

    state.initialized = true;
}

/**
 * Generate a random seed from time
 */
static unsigned long int prng_getseed()
{
    /* Basic (weak) initialization uses the current time, in seconds */
    unsigned long int seed = (unsigned long int) time(NULL);

    /* gettimeofday() adds microsecond resolution */
#if defined(USE_GETTIMEOFDAY)
    {
        struct timeval tp;
        (void) gettimeofday(&tp, NULL);

        seed *= 1000000;
        seed += tp.tv_usec;
    }
#endif

    return seed;
}

/**
 * Automatic (weak) initialization
 */
static void prng_init_auto(prng_state_s & state = prng_state)
{
    if (!state.initialized)
        prng_init(prng_getseed(), 0, 0, 0, state);
}

#endif  /* !_PRNG_HPP_ */
