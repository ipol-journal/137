/**
 * @author Nicolas Limare <nicolas.limare@cmla.ens-cachan.fr>
 * @author Yi-Qing Wang <yiqing.wang@cmla.ens-cachan.fr>
 */

#ifndef _GLOBAL_HPP
#define _GLOBAL_HPP

extern bool flag_ref_mode;
extern bool flag_random_seed;

#define LOGLEVEL_ERROR 1
#define LOGLEVEL_INFO  2
#define LOGLEVEL_DEBUG 3

/* default log level is error messages only */
#ifndef LOGLEVEL
#define LOGLEVEL 1
#endif

/**
 * logging backend
 */
#define _LOG(level, ...) {					\
	fprintf(stderr, level);					\
	fprintf(stderr, " ");					\
	fprintf(stderr, __VA_ARGS__);				\
	fprintf(stderr, " \t[from %s:%i]", __FILE__, __LINE__);	\
	fprintf(stderr, "\n");					\
    }

/**
 * error message macro
 */
#if (LOGLEVEL >= LOGLEVEL_ERROR)
#define ERROR(...) {			\
	_LOG("ERROR", __VA_ARGS__);	\
	exit(EXIT_FAILURE);		\
    }
#else
#define ERROR(...) {}
#endif

/**
 * info message macro
 */
#if (LOGLEVEL >= LOGLEVEL_INFO)
#define INFO(...) {			\
	_LOG("INFO ", __VA_ARGS__);	\
    }
#else
#define INFO(...) {}
#endif

/**
 * debug message macro
 */
#if (LOGLEVEL >= LOGLEVEL_DEBUG)
#define DEBUG(...) {			\
	_LOG("DEBUG", __VA_ARGS__);	\
    }
#else
#define DEBUG(...) {}
#endif

/* selectively allow unused functions and variables */

#if defined(__GNUC__)
/* from http://sourceforge.net/p/predef/wiki/Compilers/ */
#define _UNUSED __attribute__((__unused__))
#else
#define _UNUSED
#endif

/* timing macros, transparent if USE_TIMING not defined */
#include "timing.h"

#define TIMER_LOOP 0
#define TIMER_RAND 1
#define TIMER_AXPB 2
#define TIMER_TANH 3
#define TIMER_MMPRC 4
#define TIMER_MMT 5
#define TIMER_MTMA 6
#define TIMER_OMSQ 7
#define TIMER_SUM 8
#define TIMER_PATCH 9
#define TIMER_GEMM 10
#define TIMER_CROP 11
#define TIMER_DOWN 12
#define TIMER_MOSA 13
#define TIMER_RGGB 14
#define TIMER_MTM 15
#define TIMER_MSRC 16

#define TIMING_RESET(TIMER_ID) {		\
    TIMING_WALLCLOCK_RESET(TIMER_ID);		\
    TIMING_CPUCLOCK_RESET(TIMER_ID);		\
    }
#define TIMING_TOGGLE(TIMER_ID) {		\
    TIMING_WALLCLOCK_TOGGLE(TIMER_ID);		\
    TIMING_CPUCLOCK_TOGGLE(TIMER_ID);		\
    }
#define TIMING_LOG(STR, TIMER_ID) 					\
    TIMING_PRINTF("TIME  [" STR "] cpu:%0.6f elapsed:%0.6f\n",		\
		  TIMING_CPUCLOCK_S(TIMER_ID), \
		  TIMING_WALLCLOCK_S(TIMER_ID));

#define RGGB 0
#define GRBG 1
#define BGGR 2
#define GBRG 3

#define whiten(X, sz) axpb(X, sz, 0.0196078431372549, -2.5) // X = X / 51 - 2.5
#define blacken(X, sz) axpb(X, sz, 51., 127.5)              // X = (X + 2.5) * 51

#endif          /* !_GLOBAL_HPP */
