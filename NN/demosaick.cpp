/**
 * @author Yi-Qing Wang <yiqing.wang@cmla.ens-cachan.fr>
 * All rights reserved.
 */

#include "global.hpp"

#include <cstdlib>
#include <cstdio>
#include <cstring>

#include "appl.hpp"

#include "NeuralStruct.hpp"

/* call info */
static int _argc;
static const char * _arg0;

static void usage() {
	fprintf(stderr, \
	        "syntax:\n"
	        "	%s -h\n"
	        "	%s -c clean [-p pattern] [-s stride] [-m mosaiced] [-d demosaicked]\n"
	        "options:\n"
	        "    	-h print this help\n"
	        "	-c path to a clean input image\n"
	        "	-p Bayer color filter array pattern 0: RGGB, 1: GRBG, 2: BGGR or 3: GBRG (default 0)\n"
	        "	-s stride between two neighboring sliding windows (default 1)\n"
	        "	-m path to store the mosaiced image (default ./mosaiced.png)\n"
	        "	-d path to store the demosaicked image (default ./demosaiced.png)\n",
	        _arg0, _arg0);
	exit(EXIT_SUCCESS);
}

/**
 * increment argument index without going too far
 */
static int shift(int &i) {
	i++;
	if (i >= _argc)
		ERROR("missing parameter value");
	return i;
}

int main(int argc, char ** argv) {

	const char * clean = "unspecified";
	const char * demosaicked = "./demosaiced.png";
	const char * mosaiced = "./mosaiced.png";
	int pattern = 0;
	int stride = 1;

	// set global call info
	_argc = argc;
	_arg0 = argv[0];

	for (int i = 1; i < argc; i++) {
		if (0 == strcmp(argv[i],"-h")) {
			usage();
		} else if (0 == strcmp(argv[i],"-c")) {
			clean = argv[shift(i)];
		} else if (0 == strcmp(argv[i],"-d")) {
			demosaicked = argv[shift(i)];
		} else if (0 == strcmp(argv[i],"-m")) {
			mosaiced = argv[shift(i)];
		} else if (0 == strcmp(argv[i],"-p")) {
			pattern = atoi(argv[shift(i)]);
		} else if (0 == strcmp(argv[i],"-s")) {
			stride = atoi(argv[shift(i)]);
		} else
			ERROR("unknown parameter");

	}

	if (0 == strcmp(clean, "unspecified"))
		ERROR("Unspecified input (print the manual with %s -h)", _arg0);

	if (pattern < 0 || pattern > 3)
		ERROR("Unrecognized Bayer pattern %i (print the manual with %s -h)", pattern, _arg0);

	if (!(stride == 1 || stride == 2))
		ERROR("It makes no sense to set stride = %i at demosaicking", stride);

	demosaick(clean, demosaicked, mosaiced, pattern, stride);
}
