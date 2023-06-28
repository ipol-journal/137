#ifndef GS_H
#define GS_H

#include "global.hpp"

class WStruct {

	static float mean[];
	static float vals[];
	static float basis[];

public:

	WStruct() {}
	~WStruct() {}

	static float * getMean() {
		return mean;
	}
	static float * getBasis() {
		return basis;
	}
	static float * getVals() {
		return vals;
	}
};

#endif
