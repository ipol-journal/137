#ifndef NS_H
#define NS_H

#include "global.hpp"

class NeuralStruct
{
	static float optStackTheta[];
	static int thetaSize;
	static int border;
	static int patchSizeRows;
	static int patchSizeCols;
	static int hiddenSize;
	static int nHiddenLayers;

public:

	NeuralStruct(){}
	~NeuralStruct(){}

	static float * getTheta() { return optStackTheta; }
	static int getPatchSize(int param) { return param == 0 ? patchSizeRows : patchSizeCols; }
	static int getBorder() { return border; }
	static int getHiddenSize() { return hiddenSize; }
	static int getnHiddenLayers() { return nHiddenLayers; }
	static int getThetaSize() { return thetaSize; }
};

#endif
