#ifndef IM_H
#define IM_H

#include "global.hpp"

#include <string>
#include <vector>
#include "eigen.hpp"

using namespace Eigen;
using namespace std;

struct patchConfig {
	unsigned int rows;
	unsigned int cols;
};

MatrixXf * imread(
        const char * fileName,
        int & nRows,
        int & nCols,
        int & nChannels
);

void imwrite(
        const char * fileName,
        MatrixXf * image,
        int nChannels
);

MatrixXf * image2patches(
        MatrixXf * image,
        int stride,
        patchConfig const & patchSize,
        int nChannels
);

MatrixXf * patches2image(
        MatrixXf * patches,
        int nRows,
        int nCols,
        int stride,
        patchConfig const & patchSize,
        int nChannels
);

void display(
        MatrixXf const & data,
        int nPieces,
        patchConfig const & patchSize,
        const char * pic
);

void drawPatches(
        MatrixXf & patches,
        const char * path,
        int nPatches,
        patchConfig const & patchSize,
        vector<MatrixXf *> & images,
        int nChannels
);

void mosaicRGGB(
        MatrixXf & full,
        MatrixXf & mosaiced,
        MatrixXf & teaching,
        patchConfig const & patchSize,
        bool inverse
);

void crop(
        const MatrixXf & input,
        MatrixXf & output,
        int border,
        patchConfig const & patchSize,
        int nChannels
);

void applyCFA(
        MatrixXf * noisy,
        int CFA
);

void rotate2RGGB(
        MatrixXf * input,
        int pattern
);

void ZCA(
        const MatrixXf & in,
        MatrixXf & out,
        float filter = 0.01
);
#endif
