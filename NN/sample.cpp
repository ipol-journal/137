/**
 * @author Yi-Qing Wang <yiqing.wang@cmla.ens-cachan.fr>
 * All rights reserved.
 */

/*
 * The neural network aims to approximate a conditional expectation.
 * A rough principal component analysis helps to raise the statistical
 * profile of high frequency patterns, which do not necessarily have
 * the regularity we really want though. But it helps.
 */

#include "global.hpp"
#include "matops.hpp"
#include <fstream>
#include <iostream>
#include <cstdlib>
#include "imageio.hpp"
#include "sample.hpp"

using namespace std;
using namespace Eigen;

void sample(
        const char * path,
        const char * outFile,
        patchConfig const & patchSize,
        size_t nSize
) {
	vector<MatrixXf *> images;
	MatrixXf clean;
	drawPatches(clean, path, nSize, patchSize, images, 3);
	display(clean, 480, patchSize, "PCAsamples.png");

	MatrixXf mosaiced, teaching;
	mosaicRGGB(clean, mosaiced, teaching, patchSize, false);
	clean = mosaiced;
	whiten(clean.data(), clean.size());
	display(clean, 480, patchSize, "mosaiced.png");

	// compute the mean
	MatrixXf mean;
	mean.resize(patchSize.rows*patchSize.cols, 1);
	rsum(mean.data(), clean.data(), clean.rows(), clean.cols());
	float * ptr = mean.data();
	for (size_t k = 0; k < mean.size(); k++)
		ptr[k] /= nSize;

	// mean removal
	#pragma omp parallel for schedule(static)
	for (size_t c = 0; c < clean.cols(); c++) {
		float * out = clean.data() + c * clean.rows();
		for (size_t r = 0; r < mean.size(); r++)
			out[r] -= ptr[r];
	}
	display(clean, 480, patchSize, "meanRemoved.png");

	// compute the covariance
	MatrixXf cov;
	cov.resize(clean.rows(), clean.rows());
	mmT(cov.data(), cov.rows(), cov.cols(), clean.data(), clean.data(), clean.cols());
	ptr = cov.data();
	for (int k = 0; k < cov.size(); k++)
		ptr[k] /= nSize;

	// singular value decomposition
	JacobiSVD<MatrixXf> svd(cov, ComputeThinU);
	MatrixXf U = svd.matrixU();
	display(U, U.cols(), patchSize, "PC.png");
	VectorXf D = svd.singularValues();

	// write out
	ofstream output;
	output.open(outFile, ios_base::trunc);
	output << "#include \"WStruct.hpp\"" << endl;
	output.precision(8);
	output.setf(ios::fixed, ios::floatfield);

	ptr = mean.data();
	output << "float WStruct::mean[]={";
	for (int i = 0; i < mean.size() - 1; i++)
		output << ptr[i] << ",";
	output << ptr[mean.size() - 1] << "};" << endl;

	ptr = U.data();
	output << "float WStruct::basis[]={";
	for (int i = 0; i < U.size() - 1; i++)
		output << ptr[i] << ",";
	output << ptr[U.size() - 1] << "};" << endl;

	ptr = D.data();
	output << "float WStruct::vals[]={";
	for (int i = 0; i < D.size() - 1; i++)
		output << ptr[i] << ",";
	output << ptr[D.size() - 1] << "};" << endl;
}
