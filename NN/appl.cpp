/**
 * @author Yi-Qing Wang <yiqing.wang@cmla.ens-cachan.fr>
 * All rights reserved.
 */

#include "global.hpp"

#include "imageio.hpp"
#include "Autoencoder.hpp"
#include "NeuralStruct.hpp"

#include "WStruct.hpp"
#include "matops.hpp"

#include "appl.hpp"    // self-consistency

using namespace std;
using namespace Eigen;

void onePass(
        MatrixXf * mosaic,
        Autoencoder & encoder,
        patchConfig const & patchSize,
        int border,
        int pattern,
        MatrixXf *& demosaic
) {
	MatrixXf seen, input, rest;
	MatrixXf * patches = image2patches(mosaic, 2, patchSize, 3);

	// rotate and whiten patches in the right way
	rotate2RGGB(patches, pattern);
	mosaicRGGB(patches[0], input, rest, patchSize, false);
	whiten(input.data(), input.size());
	ZCA(input, rest);

	// neural network inference
	vector<MatrixXf> activations(encoder.nHiddenLayers+1);
	encoder.predict(rest, activations);
	MatrixXf * A = &activations[encoder.nHiddenLayers]; // short alias
	blacken(A->data(), A->size());

	// extract visible color intensities
	crop(patches[0], seen, border, patchSize, 3);
	patchConfig outPatch;
	outPatch.rows = patchSize.rows - 2*border;
	outPatch.cols = patchSize.cols - 2*border;
	mosaicRGGB(seen, input, rest, outPatch, false);

	// form the estimated patches
	mosaicRGGB(rest, input, activations[encoder.nHiddenLayers], outPatch, true);

	// inverse rotation
	if (pattern == GBRG)
		rotate2RGGB(&rest, GRBG);

	if (pattern == BGGR)
		rotate2RGGB(&rest, BGGR);

	if (pattern == GRBG)
		rotate2RGGB(&rest, GBRG);

	// aggregate to have the end result
	demosaic = patches2image(&rest, mosaic->rows()-2*border, mosaic->cols()-2*border, 2, outPatch, 3);
	delete [] patches;
}

void demosaick(
        const char * clean,
        const char * demosaicked,
        const char * mosaiced,
        int pattern,
        int stride
) {
	int rows, cols, nChannels;
	MatrixXf * cleanI = imread(clean, rows, cols, nChannels);
	assert(nChannels == 3);

	// trim to have even number of rows and cols
	// it reduces conditional loops in what follows
	if (rows % 2 == 1) {
		rows -= 1;
		for (int ch = 0; ch < 3; ch++) {
			MatrixXf tmp = cleanI[ch].block(0, 0, rows, cols);
			cleanI[ch] = tmp;
		}
	}
	if (cols % 2 == 1) {
		cols -= 1;
		for (int ch = 0; ch < 3; ch++) {
			MatrixXf tmp = cleanI[ch].block(0, 0, rows, cols);
			cleanI[ch] = tmp;
		}
	}

	// keep the original version for comparison later on
	// potentially leave out a rim of pixels because neural network might not
	// have its input equal to its output in patch size (border > 0)
	int border = NeuralStruct::getBorder();
	MatrixXf * original = new MatrixXf [nChannels];
	for (int ch = 0; ch < nChannels; ch++)
		original[ch] = cleanI[ch].block(border, border, rows - 2*border, cols - 2*border);

	// apply Bayer color filter array
	applyCFA(cleanI, pattern);
	MatrixXf * mosaic = new MatrixXf [nChannels];
	for (int ch = 0; ch < nChannels; ch++)
		mosaic[ch] = cleanI[ch];
	imwrite(mosaiced, mosaic, nChannels);

	// set up the trained neural network
	patchConfig patchSize;
	patchSize.rows = NeuralStruct::getPatchSize(0);
	patchSize.cols = NeuralStruct::getPatchSize(1);

	// rotate does not work in this case
	if (patchSize.rows != patchSize.cols)
		stride = 2;

	trainParams inParams;
	inParams.layerSizes.resize(NeuralStruct::getnHiddenLayers()+2);
	inParams.layerSizes[NeuralStruct::getnHiddenLayers()+1] = 2*(patchSize.rows-2*border)*(patchSize.cols-2*border);
	inParams.layerSizes[0] = patchSize.rows*patchSize.cols;
	for (int l = 1; l < NeuralStruct::getnHiddenLayers()+1; l++)
		inParams.layerSizes[l] = NeuralStruct::getHiddenSize();
	inParams.linear = true;
	Autoencoder encoder(inParams, NULL, NULL);
	encoder.stackTheta = Map<VectorXf>(NeuralStruct::getTheta(), NeuralStruct::getThetaSize(), 1);

	// one pass demosaicks with patch spatial offset = 2
	// for a dense demosaickage, set stride = 1
	MatrixXf * demosaic = NULL;
	onePass(cleanI, encoder, patchSize, border, pattern, demosaic);

	if (stride == 1) {

		// two more passes so that all the patches are filtered once

		MatrixXf * subcleanI = new MatrixXf [nChannels];
		for (int ch = 0; ch < 3; ch++)
			subcleanI[ch] = cleanI[ch].block(0, 1, rows, cols - 2);
		MatrixXf * subdem1 = NULL;
		int subpattern = GRBG;
		if (pattern == GRBG)
			subpattern = RGGB;
		else if (pattern == BGGR)
			subpattern = GBRG;
		else if (pattern == GBRG)
			subpattern = BGGR;
		onePass(subcleanI, encoder, patchSize, border, subpattern, subdem1);


		for (int ch = 0; ch < 3; ch++)
			subcleanI[ch] = cleanI[ch].block(1, 0, rows - 2, cols);
		MatrixXf * subdem2 = NULL;
		subpattern = GBRG;
		if (pattern == GRBG)
			subpattern = BGGR;
		else if (pattern == BGGR)
			subpattern = GRBG;
		else if (pattern == GBRG)
			subpattern = RGGB;
		onePass(subcleanI, encoder, patchSize, border, subpattern, subdem2);


		// merge

		MatrixXf weight;
		weight.resize(demosaic->rows(), demosaic->cols());
		float * ptr = weight.data();
		for (int k = 0; k < demosaic->size(); k++)
			ptr[k] = 1;

		// the effect of adding subdem1
		for (int k = demosaic->rows(); k < demosaic->size() - demosaic->rows(); k++)
			ptr[k] += 1;
		// add subdem1
		for (int ch = 0; ch < nChannels; ch++) {
			float * out = demosaic[ch].data() + demosaic->rows();
			float * in = subdem1[ch].data();
			for (int k = 0; k < demosaic->size() - 2*demosaic->rows(); k++)
				out[k] += in[k];
		}

		// the effet of adding subdem2
		for (int c = 0; c < demosaic->cols(); c++) {
			for (int r = 1; r < demosaic->rows() - 1; r++)
				ptr[r] += 1;
			ptr += demosaic->rows();
		}
		// add subdem2
		for (int ch = 0; ch < nChannels; ch++) {
			float * out = demosaic[ch].data();
			float * in = subdem2[ch].data();
			for (int c = 0; c < demosaic->cols(); c++) {
				for (int r = 1; r < demosaic->rows() - 1; r++)
					out[r] += in[r-1];
				out += demosaic->rows();
				in += demosaic->rows() - 2;
			}
		}

		// aggregate
		ptr = weight.data();
		for (int ch = 0; ch < nChannels; ch++) {
			float * out = demosaic[ch].data();
			for (int k = 0; k < demosaic->size(); k++)
				out[k] /= ptr[k];
		}

		delete [] subcleanI;
		delete [] subdem1;
		delete [] subdem2;
	}

	// calculate RMSE
	float rmse = 0;
	for (int ch = 0 ; ch < nChannels; ch++) {
		float * in = original[ch].data();
		float * de = demosaic[ch].data();
		for (int k = 0; k < original->size(); k++) {
			de[k] = de[k] > 255 ? 255 : de[k];
			de[k] = de[k] < 0 ? 0 : de[k];
			float diff = de[k] - in[k];
			rmse += diff * diff;
		}
	}
	INFO("neural network image RMSE of %s at %f", clean, sqrt(rmse/(original->size()*3)));
	imwrite("ccropped.png", original, nChannels);

	imwrite(demosaicked, demosaic, nChannels);
	delete [] cleanI;
}
