/**
 * @author Yi-Qing Wang <yiqing.wang@cmla.ens-cachan.fr>
 * All rights reserved.
 */

#include "global.hpp"

#include <iostream>
#include <vector>
#include <cassert>

#include "NeuralStruct.hpp"
#include "imageio.hpp"

#include "matops.hpp" // matrix operations

#include "Autoencoder.hpp" // self-consistency

using namespace Eigen;
using namespace std;

/**
 * represent *stackTheta* as desired
 */
void Autoencoder::theta2stack(
        vector<layerConfig> & nnStruct
) const {

	//output, hidden and input layers
	int start = 0;
	for (int l = nHiddenLayers; l >= 0; l--) {
		VectorXf theta = stackTheta.segment(start, layerSizes[l+1]*layerSizes[l]);
		nnStruct[l].W = Map<MatrixXf>(theta.data(), layerSizes[l+1], layerSizes[l]);
		start += layerSizes[l+1]*layerSizes[l];
		nnStruct[l].b = stackTheta.segment(start, layerSizes[l+1]);
		start += layerSizes[l+1];
	}
}

/**
 * represent *stackTheta* as desired
 */
void Autoencoder::stack2theta(
        vector<layerConfig> & nnStruct,
        VectorXf & theta
) const {

	// nParams could change during dropouts
	// don't make it a static variable
	int nParams = 0;
	for (int l = nHiddenLayers; l >= 0; l--)
		nParams += layerSizes[l+1]*layerSizes[l]+layerSizes[l+1];
	theta = VectorXf::Zero(nParams);

	//output, hidden and input layers
	int start = 0;
	for (int l = nHiddenLayers; l >= 0; l--) {
		theta.segment(start, layerSizes[l+1]*layerSizes[l]) = Map<VectorXf>(nnStruct[l].W.data(), layerSizes[l+1]*layerSizes[l], 1);
		start += layerSizes[l+1]*layerSizes[l];
		theta.segment(start, layerSizes[l+1]) = nnStruct[l].b;
		start += layerSizes[l+1];
	}
}

/**
 * randomly initialize the neural network
 */
void Autoencoder::randInitAE() {

	//allocate
	vector<layerConfig> nnStruct(nHiddenLayers+1);

	for (int l = 0; l <= nHiddenLayers; l++) {
		nnStruct[l].b = VectorXf::Zero(layerSizes[l+1]);
		nnStruct[l].W = MatrixXf::Random(layerSizes[l+1], layerSizes[l]) * (sqrt(6.) / sqrt(layerSizes[l] + layerSizes[l+1]));
	}

	stack2theta(nnStruct, stackTheta);
}

/**
 * compare the NN's outputs and the clean training
 * signals so as to track the progress NN makes in the process
 */
float Autoencoder::calcRMSE(
        MatrixXf const & originalOutputs,
        MatrixXf const & whitenedInputs
) {
	assert(whitenedInputs.rows() == iSize);
	assert(originalOutputs.rows() == oSize);
	assert(originalOutputs.cols() == whitenedInputs.cols());

	vector<MatrixXf> activations(nHiddenLayers+1);
	predict(whitenedInputs, activations);

	// short alias
	MatrixXf * A = &activations[nHiddenLayers];

	activations[nHiddenLayers] -= originalOutputs;

	int dataSize = originalOutputs.size();
	return Map<VectorXf>(activations[nHiddenLayers].data(), dataSize, 1).norm() / sqrt((float) dataSize);
}




/**
 * use the gradient computed in *costFunc* to perform a step of gradient descent
 */
void Autoencoder::train() {
	int thetaSize = stackTheta.size();
	VectorXf grad = VectorXf::Zero(thetaSize);

	costFunc(grad);

	//grad /= grad.norm();

	//set the learning rates
	VectorXf rates = VectorXf::Zero(thetaSize);

	float * data = rates.data();
	for (int l = nHiddenLayers; l >= 0; l--) {
		float rate = 0.1/layerSizes[l];
		for (int k = 0; k < layerSizes[l+1]*layerSizes[l] + layerSizes[l+1]; k++)
			data[k] = rate;
		data += layerSizes[l+1]*layerSizes[l] + layerSizes[l+1];
	}

	VectorXf lgrad = grad.array() * rates.array();
	stackTheta -= lgrad;
}
 
Autoencoder::Autoencoder(
        trainParams const & inParams,
        MatrixXf * learningSignals,
        MatrixXf * teachingSignals,
        vector<layerConfig> & nnStruct
) {
	nHiddenLayers = inParams.layerSizes.size() - 2;
	layerSizes = inParams.layerSizes;
	iSize = layerSizes[0];
	oSize = layerSizes[nHiddenLayers+1];
	linear = inParams.linear;

	inputs = learningSignals;
	outputs = teachingSignals;

	stack2theta(nnStruct, stackTheta);
}

Autoencoder::Autoencoder(
        trainParams const & inParams,
        MatrixXf * learningSignals,
        MatrixXf * teachingSignals
) {
	nHiddenLayers = inParams.layerSizes.size() - 2;
	layerSizes = inParams.layerSizes;
	iSize = layerSizes[0];
	oSize = layerSizes[nHiddenLayers+1];
	linear = inParams.linear;

	inputs = learningSignals;
	outputs = teachingSignals;

	randInitAE();
}

Autoencoder::~Autoencoder() {
}

/**
 * cascade of *forwardPass* that runs through the whole network
 */
void Autoencoder::predict(
        MatrixXf const & whitenedInputs,
        vector<MatrixXf> & activations
) {
	vector<layerConfig> nnStruct(nHiddenLayers+1);
	theta2stack(nnStruct);

	//feedforward through the neural network
	for (int l = 0; l <= nHiddenLayers; l++) {
		// short aliases
		MatrixXf * out = &activations[l];
		const MatrixXf * A = &nnStruct[l].W;
		const MatrixXf * B = &(l ? activations[l-1] : whitenedInputs);
		const VectorXf * C = &nnStruct[l].b;
		// set output size, if needed
		out->resize(A->rows(), B->cols());
		// out = A * B + C.replicate(1, B.cols());
		mmprc(out->data(), out->rows(), out->cols(),
		      A->data(), B->data(), A->cols(), C->data());

		if (l < nHiddenLayers || !linear)
			// out = tanh(out);
			mtanh(out->data(), out->size());
	}
}

/**
 * compute gradient with backpropagation
 */
float Autoencoder::costFunc(
        VectorXf & grad
) {
	vector<layerConfig> nnStruct(nHiddenLayers+1);
	theta2stack(nnStruct);
	int nPatches = inputs->cols();

	// TODO: check vector length
	vector<MatrixXf> activations(nHiddenLayers+1);
	vector<MatrixXf> dev(nHiddenLayers);
	vector<layerConfig> nnGrad(nHiddenLayers+1);
	vector<MatrixXf> intermediate(nHiddenLayers+1);

	// feedforward pass
	predict(*inputs, activations);

	// derivative
	for (int l=0; l<nHiddenLayers; l++) {
		// short aliases
		MatrixXf * out = &dev[l];
		const MatrixXf * A = &activations[l];
		// set output size if needed
		out->resize(A->rows(), A->cols());
		// out = 1 - A^2
		omsq(out->data(), A->data(), A->size());
	}

	// cost: matching cost
	intermediate[nHiddenLayers] = activations[nHiddenLayers] - outputs[0];
	float cost = Map<VectorXf>(intermediate[nHiddenLayers].data(),
	                           oSize*nPatches, 1).squaredNorm() / nPatches * 0.5;

	// grad
	if (linear)
		intermediate[nHiddenLayers] /= nPatches;
	else
		intermediate[nHiddenLayers] =
		        intermediate[nHiddenLayers].array() *
		        ((float) 1. - activations[nHiddenLayers].array().square()) / nPatches;

	// see the accompanying article for implementation details
	for (int l = nHiddenLayers; l > 0; l--) {
		// short aliases
		MatrixXf * out = &intermediate[l-1];
		const MatrixXf * A = &nnStruct[l].W;
		const MatrixXf * B = &intermediate[l];
		const MatrixXf * C = &dev[l-1];
		// set output size if needed
		out->resize(A->cols(), B->cols());
		// out = (A.transpose() * B).array() * C.array();
		mTma(out->data(), out->rows(), out->cols(),
		     A->data(), B->data(), A->rows(), C->data());
	}

	for (int l = 0; l<=nHiddenLayers; l++) {
		// short aliases
		MatrixXf * out = &nnGrad[l].W;
		const MatrixXf * A = &intermediate[l];
		const MatrixXf * B = &(l ? activations[l-1] : inputs[0]);
		// set output size, if needed
		out->resize(A->rows(), B->rows());
		// out = A * B.transpose();
		mmT(out->data(), out->rows(), out->cols(),
		    A->data(), B->data(), A->cols());
	}

	for (int l = 0; l <= nHiddenLayers; l++) {
		// short aliases
		VectorXf * out = &nnGrad[l].b;
		const MatrixXf * A = &intermediate[l];
		// set output size if needed
		out->resize(A->rows());
		// out = A->rowwise().sum();
		rsum(out->data(), A->data(), A->rows(), A->size());
	}

	//uncomment if you'd like to have the weight decay
	//float alpha = 1e-2;
	//cost += 0.5 * alpha * Map<VectorXf>( nnStruct[nHiddenLayers].W.data(), hiddenSize*oSize, 1 ).squaredNorm();
	//nnGrad[nHiddenLayers].W += alpha * nnStruct[nHiddenLayers].W;

	stack2theta(nnGrad, grad);

	return cost;
}

