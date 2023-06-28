/**
 * @author Yi-Qing Wang <yiqing.wang@cmla.ens-cachan.fr>
 * All rights reserved.
 */

/*
 * The backpropagation algorithm shows how to compute the
 * gradient in a closed form. To ensure its correct numerical
 * implementation, the routine gradientChecking() calculates the same
 * gradient using its elementary definition for a double checking.
 */

#include "global.hpp"

#include <iostream>
#include "Autoencoder.hpp"

#include "prng.hpp"	// external random number generator

#include "check_gradient.hpp"	// self-consistency

using namespace std;

bool gradientChecking() {

	float epsilon = 1e-2;
	float max_agreement = 1e-3;
	bool test_passed = true;

	if (flag_random_seed)
		prng_init_auto();

	for (int nHiddenLayers = 1; nHiddenLayers < 5; nHiddenLayers++) {

		// TODO: provide prng_lessthan(int) to avoid unneeded int->float->int
		int hiddenSize = 18 + prng_unif() * 6;
		int iSize = 8 + prng_unif() * 6;
		int oSize = 16 + prng_unif() * 4;
		int nPatches = 1e2 + prng_unif() * 100;

		trainParams testParams;
		testParams.layerSizes.resize(nHiddenLayers+2);
		testParams.layerSizes[0] = iSize;
		for (int l = 1; l < nHiddenLayers + 1; l++)
			testParams.layerSizes[l] = hiddenSize;
		testParams.layerSizes[nHiddenLayers+1] = oSize;
		testParams.linear = true;

		std::cout.precision(5);
		cout << std::scientific;

		MatrixXf learningSignals = MatrixXf::Random(iSize, nPatches);
		MatrixXf teachingSignals = MatrixXf::Random(oSize, nPatches);

		Autoencoder encoder(testParams, &learningSignals, &teachingSignals);
		VectorXf theta = encoder.stackTheta;
		int thetaSize = encoder.stackTheta.size();

		for (int t = 0; t < 2; t++) {
			//compute numerical gradient
			VectorXf grad = VectorXf::Zero(thetaSize);
			VectorXf nGrad = VectorXf::Zero(thetaSize);
			for (int k = 0; k < thetaSize; k++) {
				VectorXf increment = VectorXf::Zero(thetaSize);
				increment[k] = epsilon;
				encoder.stackTheta = theta - increment;
				float costLeft = encoder.costFunc(grad);
				encoder.stackTheta = theta + increment;
				float costRight = encoder.costFunc(grad);
				nGrad[k] = (costRight - costLeft) / (2 * epsilon);
			}
			encoder.stackTheta = theta;
			cout << "cost is " << encoder.costFunc(grad) << endl;
			float agreement = (nGrad - grad).norm() / (nGrad + grad).norm();
			cout << "the agreement when linear = " << encoder.linear << " and nHiddenLayers = " << nHiddenLayers << " is " << agreement << endl;
			//	assert( agreement < 1e-8 );
			encoder.linear = false;
			test_passed = test_passed && (agreement < max_agreement);
		}
	}

	return test_passed;
}

