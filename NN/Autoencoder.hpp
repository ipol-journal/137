#ifndef AE_H
#define AE_H

#include "global.hpp"

#include <vector>
#include "eigen.hpp"

using namespace std;
using namespace Eigen;

//each layer's definition

/**
 * a layer in NN is characterized by a connection weight matrix and a
 * bias vector, wrapped up here in a single structure
 */
struct layerConfig {
	MatrixXf W;
	VectorXf b;
};

//layer training parameters

/**
 * the previously described parameters that dictate the structure and
 * training objective of a NN
 */
struct trainParams {
	vector<int> layerSizes;
	int sigma;
	bool linear;
};

//AE class

/**
 * neural network (NN) class
 */
class Autoencoder {
public:

	int iSize; /** iSize NN's input is of dimension iSize */
	vector<int> layerSizes; /** number of units on all the layers */
	int oSize; /** NN's output is of dimension oSize */
	int nHiddenLayers; /** NN has *nHiddenLayers* layers */
	bool linear; /** NN's last layer is linear or not */

	//input and output
	MatrixXf * inputs; /** the NN's supervised examples */
	MatrixXf * outputs; /** the NN's supervised examples */

	//stack's definition
	VectorXf stackTheta; /**
			      * the connection weights and biases that define the
			      * NN. It may be represented as a string
			      * of matrices which comes handy
			      * when used to run the
			      * feedforward. Otherwise, it is in the
			      * form of a single column vector,
			      * convenient for doing gradient descent.
			      */

	//constructor 1
	Autoencoder(
	        trainParams const & inParams,
	        MatrixXf * learningSignals,
	        MatrixXf * teachingSignals,
	        vector<layerConfig> & nnStruct
	);

	//constructor 2
	Autoencoder(
	        trainParams const & inParams,
	        MatrixXf * learningSignals,
	        MatrixXf * teachingSignals
	);

	~Autoencoder();

	//give AE a random initialisation
	void randInitAE();

	//the neural network function
	void predict(
	        MatrixXf const & whitenedInputs,
	        vector<MatrixXf> & activations
	);

	float calcRMSE(
	        MatrixXf const & originalOutputs,
	        MatrixXf const & whitenedInputs
	);

	void train();

	void theta2stack(
	        vector<layerConfig> & nnStruct
	) const;

	void stack2theta(
	        vector<layerConfig> & nnStruct,
	        VectorXf & theta
	) const;

	float costFunc(
	        VectorXf & grad
	);
};

#endif
