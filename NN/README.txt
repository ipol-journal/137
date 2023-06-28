__usage__

* train

	syntax:
	    train -h
	    train -g
	    train -w [-p SIZE] -t LIST -o OUTPUT
	    train [-f] [-r RATIO] [-d DEPTH] [-p ROWSIZE COLSIZE] [-b BORDER] [-n NB] -t LIST -v LIST -o OUTPUT [-R]

	options:
	    -h print this help
	    -g gradient checking
	    -w create the whitening operator
	    -f training from scratch (default false)
	    -r size ratio between a hidden layer and the input (default 2)
	    -d number of hidden layers (default 2)
	    -p training patch size (default 6 6)
	    -b border cropped out from teaching signals (default 2)
	    -n number of training rounds (default 1e7)
	    -t list of training images
	    -v list of validation images
	    -o output file
	    -R reference mode for debugging:
	        no pseudo-random number generator initialization
	        random network initialization
	        RMSE compared at every round
	        no partial result saved
	        unformatted output on stdout

* demosaick

	syntax:
           demosaick -h
           demosaick -c clean [-p pattern] [-s stride] [-m mosaiced] [-d demosaicked]

	options:
	    -h print this help
	    -c path to a clean input image
	    -p Bayer color filter array pattern 0: RGGB, 1: GRBG, 2: BGGR or 3: GBRG (default 0)
	    -s stride between two neighboring sliding windows (default 1)
	    -m path to store the mosaiced image (default ./mosaiced.png)
	    -d path to store the demosaicked image (default ./demosaiced.png)

__dependencies__

* Eigen linear algebra library, versions 3. If it is not installed on
  your system (`sudo aptitude install libeigen3-dev` on Debian and
  derivatives), you can download and extract it locally with `make
  eigen`.
* BLAS with CBLAS interface. This code has been tested with Netlib
  BLAS, Atlas, OpenBLAS and Intel MKL.
* OpenMP-aware C/C++ compiler (gcc, icc and msvc support OpenMP).

__compilation__

Compilation can be achieved with the makefile provided with the source
code. Essential `make` targets are:
* make           : build the `./train` and `./demosaick` binaries
* make eigen     : download and extract locally the Eigen library
* make clean     : remove the compilation by-products
* make distclean : restore the source tree to its original state

Compilation can be configured with `make` options, which can be used
together. Essential options:
* WITH_BLAS=xxx : Choose your BLAS libraries, precised as
                  "xxx". Currently known implementations are Intel MKL
                  (WITH_BLAS=mkl), Atlas (WITH_BLAS=atlas), OpenBLAS
          (WITH_BLAS=open) and GSL (WITH_BLAS=gsl). Default is
          to use generic library names.
* WITH_TIMING=1 : Add timing information (but impacts the performance)

__source code description__

* Autoencoder.cpp     neural network (NN) class
* check_gradient.cpp  numerical validation of the backpropagation
* NeuralStruct.cpp    a NN previously learned
* build.cpp           wrapper function for the NN class and its data provider
* train.cpp           executable to launch the backpropagation
* demosaick.cpp       executable to use the learned NN
* appl.cpp            neural network demosaicking routine
* run.sh              a bash script that launches a learning process
* eigen.hpp           inclusion of Eigen and version number validation
* global.cpp          misc. global variables and macros
* prng.hpp            random number generator
* imageio.cpp         image handling functions
* io_png.c            png image input/output
* matops.cpp          matrix-oriented operations
* tanh.hpp            fast tanh() approximation
* sample.cpp          function for setting the data whitening operator
