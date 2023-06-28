#ifndef BL_H
#define BL_H

#include "Autoencoder.hpp"
#include "imageio.hpp"

//all the parameters fall in the same place
void buildAE(
        patchConfig const & patchSize,
        int const & nHiddenLayers,
        int const & border,
        int const & ratio,
        const char * tPath,  		//training
        const char * vPath,  		//validation
        unsigned int stoopt, 	        //nb rounds
        const char * outfile,	        //output file
        bool const & firstRun
);

#endif
