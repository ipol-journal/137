/**
 * @author Yi-Qing Wang <yiqing.wang@cmla.ens-cachan.fr>
 * All rights reserved.
 */

#include "global.hpp"

#include <iostream>
#include <fstream>

#include <omp.h>

#include "matops.hpp"
#include "io_png.h"
#include "prng.hpp"	//external random number generator


#include "WStruct.hpp"
#include "imageio.hpp" // self-consistency

using namespace std;
using namespace Eigen;

MatrixXf * imread(
        const char * fileName,
        int & nRows,
        int & nCols,
        int & nChannels
) {
	size_t nx, ny, nc;
	float * pixelStream = read_png_f32(fileName, &nx, &ny, &nc);
	if (pixelStream == NULL)
		ERROR("Unable to get %s", fileName);

	//return these parameters
	nCols = (int)nx;
	nRows = (int)ny;
	nChannels = (int)nc;
	DEBUG("read in %s of dimension: %lu , %lu , %i",
	      fileName, ny, nx, nChannels);

	//input stream assumes row-major while Eigen defaults to column-major
	Map<MatrixXf> parallel(pixelStream, nCols, nRows*nChannels);
	MatrixXf * image = new MatrixXf [nChannels];
	for (int ch = 0; ch < nChannels; ch++)
		image[ch] = parallel.block(0, ch*nRows, nCols, nRows).transpose();

	//release
	free(pixelStream);

	return image;
}

void imwrite(
        const char * fileName,
        MatrixXf * image,
        int nChannels
) {
	//allocate
	int nCols = image->cols();
	int nRows = image->rows();
	int pixelsPerChannel = nCols*nRows;
	float * output = new float [pixelsPerChannel*nChannels];

	//this part should be straightforward but still be careful with the order
	for (int j = 0; j < nCols; j++)
		for (int i = 0; i < nRows; i++)
			for (int ch = 0; ch < nChannels; ch++)
				output[ch*pixelsPerChannel + i*nCols + j] = image[ch](i,j);

	//output image
	write_png_f32(fileName, output, (size_t) nCols, (size_t) nRows, (size_t) nChannels);
	INFO("write %s to the local folder", fileName);

	//release
	delete [] output;
	delete [] image;
}


void display(
        MatrixXf const & data,
        int nPieces,
        patchConfig const & patchSize,
        const char * pic
) {
	assert(nPieces <= data.cols());

	//output images always have 10 columns
	int dispNcols = 24;
	int dispNrows = nPieces % dispNcols == 0 ? nPieces / dispNcols : int(nPieces / dispNcols) + 1;

	int padding = 1;
	//allocate for output
	MatrixXf * output = new MatrixXf [3];
	for (int ch = 0; ch < 3; ch++)
		output[ch] = MatrixXf::Constant(dispNrows * (patchSize.rows + padding) + padding,
		                                dispNcols * (patchSize.cols + padding) + padding,
		                                255);

	//fill
	int nRows = data.rows();
	int unit = patchSize.rows * patchSize.cols;
	int isColor = nRows == unit ? 0 : 1;
	for (int p = 0; p < nPieces; p++) {
		VectorXf pvec = data.col(p).cast<float>();
		pvec -= VectorXf::Constant(nRows, pvec.minCoeff());
		pvec *= 255 / pvec.maxCoeff();
		int row = p / dispNcols;
		int col = p % dispNcols;
		for (int ch = 0; ch < 3; ch++) {
			VectorXf pvech = pvec.segment(ch * isColor * unit, unit);
			output[ch].block(padding + row * (padding + patchSize.rows),
			                 padding + col * (padding + patchSize.cols),
			                 patchSize.rows, patchSize.cols) = Map<MatrixXf>(pvech.data(), patchSize.rows, patchSize.cols);
		}
	}

	//out
	imwrite(pic, output, 3);
}


static int pathFile2charArray(
        const char * pathFile,
        char **& addr
) {
	//initialize the returned value
	int pathCount = 0;

	for (int loop = 0; loop < 2; loop++) {

		ifstream myfile(pathFile);

		if (!myfile.is_open())
			ERROR("Unable to open %s", pathFile);

		string line;

		//an offset to make the line count (total) right
		int total = -1;

		while (myfile.good()) {

			//count lines
			getline(myfile, line);
			total += 1;

			//only do this at the second loop
			if (loop == 1 && total < pathCount) {
				//convert string to char array
				addr[total] = new char [line.size()+1];
				addr[total][line.size()] = 0;
				memcpy(addr[total], line.c_str(), line.size());
			}
		}

		myfile.close();

		//first loop just count the number of lines in the file
		if (loop == 0) {

			pathCount = total;

			if (pathCount < 1)
				ERROR("Empty %s", pathFile);

			//allocate
			addr = new char * [pathCount];
		}
	}

	return pathCount;

}

static void readImagesFromPathFile(
        const char * pathFile,
        vector<MatrixXf *> & images,
        int expectedNChannels
) {
	char ** addr = NULL;
	int nImages = pathFile2charArray(pathFile, addr);
	images.resize(nImages);

	int nRows, nCols, nChannels;
	for (int im = 0; im < nImages; im++) {
		const char * file = addr[im];
		images[im] = imread(file, nRows, nCols, nChannels);
		if (nChannels != expectedNChannels) {
			const char * type = nChannels == 1 ? "grayscale" : "RGB";
			ERROR("Detect a %s %s", type, file);
		}
	}

	//release
	for (int im = 0; im < nImages; im++)
		delete [] addr[im];
	delete [] addr;
}

/**
 * draw patches at random
 */
static void gatherPatches(
        MatrixXf & patches,
        patchConfig const & patchSize,
        int nPatches,
        vector<MatrixXf *> & images,
        int nChannels
) {
	int unit = patchSize.rows * patchSize.cols;
	patches.resize(unit*nChannels, nPatches);

	static bool firstrun = true;
	// initialize random state(s)
	static prng_state_s * state_omp; //one per thread
	if (firstrun) {
		int maxt = omp_get_max_threads();
		state_omp = new prng_state_s[maxt+1];
		for (int t = 0; t < maxt+1; t++) {
			state_omp[t] = prng_new_state();
			if (flag_random_seed)
				prng_init_auto(state_omp[t]);
		}
		firstrun = false;
	}

	DEBUG("ready to draw %i random patches of size %i x %i from images",
	      nPatches, patchSize, patchSize);

	int nImages = images.size();

	#pragma omp parallel
	{
		// TODO : out of parallel block, use "private/shared"
		float * patches_ = patches.data();
		size_t rows = patches.rows();
		// get state
		int t = omp_get_thread_num();
		prng_state_s state = state_omp[t];
		#pragma omp for schedule(static)
		for (int p = 0; p < nPatches; p++) {
			// TODO: provide prng_lessthan(int) to avoid unneeded int->float->int
			int imi = prng_unif(state) * nImages;
			int sr = (images[imi]->rows() - patchSize.rows) * prng_unif(state);
			int sc = (images[imi]->cols() - patchSize.cols) * prng_unif(state);
			DEBUG("patch %i : %i, %i, %i", p, imi, sr, sc);
			for (int ch = 0; ch < nChannels; ch++) {
				MatrixXf patch = images[imi][ch].block(sr, sc, patchSize.rows, patchSize.cols);
				float * patch_ = patch.data();
				for (int i= 0; i < unit; i++)
					patches_[p * rows + ch * unit + i] = (float) patch_[i];
			}
		}
		// save state
		state_omp[t] = state;
	}
}

static int calcNumPatches(
        int nPixels,
        int patchSize,
        int step
) {
//	it holds that for some k, nPixels = patchSize + step * k + something
//	with something = 0 to k-1
	int something = (nPixels - patchSize) % step;
	int k = (nPixels - something - patchSize)/step;
	return k + 1;
}

MatrixXf * image2patches(
        MatrixXf * image,
        int stride,
        patchConfig const & patchSize,
        int nChannels
) {

	int nRows = image->rows();
	int nCols = image->cols();

	//how many patches to draw from this image
	int rows = calcNumPatches(nRows, patchSize.rows, stride);
	int cols = calcNumPatches(nCols, patchSize.cols, stride);

	//constraints on patch position
	int row = -1 * patchSize.rows;
	int col = -1 * patchSize.cols;
	int maxRow = nRows - patchSize.rows;
	int maxCol = nCols - patchSize.cols;

	//practical stuff
	MatrixXf * patches = new MatrixXf [1];
	int unit = patchSize.rows * patchSize.cols;
	patches[0] = MatrixXf::Zero(unit*nChannels, rows*cols);

	int counter = 0;
	for (int i = 0; i < rows; i++) {
		row = max(0, min(maxRow, row + stride));
		col = -1 * patchSize.cols;
		for (int j = 0; j < cols; j++) {
			col = max(0, min(maxCol, col + stride));
			MatrixXf patch = MatrixXf::Zero(patchSize.rows, patchSize.cols);
			VectorXf cPatch = VectorXf::Zero(unit*nChannels);
			for (int ch = 0; ch < nChannels; ch++) {
				patch = image[ch].block(row, col, patchSize.rows, patchSize.cols);
				cPatch.segment(unit*ch, unit) = Map<VectorXf>(patch.data(), unit, 1).cast<float>();
			}
			patches->col(counter++) = cPatch;
		}
	}

	return patches;
}

MatrixXf * patches2image(
        MatrixXf * patches,
        int nRows,
        int nCols,
        int stride,
        patchConfig const & patchSize,
        int nChannels
) {
	MatrixXf * image = new MatrixXf [nChannels];
	for (int ch = 0; ch < nChannels; ch++)
		image[ch] = MatrixXf::Zero(nRows, nCols);

	//count how many times each pixel has been covered by some patch
	MatrixXi mask = MatrixXi::Zero(nRows, nCols);

	int rows = calcNumPatches(nRows, patchSize.rows, stride);
	int cols = calcNumPatches(nCols, patchSize.cols, stride);

	//constraints on patch position
	int col, row = -1 * patchSize.rows;
	int maxRow = nRows - patchSize.rows;
	int maxCol = nCols - patchSize.cols;

	int counter = 0;
	int unit = patchSize.rows * patchSize.cols;
	for (int i = 0; i < rows; i++) {
		row = max(0, min(maxRow,  row + stride));
		col = -1 * patchSize.cols;
		for (int j = 0; j < cols; j++) {
			col = max(0, min(maxCol, col + stride));
			VectorXf cPatch = patches->col(counter++);
			for (int ch = 0; ch < nChannels; ch++) {
				VectorXf patch = cPatch.segment(unit*ch, unit).cast<float>();
				image[ch].block(row, col, patchSize.rows, patchSize.cols) += Map<MatrixXf>(patch.data(), patchSize.rows, patchSize.cols);
			}
			mask.block(row, col, patchSize.rows, patchSize.cols) += MatrixXi::Ones(patchSize.rows, patchSize.cols);
		}
	}

	for (int j = 0; j < nCols; j++)
		for (int i = 0; i < nRows; i++)
			for (int ch = 0; ch < nChannels; ch++)
				image[ch](i, j) = (mask(i, j) == 0
				                   ? 0
				                   : image[ch](i, j) / mask(i, j));

	return image;
}


void drawPatches(
        MatrixXf & patches,
        const char * path,
        int nPatches,
        patchConfig const & patchSize,
        vector<MatrixXf *> & images,
        int nChannels
) {
	TIMING_TOGGLE(TIMER_PATCH);
	//gather images if necessary
	if (images.size() == 0) {
		readImagesFromPathFile(path, images, nChannels);
		INFO("read in %lu images in all from %s", images.size(), path);
	}

	gatherPatches(patches, patchSize, nPatches, images, nChannels);
	TIMING_TOGGLE(TIMER_PATCH);
}


void crop(
        const MatrixXf & input,
        MatrixXf & output,
        int border,
        patchConfig const & patchSize,
        int nChannels
) {
	size_t nPatches = input.cols();
	output.resize((patchSize.rows-2*border)*(patchSize.cols-2*border)*nChannels, nPatches);
	const float * in_data = input.data();
	float * out_data = output.data();

	size_t offset = border * patchSize.rows + border;

	TIMING_TOGGLE(TIMER_CROP);
	if (border == 0)
		// no border: shortcut to a single copy
		memcpy(out_data, in_data, input.size()*sizeof(float));
	else
		// loop over patches and channels
		for (size_t p = 0; p < nPatches * nChannels; p++) {
			const float * in_ptr = in_data + p * patchSize.rows * patchSize.cols;
			float * out_ptr = out_data + p * (patchSize.rows - 2*border) * (patchSize.cols - 2*border);
			// loop over patch lines and copy
			for (size_t k = 0; k < patchSize.cols - 2*border; k++)
				memcpy(out_ptr + k * (patchSize.rows - 2*border),
				       in_ptr + k * patchSize.rows + offset,
				       (patchSize.rows - 2*border) * sizeof(float));
		}
	TIMING_TOGGLE(TIMER_CROP);
}

void mosaicRGGB(
        MatrixXf & full,
        MatrixXf & mosaiced,
        MatrixXf & teaching,
        patchConfig const & patchSize,
        bool inverse
) {
	int unit = patchSize.rows * patchSize.cols;
	int dSize = unit * 3;
	int nPieces = inverse ? mosaiced.cols() : full.cols();

	if (inverse)
		full.resize(dSize, nPieces);
	else {
		mosaiced.resize(unit, nPieces);
		teaching.resize(2*unit, nPieces);
	}

	float * in_data = full.data();
	float * mdata = mosaiced.data();
	float * tdata = teaching.data();

	TIMING_TOGGLE(TIMER_RGGB);
	if (!inverse) {
		#pragma omp parallel for schedule(static)
		for (int p = 0; p < nPieces; p++) {

			float * in_ptr = in_data + p * dSize;
			float * tptr = tdata + p * 2 * unit;
			float * mptr = mdata + p * unit;

			for (int k = 0; k < patchSize.cols; k+=2) {
				for (int r = 0; r < patchSize.rows; r+=2) {
					mptr[k*patchSize.rows+r] = in_ptr[k*patchSize.rows+r];
					tptr[k*2*patchSize.rows+2*r] = in_ptr[k*patchSize.rows+r + unit];
					tptr[k*2*patchSize.rows+2*r+1] = in_ptr[k*patchSize.rows+r + 2*unit];
				}
				for (int r = 1; r < patchSize.rows; r+=2) {
					mptr[k*patchSize.rows+r] = in_ptr[k*patchSize.rows+r + unit];
					tptr[k*2*patchSize.rows+2*r] = in_ptr[k*patchSize.rows+r];
					tptr[k*2*patchSize.rows+2*r+1] = in_ptr[k*patchSize.rows+r + 2*unit];
				}
			}

			for (int k = 1; k < patchSize.cols; k+=2) {
				for (int r = 0; r < patchSize.rows; r+=2) {
					mptr[k*patchSize.rows+r] = in_ptr[k*patchSize.rows+r + unit];
					tptr[k*2*patchSize.rows+2*r] = in_ptr[k*patchSize.rows+r];
					tptr[k*2*patchSize.rows+2*r+1] = in_ptr[k*patchSize.rows+r + 2*unit];
				}
				for (int r = 1; r < patchSize.rows; r+=2) {
					mptr[k*patchSize.rows+r] = in_ptr[k*patchSize.rows+r + 2*unit];
					tptr[k*2*patchSize.rows+2*r] = in_ptr[k*patchSize.rows+r];
					tptr[k*2*patchSize.rows+2*r+1] = in_ptr[k*patchSize.rows+r + unit];
				}
			}
		}
	} else {
		#pragma omp parallel for schedule(static)
		for (int p = 0; p < nPieces; p++) {

			float * in_ptr = in_data + p * dSize;
			float * tptr = tdata + p * 2 * unit;
			float * mptr = mdata + p * unit;

			for (int k = 0; k < patchSize.cols; k+=2) {
				for (int r = 0; r < patchSize.rows; r+=2) {
					in_ptr[k*patchSize.rows+r] = mptr[k*patchSize.rows+r];
					in_ptr[k*patchSize.rows+r + unit] = tptr[k*2*patchSize.rows+2*r];
					in_ptr[k*patchSize.rows+r + 2*unit] = tptr[k*2*patchSize.rows+2*r+1];
				}
				for (int r = 1; r < patchSize.rows; r+=2) {
					in_ptr[k*patchSize.rows+r + unit] = mptr[k*patchSize.rows+r];
					in_ptr[k*patchSize.rows+r] = tptr[k*2*patchSize.rows+2*r];
					in_ptr[k*patchSize.rows+r + 2*unit] = tptr[k*2*patchSize.rows+2*r+1];
				}
			}

			for (int k = 1; k < patchSize.cols; k+=2) {
				for (int r = 0; r < patchSize.rows; r+=2) {
					in_ptr[k*patchSize.rows+r + unit] = mptr[k*patchSize.rows+r];
					in_ptr[k*patchSize.rows+r] = tptr[k*2*patchSize.rows+2*r];
					in_ptr[k*patchSize.rows+r + 2*unit] = tptr[k*2*patchSize.rows+2*r+1];
				}
				for (int r = 1; r < patchSize.rows; r+=2) {
					in_ptr[k*patchSize.rows+r + 2*unit] = mptr[k*patchSize.rows+r];
					in_ptr[k*patchSize.rows+r] = tptr[k*2*patchSize.rows+2*r];
					in_ptr[k*patchSize.rows+r + unit] = tptr[k*2*patchSize.rows+2*r+1];
				}
			}
		}
	}
	TIMING_TOGGLE(TIMER_RGGB);
}

void applyCFA(
        MatrixXf * noisy,
        int CFA
) {
	int nRows = noisy->rows();
	int nCols = noisy->cols();

	float * R = noisy[0].data();
	float * G = noisy[1].data();
	float * B = noisy[2].data();

	for (int k = 0; k < nCols; k+=2)
		for (int r = 0; r < nRows; r+=2) {
			if (CFA == RGGB) {
				G[k*nRows+r] = 0;
				B[k*nRows+r] = 0;
			} else if (CFA == BGGR) {
				G[k*nRows+r] = 0;
				R[k*nRows+r] = 0;
			} else {
				B[k*nRows+r] = 0;
				R[k*nRows+r] = 0;
			}
		}

	for (int k = 0; k < nCols; k+=2)
		for (int r = 1; r < nRows; r+=2) {
			if (CFA == GBRG) {
				G[k*nRows+r] = 0;
				B[k*nRows+r] = 0;
			} else if (CFA == GRBG) {
				G[k*nRows+r] = 0;
				R[k*nRows+r] = 0;
			} else {
				B[k*nRows+r] = 0;
				R[k*nRows+r] = 0;
			}
		}

	for (int k = 1; k < nCols; k+=2)
		for (int r = 0; r < nRows; r+=2) {
			if (CFA == GRBG) {
				G[k*nRows+r] = 0;
				B[k*nRows+r] = 0;
			} else if (CFA == GBRG) {
				G[k*nRows+r] = 0;
				R[k*nRows+r] = 0;
			} else {
				B[k*nRows+r] = 0;
				R[k*nRows+r] = 0;
			}
		}

	for (int k = 1; k < nCols; k+=2)
		for (int r = 1; r < nRows; r+=2) {
			if (CFA == BGGR) {
				G[k*nRows+r] = 0;
				B[k*nRows+r] = 0;
			} else if (CFA == RGGB) {
				G[k*nRows+r] = 0;
				R[k*nRows+r] = 0;
			} else {
				B[k*nRows+r] = 0;
				R[k*nRows+r] = 0;
			}
		}
}

void rotate2RGGB(
        MatrixXf * input,
        int pattern
) {
	if (pattern == RGGB)
		return;

	int dataSize = input->rows();
	int unit = dataSize/3;
	int patchSize = sqrt(unit);
	int nPieces = input->cols();

	MatrixXf out;
	out.resize(dataSize, nPieces);
	float * out_ptr = out.data();
	float * in_ptr = input->data();

	for (int p = 0; p < nPieces; p++) {
		for (int i = 0; i < dataSize; i++) {
			int j = i % unit;
			int c = i / unit;
			int r = j % patchSize;
			int k = j / patchSize;
			if (pattern == GRBG)
				out_ptr[(patchSize - 1 - k) + patchSize * r + c * unit ] = in_ptr[i];
			else if (pattern == BGGR)
				out_ptr[(patchSize - 1 - r) + patchSize * (patchSize - 1 - k) + c * unit] = in_ptr[i];
			else
				out_ptr[k + patchSize * (patchSize - 1 - r) + c * unit] = in_ptr[i];
		}
		in_ptr += dataSize;
		out_ptr += dataSize;
	}

	memcpy(input->data(), out.data(), out.size()*sizeof(float));
}

void ZCA(
        const MatrixXf & in,
        MatrixXf & out,
        float filter
) {
	static bool flag = true;
	static const float * mean;
	static MatrixXf op;

	if (flag) {

		// set up the whitening operator

		MatrixXf U;
		int dataSize = in.rows();
		U.resize(dataSize, dataSize);
		memcpy(U.data(), WStruct::getBasis(), U.size()*sizeof(float));
		mean = WStruct::getMean();
		float * vals = WStruct::getVals();

		MatrixXf A = U;
		float * ptr = A.data();
		for (int c = 0; c < dataSize; c++) {
			float ratio = 1./sqrt(vals[c]+filter);
			for (int r = 0; r < dataSize; r++)
				ptr[r] *= ratio;
			ptr += dataSize;
		}

		op.resize(U.rows(), A.rows());
		mmT(op.data(), op.rows(), op.cols(),
		    U.data(), A.data(), U.cols());
		flag = false;

	}

	// mean removal

	MatrixXf inter;
	int rows = in.rows();
	int cols = in.cols();
	inter.resize(rows, cols);
	memcpy(inter.data(), in.data(), in.size()*sizeof(float));

	#pragma omp parallel for schedule(static)
	for (size_t c = 0; c < cols; c++) {
		// loop on columns
		float * out_c = inter.data() + c * rows;
		for (size_t r = 0; r < rows; r++)
			out_c[r] -= mean[r];
	}

	out.resize(op.cols(), inter.cols());
	mTm(out.data(), out.rows(), out.cols(),
	    op.data(), inter.data(), op.rows());

}
