# Copyright 2014 Yi-Qing Wang <yiqing.wang@cmla.ens-cachan.fr>
#                Nicolas Limare <nicolas.limare@cmla.ens-cachan.fr>
#
# Copying and distribution of this file, with or without modification,
# are permitted in any medium without royalty provided the copyright
# notice and this notice are preserved. This file is offered as-is,
# without any warranty.

CXXSRC		= $(wildcard *.cpp)
CSRC		= $(wildcard *.c)
CXXHDR		= $(wildcard *.hpp)
CHDR		= $(wildcard *.h)

CXXOBJ		= $(CXXSRC:.cpp=.o)
COBJ		= $(CSRC:.c=.o)

CPPFLAGS	=
COPT		= -O2 -march=native -ftree-vectorize
CDBG		= -g
CFLAGS		= $(COPT) $(CDBG) -fopenmp
CXXFLAGS	= $(COPT) $(CDBG) -fopenmp
LDFLAGS		= -lgomp

default: all

ifdef WITH_TIMING
CPPFLAGS	+= -DUSE_TIMING
LDFLAGS		+= -lrt
endif

ifdef WITH_LOG_INFO
CPPFLAGS	+= -DLOGLEVEL=2
endif
ifdef WITH_LOG_DEBUG
CPPFLAGS	+= -DLOGLEVEL=3
endif

include makefile.blas

LDFLAGS		+= -lpng -lm

all: train demosaick

# generic object compilation

%.o: %.c
	$(CC) -c $< $(CPPFLAGS) $(CFLAGS) -o $@

%.o: %.cpp
	$(CHECK_MKLROOT)
	$(CXX) -c $< $(CPPFLAGS) $(CXXFLAGS) -o $@


# executable targets

train: 	Autoencoder.o NeuralStruct.o imageio.o build.o global.o WStruct.o \
		 check_gradient.o sample.o matops.o io_png.o timing.o train.o
	$(CXX) $^ $(LDFLAGS) -o $@

demosaick: Autoencoder.o NeuralStruct.o imageio.o appl.o global.o WStruct.o \
		matops.o io_png.o timing.o demosaick.o
	$(CXX) $^ $(LDFLAGS) -o $@

# doc

%.pdf: %.txt
	pandoc --from markdown \
		--standalone --normalize --smart --parse-raw \
		$< -o $@

# special targets

.PHONY:	clean distclean eigen
clean:
	$(RM) $(CXXOBJ) $(COBJ) core
distclean: clean
	$(RM) train demosaick misc/rgb2gray
	$(RM) test/timing test/timing.o test/tanh test/tanh.o test/prng test/prng.o test/tmp/out.txt
	$(RM) README.pdf
	$(RM) inputFeatures.png outputFeatures.png testin.png testout.png teaching.png noisy.png context.png
	$(RM) *.pprof *.perf *.bak *.aux *.log

# install Eigen library locally
EIGEN_DIR	= eigen3
EIGEN_TGZ	= eigen-3.2.10.tar.gz
EIGEN_URL	= https://ipolcore.ipol.im/static/jyotsna/eigen-3.2.10.tar.gz

eigen: $(EIGEN_DIR)
$(EIGEN_TGZ):
	wget -q -O $@ $(EIGEN_URL)

$(EIGEN_DIR):	$(EIGEN_TGZ)
	$(RM) -r $@
	mkdir $@
	tar xfz $< -C $@ --strip-components=1

# code dependencies

makefile.dep: $(CSRC) $(CXXSRC) $(CHDR) $(CXXHDR)
	touch $@
	makedepend -Y -f $@ $(sort $(CSRC) $(CXXSRC)) 2>/dev/null

-include makefile.dep
