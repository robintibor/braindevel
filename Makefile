PIP_FLAG = #--user
MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# if installing by hand, this should be put in file .numpy-site.cfg before installing scikits.samplerate
# with pip

define NUMPY_SITE_CFG

[samplerate]
library_dirs = $(MAKEFILE_DIR)/libsamplerate/lib/
include_dirs = $(MAKEFILE_DIR)/libsamplerate/include/

endef
export NUMPY_SITE_CFG

#put into readme or install.md
#clone-repository:
#	git clone https://robintibor@bitbucket.org/robintibor/machine-learning-for-motor-imagery.git

requirements: libs scikits-samplerate

libs:
	sudo apt-get install liblas-dev liblapack-dev gfortran libyaml-dev cython\
		libfreetype6-dev
	
scikits-samplerate:
	wget http://www.mega-nerd.com/SRC/libsamplerate-0.1.8.tar.gz
	tar -xvf libsamplerate-0.1.8.tar.gz
	cd libsamplerate-0.1.8 && \
	./configure --prefix=$(MAKEFILE_DIR)/libsamplerate/ && \
	make && \
	make install && \
	ldconfig -v

install: python-packages theano pylearn2 wyrm scikits-samplerate-pip

python-packages:
	pip install numpy scipy matplotlib scikit-learn pytest h5py $(PIP_FLAG)

theano:
	pip install git+git@github.com:Theano/Theano.git@rel-0.7rc2 $(PIP_FLAG)

pylearn2:
	pip install git+git@github.com:lisa-lab/pylearn2.git@8bd3cc2ecd4062b425d938d68024276592bce1a7 $(PIP_FLAG)

wyrm:
	pip install git+https://github.com/bbci/wyrm.git@e976e500914cce720a659025c18efc338b408721 $(PIP_FLAG)

scikits-samplerate-pip:
	(test -e ~/.numpy-site.cfg && grep -q 'samplerate' ~/.numpy-site.cfg) || echo "$$NUMPY_SITE_CFG" >> ~/.numpy-site.cfg
	# check first if file exists and already has samplerate info
	pip install scikits.samplerate $(PIP_FLAG)
