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

install: python-packages theano pylearn2 wyrm lasagne scikits-samplerate-pip braindecode

python-packages:
	pip install --upgrade pip $(PIP_FLAG)
	pip install numpy scipy matplotlib scikit-learn pytest h5py jupyter seaborn $(PIP_FLAG)

theano:
	pip install git+https://github.com/Theano/Theano.git@15c90dd3#egg=Theano==0.8.git $(PIP_FLAG)

pylearn2:
	pip install -e git+https://github.com/lisa-lab/pylearn2.git@8bd3cc2ecd4062b425d938d68024276592bce1a7#egg=pylearn2-master $(PIP_FLAG) --src pylearn2

wyrm:
	pip install -e git+https://github.com/bbci/wyrm.git@e976e500914cce720a659025c18efc338b408721#egg=Wyrm-master $(PIP_FLAG) --src wyrm

lasagne:
	pip install git+https://github.com/Lasagne/Lasagne.git@6dd88f5fada20768087f29ae89cbd83980fe0a4e $(PIP_FLAG)

scikits-samplerate-pip:
	(test -e ~/.numpy-site.cfg && grep -q 'samplerate' ~/.numpy-site.cfg) || echo "$$NUMPY_SITE_CFG" >> ~/.numpy-site.cfg
	# check first if file exists and already has samplerate info
	pip install scikits.samplerate $(PIP_FLAG)

braindecode:
	python setup.py develop $(PIP_FLAG)
