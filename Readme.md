# Installation

## Basics
If you don't have pip and/or git installed
```
sudo apt-get install python-pip git
```

## Clone the repository
```
git clone https://github.com/robintibor/braindecode.git
```

## Now make the requirements

```
cd braindecode
make requirements
```

Ignore this error:
```
/sbin/ldconfig.real: Can't create temporary cache file /etc/ld.so.cache~: Permission denied
make: *** [scikits-samplerate] Error 1
```

## Install Python packages

The following can be done in or outside a virtualenv. Make sure to have it activated if you want to use a virtualenv. The following installation steps can take quite long, even above an hour.


### Option 1 (with Makefile):
```
make install
```

or if you want to install with user flag for pip:


```
make install PIP_FLAG=--user
```

### Option 2 (with requirements.txt):

```
pip install -r requirements.txt
make scikits-samplerate-pip

```

## Test installation

add the repository to the python path, i.e. 
``` 
export PYTHONPATH=$PYTHONPATH:<repositoryfolder>
```

Start ```jupyter``` in terminal and navigate to ```braindecode/notebooks/tutorials/Artificial_Example.ipynb```. If it works, everything is fine :)


### Work with real data

Create folder ```<repositoryfolder>/data/BBCI-without-last-runs/```

Put the following files in there:

```
AnWeMoSc1S001R01_ds10_1-12.BBCI.mat
BhNoMoSc1S001R01_ds10_1-12.BBCI.mat
FaMaMoSc1S001R01_ds10_1-14.BBCI.mat
FrThMoSc1S001R01_ds10_1-11.BBCI.mat
GuJoMoSc01S001R01_ds10_1-11.BBCI.mat
JoBeMoSc01S001R01_ds10_1-11.BBCI.mat
KaUsMoSc1S001R01_ds10_1-11.BBCI.mat
LaKaMoSc1S001R01_ds10_1-9.BBCI.mat
MaGlMoSc2S001R01_ds10_1-12.BBCI.mat
MaJaMoSc1S001R01_ds10_1-11.BBCI.mat
MaVoMoSc1S001R01_ds10_1-11.BBCI.mat
NaMaMoSc1S001R01_ds10_1-11.BBCI.mat
OlIlMoSc01S001R01_ds10_1-11.BBCI.mat
PiWiMoSc1S001R01_ds10_1-11.BBCI.mat
RoBeMoSc03S001R01_ds10_1-9.BBCI.mat
RoScMoSc1S001R01_ds10_1-11.BBCI.mat
StHeMoSc01S001R01_ds10_1-10.BBCI.mat
SvMuMoSc1S001R01_ds10_1-12.BBCI.mat
```

Now you should be able to also run ```braindecode/notebooks/tutorials/Lasagne.ipynb```.
