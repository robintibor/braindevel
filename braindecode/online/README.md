# Run experiment
```
python online/server.py --host MartinSeineIP --model modelfile --noplot
# default host should work fine (gschasi)
python online/server.py --model data/models/online/cnt/shallow-uneven-trials/15 --noplot
```

## Installation Log

with user `fakelukas` on zugspitze

###Setting up repository
```
git clone https://github.com/robintibor/braindecode.git
git checkout devel
cd braindecode
```

###Installing dependencies
```
make scikits-samplerate

virtualenv venv
source venv/bin/activate
```
(Will take about 20 minutes:)
```
make install
pip install gevent
```

###Add stuff to path
Ya this is a bit old cuda, but works for now :)
```
export CUDA_HOME=/usr/local/cuda-7.0
export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-7.0/bin:$PATH
export THEANO_FLAGS=floatX=float32,device=gpu0,nvcc.fastmath=True
```

### Check Cudnn:
```
ipython
>>> import theano.sandbox.cuda.dnn; theano.sandbox.cuda.dnn.dnn_available()
Using gpu device 0: GeForce GTX TITAN X (CNMeM is disabled, CuDNN 3007)
Out[1]: True
```

### Add links to robin's data folder
```
cd braindecode (so dass du in braindecode/braindecode bist)
ln -s /media/EEG/robintibor/braindecode/data/ .
```


