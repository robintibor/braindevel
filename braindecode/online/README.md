# Run experiment

Typically run like this:

1. Remote desktop to gschasi
  1. Start two matlab 2014a instances
  2. Change to E:\WCBI\ in both matlabs
2. SSH to Zugspitze
  1. Go to braindecode folder
  2. source venv

3. Check sensor correctness
```
# auf zugspitze
python online/server.py --model data/models/online/cnt/shallow-uneven-trials/15
```
```
# in matlab 1
run TestTCP.m
# in matlab 2
# (only when sensor started message is there from python!)
run sendFromBBCIFile.m
```
Cancel everything by CTRL+C once sensors checked

4.Real experiments
```
# auf zugspitze
python online/server.py --model data/models/online/cnt/shallow-uneven-trials/15 --noplot
```
```
# in matlab 1
run TestTCP.m
# in matlab 2
# (only when sensor started message is there from python!)
run sendFromBBCIFile.m
```

when wanting to finish, press enter iun python server terminal

later use newer params like this
```
python online/server.py --model data/models/online/cnt/shallow-uneven-trials/15 --noplot --params data/models/online/cnt/shallow-uneven-trials/15.2016-04-05_15-49-58.adapted.npy
```

You can of course change model etc.

## Installation Log

with user `fakelukas` on zugspitze

###Setting up repository
```
git clone https://github.com/robintibor/braindecode.git
cd braindecode
git checkout devel
```

###Installing dependencies
```
make scikits-samplerate
```


ignore this error in this step:
```
/sbin/ldconfig.real: Can't create temporary cache file /etc/ld.so.cache~: Permission denied
make: *** [scikits-samplerate] Error 1
```

```
virtualenv venv
source venv/bin/activate
```
(Will take about 20 minutes:)
```
make install
pip install gevent
```

###Add stuff to path
Add this to .bashrc or somewhere
(I know this is a bit old cuda/cudnn, but works for now :))
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

### Copy over PyQt5
```
cp -r /usr/local/lib/python2.7/dist-packages/PyQt5/ venv/local/lib/python2.7/site-packages/
cp -r /usr/lib/python2.7/dist-packages/sip* venv/local/lib/python2.7/site-packages/
```

### Add links to robin's data folder
```
cd braindecode (so dass du in braindecode/braindecode bist)
ln -s /media/EEG/robintibor/braindecode/data/ .
```
