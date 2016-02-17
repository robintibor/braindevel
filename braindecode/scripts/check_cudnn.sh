#!/bin/bash

JOB_NUMBER=$1


LD_LIBRARY_PATH=/usr/local/cudnn-7.0/lib64:$LD_LIBRARY_PATH
LIBRARY_PATH=/usr/local/cudnn-7.0/lib64:$LIBRARY_PATH
CPATH=/usr/local/cudnn-7.0/include:$CPATH
GPU_ID=0


CUDA_VISIBLE_DEVICES=$JOB_NUMBER
PATH=/usr/local/cuda-7.5/bin/:/usr/bin/:$PATH  
THEANO_FLAGS=floatX=float32,device=gpu0,compiledir=/tmp/sudo_cudnn/

LD_LIBRARY_PATH=$LD_LIBRARY_PATH LIBRARY_PATH=$LIBRARY_PATH CPATH=$CPATH CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES PATH=$PATH THEANO_FLAGS=$THEANO_FLAGS python -c "import theano.sandbox.cuda.dnn as dnn; print dnn.dnn_available()"

