source /home/eggenspk/virtualEnvs/virtualBrain/bin/activate

typeset -i GPUID=$(cat /home/eggenspk/${JOB_ID}_${SGE_TASK_ID}_${JOB_NAME})
echo We are using GPU device ${GPUID}
export THEANO_FLAGS=floatX=float32,nvcc.fastmath=True,force_device=True,cuda.root=/usr/local/cuda,device=gpu${GPUID},exception_verbosity=high,compiledir=/tmp/${JOB_ID}.${SGE_TASK_ID}.meta_gpu-tf.q/

export CUDA_BIN=/usr/local/cuda/bin
export CUDA_LIB=/usr/local/cuda/lib64
export PATH=${CUDA_BIN}:$PATH

LD_LIBRARY_PATH=${CUDA_LIB}:$LD_LIBRARY_PATH
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/lib/atlas-base:
export LD_LIBRARY_PATH

export LD_LIBRARY_PATH=$VIRTUAL_ENV/cudnn-6.5-linux-x64-v2:$LD_LIBRARY_PATH
export LIBRARY_PATH=$VIRTUAL_ENV/cudnn-6.5-linux-x64-v2:$LIBRARY_PATH
export CPATH=$VIRTUAL_ENV/cudnn-6.5-linux-x64-v2:$CPATH

export PYTHONPATH=$PYTHONPATH:/home/eggenspk/informatikhome/eggenspk/BrainLinksBrainTools/MotorImagery/machine-learning-for-motor-imagery_ROBIN/:/home/eggenspk/informatikhome/eggenspk/BrainLinksBrainTools/MotorImagery/machine-learning-for-motor-imagery_ROBIN/motor_imagery_learning/
