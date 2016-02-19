#!/usr/bin/env python
import sys
import os
import stat

file_prefix = \
"""#!/bin/bash
#$ -o /home/schirrmr/motor-imagery/data/jobs_out/ -e /home/schirrmr/motor-imagery/data/jobs_out/

cd ${HOME}/braindecode/code/braindecode/
export PYTHONPATH=$PYTHONPATH:`pwd`/../
# add stuff for cudnn
#export LD_LIBRARY_PATH=/home/schirrmr/cudnn-6.5-linux-x64-v2:$LD_LIBRARY_PATH
#export LIBRARY_PATH=/home/schirrmr/cudnn-6.5-linux-x64-v2:$LIBRARY_PATH
#export CPATH=/home/schirrmr/cudnn-6.5-linux-x64-v2:$CPATH

## once you have cuda > 7.0 and can use cudnnv3:
export LD_LIBRARY_PATH=/home/schirrmr/cudnn-7.0-linux-x64-v.3.0-prod:$LD_LIBRARY_PATH
export LIBRARY_PATH=/home/schirrmr/cudnn-7.0-linux-x64-v.3.0-prod:$LIBRARY_PATH
export CPATH=/home/schirrmr/cudnn-7.0-linux-x64-v.3.0-prod:$CPATH

echo "Working directory is $PWD"

#export CUDA_VISIBLE_DEVICES=`cat ${HOME}/${JOB_ID}_${SGE_TASK_ID}_${JOB_NAME}`
export GPU_ID=0
echo HOME=$HOME
echo USER=$USER
echo JOB_ID=$JOB_ID
echo JOB_NAME=$JOB_NAME
echo HOSTNAME=$HOSTNAME
echo SGE_TASK_ID=$SGE_TASK_ID
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
echo GPU_ID=$GPU_ID
echo $CMD
# used to have this for separate compiledir
# but never helped with crashes or anything
# so removing it from theano flags...:
# ,compiledir=/tmp/schirrmr.${JOB_ID}.${SGE_TASK_ID}/
export THEANO_FLAGS="floatX=float32,device=gpu${GPU_ID},nvcc.fastmath=True,force_device=True"
echo THEANO_FLAGS=$THEANO_FLAGS
"""
def generate_cluster_job(sys_args):
    config_file = sys_args[0]
    arguments_for_train = sys_args[1:]
    # Expect that experiment runs sequential by default
    # so that theano flags are preserved and correct (and only one) gpu taken
    arguments_for_train.append('--quiet') # better not to create huge job output files
    train_args_string = " ".join(arguments_for_train)
    train_script = "./scripts/train_experiments.py" 
    train_command = "{:s} {:s} {:s}".format(train_script, config_file, 
        train_args_string)
    job_string = "{:s}\n{:s}\n".format(file_prefix, train_command)
    job_filename = os.path.splitext(os.path.basename(config_file))[0] + ".pbs"
    job_filepath = "data/jobs/" + job_filename
    with open(job_filepath, "w") as job_file:
        job_file.write(job_string)

    ## Make executable
    st = os.stat(job_filepath)
    os.chmod(job_filepath, st.st_mode | stat.S_IEXEC)
    print("Will run:\n{:s}".format(train_command))
    #print("Run as one of:")
    #print("qsub -q meta_gpu-tf.q " + job_filepath)
    #print("qsub -q meta_gpu-rz.q " + job_filepath)
    #print("qsub -q meta_gpux-rz.q " + job_filepath)
    return job_filepath

if __name__ == "__main__":
    generate_cluster_job(sys.argv[1:])