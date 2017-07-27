#!/usr/bin/env python
import sys
import os
import stat

file_prefix = \
"""#!/bin/bash
#$ -o /home/schirrmr/motor-imagery/data/jobs_out/ -e /home/schirrmr/motor-imagery/data/jobs_out/

cd ${HOME}/braindecode/code/braindevel/
export PYTHONPATH=$PYTHONPATH:`pwd`/../

echo "Working directory is $PWD"

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
# once cudnnv5 works or sth... ,maybe not necessary and instead remove once cudnnv5 installedon cluster
#GPU_FLAGS="device=gpu${GPU_ID},force_device=True"
#DNN_PATH_FLAGS="dnn.include_path=/home/schirrmr/cudnn-7.5-linux-x64-v5.1/include,dnn.library_path=/home/schirrmr/cudnn-7.5-linux-x64-v5.1/lib64"
#COMPILEDIR_FLAG="compiledir=/tmp/schirrmr.theano_compile_cudnn5.%(queue)s/$"
#OTHER_FLAGS="floatX=float32,nvcc.fastmath=True"
#export THEANO_FLAGS="${GPU_FLAGS},{DNN_PATH_FLAGS},{COMPILEDIR_FLAG},{OTHER_FLAGS}"
export THEANO_FLAGS="floatX=float32,device=gpu${GPU_ID},nvcc.fastmath=True,force_device=True,compiledir=/tmp/schirrmr.theano_compile.%(queue)s/"
echo THEANO_FLAGS=$THEANO_FLAGS


# exports for cudnnv3 ... maybe unneeded?
export LD_LIBRARY_PATH=/home/schirrmr/cudnn-7.0-linux-x64-v.3.0-prod:$LD_LIBRARY_PATH
export LIBRARY_PATH=/home/schirrmr/cudnn-7.0-linux-x64-v.3.0-prod:$LIBRARY_PATH
export CPATH=/home/schirrmr/cudnn-7.0-linux-x64-v.3.0-prod:$CPATH

"""

def generate_cluster_job(queue, job_args):
    # shd be ok now, try commenting line below and remove if it works
    #raise ValueError("Fix Cudnn3/5 problem first please....")#for now using cudnnv3 shd be ok atm
    config_file = job_args[0]
    arguments_for_train = job_args[1:]
    # Expect that experiment runs sequential by default
    # so that theano flags are preserved and correct (and only one) gpu taken
    arguments_for_train.append('--quiet') # better not to create huge job output files
    train_args_string = " ".join(arguments_for_train)
    train_script = "./scripts/train_experiments.py" 
    train_command = "{:s} {:s} {:s}".format(train_script, config_file, 
        train_args_string)
    
    # Replace queue in compiledir of fileprefix
    # Probably having queue as part of compiledir name
    # is not necessary sine we are using /tmp/ as our compile dir
    # which should be local to each machine(?)
    file_prefix_with_queue = file_prefix % dict(queue="rz")

    job_string = "{:s}\n{:s}\n".format(file_prefix_with_queue, train_command)
    job_filename = os.path.splitext(os.path.basename(config_file))[0] + ".pbs"
    job_filepath = "data/jobs/" + job_filename
    with open(job_filepath, "w") as job_file:
        job_file.write(job_string)

    ## Make executable
    st = os.stat(job_filepath)
    os.chmod(job_filepath, st.st_mode | stat.S_IEXEC)
    print("Will run:\n{:s}".format(train_command))
    return job_filepath

#if __name__ == "__main__":
#    generate_cluster_job(sys.argv[1:])