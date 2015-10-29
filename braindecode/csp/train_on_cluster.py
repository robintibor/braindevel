#!/usr/bin/env python
import sys
import subprocess
import os
import stat

file_prefix = \
"""#!/bin/bash
#$ -o /home/schirrmr/motor-imagery/data/jobs_out/ -e /home/schirrmr/motor-imagery/data/jobs_out/

cd ${HOME}/braindecode/code/braindecode/
export PYTHONPATH=$PYTHONPATH:`pwd`/../

echo "Working directory is $PWD"

echo HOME=$HOME
echo USER=$USER
echo JOB_ID=$JOB_ID
echo JOB_NAME=$JOB_NAME
echo HOSTNAME=$HOSTNAME
echo $CMD

# Set theano to cpu.. rest of flags probably not important
# disable gpus
export CUDA_VISIBLE_DEVICES=
export THEANO_FLAGS="floatX=float32,device=cpu,nvcc.fastmath=True"
echo THEANO_FLAGS=$THEANO_FLAGS
"""

def generate_cluster_job(sys_args):
    config_file = sys_args[0]
    arguments_for_train = sys_args[1:]
    # Expect that experiment runs sequential by default
    # so that theano flags are preserved and correct (and only one) gpu taken
    arguments_for_train.append('--quiet') # better not to create huge job output files
    train_args_string = " ".join(arguments_for_train)
    train_script = "./csp/train_experiment.py" 
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
    print("Run as:")
    print("qsub -q meta_core.q " + job_filepath)
    return job_filepath

if __name__ == "__main__":
    job_args = sys.argv[1:]
    job_filepath = generate_cluster_job(job_args)
    command = "qsub -q meta_core.q {:s}".format(job_filepath)
    print("Running:\n" + command)
    subprocess.call([command],shell=True)
    