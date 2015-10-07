#!/usr/bin/env python
from generate_cluster_job import generate_cluster_job
import sys
import subprocess

if __name__ == "__main__":
    queue = sys.argv[1]
    assert queue == 'tf' or queue == 'rz', ("only know rz and tf queues, "
    "not: " + queue)
    job_args = sys.argv[2:]
    job_filepath = generate_cluster_job(job_args)
    command = "qsub -q meta_gpu-{:s}.q {:s}".format(queue, job_filepath)
    print("Running:\n" + command)
    subprocess.call([command],shell=True)
    