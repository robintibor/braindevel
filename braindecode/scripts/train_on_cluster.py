#!/usr/bin/env python
from generate_cluster_job import generate_cluster_job
import sys
import subprocess

if __name__ == "__main__":
    queue = sys.argv[1]
    assert queue in['tf', 'rz', 'rzx', 'test'], ("only know "
        "rz, rzx, tf and test queues, not: " + queue)
    queue_name = "meta_gpu-{:s}"
    if queue == 'rzx':
        queue_name = "meta_gpux-rz"
    if sys.argv[2].startswith('metagpu'):
        hostname = sys.argv[2]
        print("Running on {:s}".format(hostname))
        job_args = sys.argv[3:]
    else:
        job_args = sys.argv[2:]
        hostname = None
    job_filepath = generate_cluster_job(job_args)
    if hostname is not None:
        command = "qsub -l hostname={:s} -q meta_gpu-{:s}.q {:s}".format(
            hostname, queue, job_filepath)
    else:
        command = "qsub -q {:s}.q {:s}".format(queue_name, job_filepath)
    print("Running:\n" + command)
    subprocess.call([command],shell=True)
    