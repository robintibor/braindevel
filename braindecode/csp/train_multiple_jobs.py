#!/usr/bin/env python

import sys
import subprocess
import time

if __name__ == "__main__":
    if len(sys.argv) < 2 or (sys.argv[1] in ['-h', '--help']):
        print("Usage: ./csp/train_multiple_jobs.py configfilename [start] [stop] [step] [waittime] [train_flags]")
        sys.exit(0)
    config_filename = sys.argv[1]
    start = 1
    stop = 144
    step = 1
    waittime = 60
    if len(sys.argv) > 2:
        start = int(sys.argv[2])
    if len(sys.argv) > 3:
        stop = int(sys.argv[3])
    if len(sys.argv) > 4:
        step = int(sys.argv[4])
    if len(sys.argv) > 5:
        waittime = int(sys.argv[5])
    hostname = None
    if len(sys.argv) > 6 and sys.argv[6].startswith("metaex"):
        hostname = sys.argv[6]
        train_arg_string = " ".join(sys.argv[7:])
    else:
        train_arg_string = " ".join(sys.argv[6:])
    

    train_script_file = "./csp/train_on_cluster.py"

    for i_start in range(start,stop+1, step):
        if i_start > start:
            print("Sleeping {:d} sec until starting next experiment...".format(
                waittime))
            time.sleep(waittime)
        i_stop = min(i_start + step - 1, stop)
        if hostname is not None:
            command = "{:s} {:s} {:s} --start {:d} --stop {:d} {:s}".format(
                train_script_file, hostname, config_filename, i_start, i_stop, 
                train_arg_string)
        else:
            command = "{:s} {:s} --start {:d} --stop {:d} {:s}".format(
                train_script_file, config_filename, i_start, i_stop, 
                train_arg_string)
            
        subprocess.call([command],shell=True)
        #print command
