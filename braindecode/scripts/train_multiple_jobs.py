#!/usr/bin/env python

import sys
import subprocess
import time

if __name__ == "__main__":
    if len(sys.argv) < 2 or (sys.argv[1] in ['-h', '--help']):
        print("Usage: ./scripts/train_multiple_jobs.py queuename configfilename  [start] [stop] [step] [waiting_sec] [train_flags]")
        sys.exit(0)
    queue = sys.argv[1]
    assert queue in  ["rz", "tf", "test", "rzx"]
    
    config_filename = sys.argv[2]
    start = 1
    stop = 144
    step = 1
    wait_time = 60
    if len(sys.argv) > 3:
        start = int(sys.argv[3])
    if len(sys.argv) > 4:
        stop = int(sys.argv[4])
    if len(sys.argv) > 5:
        step = int(sys.argv[5])
    if len(sys.argv) > 6:
        wait_time = float(sys.argv[6])
    if len(sys.argv) > 7:
        train_arg_string = " ".join(sys.argv[7:])
    else:
        train_arg_string = ""

    train_script_file = "./scripts/train_on_cluster.py"

    for i_start in range(start,stop+1, step):
        if i_start > start:
            print("Sleeping {:.1f} seconds until starting next experiment...".format(
                wait_time))
            time.sleep(wait_time)
        i_stop = min(i_start + step - 1, stop)
        command = "{:s} {:s} {:s} --start {:d} --stop {:d} {:s}".format(
            train_script_file, queue, config_filename, i_start, i_stop, 
            train_arg_string)
        
        subprocess.call([command],shell=True)
