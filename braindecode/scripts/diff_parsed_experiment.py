#!/usr/bin/env python
"""Check whether the parse of a yaml experiment is the same as before/
show diff between two parsed experiments"""

import argparse
import os
from glob import glob
import difflib
from braindecode.experiments.parse import create_experiment_yaml_strings_from_files

_train_str_seperator = "#####TRAINSTRING#####"

def parse_command_line_arguments():
    parser = argparse.ArgumentParser(
        description="""Diff parsed experiments.
        
        Examples:
        
        ./scripts/diff_parsed_experiment.py compare configs/experiments/bci_competition/combined/fb_net_150_fs.yaml 
        ./scripts/diff_parsed_experiment.py compare configs/experiments/bci_competition/combined/fb_net_150_fs.yaml --dir data/models/bci-competition/combined/raw-net-250-fs/cross-validation/
        ./scripts/diff_parsed_experiment.py compare configs/experiments/bci_competition/combined/fb_net_150_fs.yaml --file configs/experiments/combined/fb_net_150_fs.yaml.old_train_strs
        ./scripts/diff_parsed_experiment.py store configs/experiments/bci_competition/combined/fb_net_150_fs.yaml 

    """
    )
    parser.add_argument('experiments_file_name',  action='store',
        help='A YAML configuration file specifying the experiment.')
    subparsers = parser.add_subparsers(dest="subparser_name")
    compare_parser = subparsers.add_parser('compare', description="""
        Compare current experiment and either a folder with yaml experiment strings
        or a file with old yaml strings created with the store command of this script. 
        
    """)
    
    compare_opts = compare_parser.add_mutually_exclusive_group(required=False)
    compare_opts.add_argument('--dir', default=None, type=str,
                        help='''A directory of yaml files to compare 
                        the current parse to.''')
    compare_opts.add_argument('--file', default=None, type=str,
                        help='''A file with old yaml strings to compare 
                        the current parse to.''')
    subparsers.add_parser('store', description="""
        Store current parse of an experiment into a file with all experiment strings.
        Example:
        ./scripts/diff_parsed_experiment.py store configs/experiments/bci_competition/combined/fb_net_150_fs.yaml 
    """)
    args = parser.parse_args()
    
    
    if (args.file is None) and (args.dir is None):
        # if neither file nor dir given, assume old train strs exist...
        file_name = args.experiments_file_name + ".old_train_strs"
        if not os.path.isfile(file_name):
            raise ValueError("No file or directory argument given and "
                "{:s} does not exist".format(file_name))
        args.file = file_name
    
    return args

def store_experiment_yaml_strings(experiments_file_name):
    train_strs = create_experiment_yaml_strings_from_files(
        experiments_file_name, 
        main_template_filename='configs/eegnet_template.yaml')
    train_str_file_name = experiments_file_name + ".old_train_strs"
    with open(train_str_file_name, 'w') as tmp_file:
        tmp_file.write(_train_str_seperator.join(train_strs))

def compare_experiment_yaml_strings(experiments_file_name, dir_name,
            file_name):
        assert (not ((dir_name is None) and (file_name is None)))
        old_train_strs = extract_old_train_strings(dir_name, file_name)
        # from http://stackoverflow.com/a/845432/1469195
        
        
        new_train_strs = create_experiment_yaml_strings_from_files(
            experiments_file_name, 
            main_template_filename='configs/eegnet_template.yaml')
        
        for i_str, (old_str, new_str) in enumerate(
                zip(old_train_strs, new_train_strs)):
            new_lines = new_str.splitlines()
            old_lines = old_str.splitlines()
            # ignore inputlayer line
            new_lines = [l for l in new_lines if (not "InputLayer" in l)]
            old_lines = [l for l in old_lines if (not "InputLayer" in l)]
            diff_str = '\n'.join(difflib.unified_diff(old_lines, new_lines))
        
            if diff_str != '':
                print("Experiment {:d}".format(i_str + 1))
                print(diff_str)
                print('\n')
            print("Experiment {:2d} ok.".format(i_str + 1))
                
        if (len(old_train_strs) != len(new_train_strs)):
            print("Could not compare all experiments.")
            print("Old experiments: {:d}".format(len(old_train_strs)))
            print("New experiments: {:d}".format(len(new_train_strs)))

def extract_old_train_strings(dir_name, file_name):
    if dir_name is not None: # sort numerically by last part of filename (between slash and dot,
        # there is the experiment id)
        yaml_filenames = sorted(glob(os.path.join(dir_name, '*.yaml')), key=lambda f:int(f.split('/')[-1].split('.')[0]))
        old_train_strs = []
        for filename in yaml_filenames:
            with open(filename, 'r') as yaml_file:
                train_str = yaml_file.read()
            old_train_strs.append(train_str)
    
    if file_name is not None:
        with open(file_name, 'r') as tmp_file:
            old_train_strs = tmp_file.read().split(_train_str_seperator)
    return old_train_strs

if __name__ == "__main__":
    args = parse_command_line_arguments()
    if args.subparser_name == 'store':
        store_experiment_yaml_strings(args.experiments_file_name)
    if args.subparser_name == 'compare':
        compare_experiment_yaml_strings(args.experiments_file_name,
            args.dir, args.file)

