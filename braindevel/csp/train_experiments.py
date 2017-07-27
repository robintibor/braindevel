#!/usr/bin/env python
import time
import logging
import yaml
from pylearn2.utils.logger import (
    CustomStreamHandler, CustomFormatter)
import argparse
from pylearn2.config import yaml_parse
import lasagne
import os
from numpy.random import RandomState
import pickle
import numpy as np
from braindevel.experiments.experiment_runner import (
    ExperimentsRunner)
from braindevel.experiments.parse import (
    create_experiment_yaml_strings_from_files)
from braindevel.csp.results import CSPResult, CSPModel
from braindevel.csp.print_results import CSPResultPrinter
from pprint import pprint
log = logging.getLogger(__name__)

class CSPExperimentsRunner(ExperimentsRunner):
    def _run_experiments_with_string(self, experiment_index, train_str):
        lasagne.random.set_rng(RandomState(9859295))
        starttime = time.time()
        
        train_dict = yaml_parse.load(train_str)
        self._save_train_string(train_str, experiment_index)
        if not self._quiet:
            pprint(train_dict['original_params'])
        if not self._dry_run:
            csp_train = train_dict['csp_train']
            
            if not self._cross_validation:
                csp_train.only_last_fold=True
            else: # cross validation
                csp_train.only_last_fold=False
            if self._shuffle:
                csp_train.shuffle = True
            
            csp_train.run()
            endtime = time.time() 
            result = CSPResult(
                    csp_trainer=csp_train,
                    parameters=train_dict['original_params'],
                    training_time=endtime - starttime)   
            folder_path = self._folder_paths[experiment_index]
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            result_file_name = self._get_result_save_path(experiment_index)
            with open(result_file_name, 'w') as resultfile:
                pickle.dump(result, resultfile)
            
            if train_dict['save_model']:
                model = CSPModel(csp_train)
                model_save_path = self._get_model_save_path(experiment_index)
                model.save(model_save_path)

    def _print_results(self):
        valid_folder_paths = [p for p in self._folder_paths 
            if p is not None]
        for folder_path in np.unique(valid_folder_paths):
            res_printer = CSPResultPrinter(folder_path)
            res_printer.print_results()
            print("\n")
    
def setup_logging():
    """ Set up a root logger so that other modules can use logging
    Adapted from scripts/train.py from pylearn"""
        
    root_logger = logging.getLogger()
    prefix = '%(asctime)s '
    formatter = CustomFormatter(prefix=prefix)
    handler = CustomStreamHandler(formatter=formatter)
    root_logger.handlers  = []
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

def parse_command_line_arguments():
    parser = argparse.ArgumentParser(
        description="""Launch an experiment from a YAML experiment file.
        Example: ./train_experiments.py yaml-scripts/experiments.yaml """
    )
    parser.add_argument('experiments_file_name', action='store',
                        choices=None,
                        help='A YAML configuration file specifying the '
                             'experiment')
    parser.add_argument('--template_file_name', action='store',
                        default='configs/csp_template.yaml',
                        help='A YAML configuration file specifying the '
                             'template for all experiments')
    parser.add_argument('--quiet', action="store_true",
        help="Run algorithm quietly without progress output")
    parser.add_argument('--debug', action="store_true",
        help="Run with debug options.")
    parser.add_argument('--test', action="store_true",
        help="Run experiment on less features and less data to test it")
    parser.add_argument('--dryrun', action="store_true",
        help="Only show parameters for experiment, don't train.")
    parser.add_argument('--cv', action="store_true", 
        help="Use cross validation instead of train test split")
    parser.add_argument('--shuffle', action="store_true", 
        help="Use shuffle (only use together with --cv)")
    parser.add_argument('--params', nargs='*', default=[],
                        help='''Parameters to override default values/other values given in experiment file.
                        Supply it in the form parameter1=value1 parameters2=value2, ...''')
    parser.add_argument('--startid', type=int,
                        help='''Start with experiment at specified id....''')
    parser.add_argument('--stopid', type=int,
                        help='''Stop with experiment at specified id....''')
    parser.add_argument('--filters', nargs='*', default=[],
                        help='''Filter experiments by parameter values.
                        Only run those experiments where the parameter matches the given value.
                        Supply it in the form parameter1=value1 parameters2=value2, ...''')
    parser.add_argument('--skipexisting', action="store_true",
                        help='''Only run those experiments that werent run yet.''')
    args = parser.parse_args()
    assert (not (args.shuffle and (not args.cv))), ("Use shuffle only "
        "together with cross validation.")
    # dictionary values are given with = inbetween, parse them here by hand
    param_dict =  dict([param_and_value.split('=') 
                        for param_and_value in args.params])
    args.params = param_dict
    # already load as yaml here to compare to later loaded params...
    filter_dict =  dict([(param_and_value.split('=')[0], 
                    yaml.load(param_and_value.split('=')[1]))
                        for param_and_value in args.filters])
    args.filters = filter_dict
    if (args.startid is  not None):
        # model ids printed are 1-based, python is zerobased
        # stop id can remain same as stop implies exclusive index
        # and 1-based inclusive == 0-based exclusive
        args.startid = args.startid - 1 
    return args

if __name__ == "__main__":
    setup_logging()
    args = parse_command_line_arguments()

    all_train_strs = create_experiment_yaml_strings_from_files(
        args.experiments_file_name, args.template_file_name,
        debug=args.debug, command_line_params=args.params,
        filter_params=args.filters)
    exp_runner = CSPExperimentsRunner(quiet=args.quiet, start_id=args.startid,
        stop_id=args.stopid, cross_validation=args.cv, shuffle=args.shuffle,
        skip_existing=args.skipexisting, dry_run=args.dryrun)
    exp_runner.run(all_train_strs)

