#!/usr/bin/env python
import logging
from braindevel.experiments.parse import (
    create_experiment_yaml_strings_from_files)
from pylearn2.utils.logger import (CustomStreamHandler, CustomFormatter)
from braindevel.experiments.experiment_runner import ExperimentsRunner
import argparse
import yaml

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
        default='configs/eegnet_template.yaml',
        help='A YAML configuration file specifying the '
        'template for all experiments')
    parser.add_argument('--quiet', action="store_true",
        help="Run algorithm quietly without progress output")
    parser.add_argument('--debug', action="store_true",
        help="Run with debug options.")
    parser.add_argument('--batchtest', action="store_true",
        help="Try which batch size still works for the memory on the gpu....")
    parser.add_argument('--test', action="store_true",
        help="Run experiment on less features and less data to test it")
    parser.add_argument('--dryrun', action="store_true",
        help="Only show parameters for experiment, don't train.")
    parser.add_argument('--cv', action="store_true", 
        help="Use cross validation instead of train test split")
    parser.add_argument('--shuffle', action="store_true", 
        help="Use shuffle (only use together with --cv)")
    parser.add_argument('--firstsets', type=int, default=None,
            help='''Use only first n datasets.''')

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
    parser.add_argument('--nopredlosshack', action="store_true",
                        help='''Do not use pred loss hack for epilepsy experiments.''')
    args = parser.parse_args()

    if args.firstsets is None:
        args.firstsets = False # False means take all
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
        # stop id can remain same as stop implies exlusive index
        # and 1-based inclusive == 0-based exclusive
        args.startid = args.startid - 1 
    
    return args

if __name__ == "__main__":
    setup_logging()
    args = parse_command_line_arguments()
    all_train_strs = create_experiment_yaml_strings_from_files(
        args.experiments_file_name, args.template_file_name, args.debug,
        command_line_params=args.params,
        only_first_n_sets=args.firstsets,
        filter_params=args.filters)
    exp_runner = ExperimentsRunner(quiet=args.quiet, start_id=args.startid,
        stop_id=args.stopid, cross_validation=args.cv, shuffle=args.shuffle,
        debug=args.debug, dry_run=args.dryrun, 
        only_first_n_sets=args.firstsets, batch_test=args.batchtest,
        skip_existing=args.skipexisting, pred_loss_hack=not args.nopredlosshack)
    exp_runner.run(all_train_strs)

