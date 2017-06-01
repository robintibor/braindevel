import os
import sys
import time

import braindecode.hyperopt.hyperopt as hyperopt

__authors__ = ["Katharina Eggensperger"]
__contact__ = "automl.org"

import logging

def parse_cli():
    """
    Provide a generic command line interface for benchmarks. It will just parse
    the command line according to simple rules and return two dictionaries, one
    containing all arguments for the benchmark algorithm like dataset,
    crossvalidation metadata etc. and the containing all learning algorithm
    hyperparameters.

    Parsing rules:
    - Arguments with two minus signs are treated as benchmark arguments, Xalues
     are not allowed to start with a minus. The last argument must --params,
     starting the hyperparameter arguments.
    - All arguments after --params are treated as hyperparameters to the
     learning algorithm. Every parameter name must start with one minus and must
     have exactly one value which has to be given in single quotes.
    - Arguments with no value before --params are treated as boolean arguments

    Example:
    python neural_network.py --folds 10 --fold 1 --dataset convex  --params
        -depth '3' -n_hid_0 '1024' -n_hid_1 '1024' -n_hid_2 '1024' -lr '0.01'
    """
    args = {}
    arg_name = None
    arg_values = None
    parameters = {}

    cli_args = sys.argv
    found_params = False
    skip = True
    iterator = enumerate(cli_args)

    for idx, arg in iterator:
        if skip:
            skip = False
            continue
        else:
            skip = True

        if arg == "--params":
            if arg_name:
                args[arg_name] = " ".join(arg_values)
            found_params = True
            skip = False

        elif arg[0:2] == "--" and not found_params:
            if arg_name:
                args[arg_name] = " ".join(arg_values)
            arg_name = arg[2:]
            arg_values = []
            skip = False

        elif arg[0:2] == "--" and found_params:
            raise ValueError("You are trying to specify an argument after the "
                             "--params argument. Please change the order.")

        elif arg[0] == "-" and arg[0:2] != "--" and found_params:
            parameters[cli_args[idx][1:]] = cli_args[idx+1]

        elif arg[0] == "-" and arg[0:2] != "--" and not found_params:
            raise ValueError("You either try to use arguments with only one lea"
                             "ding minus or try to specify a hyperparameter bef"
                             "ore the --params argument. %s" %
                             " ".join(cli_args))
        elif arg[0:2] != "--" and not found_params:
            arg_values.append(arg)
            skip = False

        elif not found_params:
            raise ValueError("Illegal command line string, expected an argument"
                             " starting with -- but found %s" % (arg,))

        else:
            raise ValueError("Illegal command line string, expected a hyperpara"
                             "meter starting with - but found %s" % (arg,))

    return args, parameters


def main(params, **kwargs):
    #print 'Params: ', params,
    print "kwargs", kwargs
    if "debug" in kwargs and kwargs['debug'] == '1':
        print "#" * 80
        print "# DEBUGGING "
        print "#" * 80
        logging.basicConfig(level=logging.DEBUG)
    y = hyperopt.train_hyperopt(params)
    print 'Result: ', y
    return y

if __name__ == "__main__":
    starttime = time.time()
    args, params = parse_cli()
    print params

    dataset_dir = args['dataset_dir']
    fold = int(float(args['fold']))

    dataset_list = ['A01TE.mat',
        'A02TE.mat',
        'A03TE.mat',
        'A04TE.mat',
        'A05TE.mat',
        'A06TE.mat',
        'A07TE.mat',
        'A08TE.mat',
        'A09TE.mat']
    
    # We assume we split our datasets into 10 folds
    # the given fold variable is referring to all folds of all datasets
    # concatenated, so e.g. fold 37 is 
    # dataset 3, fold 7 in 0-based indexing

    assert len(dataset_list) == (int(float(args['folds'])) / 10)
    dataset_nr = fold / 10
    inner_fold_nr = fold % 10
    
    # debug: (should be 0.649123 for rawnet)
    #dataset_nr = 0
    #inner_fold_nr = 9

    params['dataset_filename'] = os.path.join(dataset_dir, dataset_list[dataset_nr])
    params['i_test_fold'] = inner_fold_nr
    assert os.path.isfile(params['dataset_filename'])
    print "Using dataset %s" % params['dataset_filename']

    result = main(params, **args)
    duration = time.time() - starttime
    print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
        ("SAT", abs(duration), result, -1, str(__file__))