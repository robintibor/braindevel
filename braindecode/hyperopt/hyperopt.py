#!/usr/bin/env python

"""
This is a script to make hyperparameter optimization with braindecode.
"""
import logging
import os
from pprint import pprint
from braindecode.experiments.parse import (
    create_config_strings, create_config_objects,
    create_templates_variants_from_config_objects,
    process_parameters_by_templates, process_templates)
from string import Template
from pylearn2.config import yaml_parse
import yaml
from braindecode.experiments.experiment import Experiment
import lasagne
from numpy.random import RandomState

def train_hyperopt(params):
    """ Runs one fold with given parameters and returns test misclass."""
    lasagne.random.set_rng(RandomState(9859295))

    template_name = params.pop('template_name')    
    params = adjust_params_for_hyperopt(params)
    
    config_strings = create_config_strings(template_name)
    config_objects = create_config_objects(config_strings)
    templates, _ = create_templates_variants_from_config_objects(
        config_objects)
    
    
    processed_templates, params_without_template_params  = process_templates(
                templates, params)
    final_params = process_parameters_by_templates(params_without_template_params, 
        processed_templates)
    
    # go to directory above this source-file
    main_template_filename = os.path.dirname(os.path.abspath(os.path.dirname(
        __file__)))
    # then complete path to config
    main_template_filename = os.path.join(main_template_filename, "configs", 
        "eegnet_template.yaml")
    
    with (open(main_template_filename, 'r')) as main_template_file:
        main_template_str = main_template_file.read()
        
        
    final_params['original_params'] = 'dummy'
    train_str = Template(main_template_str).substitute(final_params)
    
    def do_not_load_constructor(loader, node):
        return None
    yaml.add_constructor(u'!DoNotLoad', do_not_load_constructor)
    modified_train_str = train_str.replace('layers: ', 'layers: !DoNotLoad ')
    train_dict = yaml_parse.load(modified_train_str) 
    dataset = train_dict['dataset'] 
    dataset.load()
    dataset_provider = train_dict['dataset_provider']
    
    assert 'in_sensors' in train_str
    assert 'in_rows' in train_str
    assert 'in_cols' in train_str
    
    train_str = train_str.replace('in_sensors',
        str(dataset.get_topological_view().shape[1]))
    train_str = train_str.replace('in_rows',
        str(dataset.get_topological_view().shape[2]))
    train_str = train_str.replace('in_cols', 
        str(dataset.get_topological_view().shape[3]))
    
    train_dict =  yaml_parse.load(train_str)
    layers = train_dict['layers']
    final_layer = layers[-1]

    # turn off debug/info logging
    logging.getLogger("pylearn2").setLevel(logging.WARN)
    logging.getLogger("braindecode").setLevel(logging.WARN)
    exp = Experiment()
    exp.setup(final_layer, dataset_provider, **train_dict['exp_args'])
    exp.run()
    final_misclass = exp.monitor_chans['test_misclass'][-1]
    print("Result for")
    pprint(params)
    print("Final Test misclass: {:5.4f}".format(float(final_misclass)))
    return final_misclass

def adjust_params_for_hyperopt(params):
    for key in params:
        val = params[key]
        if isinstance(val, basestring) and '**' in val:
            params[key] = val.replace('**', '$')
    params['save_path'] = 'null' # needs some value
    params['num_folds'] = '10'
    return params


if __name__ == "__main__":
    params = {'dataset': '**raw_set',
         'dataset_filename': 'data/bci-competition-iv/2a-combined/A01TE.mat',
         'dataset_provider': '*preprocessed_provider',
         'filter_time_length': 15,
         'layers': '**raw_net_layers',
         'loss_var_func': '**categorical_crossentropy',
         'low_cut_off_hz': 2,
         'max_epochs': 5,
         'max_increasing_epochs': 150,
         'pool_time_length': 50,
         'pool_time_stride': 10,
         'preprocessor': '**online_chan_freq_wise',
         'resample_fs': 150,
         'sensor_names': 'null',
         'trial_start': 0,
         'trial_stop': 4000,
         'updates_var_func': '**adam',
         'num_folds': 10,
         'i_test_fold': 9,
         'template_name': '/home/schirrmr/braindecode/code/braindecode/hyperopt/robin_rawnet/templates.yaml'}
        
    train_hyperopt(params)

