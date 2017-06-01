#!/usr/bin/env python

import argparse
from braindecode.experiments.parse import (
    create_experiment_yaml_strings_from_files, create_config_strings, create_config_objects,
    create_templates_variants_from_config_objects,
    process_parameters_by_templates, process_templates)
import numbers

def parse_command_line_arguments():
    parser = argparse.ArgumentParser(
        description="""Print results stored in a folder.
        Example: ./scripts/create_hyperopt_files.py configs/experiments/bci_competition/combined/raw_net_150_fs.yaml """
    )
    parser.add_argument('experiments_file_name', action='store',
                        choices=None,
                        help='Yaml experiment file to base hyperopt config on.')
    args = parser.parse_args()
    return args


def create_hyperopt_files(experiments_file_name):
    ## Get templates and Variants
    config_strings = create_config_strings(experiments_file_name)
    config_objects = create_config_objects(config_strings)
    templates, variants = create_templates_variants_from_config_objects(config_objects)
    
    
    ## Create template string and save to file
    template_str = "{\n    templates: {\n"
    for key in templates:
        template_str += "        {:s}: {:s},\n".format(key, templates[key])
        
    template_str += "}}\n\n"
    
    with open('hyperopt_template.yaml', 'w') as template_file:
        template_file.write(template_str)


    ## Create parameter ranges and save to .pcs-file
    
    # Fold param variants into one dict param -> list of possibilities    
    parameter_ranges = dict()
    for param_dict in variants:
        for key in param_dict:
            if key in parameter_ranges and (param_dict[key] not in parameter_ranges[key]):
                parameter_ranges[key].append(param_dict[key])
            if key not in parameter_ranges:
                parameter_ranges[key]  = [param_dict[key]]
            
    # Delete unnecessary stuff, add template name reminder
    parameter_ranges.pop('dataset_filename')
    parameter_ranges.pop('save_path')
    parameter_ranges['template_name'] = ['!ADD_TEMPLATE_FILE_NAME!']
    # Build string
    hyperopt_param_string = ""
    for key, values in parameter_ranges.iteritems():
        # take middle value as default value
        default_str = "[{:s}]".format(str(values[len(values) // 2]))
        if len(values) == 1:
            val_str = "{{{:s}}}".format(str(values[0]))
        else:
            is_integer = False
            if all(isinstance(val, numbers.Number) for val in values):
                val_str = "[{:s}, {:s}]".format(str(values[0]), str(values[-1]))
                is_integer = all(val.is_integer() for val in values)
                if (is_integer):
                    default_str += 'i'
            else:
                val_str = str(values).replace('(', '{').replace(')', '}')
        line =  "{:30s} {:30s} {:s}\n".format(str(key), val_str, default_str)
        
        line = line.replace("$", "**")
        # correct indentation
        line = line.replace(" [**", "[**")
        hyperopt_param_string += line
        
        with open('hyperopt_params.pcs', 'w') as param_file:
            param_file.write(hyperopt_param_string)

if __name__ == "__main__":
    args = parse_command_line_arguments()
    create_hyperopt_files(args.experiments_file_name)