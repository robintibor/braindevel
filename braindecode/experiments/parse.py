import itertools
from copy import deepcopy, copy
import yaml
from string import Template
from collections import deque
import numpy as np
import re
from braindecode.util import add_message_to_exception, merge_dicts
import logging
import pprint
log = logging.getLogger(__name__)

def transform_vals_to_string_constructor(loader, node):
    return dict([(v[0].value, yaml.serialize(v[1])) for v in node.value])

def create_experiment_yaml_strings_from_files(config_filename, 
        main_template_filename, debug=False, command_line_params=None,
        only_first_n_sets=False, filter_params=()):
    # First read out all files (check for extends attribute)
    # and transform to files to strings...
    # Then call creation of experiment yaml strings
    config_strings = create_config_strings(config_filename)
    with open(main_template_filename, 'r') as main_template_file:
        main_template_str = main_template_file.read()
    return create_experiment_yaml_strings(config_strings, main_template_str,
        debug=debug, command_line_params=command_line_params,
        only_first_n_sets=only_first_n_sets,
        filter_params=filter_params)

def create_config_strings(config_filename):
    yaml.add_constructor(u'!TransformValsToString', transform_vals_to_string_constructor)
    config_strings = []
    config_filename_stack = deque([config_filename])
    while len(config_filename_stack) > 0:
        config_filename = config_filename_stack.pop()
        with open(config_filename, 'r') as config_file:
            config_str = config_file.read()
        config_obj = yaml.load(config_str.replace("templates:", "templates: !TransformValsToString"))
        if 'extends' in config_obj:
            other_filenames = config_obj['extends']
            config_filename_stack.extend(other_filenames)
        config_strings.append(config_str)
    # Need to reverse as top file needs to be first config string
    # (assumption by function create_templates_variants_from_config_objects)
    config_strings = config_strings[::-1]
    return config_strings

def create_experiment_yaml_strings(all_config_strings, main_template_str,
        debug=False, command_line_params=None, only_first_n_sets=False,
        filter_params=()):
    """ Config strings should be from top file (the "start" file, which you 
    passed as argument to create_experiment_yaml_strings_from_files), 
    to bottom file."""
    parser = ConfigParser()
    train_strings = parser.parse(all_config_strings,
        main_template_str, debug=debug, command_line_params=command_line_params,
        only_first_n_sets=only_first_n_sets, filter_params=filter_params)
    return train_strings

class ConfigParser(object):
    def parse(self, all_config_strings, main_template_str, debug=False,
        command_line_params=None, only_first_n_sets=False, filter_params=()):
        self.debug = debug
        self.command_line_params=command_line_params
        self.only_first_n_sets = only_first_n_sets
        self.filter_params = filter_params
        config_objects = create_config_objects(all_config_strings)
        final_params, templates_to_used_params = self._create_params(
            config_objects)
        train_strings = self._substitute_in_main_template(final_params, 
            main_template_str, templates_to_used_params)
        return train_strings
        
    def _create_params(self, config_objects):
        # Create templates and variants
        templates, variants = self._create_templates_variants(config_objects)
        final_params, templates_to_used_params = merge_parameters_and_templates(
            variants, templates)
        # add original params for later printing
        for i_config in range(len(final_params)):
            final_params[i_config]['original_params'] = yaml.dump(
                variants[i_config], default_flow_style=True)
        # Remove equal params. 
        # Can happen if param in upper file has several possible
        # values and in lower file is set to be always the same one, e.g.,
        # upper: resample_fs: [300,150], lower: resample_fs: [200]
        _, unique_inds = np.unique(final_params, return_index=True)
        unique_final_params = np.array(final_params)[np.sort(unique_inds)]
        # also remove for template to used params dicts
        templates_to_used_params = np.array(templates_to_used_params)[
            np.sort(unique_inds)]
        return unique_final_params, templates_to_used_params
    
    def _create_templates_variants(self, config_objects):
        # Create variants
        all_variants = []
        templates = dict()
        for config_obj in config_objects:
            if 'variants' in config_obj:
                sub_variants = create_variants_recursively(
                    config_obj['variants'])
                all_variants = product_of_lists_of_dicts(
                    all_variants, sub_variants)
            if 'templates' in config_obj:
                templates.update(config_obj['templates'])

        # Possibly filter variants
        filtered_variants = []
        for variant in all_variants:
            include_variant = True
            for key in self.filter_params:
                if variant.get(key, 'undefined') != self.filter_params[key]:
                    include_variant = False
            if include_variant:
                filtered_variants.append(variant)
        all_variants = filtered_variants
        
        # Constrain to only first n datasets...
        if self.only_first_n_sets is not False:
            all_filenames = []
            for variant in all_variants:
                if variant['dataset_filename'] not in all_filenames:
                    all_filenames.append(variant['dataset_filename'])
            wanted_filenames = all_filenames[:self.only_first_n_sets]
            all_variants = [var 
                for var in all_variants 
                if var['dataset_filename'] in wanted_filenames]
             
        # (possibly remove this)
        # Regression check since that is confusing 
        for var in all_variants:
            assert not 'only_last_fold' in var, ("Will be set by cross validation "
             "argument of the runner")
        
        # Set debug parameters if wanted
        if self.debug:
            log.info("Setting debug parameters")
            for variant in all_variants:
                if 'max_epochs' in variant:
                    variant['max_epochs'] = 1
                if 'n_top_bottom_csp_filters' in variant:
                    variant['n_top_bottom_csp_filters'] = 1
                if 'sensor_names' in variant:
                    variant['sensor_names'] = ['C3', 'C4', 'Cz']
                if 'load_sensor_names' in variant:
                    variant['load_sensor_names'] = ['C3', 'C4', 'Cz']
                if 'n_used_seizures' in variant:
                    variant['n_train_seizures'] = 4
                    variant['n_valid_seizures'] = 2
                    variant['n_used_seizures'] = 8
                    variant['non_seizure_example_stride'] = 15
                    #variant['n_middle_crop'] = 30000
                    
                #variant['last_subject'] = 1
                #ignore_before = variant.pop('ignore_unused', [])
                #variant['ignore_unused'] = ignore_before + ['last_subject']
        
        # Update all params with command line params
        if self.command_line_params is not None:
            for variant in all_variants:
                variant.update(self.command_line_params)
                
        # Remove "ignore unused" params
        for variant in all_variants:
            ignore_unused = variant.pop('ignore_unused', [])
            for param in ignore_unused:
                try:
                    variant.pop(param)
                except KeyError, exc:
                    additional_message = ' (when removing unused params})'
                    add_message_to_exception(exc, additional_message)
                    raise
        return templates, all_variants
    
    def _substitute_in_main_template(self, final_params, main_template_str,
            templates_to_used_params):
        train_strings = []
        for i_config in range(len(final_params)):
            this_params = final_params[i_config]
            this_templates_to_params = templates_to_used_params[i_config]
            
            # Check that parameters are all being used while substituting
            # Keep in mind parameters specified to be unused
            # have already been removed
            ignore_unused = ['original_params']
            params_in_templates = []
            for template in this_templates_to_params:
                params_in_templates.extend(this_templates_to_params[template])
            ignore_unused.extend(params_in_templates)
            train_str = substitute_check_superfluous_mappings(
                main_template_str, this_params, ignore_unused)
            train_strings.append(train_str)
        return train_strings

def create_config_objects(all_config_strings):
    yaml.add_constructor(u'!TransformValsToString', transform_vals_to_string_constructor)
    config_objects = [yaml.load(conf_str.replace("templates:", 
                "templates: !TransformValsToString")) for 
        conf_str in all_config_strings]
    return config_objects

def create_variants_recursively(variants):
    """
    Create Variants, variants are like structured like trees of ranges basically...
    >>> variant_dict_list = [[{'batches': [1, 2]}]]
    >>> create_variants_recursively(variant_dict_list)
    [{'batches': 1}, {'batches': 2}]
    >>> variant_dict_list = [[{'batches': [1, 2], 'algorithm': ['bgd', 'sgd']}]]
    >>> create_variants_recursively(variant_dict_list)
    [{'batches': 1, 'algorithm': 'bgd'}, {'batches': 1, 'algorithm': 'sgd'}, {'batches': 2, 'algorithm': 'bgd'}, {'batches': 2, 'algorithm': 'sgd'}]
    
    >>> variant_dict_list = [[{'batches': [1,2]}, {'algorithm': ['bgd', 'sgd']}]]
    >>> create_variants_recursively(variant_dict_list)
    [{'batches': 1}, {'batches': 2}, {'algorithm': 'bgd'}, {'algorithm': 'sgd'}]

    >>> variant_dict_list = [[{'algorithm': ['bgd'], 'variants': [[{'batches': [1, 2]}]]}]]
    >>> create_variants_recursively(variant_dict_list)
    [{'batches': 1, 'algorithm': 'bgd'}, {'batches': 2, 'algorithm': 'bgd'}]
    
    >>> variant_dict_list = [[{'batches': [1, 2]}], [{'algorithm': ['bgd', 'sgd']}]]
    >>> create_variants_recursively(variant_dict_list)
    [{'batches': 1, 'algorithm': 'bgd'}, {'batches': 1, 'algorithm': 'sgd'}, {'batches': 2, 'algorithm': 'bgd'}, {'batches': 2, 'algorithm': 'sgd'}]


    """
    list_of_lists_of_all_dicts = []
    for dict_list in variants:
        list_of_lists_of_dicts = []
        for param_dict in dict_list:
            param_dict = deepcopy(param_dict)
            variants = param_dict.pop('variants', None) # pop in case it exists
            param_dicts = cartesian_dict_of_lists_product(param_dict)
            if (variants is not None):
                list_of_variant_param_dicts =  create_variants_recursively(variants)
                param_dicts = product_of_lists_of_dicts(param_dicts, list_of_variant_param_dicts)
            list_of_lists_of_dicts.append(param_dicts)
        list_of_dicts =  merge_lists(list_of_lists_of_dicts)
        list_of_lists_of_all_dicts.append(list_of_dicts)
    return product_of_list_of_lists_of_dicts(list_of_lists_of_all_dicts)
        

def cartesian_dict_of_lists_product(params):
    # from stackoverflow somewhere (with different func name) :)
    value_product = [x for x in apply(itertools.product, params.values())]
    return [dict(zip(params.keys(), values)) for values in value_product]

def merge_lists(list_of_lists):
    return list(itertools.chain(*list_of_lists))

def merge_pairs_of_dicts_in_list(l):
    return [dict(a[0].items() + a[1].items()) for a in l]

def product_of_lists_of_dicts(a, b):
    ''' For two lists of dicts, first compute the product as list of dicts.
    First compute the cartesian product of both lists.
    Then merge the pairs of dicts into one list.
    If one of the lists is empty, return the other list.
    Examples:
    >>> a = [{'color':'red', 'number': 2}, {'color':'blue', 'number': 3}]
    >>> b = [{'name':'Jackie'}, {'name':'Hong'}]
    >>> product = product_of_lists_of_dicts(a,b)
    >>> from pprint import pprint
    >>> pprint(product)
    [{'color': 'red', 'name': 'Jackie', 'number': 2},
     {'color': 'red', 'name': 'Hong', 'number': 2},
     {'color': 'blue', 'name': 'Jackie', 'number': 3},
     {'color': 'blue', 'name': 'Hong', 'number': 3}]
    >>> a = [{'color':'red', 'number': 2}, {'color':'blue', 'number': 3}]
    >>> product_of_lists_of_dicts(a, [])
    [{'color': 'red', 'number': 2}, {'color': 'blue', 'number': 3}]
    >>> product_of_lists_of_dicts([], a)
    [{'color': 'red', 'number': 2}, {'color': 'blue', 'number': 3}]
    >>> product_of_lists_of_dicts([], [])
    []
    '''
    if (len(a) > 0 and len(b) > 0):
        product = itertools.product(a, b)
        merged_product = merge_pairs_of_dicts_in_list(product)
        return merged_product
    else:
        return (a if len(a) > 0 else b)
    
def product_of_list_of_lists_of_dicts(list_of_lists):
    '''
    >>> a = [{'color':'red', 'number': 2}, {'color':'blue', 'number': 3}]
    >>> b = [{'name':'Jackie'}, {'name':'Hong'}]
    >>> c = [{'hair': 'grey'}]
    >>> d = []
    >>> product = product_of_list_of_lists_of_dicts([a, b, c, d])
    >>> from pprint import pprint
    >>> pprint(product)
    [{'color': 'red', 'hair': 'grey', 'name': 'Jackie', 'number': 2},
     {'color': 'red', 'hair': 'grey', 'name': 'Hong', 'number': 2},
     {'color': 'blue', 'hair': 'grey', 'name': 'Jackie', 'number': 3},
     {'color': 'blue', 'hair': 'grey', 'name': 'Hong', 'number': 3}]
     '''
    return reduce(product_of_lists_of_dicts, list_of_lists)

def merge_parameters_and_templates(all_parameters, templates):
    all_final_params = []
    templates_to_used_params_per_config = []
    for param_config in all_parameters:
        processed_templates, templates_to_used_params = process_templates(
            templates, param_config)
        templates_to_used_params_per_config.append(templates_to_used_params)
        final_params = process_parameters_by_templates(param_config,
            processed_templates)
        all_final_params.append(final_params)
    return all_final_params, templates_to_used_params_per_config

def process_templates(templates, parameters):
    """Substitute parameters within templates, return substituted templates, 
    only returns those templates actually needed by the parameters. """
    processed_templates = {}
    # we need templates to replace placeholders in parameters
    # placeholders defined with $
    needed_template_names = []
    for value in parameters.values():
        param_str = str(value)
        needed_templates = get_placeholders(param_str)
        for template_name in needed_templates:
            needed_template_names.append(template_name)
    
    # now also go through template strings see if a template appears there
    new_template_found = True
    while new_template_found:
        new_template_found = False
        for a_needed_template_name in needed_template_names:
            assert a_needed_template_name in templates, ("Template should "
                "exist: {:s}".format(a_needed_template_name))
            template = templates[a_needed_template_name]
            for template_name in templates:
                if (('$' + template_name) in template and 
                        (template_name not in needed_template_names)):
                    needed_template_names.append(template_name)
                    new_template_found = True
    
    # now for any needed template, first substitute any $-placeholders
    # _within_ the template with a value from the parameter.
    # Then replace the parameter itself with the template
    # e.g. parameters .. {layers: "$flat", hidden_neurons: 8,...},
    # template: flat: [...$hidden_neurons...]
    # 1 => template: flat: [...8...]
    # 2 => processed_parameters .. {layers: "[...8...]", hidden_neurons:8, ...}
    
    # we make another hack: actual template variables within templates
    # should not be replaced as they will be properly replaced in next step
    # therefore create a dict that maps template name back to itself (with
    # $ infront as marker again)
    template_names = templates.keys()
    template_to_template = dict(zip(template_names, ['$' + k for k in template_names]))
    templates_and_parameters = merge_dicts(template_to_template, parameters)
    
    # Remember which parameters were used in which templates
    # to later check if all parameters were used anywhere
    templates_to_used_params = dict()
    
    for template_name in needed_template_names:
        template_string = templates[template_name]
        params_for_this_template = get_placeholders(template_string)
        params_for_this_template = set(params_for_this_template)# - 
        #    set(template_names))
        templates_to_used_params[template_name] = params_for_this_template
        try:
            template_string = Template(template_string).substitute(
                templates_and_parameters)
            processed_templates[template_name] = template_string
        except KeyError, exc:
            additional_message = ' (when substituting variables in template {:s})'.format(
                template_name)
            add_message_to_exception(exc, additional_message)
            raise
    

    # Now it can still happen that a template has been replaced by another template
    # This is fixed in this loop
    # We do it until there are no more dollars in the parameters..
    unprocessed_template_exists = True
    i = 0
    while unprocessed_template_exists and i < 20:
        unprocessed_template_exists = False
        for template_name in processed_templates.keys():
            template_string = processed_templates[template_name]
            if '$' in template_string:
                new_str = Template(template_string).substitute(processed_templates)
                processed_templates[template_name] = new_str
                unprocessed_template_exists = True
        i+=1
    if unprocessed_template_exists: # just to prevent infinite loops
        raise ValueError("Could not replace all templates")
    return processed_templates, templates_to_used_params


def process_parameters_by_templates(parameters, templates):
    processed_parameters = copy(parameters)
    for key in parameters.keys():
        value = parameters[key]
        if isinstance(value, basestring) and value.startswith('$'):
            value = templates[value[1:]] # [1:] to remove the $ at start
            processed_parameters[key] = value
    return processed_parameters

def substitute_check_superfluous_mappings(string, params, ignore_unused=()):
    # Substitute, will fail if params are missing
    sustitution = Template(string).substitute(params)
    
    # Now check if any params were superfluous
    needed_patterns = get_placeholders(string)
    superfluous_params = set(params.keys()) - set(needed_patterns)
    superfluous_params = superfluous_params - set(ignore_unused)
    # for now hack, remove this again!! in process templates have to take care of new templates
    superfluous_params -= set(['n_increase_filter_factor', 'conv_nonlin', 'n_start_filters',
        'n_post_filters'])
    if len(superfluous_params) > 0:
        raise ValueError("Unused parameters: {:s}".format(
                str(list(superfluous_params))))
    return sustitution
    
def get_placeholders(string):
    id_pattern = '\$' + Template.idpattern # '$[_a-z][_a-z0-9]*'
    #print ("id pattern")
    #print(id_pattern)
    needed_patterns = re.findall(id_pattern, string)
    # remove dollar at start
    needed_patterns = [p[1:] for p in needed_patterns]
    #print("needed patterns", needed_patterns)
    return needed_patterns