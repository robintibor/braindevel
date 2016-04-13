import yaml
from braindecode.experiments.parse import (create_variants_recursively, 
    merge_parameters_and_templates, transform_vals_to_string_constructor,
    create_experiment_yaml_strings, 
    substitute_check_superfluous_mappings)
import ConfigParser

def test_variants_simple():
    config_obj = {
        'templates':{}, 
        'variants':[
            [{'range':[2,3,4]}]
        ]
    }
    parser = ConfigParser()
    parser.filter_params= ()
    parser.only_first_n_sets = False
    parser.debug = False
    parser.command_line_params= None
    
    templates, variants = parser._create_templates_variants([config_obj])
    assert templates == {}
    assert variants == [{'range': 2}, {'range': 3}, {'range': 4}]


def test_superfluous_param():
    ## superfluous param
    substituted = substitute_check_superfluous_mappings('$param', dict(param=1))
    assert substituted == '1'
    
    with pytest.raises(ValueError) as excinfo:
        substitute_check_superfluous_mappings('$param', dict(param=1, param2=2))
    assert excinfo.value.message == "Unused parameters: ['param2']"

def test_ignore_superfluous_param():
    # superfluous ignored
    substituted = substitute_check_superfluous_mappings('$param',
        dict(param=1, param2=2), ignore_unused=['param2'])
    assert substituted == '1'    

def test_single_param():
    template_str = "$test_param"
    experiment_str = """
    {
        variants: [[{
           test_param: [1], 
        }]]
    }"""
    
    
    all_train_strs = create_experiment_yaml_strings([experiment_str],
        template_str)
        
    assert all_train_strs[0] == "1"

def test_using_param_two_times():
    template_str = "$test_param, template: $the_template"
    experiment_str = """
    {
        templates: {
            test_template: {number: $test_param},
        },
        variants: [[{
           test_param: [1,2], 
           the_template: [$test_template],
        }]]
        
    }
    """
    all_train_strs = create_experiment_yaml_strings([experiment_str],
        template_str)
    assert all_train_strs[0] == "1, template: {number: 1}\n"
    assert all_train_strs[1] == "2, template: {number: 2}\n"

    
    
def test_template_within_template():
    config_str = """
    {
        templates: {
            a_layer: {
                weight_init: $weight_init,
            },
            glorotuniform: 'funny glorot uniform template $range',
        },
        variants: [[
            {
                weight_init: [$glorotuniform], 
                range: [2], 
                layer:[$a_layer],
            },
        ]]
    }
    """
    
    yaml.add_constructor(u'!TransformValsToString', transform_vals_to_string_constructor)
    
    config_obj = yaml.load(config_str.replace("templates:", 
        "templates: !TransformValsToString"))
    
    all_params =  create_variants_recursively(config_obj['variants'])
    templates = config_obj['templates']
    final_params, templates_to_parameters = merge_parameters_and_templates(all_params, templates)
    assert final_params == [{'weight_init': "'funny glorot uniform template 2'\n",
        'range': 2,
        'layer': "{weight_init: 'funny glorot uniform template 2'\n}\n"}]
     
    assert templates_to_parameters ==  [dict(a_layer=set(['weight_init']),
                                   glorotuniform=set(['range']))]
def test_missing_param():
    ## Failing missing param
    template_str = "$test_param"
    experiment_str = """
    {
        variants: [[{
           test_param_2: [2], 
        }]]
    }"""
    
    with pytest.raises(AssertionError) as excinfo:
        all_train_strs = create_experiment_yaml_strings([experiment_str],
            template_str)
    assert excinfo.value.message == 'test_param'

def test_unused_param():
    template_str = "$test_param"
    experiment_str = """
    {
        variants: [[{
            test_param: [1],
           test_param_2: [2], 
        }]]
    }"""
    with pytest.raises(AssertionError) as excinfo:
        all_train_strs = create_experiment_yaml_strings([experiment_str],
            template_str)
    assert excinfo.value.message == "Unused parameters: ['test_param_2']"
    
def test_ignore_unused_param():
    template_str = "$test_param"
    experiment_str = """
    {
        variants: [[{
           test_param: [1],
           test_param_2: [2],
           ignore_unused: [['test_param_2']],
        }]]
    }"""
    all_train_strs = create_experiment_yaml_strings([experiment_str],
        template_str)
    assert all_train_strs[0] == '1'

def test_param_used_in_template():
    template_str = "$the_template"
    experiment_str = """
    
    {
        templates: {
            test_template: $test_param,
        },
        variants: [[{
            test_param: [1],
            the_template: [$test_template]
        }]]
    }"""
    all_train_strs = create_experiment_yaml_strings([experiment_str],
        template_str)
    assert all_train_strs[0] == '1\n...\n' # somehow our create_config_objects add these \n...\n ...

def test_cycle_in_param_template_logic():
    template_str = "$test_param"
    experiment_str = """
    
    {
        templates: {
            test_template: $test_param_2,
        },
        variants: [[{
            test_param: [1],
            test_param_2: [$test_template],
        }]]
    }"""
        
    with pytest.raises(AssertionError) as excinfo:
        create_experiment_yaml_strings([experiment_str],
            template_str)
    assert excinfo.value.message == "Could not replace all templates"

def test_param_in_unused_template():
    template_str = "$test_param"
    experiment_str = """
    
    {
        templates: {
            test_template: $test_param_2,
        },
        variants: [[{
            test_param: [1],
            test_param_2: [2],
        }]]
    }"""
        
    with pytest.raises(AssertionError) as excinfo:
        create_experiment_yaml_strings([experiment_str],
            template_str)
    assert excinfo.value.message == "Unused parameters: ['test_param_2']"

def test_param_in_nested_template():
    template_str = "$test_param"
    experiment_str = """
    
    {
        templates: {
            top_template: $test_template,
            test_template: $test_param_2,
        },
        variants: [[{
            test_param: [$top_template],
            test_param_2: [2],
        }]]
    }"""
        
    all_train_strs = create_experiment_yaml_strings([experiment_str],
        template_str)
    assert all_train_strs == [u'2\n...\n\n...\n']
    
def test_param_in_unused_nested_template():
    template_str = ""
    experiment_str = """
    
    {
        templates: {
            top_template: $test_template,
            test_template: $test_param_2,
        },
        variants: [[{
            test_param: [$top_template],
            test_param_2: [2],
            ignore_unused: [['test_param']],
        }]]
    }"""
        
    with pytest.raises(AssertionError) as excinfo:
        create_experiment_yaml_strings([experiment_str],
            template_str)
    assert excinfo.value.message == "Unused parameters: ['test_param_2']"