
import yaml
from braindecode.experiments.parse import (create_variants_recursively, 
    merge_parameters_and_templates, transform_vals_to_string_constructor,
    create_templates_variants_from_config_objects,
    create_experiment_yaml_strings)

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
    # possibly remove equal params?
    templates = config_obj['templates']
    final_params = merge_parameters_and_templates(all_params, templates)
    assert final_params == [{'weight_init': "'funny glorot uniform template 2'\n",
        'range': 2,
        'layer': "{weight_init: 'funny glorot uniform template 2'\n}\n"}]
    
def test_variants_simple():
    config_obj = {
        'templates':{}, 
        'variants':[
            [{'range':[2,3,4]}]
        ]
    }
    templates, variants = create_templates_variants_from_config_objects(
        [config_obj])
    assert templates == {}
    assert variants == [{'range': 2}, {'range': 3}, {'range': 4}]
    
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

    