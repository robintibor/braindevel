
import yaml
from braindecode.experiments.parse import (create_variants_recursively, 
    merge_parameters_and_templates, transform_vals_to_string_constructor)

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
    assert final_params == final_params == [{'layer': 
        "{weight_init: 'funny glorot uniform template 2'\n}\n"}]