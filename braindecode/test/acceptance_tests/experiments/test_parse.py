from tempfile import NamedTemporaryFile

from braindecode.experiments.parse import (
    create_experiment_yaml_strings_from_files)

def test_extended_filenames():
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
    
    config_str_sub = """
    {
        extends: ['extendsfilename'],
        
        variants: [[{
            range: [8,10], 
        }]]
    }
    """
    main_template_str = """layer: $layer"""
    with NamedTemporaryFile() as (config_top_file
        ), NamedTemporaryFile() as (config_bottom_file
        ), NamedTemporaryFile() as main_template_file:
        config_str_sub = config_str_sub.replace('extendsfilename',
                                               config_top_file.name)
        config_top_file.write(config_str)
        config_bottom_file.write(config_str_sub)
        main_template_file.write(main_template_str)
        config_top_file.flush()
        config_bottom_file.flush()
        main_template_file.flush()
        train_strings = create_experiment_yaml_strings_from_files(
            config_bottom_file.name, main_template_file.name)
        assert train_strings == [
            "layer: {weight_init: 'funny glorot uniform template 8'\n}\n",
            "layer: {weight_init: 'funny glorot uniform template 10'\n}\n"]
        
        
def test_multiple_extended_filenames():
    config_str = """
    {
        templates: {
            a_layer: {
                weight_init: $weight_init,
                attribute_sub: $attribute_sub,
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
    
    config_str_sub = """
    {
        extends: ['topfilename'],
        
        variants: [[{
            range: [8,10], 
        }]]
    }
    """
    
    config_str_sub_sub = """
    {
        extends: ['subfilename'],
        
        variants: [[{
            attribute_sub: [2,6,3], 
        }]]
    }
    """
    main_template_str = """layer: $layer"""
    with NamedTemporaryFile() as (config_top_file
        ), NamedTemporaryFile() as (config_sub_file
        ), NamedTemporaryFile() as (config_sub_sub_file
        ), NamedTemporaryFile() as main_template_file:
        config_str_sub = config_str_sub.replace('topfilename',
                                               config_top_file.name)
        config_str_sub_sub = config_str_sub_sub.replace('subfilename',
                                               config_sub_file.name)
        config_top_file.write(config_str)
        config_sub_file.write(config_str_sub)
        config_sub_sub_file.write(config_str_sub_sub)
        main_template_file.write(main_template_str)
        config_top_file.flush()
        config_sub_file.flush()
        config_sub_sub_file.flush()
        main_template_file.flush()
        train_strings = create_experiment_yaml_strings_from_files(
            config_sub_sub_file.name, main_template_file.name)
        assert train_strings == [
            "layer: {weight_init: 'funny glorot uniform template 8'\n, attribute_sub: 2}\n",
            "layer: {weight_init: 'funny glorot uniform template 8'\n, attribute_sub: 6}\n",
            "layer: {weight_init: 'funny glorot uniform template 8'\n, attribute_sub: 3}\n",
            "layer: {weight_init: 'funny glorot uniform template 10'\n, attribute_sub: 2}\n",
            "layer: {weight_init: 'funny glorot uniform template 10'\n, attribute_sub: 6}\n",
            "layer: {weight_init: 'funny glorot uniform template 10'\n, attribute_sub: 3}\n"]       