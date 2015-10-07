import logging
from braindecode.experiments.parse import create_experiment_yaml_strings
from pylearn2.utils.logger import (CustomStreamHandler, CustomFormatter)
from braindecode.experiments.experiment_runner import ExperimentsRunner


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
    
setup_logging()
    
with open('configs/experiments/debug/single_filter_net.yaml', 'r') as f:
    config_str = f.read()

with open('configs/eegnet_template.yaml', 'r') as f:
    main_template_str = f.read()
all_train_strs = create_experiment_yaml_strings(config_str, main_template_str)

exp_runner = ExperimentsRunner()
exp_runner.run(all_train_strs)
train_str = all_train_strs[0]
