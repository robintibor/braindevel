from braindecode.experiments.parse import create_experiment_yaml_strings
from braindecode.util import FuncAndArgs
from braindecode.experiments.experiment import Experiment
import yaml
import numpy as np
from braindecode.datasets.bbci_pylearn_dataset import BBCIPylearnCleanDataset
from braindecode.mywyrm.processing import highpass_cnt
from braindecode.mywyrm.clean import BBCISetNoCleaner
import lasagne  
from numpy.random import RandomState  
lasagne.random.set_rng(RandomState(9859295))

with open('configs/experiments/debug/single_filter_net.yaml', 'r') as f:
    config_str = f.read()

with open('configs/eegnet_template.yaml', 'r') as f:
    main_template_str = f.read()

def load_raw_data(filename='data/BBCI-without-last-runs/BhNoMoSc1S001R01_ds10_1-12.BBCI.mat',
                 sensors=['CPz', 'CP1', 'CP2']):
    raw_dataset = BBCIPylearnCleanDataset(filenames=filename,
                               load_sensor_names=sensors, cleaner=BBCISetNoCleaner(),
                               cnt_preprocessors=[(highpass_cnt, {'low_cut_off_hz': 0.5})]
    )

    raw_dataset.load()
    raw_topo_view = raw_dataset.get_topological_view()
    return raw_topo_view, raw_dataset.y

all_train_strs = create_experiment_yaml_strings(config_str, main_template_str)

train_str = all_train_strs[0]

X, y = load_raw_data(filename='data/BBCI-without-last-runs/MaJaMoSc1S001R01_ds10_1-11.BBCI.mat')
classes = np.argmax(y, axis=1)

right_rest_mask = np.logical_or(classes == 0, classes==2)

X_right_rest = X[right_rest_mask].astype(np.float32)
y_right_rest = y[right_rest_mask].astype(np.int32)
classes_right_rest = classes[right_rest_mask].astype(np.int32)


assert 'in_sensors' in train_str
assert 'in_rows' in train_str
assert 'in_cols' in train_str

train_str = train_str.replace('in_sensors', '3')
train_str = train_str.replace('in_rows', '2000')
train_str = train_str.replace('in_cols', '1')

train_dict = yaml.load(train_str)

layers = train_dict['layers']

network = layers[-1]

exp = Experiment()
exp.setup(network, X_right_rest.astype(np.float32), classes_right_rest.astype(np.int32) == 2,
          loss_var_func=FuncAndArgs(lasagne.objectives.categorical_crossentropy), 
          updates_var_func=FuncAndArgs(lasagne.updates.adam))
exp.run()

