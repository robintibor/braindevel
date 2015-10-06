from braindecode.experiments.parse import create_experiment_yaml_strings
from braindecode.experiments.experiment import Experiment
import yaml
import numpy as np
from braindecode.datasets.bbci_pylearn_dataset import BBCIPylearnCleanDataset
from braindecode.mywyrm.processing import highpass_cnt
from braindecode.mywyrm.clean import BBCISetNoCleaner
import lasagne
from numpy.random import RandomState  
from braindecode.datasets.preprocessing import RestrictToTwoClasses
from braindecode.datasets.dataset_splitters import DatasetSingleFoldSplitter
from braindecode.datasets.batch_iteration import get_balanced_batches
lasagne.random.set_rng(RandomState(9859295))

with open('configs/experiments/debug/single_filter_net.yaml', 'r') as f:
    config_str = f.read()

with open('configs/eegnet_template.yaml', 'r') as f:
    main_template_str = f.read()

all_train_strs = create_experiment_yaml_strings(config_str, main_template_str)

train_str = all_train_strs[0]

filename='data/BBCI-without-last-runs/MaJaMoSc1S001R01_ds10_1-11.BBCI.mat'
raw_dataset = BBCIPylearnCleanDataset(filenames=filename,
       cnt_preprocessors=[(highpass_cnt, {'low_cut_off_hz': 0.5})],
       cleaner=BBCISetNoCleaner(),
       load_sensor_names=['CPz', 'CP1', 'CP2'],
        unsupervised_preprocessor=RestrictToTwoClasses([0,2])
    )

raw_dataset.load()

# for now format y back to classes
raw_dataset.y = np.argmax(raw_dataset.y, axis=1).astype(np.int32)

dataset_splitter = DatasetSingleFoldSplitter(raw_dataset, num_folds=10, 
    test_fold_nr=9)

assert 'in_sensors' in train_str
assert 'in_rows' in train_str
assert 'in_cols' in train_str

train_str = train_str.replace('in_sensors',
    str(raw_dataset.get_topological_view().shape[1]))
train_str = train_str.replace('in_rows',
    str(raw_dataset.get_topological_view().shape[2]))
train_str = train_str.replace('in_cols', 
    str(raw_dataset.get_topological_view().shape[3]))

train_dict = yaml.load(train_str)

layers = train_dict['layers']
final_layer = layers[-1]

exp = Experiment()
exp.setup(final_layer, dataset_splitter,
          loss_var_func=lasagne.objectives.categorical_crossentropy, 
          updates_var_func=lasagne.updates.adam,
          batch_iter_func=get_balanced_batches)
exp.run()
