#!/usr/bin/env python
import numpy as np
import pickle
from glob import glob
from braindecode.experiments.load import load_model
from braindecode.experiments.experiment import create_experiment
import sys
import logging
log = logging.getLogger(__name__)

def update_result_to_new_iterator(basename):
    exp = create_experiment(basename + '.yaml')
    model = load_model(basename + '.pkl')
    exp.final_layer = model
    exp.setup()
    datasets = exp.dataset_provider.get_train_merged_valid_test(exp.dataset)
    exp.create_monitors(datasets)
    exp.monitor_manager.monitor_epoch(exp.monitor_chans, datasets, 
            exp.iterator)
    
    result = np.load(basename + '.result.pkl')

    for set_name in ['train', 'valid', 'test']:
        for chan_name in 'loss', 'sample_misclass':
            full_chan_name = set_name + '_' + chan_name

            assert np.allclose(result.monitor_channels[full_chan_name][-1], 
                        exp.monitor_chans[full_chan_name][-1], 
                        rtol=1e-3, atol=1e-3), ( 
                        "Not close: old {:f}, new: {:f}".format(result.monitor_channels[full_chan_name][-1],
                        exp.monitor_chans[full_chan_name][-1]))

    for set_name in ['train', 'valid', 'test']:
        full_chan_name = set_name + '_' + 'misclass'
        result.monitor_channels[full_chan_name][-1] = exp.monitor_chans[full_chan_name][-1]

    result_filename = basename + '.result.pkl'
    pickle.dump(result, open(result_filename, 'w'))
    
if __name__ == '__main__':
    start = int(sys.argv[1])
    stop = int(sys.argv[2])
    all_result_names = sorted(glob('data/models/paper/ours/cnt/shallow/*.result.pkl'),
      key=lambda s:int(s.split('.result.pkl')[0].split('/')[-1]))
    for i_result, result_name in enumerate(all_result_names[start:stop]):
        log.info("Running {:d} of {:d}".format(i_result, len(all_result_names[start:stop])))
        basename = result_name.replace('.result.pkl', '')
        update_result_to_new_iterator(basename)
