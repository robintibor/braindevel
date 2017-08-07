import logging
import os.path
import time
from collections import OrderedDict

import numpy as np
from torch import nn
from torch.nn import init
from torch.nn.functional import elu
import torch.nn.functional as F
from torch import optim

from braindecode.torch_ext.init import glorot_weight_zero_bias
from braindecode.torch_ext.modules import Expression
from braindecode.torch_ext.util import np_to_var

from hyperoptim.parse import cartesian_dict_of_lists_product, \
    product_of_list_of_lists_of_dicts
from hyperoptim.util import save_pkl_artifact

from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from braindecode.experiments.experiment import Experiment
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, \
    RuntimeMonitor
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.datautil.iterators import BalancedBatchSizeIterator
from braindecode.datautil.splitters import split_into_two_sets
from braindecode.torch_ext.util import set_random_seeds, np_to_var
from braindecode.mne_ext.signalproc import mne_apply, resample_cnt
from braindecode.datautil.signalproc import (bandpass_cnt,
                                             exponential_running_standardize)
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
from braindecode.datautil.splitters import concatenate_sets
from braindecode.torch_ext.losses import l1_loss, l2_loss

log = logging.getLogger(__name__)

class EEGNet(object):
    """
    EEGNet model from [EEGNet]_.

    Notes
    -----
    This one with timewise padding just for more exact trial-wise replication,
    does not make sense for cropped training.
    This implementation is not guaranteed to be correct, has not been checked
    by original authors, only reimplemented from the paper description.

    References
    ----------

    .. [EEGNet] Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon,
       S. M., Hung, C. P., & Lance, B. J. (2016).
       EEGNet: A Compact Convolutional Network for EEG-based
       Brain-Computer Interfaces.
       arXiv preprint arXiv:1611.08024.
    """
    def __init__(self, in_chans,
                 n_classes,
                 final_conv_length='auto',
                 input_time_length=None,
                 pool_mode='max',
                 second_kernel_size=(2,32),
                 third_kernel_size=(8,4),
                 drop_prob=0.25
                 ):

        if final_conv_length == 'auto':
            assert input_time_length is not None
        self.__dict__.update(locals())
        del self.self

    def create_network(self):
        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]
        model = nn.Sequential()
        n_filters_1 = 16
        model.add_module('conv_1', nn.Conv2d(
            self.in_chans, n_filters_1, (1, 1), stride=1, bias=True))
        model.add_module('bnorm_1', nn.BatchNorm2d(
            n_filters_1, momentum=0.01, affine=True, eps=1e-3),)
        model.add_module('elu_1', Expression(elu))
        # transpose to examples x 1 x (virtual, not EEG) channels x time
        model.add_module('permute_1', Expression(lambda x: x.permute(0,3,1,2)))

        model.add_module('drop_1', nn.Dropout(p=self.drop_prob))

        n_filters_2 = 4
        # keras padds unequal padding more in front:
        # https://stackoverflow.com/questions/43994604/padding-with-even-kernel-size-in-a-convolutional-layer-in-keras-theano

        model.add_module('conv_2', nn.Conv2d(
            1, n_filters_2, self.second_kernel_size, stride=1,
            padding=(self.second_kernel_size[0] // 2, self.second_kernel_size[1] // 2),
            bias=True))
        model.add_module('bnorm_2',nn.BatchNorm2d(
            n_filters_2, momentum=0.01, affine=True, eps=1e-3),)
        model.add_module('elu_2', Expression(elu))
        model.add_module('pool_2', pool_class(
            kernel_size=(2, 4), stride=(2, 4)))
        model.add_module('drop_2', nn.Dropout(p=self.drop_prob))

        n_filters_3 = 4
        model.add_module('conv_3', nn.Conv2d(
            n_filters_2, n_filters_3, self.third_kernel_size, stride=1,
            padding=(self.third_kernel_size[0] // 2, self.third_kernel_size[1] // 2),
            bias=True))
        model.add_module('bnorm_3',nn.BatchNorm2d(
            n_filters_3, momentum=0.01, affine=True, eps=1e-3),)
        model.add_module('elu_3', Expression(elu))
        model.add_module('pool_3', pool_class(
            kernel_size=(2, 4), stride=(2, 4)))
        model.add_module('drop_3', nn.Dropout(p=self.drop_prob))



        out = model(np_to_var(np.ones(
            (1, self.in_chans, self.input_time_length, 1),
            dtype=np.float32)))
        n_out_virtual_chans = out.cpu().data.numpy().shape[2]

        if self.final_conv_length == 'auto':
            n_out_time = out.cpu().data.numpy().shape[3]
            self.final_conv_length = n_out_time

        model.add_module('conv_classifier', nn.Conv2d(
            n_filters_3, self.n_classes,
            (n_out_virtual_chans, self.final_conv_length,), bias=True))
        model.add_module('softmax', nn.LogSoftmax())
        # Transpose back to the the logic of braindecode,
        # so time in third dimension (axis=2)
        model.add_module('permute_2', Expression(lambda x: x.permute(0,1,3,2)))
        # remove empty dim at end and potentially remove empty time dim
        # do not just use squeeze as we never want to remove first dim
        def squeeze_output(x):
            assert x.size()[3] == 1
            x = x[:,:,:,0]
            if x.size()[2] == 1:
                x = x[:,:,0]
            return x
        model.add_module('squeeze',  Expression(squeeze_output))
        glorot_weight_zero_bias(model)
        return model


def get_templates():
    return {}


def get_grid_param_list():
    dictlistprod = cartesian_dict_of_lists_product
    default_params = [{
        'save_folder': './data/models/pytorch/bcic-iv-2a/eegnet-cross-subject-time-pad/',
        'only_return_exp': False,
    }]
    subject_folder_params = dictlistprod({
        'subject_id': range(1, 10),
        'data_folder': ['/home/schirrmr/data/bci-competition-iv/2a-gdf/', ]
    })

    preproc_params = dictlistprod({
        'low_cut_hz': [0,4],
    })

    stop_params = dictlistprod({
        'max_epochs': [500],
    })

    grid_params = product_of_list_of_lists_of_dicts([
        default_params,
        subject_folder_params,
        preproc_params,
        stop_params,
    ])

    return grid_params


def sample_config_params(rng, params):
    return params


def load_and_create_set(loader, low_cut_hz):
    # Preprocessing
    high_cut_hz = 40.0
    # lets convert to millvolt for numerical stability of next operations
    cnt = loader.load()

    cnt = cnt.drop_channels(['STI 014', 'EOG-left',
                             'EOG-central', 'EOG-right'])
    assert len(cnt.ch_names) == 22
    cnt = mne_apply(lambda a: a * 1e6, cnt)
    cnt = resample_cnt(cnt, 128.0)
    cnt = mne_apply(lambda a: bandpass_cnt(
        a, low_cut_hz, high_cut_hz, cnt.info['sfreq'], filt_order=3, axis=1),
                    cnt)

    # Trial segementation
    marker_def = OrderedDict([('Left Hand', [1]), ('Right Hand', [2],),
                                  ('Foot', [3]), ('Tongue', [4])])
    ival = [500, 2500]

    dataset = create_signal_target_from_raw_mne(cnt, marker_def, ival)
    return dataset

def run_exp(data_folder, subject_id, low_cut_hz, cuda,
            max_epochs):
    test_subject_id = subject_id
    train_subject_ids = [sub_id for sub_id in range(1, 10) if
                         sub_id != test_subject_id]
    train_filenames = ['A{:02d}T.gdf'.format(sub_id) for sub_id in
                       train_subject_ids]
    valid_filename = 'A{:02d}T.gdf'.format(test_subject_id)
    test_filename = 'A{:02d}E.gdf'.format(test_subject_id)

    train_filepaths = [os.path.join(data_folder, fname) for fname in
                       train_filenames]
    valid_filepath = os.path.join(data_folder, valid_filename)
    test_filepath = os.path.join(data_folder, test_filename)
    train_loaders = [BCICompetition4Set2A(
        fpath,  labels_filename=fpath.replace('.gdf','.mat'))
        for fpath in train_filepaths]

    valid_loader = BCICompetition4Set2A(
        valid_filepath, labels_filename=valid_filepath.replace('.gdf', '.mat'))

    test_loader = BCICompetition4Set2A(
        test_filepath, labels_filename=test_filepath.replace('.gdf', '.mat'))


    train_sets = [load_and_create_set(loader, low_cut_hz=low_cut_hz)
                  for loader in train_loaders]

    train_set = concatenate_sets(train_sets)

    valid_set = load_and_create_set(valid_loader, low_cut_hz=low_cut_hz)
    test_set = load_and_create_set(test_loader, low_cut_hz=low_cut_hz)

    set_random_seeds(seed=20190706, cuda=cuda)

    n_classes = 4
    n_chans = int(train_set.X.shape[1])
    input_time_length = train_set.X.shape[2]
    model = EEGNet(n_chans, n_classes,
                   input_time_length=input_time_length, ).create_network()

    if cuda:
        model.cuda()
    log.info("Model: \n{:s}".format(str(model)))

    optimizer = optim.Adam(model.parameters())

    def elastic_net_loss(model):
        return l2_loss(model) * 1e-4 + l1_loss(model) * 1e-4

    iterator = BalancedBatchSizeIterator(batch_size=16)

    stop_criterion = MaxEpochs(max_epochs)

    monitors = [LossMonitor(), MisclassMonitor(), RuntimeMonitor()]

    model_constraint = None

    model_loss_function = elastic_net_loss

    # todo not clear if final model or best from validation loss
    # Right now it will take the one with best validation loss
    exp = Experiment(model, train_set, valid_set, test_set, iterator=iterator,
                 loss_function=F.nll_loss, optimizer=optimizer,
                 model_constraint=model_constraint,
                 monitors=monitors,
                 stop_criterion=stop_criterion,
                 remember_best_column='valid_loss',# todo valid loss
                 run_after_early_stop=False, cuda=cuda,
                model_loss_function=model_loss_function)

    exp.run()
    return exp


def run(ex, data_folder, subject_id, low_cut_hz, only_return_exp, max_epochs):
    cuda = True
    start_time = time.time()
    assert only_return_exp is False
    assert (only_return_exp is False) or (n_chans is not None)
    ex.info['finished'] = False


    exp = run_exp(data_folder, subject_id, low_cut_hz, cuda,
                  max_epochs=max_epochs)
    last_row = exp.epochs_df.iloc[-1]
    end_time = time.time()
    run_time = end_time - start_time
    ex.info['finished'] = True

    for key, val in last_row.iteritems():
        ex.info[key] = float(val)
    ex.info['runtime'] = run_time
    save_pkl_artifact(ex, exp.epochs_df, 'epochs_df.pkl')
    save_pkl_artifact(ex, exp.before_stop_df, 'before_stop_df.pkl')

