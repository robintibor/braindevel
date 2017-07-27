import logging
import os.path
import time
from collections import OrderedDict

import numpy as np
import torch.nn.functional as F
from torch import optim
import torch as th
from torch import nn
from torch.nn import init

from braindecode.models.deep4 import Deep4Net
from braindecode.models.util import to_dense_prediction_model
from braindecode.torch_ext.functions import square, safe_log
from braindecode.torch_ext.modules import Expression
from hyperoptim.parse import cartesian_dict_of_lists_product, \
    product_of_list_of_lists_of_dicts
from hyperoptim.util import save_pkl_artifact

from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from braindecode.experiments.experiment import Experiment
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, \
    RuntimeMonitor, CroppedTrialMisclassMonitor
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.datautil.splitters import split_into_two_sets
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.torch_ext.util import set_random_seeds, np_to_var
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.signalproc import (bandpass_cnt,
                                             exponential_running_standardize)
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne

log = logging.getLogger(__name__)


class ResidualConv(nn.Module):
    def __init__(self, in_channels, **conv_kwargs):
        super(ResidualConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, (1,1), **conv_kwargs)

    def forward(self, inputs):
        out_conv = self.conv(inputs)
        return out_conv + inputs


class ShallowFBCSPNetWith1x1Conv(object):
    """
    Shallow ConvNet model from [2]_.

    References
    ----------

    .. [2] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., 
       Glasstetter, M., Eggensperger, K., Tangermann, M., ... & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       arXiv preprint arXiv:1703.05051.
    """
    def __init__(self, in_chans,
                 n_classes,
                 input_time_length=None,
                 n_filters_time=40,
                 filter_time_length=25,
                 n_filters_spat=40,
                 pool_time_length=75,
                 pool_time_stride=15,
                 final_conv_length=30,
                 conv_nonlin=square,
                 pool_mode='mean',
                 pool_nonlin=safe_log,
                 split_first_layer=True,
                 batch_norm=True,
                 batch_norm_alpha=0.1,
                 drop_prob=0.5,
                 add_1x1=True,
                 batch_norm_1x1=False,
                 residual_1x1=False,
                 grouped_1x1=False):
        if final_conv_length == 'auto':
            assert input_time_length is not None
        self.__dict__.update(locals())
        del self.self

    def create_network(self):
        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]
        model = nn.Sequential()
        if self.split_first_layer:
            model.add_module('dimshuffle',
                             Expression(lambda x: x.permute(0, 3, 2, 1)))
            model.add_module('conv_time', nn.Conv2d(1, self.n_filters_time,
                                                    (
                                                    self.filter_time_length, 1),
                                                    stride=1, ))
            model.add_module('conv_spat',
                             nn.Conv2d(self.n_filters_time, self.n_filters_spat,
                                       (1, self.in_chans), stride=1,
                                       bias=not self.batch_norm))
            n_filters_conv = self.n_filters_spat
        else:
            model.add_module('conv_time',
                             nn.Conv2d(self.in_chans, self.n_filters_time,
                                       (self.filter_time_length, 1),
                                       stride=1,
                                       bias=not self.batch_norm))
            n_filters_conv = self.n_filters_time
        if self.batch_norm:
            model.add_module('bnorm',
                             nn.BatchNorm2d(n_filters_conv,
                                            momentum=self.batch_norm_alpha,
                                            affine=True),)
        model.add_module('conv_nonlin', Expression(self.conv_nonlin))
        model.add_module('pool',
                         pool_class(kernel_size=(self.pool_time_length, 1),
                                    stride=(self.pool_time_stride, 1)))

        if self.add_1x1:
            if self.grouped_1x1:
                groups_1x1 = n_filters_conv // 2
                if self.residual_1x1:
                    out_1x1_conv = n_filters_conv
                else:
                    out_1x1_conv = n_filters_conv // 2

            else:
                groups_1x1 = 1
                out_1x1_conv = n_filters_conv
            if self.residual_1x1:
                model.add_module('conv_1x1',
                                 ResidualConv(n_filters_conv,
                                              bias=not self.batch_norm_1x1,
                                              groups=groups_1x1))

            else:
                model.add_module('conv_1x1',
                             nn.Conv2d(n_filters_conv, out_1x1_conv,
                                       (1, 1),
                                       stride=1,
                                       bias=not self.batch_norm_1x1,
                                       groups=groups_1x1))

            n_filters_conv = out_1x1_conv
            if self.batch_norm_1x1:
                model.add_module('bnorm_1x1',
                                 nn.BatchNorm2d(n_filters_conv,
                                                momentum=self.batch_norm_alpha,
                                                affine=True),)

        model.add_module('pool_nonlin', Expression(self.pool_nonlin))
        model.add_module('drop', nn.Dropout(p=self.drop_prob))
        if self.final_conv_length == 'auto':
            out = model(np_to_var(np.ones(
                (1, self.in_chans, self.input_time_length,1),
                dtype=np.float32)))
            n_out_time = out.cpu().data.numpy().shape[2]
            self.final_conv_length = n_out_time
        model.add_module('conv_classifier',
                             nn.Conv2d(n_filters_conv, self.n_classes,
                                       (self.final_conv_length, 1), bias=True))
        model.add_module('softmax', nn.LogSoftmax())

        # remove empty dim at end and potentially remove empty time dim
        # do not just use squeeze as we never want to remove first dim
        def squeeze_output(x):
            assert x.size()[3] == 1
            x = x[:,:,:,0]
            if x.size()[2] == 1:
                x = x[:,:,0]
            return x
        model.add_module('squeeze',  Expression(squeeze_output))

        # Initialization, xavier is same as in paper...
        init.xavier_uniform(model.conv_time.weight, gain=1)
        # maybe no bias in case of no split layer and batch norm
        if self.split_first_layer or (not self.batch_norm):
            init.constant(model.conv_time.bias, 0)

        if self.split_first_layer:
            init.xavier_uniform(model.conv_spat.weight, gain=1)
            if not self.batch_norm:
                init.constant(model.conv_spat.bias, 0)
        if self.batch_norm:
            init.constant(model.bnorm.weight, 1)
            init.constant(model.bnorm.bias, 0)

        if self.add_1x1:
            if self.residual_1x1:
                init.xavier_uniform(model.conv_1x1.conv.weight, gain=1)
            elif self.grouped_1x1:
                init.xavier_uniform(model.conv_1x1.weight, gain=1)
            else:
                init.dirac(model.conv_1x1.weight)
            if not self.batch_norm_1x1:
                if self.residual_1x1:
                    init.constant(model.conv_1x1.conv.bias, 0)
                else:
                    init.constant(model.conv_1x1.bias, 0)
            else:
                init.constant(model.bnorm_1x1.weight, 1)
                init.constant(model.bnorm_1x1.bias, 0)

        init.xavier_uniform(model.conv_classifier.weight, gain=1)
        init.constant(model.conv_classifier.bias, 0)

        return model


def get_templates():
    return {}


def get_grid_param_list():
    dictlistprod = cartesian_dict_of_lists_product
    default_params = [{
        'save_folder': './data/models/pytorch/shallow_1x1/bcic-iv-2a-half-groups-11/',
        'only_return_exp': False,
    }]
    subject_folder_params = dictlistprod({
        'subject_id': list(range(1, 10)),
        'data_folder': ['/home/schirrmr/data/bci-competition-iv/2a-gdf/', ]
    })

    preproc_params = dictlistprod({
        'low_cut_hz': [4],
    })

    model_params = dictlistprod({
        'residual_1x1': [False],#True,
        'model': ['shallow_1x1',],#shallow
        'batch_norm_1x1': [True, False],#False,
        'grouped_1x1': [True],
        'add_1x1': [False]
    })

    grid_params = product_of_list_of_lists_of_dicts([
        default_params,
        subject_folder_params,
        preproc_params,
        model_params,
    ])

    return grid_params


def sample_config_params(rng, params):
    return params


def run_exp(data_folder, subject_id, low_cut_hz, model, cuda, batch_norm_1x1,
            add_1x1, residual_1x1, grouped_1x1):
    train_filename = 'A{:02d}T.gdf'.format(subject_id)
    test_filename = 'A{:02d}E.gdf'.format(subject_id)
    train_filepath = os.path.join(data_folder, train_filename)
    test_filepath = os.path.join(data_folder, test_filename)
    train_label_filepath = train_filepath.replace('.gdf', '.mat')
    test_label_filepath = test_filepath.replace('.gdf', '.mat')

    train_loader = BCICompetition4Set2A(
        train_filepath, labels_filename=train_label_filepath)
    test_loader = BCICompetition4Set2A(
        test_filepath, labels_filename=test_label_filepath)
    train_cnt = train_loader.load()
    test_cnt = test_loader.load()

    # Preprocessing
    train_cnt = train_cnt.drop_channels(['STI 014', 'EOG-left',
                                         'EOG-central', 'EOG-right'])
    assert len(train_cnt.ch_names) == 22
    # lets convert to millvolt for numerical stability of next operations
    train_cnt = mne_apply(lambda a: a * 1e6, train_cnt)
    train_cnt = mne_apply(
        lambda a: bandpass_cnt(a, low_cut_hz, 38, train_cnt.info['sfreq'],
                               filt_order=3,
                               axis=1), train_cnt)
    train_cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=1e-3,
                                                  init_block_size=1000,
                                                  eps=1e-4).T,
        train_cnt)

    test_cnt = test_cnt.drop_channels(['STI 014', 'EOG-left',
                                       'EOG-central', 'EOG-right'])
    assert len(test_cnt.ch_names) == 22
    test_cnt = mne_apply(lambda a: a * 1e6, test_cnt)
    test_cnt = mne_apply(
        lambda a: bandpass_cnt(a, low_cut_hz, 38, test_cnt.info['sfreq'],
                               filt_order=3,
                               axis=1), test_cnt)
    test_cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=1e-3,
                                                  init_block_size=1000,
                                                  eps=1e-4).T,
        test_cnt)

    marker_def = OrderedDict([('Left Hand', [1]), ('Right Hand', [2],),
                              ('Foot', [3]), ('Tongue', [4])])
    ival = [-500, 4000]

    train_set = create_signal_target_from_raw_mne(train_cnt, marker_def, ival)
    test_set = create_signal_target_from_raw_mne(test_cnt, marker_def, ival)

    train_set, valid_set = split_into_two_sets(train_set,
                                               first_set_fraction=0.8)

    set_random_seeds(seed=20190706, cuda=True)

    n_classes = 4
    n_chans = int(train_set.X.shape[1])
    input_time_length=1000
    if model == 'shallow':
        model = ShallowFBCSPNet(n_chans, n_classes, input_time_length=input_time_length,
                            final_conv_length=30).create_network()
    elif model == 'deep':
        model = Deep4Net(n_chans, n_classes, input_time_length=input_time_length,
                            final_conv_length=2).create_network()
    elif model == 'shallow_1x1':
        model = ShallowFBCSPNetWith1x1Conv(
            n_chans, n_classes, input_time_length=input_time_length,
            final_conv_length=30,
            add_1x1=add_1x1,
            batch_norm_1x1=batch_norm_1x1,
            residual_1x1=residual_1x1,
            grouped_1x1=grouped_1x1,
        ).create_network()
    else:
        assert False

    to_dense_prediction_model(model)
    if cuda:
        model.cuda()
    log.info("Model: \n{:s}".format(str(model)))

    dummy_input = np_to_var(train_set.X[:1, :, :, None])
    if cuda:
        dummy_input = dummy_input.cuda()
    out = model(dummy_input)

    n_preds_per_input = out.cpu().data.numpy().shape[2]

    optimizer = optim.Adam(model.parameters())

    iterator = CropsFromTrialsIterator(batch_size=60,
                                       input_time_length=input_time_length,
                                       n_preds_per_input=n_preds_per_input)

    stop_criterion = Or([MaxEpochs(800),
                         NoDecrease('valid_misclass', 80)])

    monitors = [LossMonitor(), MisclassMonitor(col_suffix='sample_misclass'),
                CroppedTrialMisclassMonitor(
                    input_time_length=input_time_length), RuntimeMonitor()]

    model_constraint = MaxNormDefaultConstraint()

    loss_function = lambda preds, targets: F.nll_loss(
        th.mean(preds, dim=2)[:, :, 0], targets)

    exp = Experiment(model, train_set, valid_set, test_set, iterator=iterator,
                     loss_function=loss_function, optimizer=optimizer,
                     model_constraint=model_constraint,
                     monitors=monitors,
                     stop_criterion=stop_criterion,
                     remember_best_column='valid_misclass',
                     run_after_early_stop=True, cuda=cuda)
    exp.run()
    return exp


def run(ex, data_folder, subject_id, low_cut_hz, model, only_return_exp,
        add_1x1, batch_norm_1x1, residual_1x1, grouped_1x1):
    cuda = True
    start_time = time.time()
    assert only_return_exp is False
    assert (only_return_exp is False) or (n_chans is not None)
    ex.info['finished'] = False

    exp = run_exp(data_folder, subject_id, low_cut_hz, model, cuda,
                  add_1x1=add_1x1,
                  batch_norm_1x1=batch_norm_1x1,
                  residual_1x1=residual_1x1,
                  grouped_1x1=grouped_1x1)
    last_row = exp.epochs_df.iloc[-1]
    end_time = time.time()
    run_time = end_time - start_time
    ex.info['finished'] = True

    for key, val in last_row.iteritems():
        ex.info[key] = float(val)
    ex.info['runtime'] = run_time
    save_pkl_artifact(ex, exp.epochs_df, 'epochs_df.pkl')
    save_pkl_artifact(ex, exp.before_stop_df, 'before_stop_df.pkl')

