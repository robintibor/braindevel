import numpy as np
import theano
from theano.tensor.nnet import conv2d
import theano.tensor as T
import logging
from collections import OrderedDict
from braindecode.veganlasagne.pool import SumPool2dLayer
from lasagne.layers.special import NonlinearityLayer
from lasagne.layers.noise import DropoutLayer
from braindecode.datasets.grasp_lift import KaggleGraspLiftSet
from braindecode.datahandling.splitters import KaggleTrainValidTestSplitter,\
    FixedTrialSplitter
from braindecode.datahandling.preprocessing import OnlineAxiswiseStandardize
from braindecode.datahandling.batch_iteration import CntWindowsFromCntIterator
from braindecode.util import FuncAndArgs
from braindecode.veganlasagne.objectives import weighted_binary_cross_entropy
from lasagne.updates import adam, sgd
from braindecode.veganlasagne.update_modifiers import MaxNormConstraint
from braindecode.veganlasagne.monitors import LossMonitor,\
    AUCMeanMisclassMonitor, RuntimeMonitor, MisclassMonitor, auc_classes_mean,\
    Monitor
from braindecode.veganlasagne.stopping import Or, NoDecrease, MaxEpochs
from lasagne.objectives import categorical_crossentropy
from braindecode.datasets.pylearn import DenseDesignMatrixWrapper
logging.basicConfig(level=logging.DEBUG)
import lasagne.layers
from lasagne.layers import (InputLayer, DimshuffleLayer, Conv2DLayer)
from lasagne.nonlinearities import identity, sigmoid
from braindecode.veganlasagne.nonlinearities import safe_log
from braindecode.veganlasagne.layers import (StrideReshapeLayer, FinalReshapeLayer,
    Conv2DAllColsLayer)
from braindecode.experiments.experiment import Experiment
from numpy.random import RandomState

class FakeAUCMeanMisclassMonitor(Monitor):
    def __init__(self, input_time_length=None, n_sample_preds=None):
        self.input_time_length = input_time_length
        self.n_sample_preds = n_sample_preds
    
    def setup(self, monitor_chans, datasets):
        for setname in datasets:
            assert setname in ['train', 'valid', 'test']
            monitor_key = "{:s}_misclass".format(setname)
            monitor_chans[monitor_key] = []
        self.rng = RandomState(328774)

    def monitor_epoch(self, monitor_chans):
        return

    def monitor_set(self, monitor_chans, setname, all_preds, losses, 
            all_batch_sizes, targets, dataset):
        # remove last preds that were duplicates due to overlap of final windows
        n_samples = len(dataset.y)
        all_preds_arr = np.concatenate(all_preds)
        targets = np.round(self.rng.rand(*all_preds_arr.shape)).astype(np.int32)
        auc_mean = auc_classes_mean(targets, all_preds_arr)
        misclass = 1 - auc_mean
        monitor_key = "{:s}_misclass".format(setname)
        monitor_chans[monitor_key].append(float(misclass))
        
        
class DummyIterator(object):
    def __init__(self):
        self.rng = RandomState(328774)
    
    def reset_rng(self):
        self.rng = RandomState(328774)
    
    def get_batches(self, dataset, shuffle):
        batch_size = 20
        
        for _ in xrange(45):
            topo = dataset.get_topological_view()
            batch_topo = np.float32(np.ones((batch_size, topo.shape[1],
                 2000, topo.shape[3])))
            batch_y = np.ones((1392 * batch_size, dataset.y.shape[1])).astype(np.int32)
        
       
            yield batch_topo, batch_y 

def sgd_iterator_fake_auc():
    # no crash ?!
    # 4) adam durch sgd mit fester lernrate ersetzen
    # 5) iterator durch neu geschriebenen der einfach batches randommaessig holt und achsen switched...
    # 6) monitor mit fake auc mean misclass ersetzen
    network = InputLayer([None,32,2000,1])
    network = DimshuffleLayer(network, [0,3,2,1])
    network = Conv2DLayer(network, num_filters=40, filter_size=[30,1],
        nonlinearity=identity, name='time_conv')
    network = Conv2DAllColsLayer(network, num_filters=40, filter_size=[1,-1],
        nonlinearity=T.sqr, name='spat_conv')
    network = SumPool2dLayer(network, pool_size=[50,1], stride=[1,1],
        mode='average_exc_pad')
    network=StrideReshapeLayer(network, n_stride=10)
    network = NonlinearityLayer(network, nonlinearity=safe_log)
    network = DropoutLayer(network, p=0.5)
    network = Conv2DLayer(network, num_filters=6, filter_size=[54,1],
        nonlinearity=identity, name='final_dense')
    network = FinalReshapeLayer(network)
    network = NonlinearityLayer(network, nonlinearity=sigmoid)

    preprocessor = None

    iterator = DummyIterator()
    loss_expression = categorical_crossentropy
    updates_expression = FuncAndArgs(sgd, learning_rate=0.1)
    updates_modifier = None #AUCMeanMisclassMonitor(input_time_length=2000,
        #n_sample_preds=1392), 
    monitors = [LossMonitor(),FakeAUCMeanMisclassMonitor(), RuntimeMonitor()]
    stop_criterion = MaxEpochs(1000)
    rng = RandomState(30493049)
    random_topo = rng.randn(1334972, 32, 1, 1)
    random_y = np.round(rng.rand(1334972, 6)).astype(np.int32)

    dataset = DenseDesignMatrixWrapper(topo_view=random_topo, y=random_y,
        axes=('b', 'c', 0, 1))
    splitter = FixedTrialSplitter(n_train_trials=1234972, valid_set_fraction=0.1)
    
    exp = Experiment(network, dataset, splitter, preprocessor, iterator, loss_expression, 
        updates_expression, updates_modifier, monitors, stop_criterion, remember_best_chan='valid_misclass')
    exp.setup()
    datasets = exp.dataset_provider.get_train_merged_valid_test(dataset)
    exp.create_monitors(datasets)
    exp.run_until_second_stop()

def sgd_iterator():
    # no crash?!
    # 4) adam durch sgd mit fester lernrate ersetzen
    # 5) iterator durch neu geschriebenen der einfach batches randommaessig holt und achsen switched...
    network = InputLayer([None,32,2000,1])
    network = DimshuffleLayer(network, [0,3,2,1])
    network = Conv2DLayer(network, num_filters=40, filter_size=[30,1],
        nonlinearity=identity, name='time_conv')
    network = Conv2DAllColsLayer(network, num_filters=40, filter_size=[1,-1],
        nonlinearity=T.sqr, name='spat_conv')
    network = SumPool2dLayer(network, pool_size=[50,1], stride=[1,1],
        mode='average_exc_pad')
    network=StrideReshapeLayer(network, n_stride=10)
    network = NonlinearityLayer(network, nonlinearity=safe_log)
    network = DropoutLayer(network, p=0.5)
    network = Conv2DLayer(network, num_filters=6, filter_size=[54,1],
        nonlinearity=identity, name='final_dense')
    network = FinalReshapeLayer(network)
    network = NonlinearityLayer(network, nonlinearity=sigmoid)

    preprocessor = None

    iterator = DummyIterator()
    loss_expression = categorical_crossentropy
    updates_expression = FuncAndArgs(sgd, learning_rate=0.1)
    updates_modifier = None #AUCMeanMisclassMonitor(input_time_length=2000,
        #n_sample_preds=1392), 
    monitors = [LossMonitor(),MisclassMonitor(), RuntimeMonitor()]
    stop_criterion = MaxEpochs(1000)
    rng = RandomState(30493049)
    random_topo = rng.randn(1334972, 32, 1, 1)
    random_y = np.round(rng.rand(1334972, 6)).astype(np.int32)

    dataset = DenseDesignMatrixWrapper(topo_view=random_topo, y=random_y,
        axes=('b', 'c', 0, 1))
    splitter = FixedTrialSplitter(n_train_trials=1234972, valid_set_fraction=0.1)
    
    exp = Experiment(network, dataset, splitter, preprocessor, iterator, loss_expression, 
        updates_expression, updates_modifier, monitors, stop_criterion, remember_best_chan='valid_misclass')
    exp.setup()
    datasets = exp.dataset_provider.get_train_merged_valid_test(dataset)
    exp.create_monitors(datasets)
    exp.run_until_second_stop()
    
def sgd_exp():
    # successfully crashed Monday, 15th february 19:18
    # 4) adam durch sgd ersetzen
    network = InputLayer([None,32,2000,1])
    network = DimshuffleLayer(network, [0,3,2,1])
    network = Conv2DLayer(network, num_filters=40, filter_size=[30,1],
        nonlinearity=identity, name='time_conv')
    network = Conv2DAllColsLayer(network, num_filters=40, filter_size=[1,-1],
        nonlinearity=T.sqr, name='spat_conv')
    network = SumPool2dLayer(network, pool_size=[50,1], stride=[1,1],
        mode='average_exc_pad')
    network=StrideReshapeLayer(network, n_stride=10)
    network = NonlinearityLayer(network, nonlinearity=safe_log)
    network = DropoutLayer(network, p=0.5)
    network = Conv2DLayer(network, num_filters=6, filter_size=[54,1],
        nonlinearity=identity, name='final_dense')
    network = FinalReshapeLayer(network)
    network = NonlinearityLayer(network, nonlinearity=sigmoid)

    preprocessor = None

    iterator = CntWindowsFromCntIterator(batch_size=20,
        input_time_length=2000, n_sample_preds=1392,
        oversample_targets=True, remove_baseline_mean=False)
    loss_expression = categorical_crossentropy
    updates_expression =  FuncAndArgs(sgd, learning_rate=0.1)
    updates_modifier = None
    monitors = [LossMonitor(), AUCMeanMisclassMonitor(input_time_length=2000,
        n_sample_preds=1392), RuntimeMonitor()]
    stop_criterion = MaxEpochs(1000)
    rng = RandomState(30493049)
    random_topo = rng.randn(1334972, 32, 1, 1)
    random_y = np.round(rng.rand(1334972, 6)).astype(np.int32)

    dataset = DenseDesignMatrixWrapper(topo_view=random_topo, y=random_y,
        axes=('b', 'c', 0, 1))
    splitter = FixedTrialSplitter(n_train_trials=1234972, valid_set_fraction=0.1)
    
    exp = Experiment(network, dataset, splitter, preprocessor, iterator, loss_expression, 
        updates_expression, updates_modifier, monitors, stop_criterion, remember_best_chan='valid_misclass')
    exp.setup()
    datasets = exp.dataset_provider.get_train_merged_valid_test(dataset)
    exp.create_monitors(datasets)
    exp.run_until_second_stop()

def random_set():
    # successfully crashed
    # 3) kaggle grasp lift set durch random set ersetzen, entsprechender splitter auch
    network = InputLayer([None,32,2000,1])
    network = DimshuffleLayer(network, [0,3,2,1])
    network = Conv2DLayer(network, num_filters=40, filter_size=[30,1],
        nonlinearity=identity, name='time_conv')
    network = Conv2DAllColsLayer(network, num_filters=40, filter_size=[1,-1],
        nonlinearity=T.sqr, name='spat_conv')
    network = SumPool2dLayer(network, pool_size=[50,1], stride=[1,1],
        mode='average_exc_pad')
    network=StrideReshapeLayer(network, n_stride=10)
    network = NonlinearityLayer(network, nonlinearity=safe_log)
    network = DropoutLayer(network, p=0.5)
    network = Conv2DLayer(network, num_filters=6, filter_size=[54,1],
        nonlinearity=identity, name='final_dense')
    network = FinalReshapeLayer(network)
    network = NonlinearityLayer(network, nonlinearity=sigmoid)

    #dataset = KaggleGraspLiftSet(data_folder='data/kaggle-grasp-lift/',
    #    i_subject=8, resample_half=True, standardize=False)
    #splitter = KaggleTrainValidTestSplitter(use_test_as_valid=False)
    preprocessor = None

    iterator = CntWindowsFromCntIterator(batch_size=20,
        input_time_length=2000, n_sample_preds=1392,
        oversample_targets=True, remove_baseline_mean=False)
    loss_expression = categorical_crossentropy
    updates_expression = adam
    updates_modifier = None
    monitors = [LossMonitor(), AUCMeanMisclassMonitor(input_time_length=2000,
        n_sample_preds=1392), RuntimeMonitor()]
    stop_criterion = MaxEpochs(1000)
    rng = RandomState(30493049)
    random_topo = rng.randn(1334972, 32, 1, 1)
    random_y = np.round(rng.rand(1334972, 6)).astype(np.int32)

    dataset = DenseDesignMatrixWrapper(topo_view=random_topo, y=random_y,
        axes=('b', 'c', 0, 1))
    splitter = FixedTrialSplitter(n_train_trials=1234972, valid_set_fraction=0.1)
    
    exp = Experiment(network, dataset, splitter, preprocessor, iterator, loss_expression, 
        updates_expression, updates_modifier, monitors, stop_criterion, remember_best_chan='valid_misclass')
    exp.setup()
    datasets = exp.dataset_provider.get_train_merged_valid_test(dataset)
    exp.create_monitors(datasets)
    exp.run_until_second_stop()

def after_early_stop():
    # 2) nurnoch nach early stop weiterlaufen
    network = InputLayer([None,32,2000,1])
    network = DimshuffleLayer(network, [0,3,2,1])
    network = Conv2DLayer(network, num_filters=40, filter_size=[30,1],
        nonlinearity=identity, name='time_conv')
    network = Conv2DAllColsLayer(network, num_filters=40, filter_size=[1,-1],
        nonlinearity=T.sqr, name='spat_conv')
    network = SumPool2dLayer(network, pool_size=[50,1], stride=[1,1],
        mode='average_exc_pad')
    network=StrideReshapeLayer(network, n_stride=10)
    network = NonlinearityLayer(network, nonlinearity=safe_log)
    network = DropoutLayer(network, p=0.5)
    network = Conv2DLayer(network, num_filters=6, filter_size=[54,1],
        nonlinearity=identity, name='final_dense')
    network = FinalReshapeLayer(network)
    network = NonlinearityLayer(network, nonlinearity=sigmoid)
    
    dataset = KaggleGraspLiftSet(data_folder='data/kaggle-grasp-lift/',
        i_subject=8, resample_half=True, standardize=False)
    splitter = KaggleTrainValidTestSplitter(use_test_as_valid=False)
    preprocessor = None
    
    iterator = CntWindowsFromCntIterator(batch_size=20,
        input_time_length=2000, n_sample_preds=1392,
        oversample_targets=True, remove_baseline_mean=False)
    loss_expression = categorical_crossentropy
    updates_expression = adam
    updates_modifier = None
    monitors = [LossMonitor(), AUCMeanMisclassMonitor(input_time_length=2000,
        n_sample_preds=1392), RuntimeMonitor()]
    stop_criterion = MaxEpochs(1000)
    
    exp = Experiment(network, dataset, splitter, preprocessor, iterator, loss_expression, 
        updates_expression, updates_modifier, monitors, stop_criterion, remember_best_chan='valid_misclass')
    exp.setup()
    exp.run_after_early_stop()

def with_lasagne_reduced():
    # successfully crashed
    # 1 ) preprocessor None, nur max epochs update modifier weg, categorical cross entropy
    network = InputLayer([None,32,2000,1])
    network = DimshuffleLayer(network, [0,3,2,1])
    network = Conv2DLayer(network, num_filters=40, filter_size=[30,1],
        nonlinearity=identity, name='time_conv')
    network = Conv2DAllColsLayer(network, num_filters=40, filter_size=[1,-1],
        nonlinearity=T.sqr, name='spat_conv')
    network = SumPool2dLayer(network, pool_size=[50,1], stride=[1,1],
        mode='average_exc_pad')
    network=StrideReshapeLayer(network, n_stride=10)
    network = NonlinearityLayer(network, nonlinearity=safe_log)
    network = DropoutLayer(network, p=0.5)
    network = Conv2DLayer(network, num_filters=6, filter_size=[54,1],
        nonlinearity=identity, name='final_dense')
    network = FinalReshapeLayer(network)
    network = NonlinearityLayer(network, nonlinearity=sigmoid)
    
    dataset = KaggleGraspLiftSet(data_folder='data/kaggle-grasp-lift/',
        i_subject=8, resample_half=True, standardize=False)
    splitter = KaggleTrainValidTestSplitter(use_test_as_valid=False)
    preprocessor = None
    iterator = CntWindowsFromCntIterator(batch_size=20,
        input_time_length=2000, n_sample_preds=1392,
        oversample_targets=True, remove_baseline_mean=False)
    loss_expression = categorical_crossentropy
    updates_expression = adam
    updates_modifier = None
    monitors = [LossMonitor(), AUCMeanMisclassMonitor(input_time_length=2000,
        n_sample_preds=1392), RuntimeMonitor()]
    stop_criterion = MaxEpochs(1000)
    
    exp = Experiment(network, dataset, splitter, preprocessor, iterator, loss_expression, 
        updates_expression, updates_modifier, monitors, stop_criterion, remember_best_chan='valid_misclass')
    exp.setup()
    exp.run()
    

def with_lasagne():
    # Crashed successfully after first epoch at one GPU
    # Als naechstes raus: 
    # 1 ) preprocessor, nur max epochs update modifier weg, categorical cross entropy
    # 2) nurnoch nach early stop weiterlaufen
    # 3) kaggle grasp lift set durch random set ersetzen, entsprechender splitter auch
    # 4) adam durch sgd mit fester lernrate ersetzen
    # 5) creat escript with just the sgd_exp, copy oer all classes necessary, do not import anything
    
    #... here it no longer crashed :/
    # 5) iterator durch neu geschriebenen der einfach batches randommaessig holt und achsen switched...
    # 6) komplette experimentklasse ersetzen, funktionen hierhin kopieren
    network = InputLayer([None,32,2000,1])
    network = DimshuffleLayer(network, [0,3,2,1])
    network = Conv2DLayer(network, num_filters=40, filter_size=[30,1],
        nonlinearity=identity, name='time_conv')
    network = Conv2DAllColsLayer(network, num_filters=40, filter_size=[1,-1],
        nonlinearity=T.sqr, name='spat_conv')
    network = SumPool2dLayer(network, pool_size=[50,1], stride=[1,1],
        mode='average_exc_pad')
    network=StrideReshapeLayer(network, n_stride=10)
    network = NonlinearityLayer(network, nonlinearity=safe_log)
    network = DropoutLayer(network, p=0.5)
    network = Conv2DLayer(network, num_filters=6, filter_size=[54,1],
        nonlinearity=identity, name='final_dense')
    network = FinalReshapeLayer(network)
    network = NonlinearityLayer(network, nonlinearity=sigmoid)
    
    dataset = KaggleGraspLiftSet(data_folder='data/kaggle-grasp-lift/',
        i_subject=8, resample_half=True, standardize=False)
    splitter = KaggleTrainValidTestSplitter(use_test_as_valid=False)
    preprocessor = OnlineAxiswiseStandardize(axis= ['c', 1])
    iterator = CntWindowsFromCntIterator(batch_size=20,
        input_time_length=2000, n_sample_preds=1392,
        oversample_targets=True, remove_baseline_mean=False)
    loss_expression = FuncAndArgs(weighted_binary_cross_entropy,
        imbalance_factor=20)
    updates_expression = adam
    updates_modifier = MaxNormConstraint(layer_names_to_norms=
        dict(time_conv=2.0, spat_conv=2.0, final_dense=0.5))
    monitors = [LossMonitor(), AUCMeanMisclassMonitor(input_time_length=2000,
        n_sample_preds=1392), RuntimeMonitor()]
    stop_criterion = Or(
        stop_criteria=[NoDecrease(chan_name='valid_misclass', num_epochs=100,
        min_decrease=0.),
        MaxEpochs(1000)])

    exp = Experiment(network, dataset, splitter, preprocessor, iterator, loss_expression, 
        updates_expression, updates_modifier, monitors, stop_criterion, remember_best_chan='valid_misclass')
    exp.setup()
    exp.run()

def without_lasagne():
    logging.info("Generating random data...")
    rng = RandomState(9387498374)
    in_data = rng.randn(2000,45,2000,1).astype(np.float32)
    logging.info("... Done.")
    
    params = []
    
    n_layers = 5
    
    input_sym = T.ftensor4()
    
    cur_out = input_sym
    out_shape = in_data.shape
    for i_layer in xrange(n_layers):
        cur_out = cur_out.dimshuffle(0,3,2,1)
        W1 = theano.shared(rng.randn(40,1,50,1).astype(np.float32) * 0.1)
        cur_out = conv2d(cur_out, W1, border_mode='valid')
        params.append(W1)
        W2 = theano.shared(rng.randn(45,40,1,45).astype(np.float32) * 0.1)
        cur_out = conv2d(cur_out, W2, border_mode='valid')
        params.append(W2)
        out_shape = (out_shape[0], out_shape[1], out_shape[2] - 49, 1)
        cur_out = T.sqrt(cur_out * cur_out)
        # here maybe more complicated stuff like set subtensor?
    logging.info("out shape {:s}".format(out_shape))
    
    target_sym = T.ftensor4()
    
    cost = T.mean(T.abs_(cur_out - target_sym))
    
    grads = T.grad(cost, params)
    
    updates = OrderedDict()
    for i_p, p in enumerate(params):
        updated_p = p - 0.001 * grads[i_p]
        updates[p] = updated_p
        
        
    logging.info("Compiling...")
    update_fn = theano.function([input_sym, target_sym],
        [cur_out, cost], updates=updates)
    logging.info("Done.")
    
    n_epochs=2000
    overall_cost = 0
    logging.info("Running...")
    for i_epoch in xrange(n_epochs):
        this_inds = rng.choice(in_data.shape[0], size=15, replace=False)
        this_in_data = in_data[this_inds]
        target_data = rng.randn(15, *out_shape[1:]).astype(np.float32)
        cur_out, cost = update_fn(this_in_data, target_data)
        #numpy_cost = np.mean(np.abs(cur_out - target_data)) 
        auc = auc_classes_mean(cur_out, cur_out.shape)
        print auc
        #corr = np.corrcoef(cur_out[:3].flatten() , target_data[:3].flatten())
        #verall_cost += cost * 0.001 + numpy_cost * 0.001# + 0.001 * np.mean(corr)
        if i_epoch % (n_epochs // 100) == 0:
            print i_epoch, overall_cost 
    logging.info("Done.")
        
    print overall_cost

if __name__ == "__main__":
    sgd_iterator_fake_auc()
