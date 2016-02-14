from numpy.random import RandomState
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
from braindecode.datahandling.splitters import KaggleTrainValidTestSplitter
from braindecode.datahandling.preprocessing import OnlineAxiswiseStandardize
from braindecode.datahandling.batch_iteration import CntWindowsFromCntIterator
from braindecode.util import FuncAndArgs
from braindecode.veganlasagne.objectives import weighted_binary_cross_entropy
from lasagne.updates import adam
from braindecode.veganlasagne.update_modifiers import MaxNormConstraint
from braindecode.veganlasagne.monitors import LossMonitor,\
    AUCMeanMisclassMonitor, RuntimeMonitor
from braindecode.veganlasagne.stopping import Or, NoDecrease, MaxEpochs
from lasagne.objectives import categorical_crossentropy
logging.basicConfig(level=logging.DEBUG)
import lasagne.layers
from lasagne.layers import (InputLayer, DimshuffleLayer, Conv2DLayer)
from lasagne.nonlinearities import identity, sigmoid
from braindecode.veganlasagne.nonlinearities import safe_log
from braindecode.veganlasagne.layers import (StrideReshapeLayer, FinalReshapeLayer,
    Conv2DAllColsLayer)
from braindecode.experiments.experiment import Experiment



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
        numpy_cost = np.mean(np.abs(cur_out - target_data)) 
        #corr = np.corrcoef(cur_out[:3].flatten() , target_data[:3].flatten())
        overall_cost += cost * 0.001 + numpy_cost * 0.001# + 0.001 * np.mean(corr)
        if i_epoch % (n_epochs // 100) == 0:
            print i_epoch, overall_cost 
    logging.info("Done.")
        
    print overall_cost
if __name__ == "__main__":
    with_lasagne_reduced()
