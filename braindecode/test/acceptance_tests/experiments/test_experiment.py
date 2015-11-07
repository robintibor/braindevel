import braindecode.veganlasagne.monitors
from braindecode.datahandling.preprocessing import OnlineAxiswiseStandardize
from braindecode.datahandling.splitters import FixedTrialSplitter
from braindecode.datahandling.batch_iteration import (BalancedBatchIterator,
    WindowsIterator)
from braindecode.datasets.pylearn import DenseDesignMatrixWrapper
from braindecode.experiments.experiment import Experiment
from lasagne.layers import (DenseLayer, InputLayer)

import lasagne
from numpy.random import RandomState
import numpy as np
from braindecode.veganlasagne.update_modifiers import MaxNormConstraint

def test_experiment_fixed_split():
    """ Regression test, checking that values have not changed from original run"""
    data_rng = RandomState(398765905)
    rand_topo = data_rng.rand(200,10,10,3).astype(np.float32)
    rand_y = np.int32(data_rng.rand(200) > 0.5)
    rand_topo[rand_y == 1] += 0.01
    rand_set = DenseDesignMatrixWrapper(topo_view=rand_topo, y=rand_y)

    lasagne.random.set_rng(RandomState(9859295))
    in_layer = InputLayer(shape= [None, 10,10,3])
    network = DenseLayer(incoming=in_layer, name="softmax",
        num_units=2, nonlinearity=lasagne.nonlinearities.softmax)
    
    
    updates_modifier = MaxNormConstraint({'softmax': 0.5})
    dataset = rand_set
    
    dataset_iterator = BalancedBatchIterator(batch_size=60)
    
    preprocessor = OnlineAxiswiseStandardize (axis=['c', 1])
    dataset_splitter=FixedTrialSplitter(n_train_trials=150, valid_set_fraction=0.1)
    updates_var_func=lasagne.updates.adam
    loss_var_func= lasagne.objectives.categorical_crossentropy
    monitors=[braindecode.veganlasagne.monitors.LossMonitor (),
                    braindecode.veganlasagne.monitors.MisclassMonitor(),
                    braindecode.veganlasagne.monitors.RuntimeMonitor()]
    stop_criterion= braindecode.veganlasagne.stopping.MaxEpochs(num_epochs=30)
    
    
    exp = Experiment(network, dataset, dataset_splitter, preprocessor,
              dataset_iterator, loss_var_func, updates_var_func, 
              updates_modifier, monitors,
              stop_criterion)
    exp.setup()
    exp.run()
    assert np.allclose(
        [0.548148, 0.540741, 0.503704, 0.451852, 0.392593, 0.370370, 
        0.340741, 0.281481, 0.237037, 0.207407, 0.192593, 0.177778, 
        0.133333, 0.111111, 0.111111, 0.103704, 0.096296, 0.088889, 
        0.088889, 0.081481, 0.074074, 0.066667, 0.066667, 0.059259, 
        0.059259, 0.051852, 0.037037, 0.037037, 0.029630, 0.029630, 
        0.029630, 0.053333, 0.053333, 0.053333, 0.053333, 0.040000, 
        0.040000, 0.026667, 0.026667, 0.026667, 0.026667, 0.033333, 
        0.033333, 0.033333, 0.033333, 0.026667, 0.020000, 0.020000, 
        0.020000],
        exp.monitor_chans['train_misclass'],
        rtol=1e-4, atol=1e-4)
    assert np.allclose(
        [0.400000, 0.400000, 0.400000, 0.400000, 0.400000, 0.400000, 
        0.400000, 0.400000, 0.333333, 0.333333, 0.333333, 0.266667, 
        0.266667, 0.266667, 0.266667, 0.266667, 0.266667, 0.266667, 
        0.266667, 0.266667, 0.266667, 0.266667, 0.266667, 0.266667, 
        0.333333, 0.333333, 0.333333, 0.266667, 0.266667, 0.266667, 
        0.266667, 0.266667, 0.266667, 0.266667, 0.266667, 0.200000, 
        0.200000, 0.133333, 0.133333, 0.133333, 0.133333, 0.133333, 
        0.133333, 0.133333, 0.133333, 0.066667, 0.000000, 0.000000, 
        0.000000],
        exp.monitor_chans['valid_misclass'],
        rtol=1e-4, atol=1e-4)
    assert np.allclose(
        [0.440000, 0.420000, 0.400000, 0.420000, 0.420000, 0.440000, 
        0.420000, 0.420000, 0.400000, 0.400000, 0.380000, 0.380000, 
        0.400000, 0.400000, 0.400000, 0.380000, 0.400000, 0.400000, 
        0.380000, 0.380000, 0.380000, 0.380000, 0.360000, 0.360000, 
        0.380000, 0.380000, 0.380000, 0.400000, 0.420000, 0.420000, 
        0.420000, 0.420000, 0.420000, 0.420000, 0.420000, 0.380000, 
        0.380000, 0.380000, 0.380000, 0.400000, 0.400000, 0.400000, 
        0.400000, 0.380000, 0.380000, 0.380000, 0.400000, 0.400000, 
        0.380000],
        exp.monitor_chans['test_misclass'],
        rtol=1e-4, atol=1e-4)
    assert np.allclose(
        [1.200389, 0.777420, 0.740212, 0.705151, 0.672329, 0.641764, 
        0.613245, 0.586423, 0.561397, 0.538399, 0.517073, 0.497741, 
        0.479949, 0.463601, 0.448505, 0.434583, 0.421652, 0.409739, 
        0.398721, 0.388490, 0.378988, 0.370121, 0.361965, 0.354295, 
        0.347159, 0.340496, 0.334237, 0.328328, 0.322803, 0.317624, 
        0.312765, 0.340091, 0.335658, 0.330868, 0.325923, 0.320895, 
        0.316027, 0.311290, 0.306683, 0.302364, 0.298264, 0.294475, 
        0.290957, 0.287673, 0.284664, 0.281860, 0.279309, 0.276918, 
        0.274709],
        exp.monitor_chans['train_loss'],
        rtol=1e-4, atol=1e-4)
    assert np.allclose(
        [0.767627, 0.643132, 0.637843, 0.630741, 0.624525, 0.619630, 
        0.614651, 0.610087, 0.606251, 0.602312, 0.598985, 0.595385, 
        0.592522, 0.590259, 0.588366, 0.586463, 0.584568, 0.582393, 
        0.581478, 0.580471, 0.580075, 0.579705, 0.579723, 0.579708, 
        0.579832, 0.580391, 0.581109, 0.581944, 0.582418, 0.583385, 
        0.584484, 0.585879, 0.582269, 0.571548, 0.555956, 0.536982, 
        0.517474, 0.496652, 0.474400, 0.453094, 0.432208, 0.412533, 
        0.394271, 0.377036, 0.361311, 0.346461, 0.333406, 0.321266, 
        0.310158],
        exp.monitor_chans['valid_loss'],
        rtol=1e-4, atol=1e-4)
    assert np.allclose(
        [1.069735, 0.751960, 0.746601, 0.741921, 0.737756, 0.734315, 
        0.731448, 0.728690, 0.726593, 0.724205, 0.722386, 0.721009, 
        0.719658, 0.718555, 0.717857, 0.717533, 0.717486, 0.717510, 
        0.717721, 0.717611, 0.717790, 0.717982, 0.718380, 0.718953, 
        0.719711, 0.720474, 0.721219, 0.721976, 0.722723, 0.723320, 
        0.723895, 0.723661, 0.724509, 0.725173, 0.725513, 0.726169, 
        0.726820, 0.727371, 0.727916, 0.728548, 0.729049, 0.729874, 
        0.730807, 0.731600, 0.732750, 0.733926, 0.735067, 0.736198, 
        0.737148],
        exp.monitor_chans['test_loss'],
        rtol=1e-4, atol=1e-4)
    
    
    
def test_experiment_sample_windows():
    data_rng = RandomState(398765905)
    rand_topo = data_rng.rand(200,10,10,3).astype(np.float32)
    rand_y = np.int32(data_rng.rand(200) > 0.5)
    rand_topo[rand_y == 1] += 0.1
    rand_set = DenseDesignMatrixWrapper(topo_view=rand_topo, y=rand_y)
    
    lasagne.random.set_rng(RandomState(9859295))
    in_layer = InputLayer(shape= [None, 10,5,3])
    network = DenseLayer(incoming=in_layer, name='softmax',
        num_units=2, nonlinearity=lasagne.nonlinearities.softmax)
    updates_modifier = MaxNormConstraint({'softmax': 0.5})
    
    dataset = rand_set
    
    dataset_iterator = WindowsIterator(n_samples_per_window=5, 
                                             batch_size=60)
    
    preprocessor = OnlineAxiswiseStandardize(axis=['c', 1])
    dataset_splitter=FixedTrialSplitter(n_train_trials=150, valid_set_fraction=0.1)
    updates_var_func=lasagne.updates.adam
    loss_var_func= lasagne.objectives.categorical_crossentropy
    monitors=[braindecode.veganlasagne.monitors.LossMonitor (),
                    braindecode.veganlasagne.monitors.WindowMisclassMonitor(),
                    braindecode.veganlasagne.monitors.RuntimeMonitor()]
    stop_criterion= braindecode.veganlasagne.stopping.MaxEpochs(num_epochs=5)
    
    
    exp = Experiment(network, dataset, dataset_splitter, preprocessor,
              dataset_iterator, loss_var_func, updates_var_func, 
              updates_modifier, monitors, stop_criterion)
    exp.setup()
    exp.run()
    
    assert np.allclose(
        [0.629630,0.140741,0.029630,0.022222,0.000000,0.000000,0.000000],
        exp.monitor_chans['train_misclass'],
        rtol=1e-4, atol=1e-4)
    assert np.allclose(
        [0.400000,0.133333,0.066667,0.000000,0.000000,0.000000,0.000000],
        exp.monitor_chans['valid_misclass'],
        rtol=1e-4, atol=1e-4)
    assert np.allclose(
        [0.560000,0.060000,0.000000,0.000000,0.000000,0.000000,0.000000],
        exp.monitor_chans['test_misclass'],
        rtol=1e-4, atol=1e-4)
    assert np.allclose(
        [1.180485, 0.574264, 0.420023, 0.330909, 0.278569, 0.245692, 0.242845],
        exp.monitor_chans['train_loss'],
        rtol=1e-4, atol=1e-4)
    assert np.allclose(
        [1.017478, 0.514316, 0.370705, 0.289101, 0.241017, 0.211246, 0.215967],
        exp.monitor_chans['valid_loss'],
        rtol=1e-4, atol=1e-4)
    assert np.allclose(
        [1.031094, 0.504486, 0.352198, 0.269623, 0.223666, 0.196412, 0.197652],
        exp.monitor_chans['test_loss'],
        rtol=1e-4, atol=1e-4)