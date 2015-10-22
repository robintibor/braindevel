from lasagne.layers import (DenseLayer, DropoutLayer, Conv2DLayer,
    DimshuffleLayer, InputLayer, NonlinearityLayer)

import lasagne
from numpy.random import RandomState
import numpy as np
import braindecode.veganlasagne.monitors
from braindecode.datasets.preprocessing import OnlineAxiswiseStandardize
from braindecode.datasets.dataset_splitters import DatasetFixedTrialSplitter
from braindecode.datasets.batch_iteration import BalancedBatchIterator
from braindecode.datasets.pylearn import DenseDesignMatrixWrapper
from braindecode.experiments.experiment import Experiment

def test_experiment_fixed_split():
    """ Regression test, checking that values have not changed from original run"""
    data_rng = RandomState(398765905)
    rand_topo = data_rng.rand(200,10,10,3).astype(np.float32)
    rand_y = np.int32(data_rng.rand(200) > 0.5)
    rand_topo[rand_y == 1] += 0.01
    rand_set = DenseDesignMatrixWrapper(topo_view=rand_topo, y=rand_y)

    lasagne.random.set_rng(RandomState(9859295))
    in_layer = InputLayer(shape= [None, 10,10,3])
    network = DenseLayer(incoming=in_layer,
        num_units=2, nonlinearity=lasagne.nonlinearities.softmax)
    
    
    dataset = rand_set
    
    dataset_iterator = BalancedBatchIterator(batch_size=60)
    
    preprocessor = OnlineAxiswiseStandardize (axis=['c', 1])
    dataset_splitter=DatasetFixedTrialSplitter(n_train_trials=150, valid_set_fraction=0.1)
    updates_var_func=lasagne.updates.adam
    loss_var_func= lasagne.objectives.categorical_crossentropy
    monitors=[braindecode.veganlasagne.monitors.LossMonitor (),
                    braindecode.veganlasagne.monitors.MisclassMonitor(),
                    braindecode.veganlasagne.monitors.RuntimeMonitor()]
    stop_criterion= braindecode.veganlasagne.stopping.MaxEpochs(num_epochs=30)
    
    
    exp = Experiment()
    exp.setup(network, dataset, dataset_splitter, preprocessor,
              dataset_iterator, loss_var_func, updates_var_func, monitors, stop_criterion)
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
        0.266667, 0.266667, 0.266667, 0.266667, 0.266667, 0.333333, 
        0.333333, 0.333333, 0.333333, 0.266667, 0.266667, 0.266667, 
        0.266667, 0.266667, 0.266667, 0.266667, 0.266667, 0.200000, 
        0.200000, 0.133333, 0.133333, 0.133333, 0.133333, 0.133333, 
        0.133333, 0.133333, 0.133333, 0.066667, 0.000000, 0.000000, 
        0.000000],
        exp.monitor_chans['valid_misclass'],
        rtol=1e-4, atol=1e-4)
    assert np.allclose(
        [0.440000, 0.440000, 0.400000, 0.420000, 0.420000, 0.440000, 
        0.420000, 0.420000, 0.400000, 0.400000, 0.380000, 0.380000, 
        0.400000, 0.400000, 0.400000, 0.380000, 0.400000, 0.420000, 
        0.400000, 0.400000, 0.400000, 0.400000, 0.380000, 0.380000, 
        0.380000, 0.380000, 0.400000, 0.400000, 0.420000, 0.420000, 
        0.420000, 0.420000, 0.420000, 0.420000, 0.420000, 0.400000, 
        0.380000, 0.380000, 0.380000, 0.400000, 0.400000, 0.400000, 
        0.400000, 0.380000, 0.360000, 0.360000, 0.380000, 0.380000, 
        0.380000],
        exp.monitor_chans['test_misclass'],
        rtol=1e-4, atol=1e-4)
    assert np.allclose(
        [1.200157, 0.777377, 0.740182, 0.705133, 0.672322, 0.641766, 
        0.613255, 0.586440, 0.561420, 0.538427, 0.517106, 0.497779, 
        0.479991, 0.463645, 0.448552, 0.434632, 0.421703, 0.409792, 
        0.398776, 0.388545, 0.379045, 0.370180, 0.362024, 0.354355, 
        0.347220, 0.340558, 0.334299, 0.328391, 0.322866, 0.317688, 
        0.312830, 0.340149, 0.335718, 0.330929, 0.325984, 0.320957, 
        0.316090, 0.311353, 0.306748, 0.302429, 0.298330, 0.294541, 
        0.291024, 0.287741, 0.284732, 0.281929, 0.279378, 0.276987, 
        0.274778],
        exp.monitor_chans['train_loss'],
        rtol=1e-4, atol=1e-4)
    assert np.allclose(
        [0.760241, 0.643337, 0.638225, 0.631345, 0.625300, 0.620540, 
        0.615686, 0.611239, 0.607489, 0.603653, 0.600413, 0.596902, 
        0.594116, 0.591904, 0.590056, 0.588196, 0.586344, 0.584209, 
        0.583298, 0.582295, 0.581890, 0.581510, 0.581513, 0.581483, 
        0.581583, 0.582107, 0.582787, 0.583576, 0.584011, 0.584935, 
        0.585983, 0.585884, 0.582275, 0.571557, 0.555970, 0.537003, 
        0.517502, 0.496685, 0.474440, 0.453141, 0.432260, 0.412591, 
        0.394333, 0.377102, 0.361379, 0.346532, 0.333479, 0.321340, 
        0.310233]
        ,
        exp.monitor_chans['valid_loss'],
        rtol=1e-4, atol=1e-4)
    assert np.allclose(
        [1.036242, 0.747390, 0.742338, 0.737899, 0.733926, 0.730600, 
        0.727779, 0.725068, 0.722926, 0.720520, 0.718643, 0.717209, 
        0.715787, 0.714612, 0.713813, 0.713373, 0.713178, 0.713057, 
        0.713117, 0.712879, 0.712908, 0.712968, 0.713224, 0.713647, 
        0.714233, 0.714829, 0.715419, 0.716025, 0.716617, 0.717075, 
        0.717519, 0.717882, 0.718591, 0.719128, 0.719370, 0.719903, 
        0.720425, 0.720859, 0.721281, 0.721780, 0.722171, 0.722853, 
        0.723639, 0.724292, 0.725274, 0.726277, 0.727251, 0.728230, 
        0.729031],
        exp.monitor_chans['test_loss'],
        rtol=1e-4, atol=1e-4)