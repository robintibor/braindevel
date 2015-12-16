from numpy.random import RandomState
import numpy as np
from lasagne.layers import InputLayer, GlobalPoolLayer
import theano.tensor as T
from braindecode.datahandling.preprocessing import exponential_running_standardize
from braindecode.online.data_processor import StandardizeProcessor
from braindecode.online.predictor import OnlinePredictor
from braindecode.online.model import OnlineModel

def test_online_predictor():
    """ Test whether predictions are done at correct timepoints.
    Model actually just returns input """
    
    rng = RandomState(3904890384)
    n_samples_in_buffer = 1000
    dataset = rng.rand(n_samples_in_buffer*2,5).astype(np.float32)
    
    factor_new=0.001
    n_stride = 10
    pred_freq = 11
    standardized = exponential_running_standardize(dataset,
        factor_new=factor_new, init_block_size=n_stride)
    model = InputLayer([1,1,1,1])
    
    processor = StandardizeProcessor(factor_new=factor_new,
        n_samples_in_buffer=n_samples_in_buffer)
    
    online_model = OnlineModel(model)
    online_pred = OnlinePredictor(processor, online_model, pred_freq=pred_freq)
    
    online_pred.initialize(n_chans=dataset.shape[1])
    all_preds = []
    for i_start_sample in xrange(0,dataset.shape[0]-n_stride+1,n_stride):
        online_pred.receive_samples(dataset[i_start_sample:i_start_sample+n_stride])
        if online_pred.has_new_prediction():
            pred, _ = online_pred.pop_last_prediction_and_sample_ind()
            all_preds.append(pred)
        
    assert np.array_equal(np.array(all_preds).squeeze(), standardized[10::pred_freq])
    
def test_sum_prediction():
    """ Test with a model that predicts sum over four samples """
    rng = RandomState(3904890384)
    n_samples_in_buffer = 1000
    dataset = rng.rand(n_samples_in_buffer*2,5).astype(np.float32)
    
    factor_new=0.001
    n_stride = 10
    pred_freq = 11
    standardized = exponential_running_standardize(dataset,
        factor_new=factor_new, init_block_size=n_stride)
    model = InputLayer([1,1,4,1])
    model = GlobalPoolLayer(model,pool_function=T.sum)
    
    expected = [np.sum(standardized[stop-4:stop], axis=0) for stop in xrange(11, dataset.shape[0], 11)]
    expected = np.array(expected)
    
    processor = StandardizeProcessor(factor_new=factor_new,
        n_samples_in_buffer=n_samples_in_buffer)
    
    online_model = OnlineModel(model)
    online_pred = OnlinePredictor(processor, online_model, pred_freq=pred_freq)
    
    online_pred.initialize(n_chans=dataset.shape[1])
    all_preds = []
    for i_start_sample in xrange(0,dataset.shape[0]-n_stride+1,n_stride):
        online_pred.receive_samples(dataset[i_start_sample:i_start_sample+n_stride])
        if online_pred.has_new_prediction():
            pred, _ = online_pred.pop_last_prediction_and_sample_ind()
            all_preds.append(pred)
        
    assert np.allclose(np.array(all_preds).squeeze(), expected, rtol=1e-3)