import lasagne
import theano
import theano.tensor as T
import numpy as np
from braindecode.veganlasagne.layers import StrideReshapeLayer, FinalReshapeLayer
from braindecode.test.util import to_4d_time_array, equal_without_nans,\
    allclose_without_nans
    
from lasagne.nonlinearities import softmax, identity
from numpy.random import RandomState
from braindecode.veganlasagne.pool import SumPool2dLayer
from braindecode.veganlasagne.nonlinearities import safe_log

def get_input_shape(network):
    return lasagne.layers.get_all_layers(network)[0].output_shape

def test_stride_reshape_layer():
    input_var = T.tensor4('inputs').astype(theano.config.floatX)
    network = lasagne.layers.InputLayer(shape=[None,1,15,1], input_var=input_var)
    network = lasagne.layers.Conv2DLayer(network, num_filters=1,filter_size=[3, 1],
                                         W=lasagne.init.Constant(1), stride=(1,1))
    network = StrideReshapeLayer(network, n_stride=2, invalid_fill_value=np.nan)
    network = lasagne.layers.Conv2DLayer(network, num_filters=1,filter_size=[2, 1],
                                         W=lasagne.init.Constant(1), stride=(1,1))
    network = StrideReshapeLayer(network, n_stride=2, invalid_fill_value=np.nan)
    print network.output_shape
    network = lasagne.layers.Conv2DLayer(network, num_filters=4, filter_size=[2, 1],
                                         W=to_4d_time_array([[1,1], [-1,-1], [0.1,0.1], [-0.1,-0.1]]), stride=(1,1),
                                        nonlinearity=lasagne.nonlinearities.identity)
    network = FinalReshapeLayer(network, remove_invalids=False)
    
    preds_cnt = lasagne.layers.get_output(lasagne.layers.get_all_layers(network)[1:])
    pred_cnt_func = theano.function([input_var], preds_cnt)
    layer_activations = pred_cnt_func(to_4d_time_array([range(1,16), range(16,31)]))
    assert equal_without_nans(np.array([[[[  6.], [  9.], [ 12.], [ 15.], [ 18.], [ 21.], 
                           [ 24.], [ 27.], [ 30.], [ 33.], [ 36.], [ 39.], 
                           [ 42.]]],
        [[[ 51.], [ 54.], [ 57.], [ 60.], [ 63.], [ 66.], 
          [ 69.], [ 72.], [75.], [ 78.], [ 81.], [ 84.], 
          [ 87.]]]], 
            dtype=np.float32),
        layer_activations[0])
    assert equal_without_nans(np.array(
        [[[[  6.], [ 12.], [ 18.], [ 24.], [ 30.], [ 36.], [ 42.]]],
       [[[ 51.], [ 57.], [ 63.], [ 69.], [ 75.], [ 81.], [ 87.]]],
       [[[  9.], [ 15.], [ 21.], [ 27.], [ 33.], [ 39.], [ np.nan]]],
       [[[ 54.], [ 60.], [ 66.], [ 72.], [ 78.], [ 84.], [ np.nan]]]],
       dtype=np.float32),
       layer_activations[1])
    
    assert equal_without_nans(np.array([[[[  18.], [  30.], [  42.], [  54.], [  66.], [  78.]]],
       [[[ 108.], [ 120.], [ 132.], [ 144.], [ 156.], [ 168.]]],
       [[[  24.], [  36.], [  48.], [  60.], [  72.], [  np.nan]]],
       [[[ 114.], [ 126.], [ 138.], [ 150.], [ 162.], [  np.nan]]]],
       dtype=np.float32),
       layer_activations[2])
    
    assert equal_without_nans(np.array([[[[  18.], [  42.], [  66.]]],
        [[[ 108.], [ 132.], [ 156.]]],
        [[[  24.], [  48.], [  72.]]],
        [[[ 114.], [ 138.], [ 162.]]],
        [[[  30.], [  54.], [  78.]]],
        [[[ 120.], [ 144.], [ 168.]]],
        [[[  36.], [  60.], [  np.nan]]],
        [[[ 126.], [ 150.], [  np.nan]]]],
        dtype=np.float32),
        layer_activations[3])
    
    assert allclose_without_nans(np.array(
        [[[[  60.        ], [ 108.        ]], [[ -60.        ], [-108.        ]],
        [[   6.00000048], [  10.80000019]], [[  -6.00000048], [ -10.80000019]]],
        [[[ 240.        ], [ 288.        ]], [[-240.        ], [-288.        ]],
        [[  24.        ], [  28.80000114]], [[ -24.        ], [ -28.80000114]]],
        [[[  72.        ], [ 120.        ]], [[ -72.        ], [-120.        ]],
        [[   7.20000029], [  12.        ]], [[  -7.20000029], [ -12.        ]]],
        [[[ 252.        ], [ 300.        ]], [[-252.        ], [-300.        ]],
        [[  25.20000076], [  30.00000191]], [[ -25.20000076], [ -30.00000191]]],
        [[[  84.        ], [ 132.        ]], [[ -84.        ], [-132.        ]],
        [[   8.40000057], [  13.19999981]], [[  -8.40000057], [ -13.19999981]]],
        [[[ 264.        ], [ 312.        ]], [[-264.        ], [-312.        ]],
        [[  26.40000153], [  31.20000076]], [[ -26.40000153], [ -31.20000076]]],
        [[[  96.        ], [          np.nan]], [[ -96.        ], [          np.nan]],
        [[   9.60000038], [          np.nan]], [[  -9.60000038], [          np.nan]]],
        [[[ 276.        ], [          np.nan]], [[-276.        ], [          np.nan]],
        [[  27.60000038], [          np.nan]], [[ -27.60000038], [          np.nan]]]],
        dtype=np.float32),
        layer_activations[4])
    
    assert allclose_without_nans(np.array(
        [[  60.        ,  -60.        ,    6.0,   -6.],
        [ 240.        , -240.        ,   24.        ,  -24.        ],
        [  72.        ,  -72.        ,    7.2,   -7.2],
        [ 252.        , -252.        ,   25.2,  -25.2],
        [  84.        ,  -84.        ,    8.4,   -8.4],
        [ 264.        , -264.        ,   26.4,  -26.4],
        [  96.        ,  -96.        ,    9.6,   -9.6],
        [ 276.        , -276.        ,   27.6,  -27.6],
        [ 108.        , -108.        ,   10.8,  -10.8],
        [ 288.        , -288.        ,   28.8,  -28.8],
        [ 120.        , -120.        ,   12. ,  -12.        ],
        [ 300.        , -300.        ,   30.0,  -30.0],
        [ 132.        , -132.        ,   13.2,  -13.2],
        [ 312.        , -312.        ,   31.2,  -31.2],
        [          np.nan,           np.nan,           np.nan,           np.nan],
        [          np.nan,           np.nan,           np.nan,           np.nan]],
        dtype=np.float32),
        layer_activations[5])
    
def test_raw_net_trial_based_and_continuous():
    softmax_rng = RandomState(3094839048)
    orig_softmax_weights = softmax_rng.randn(4,20,54,1).astype(theano.config.floatX) * 0.01
    
    
    lasagne.random.set_rng(RandomState(23903823))
    
    input_var = T.tensor4('inputs')
    
    epo_network = lasagne.layers.InputLayer(shape=[None,22,1200,1],
                                        input_var=input_var)
    # we have to switch channel dimension to height axis to not squash/convolve them away
    epo_network = lasagne.layers.DimshuffleLayer(epo_network, pattern=(0,3,2,1))
    epo_network = lasagne.layers.Conv2DLayer(epo_network, num_filters=20,filter_size=[30, 1], nonlinearity=identity)
    epo_network = lasagne.layers.DropoutLayer(epo_network, p=0.5)
    epo_network = lasagne.layers.Conv2DLayer(epo_network, num_filters=20,filter_size=[1, 22], nonlinearity=T.sqr)
    epo_network = SumPool2dLayer(epo_network, pool_size=(100,1), stride=(20,1), mode='average_exc_pad')
    epo_network = lasagne.layers.NonlinearityLayer(epo_network, nonlinearity=safe_log)
    epo_network = lasagne.layers.DropoutLayer(epo_network, p=0.5)
    epo_network = lasagne.layers.DenseLayer(epo_network, num_units=4,nonlinearity=softmax,
                                        W=orig_softmax_weights.reshape(4,-1).T)
    
    preds = lasagne.layers.get_output(epo_network, deterministic=True)
    pred_func = theano.function([input_var], preds)
    
    n_trials = 20
    n_samples = 1200 + n_trials - 1
    
    rng = RandomState(343434216)
    
    orig_inputs = rng.randn(1, get_input_shape(epo_network)[1],n_samples,1).astype(
        theano.config.floatX)
    
    # reshape to 2000 trials
    
    trialwise_inputs = [orig_inputs[:,:,start_i:start_i+1200] for start_i in range(n_trials)]
    
    trialwise_inputs = np.array(trialwise_inputs)[:,0]
    
    
    lasagne.random.set_rng(RandomState(23903823))
    input_var = T.tensor4('inputs')
    
    cnt_network = lasagne.layers.InputLayer(shape=[None,22,n_samples,1],
                                        input_var=input_var)
    # we have to switch channel dimension to height axis to not squash/convolve them away
    cnt_network = lasagne.layers.DimshuffleLayer(cnt_network, pattern=(0,3,2,1))
    cnt_network = lasagne.layers.Conv2DLayer(cnt_network, num_filters=20,
        filter_size=[30, 1], nonlinearity=identity)
    cnt_network = lasagne.layers.DropoutLayer(cnt_network, p=0.5)
    cnt_network = lasagne.layers.Conv2DLayer(cnt_network, num_filters=20,
                filter_size=[1, 22], nonlinearity=T.sqr)
    cnt_network = SumPool2dLayer(cnt_network, pool_size=(100,1),
        stride=(1,1), mode='average_exc_pad')
    cnt_network = lasagne.layers.NonlinearityLayer(cnt_network,
        nonlinearity=safe_log)
    cnt_network = StrideReshapeLayer(cnt_network, n_stride=20)
    cnt_network = lasagne.layers.DropoutLayer(cnt_network, p=0.5)
    cnt_network = lasagne.layers.Conv2DLayer(cnt_network, num_filters=4,
        filter_size=[54, 1], W=orig_softmax_weights[:,:,::-1,:], stride=(1,1),
        nonlinearity=lasagne.nonlinearities.identity)
    cnt_network = FinalReshapeLayer(cnt_network)
    cnt_network = lasagne.layers.NonlinearityLayer(cnt_network,
        nonlinearity=softmax)
    preds_cnt = lasagne.layers.get_output(cnt_network, deterministic=True)
    pred_cnt_func = theano.function([input_var], preds_cnt)
    
    
    results = []
    batch_size = 5
    for i_trial in xrange(0,len(trialwise_inputs),batch_size):
        res =  pred_func(trialwise_inputs[i_trial:min(len(trialwise_inputs),
            i_trial+batch_size)])
        results.append(res)
    results = np.array(results).squeeze()
    res_cnt = pred_cnt_func(orig_inputs)
    reshaped_results = np.concatenate(results)
    assert np.allclose(reshaped_results, res_cnt[:n_trials])
