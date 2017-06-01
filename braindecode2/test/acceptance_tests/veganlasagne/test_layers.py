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
        [  72.        ,  -72.        ,    7.2,   -7.2],
        [  84.        ,  -84.        ,    8.4,   -8.4],
        [  96.        ,  -96.        ,    9.6,   -9.6],
        [ 108.        , -108.        ,   10.8,  -10.8],
        [ 120.        , -120.        ,   12. ,  -12.        ],
        [ 132.        , -132.        ,   13.2,  -13.2],
        [          np.nan,           np.nan,           np.nan,           np.nan],
        [ 240.        , -240.        ,   24.        ,  -24.        ],
        [ 252.        , -252.        ,   25.2,  -25.2],
        [ 264.        , -264.        ,   26.4,  -26.4],
        [ 276.        , -276.        ,   27.6,  -27.6],
        [ 288.        , -288.        ,   28.8,  -28.8],
        [ 300.        , -300.        ,   30.0,  -30.0],
        [ 312.        , -312.        ,   31.2,  -31.2],
        [          np.nan,           np.nan,           np.nan,           np.nan]],
        dtype=np.float32),
        layer_activations[5])
    
def test_stride_reshape_layer_with_nonempty_3rd_dim():
    input_var = T.tensor4('inputs').astype(theano.config.floatX)
    network = lasagne.layers.InputLayer(shape=[None,1,15,2], input_var=input_var)
    network = lasagne.layers.Conv2DLayer(network, num_filters=1,filter_size=[3, 1],
                                         W=lasagne.init.Constant(1), stride=(1,1))
    network = StrideReshapeLayer(network, n_stride=2, invalid_fill_value=np.nan)
    network = lasagne.layers.Conv2DLayer(network, num_filters=1,filter_size=[2, 1],
                                         W=lasagne.init.Constant(1), stride=(1,1))
    network = StrideReshapeLayer(network, n_stride=2, invalid_fill_value=np.nan)
    network = lasagne.layers.Conv2DLayer(network, num_filters=4, filter_size=[2, 1],
                                         W=to_4d_time_array([[1,1], [-1,-1], [0.1,0.1], [-0.1,-0.1]]), stride=(1,1),
                                        nonlinearity=lasagne.nonlinearities.identity)
    network = lasagne.layers.Conv2DLayer(network, num_filters=3, filter_size=[1, 2],
                                         W=np.array([[[[1,1]],[[0.5,0.5]], [[1,1]], [[0.5,0.5]]],
                                                      [[[-1,-1]],[[-0.5,-0.5]], [[-1,-1]], [[0,0]]],
                                                     [[[0,0]],[[0,0]], [[-1,1]], [[0,0]]]], dtype=np.float32), stride=(1,1),
                                        nonlinearity=lasagne.nonlinearities.identity)
    network = FinalReshapeLayer(network, remove_invalids=False)
    
    preds_cnt = lasagne.layers.get_output(lasagne.layers.get_all_layers(network)[1:])
    pred_cnt_func = theano.function([input_var], preds_cnt)
    inputs= np.array([[range(1,16), range(101,116)],
                     [range(16,31), range(116,131)]]).astype(np.float32)
    inputs = inputs.swapaxes(1,2)[:,np.newaxis,:,:]
    layer_activations = pred_cnt_func(inputs)
    assert equal_without_nans(np.array([[[[  6., 306], [  9., 309], [ 12., 312], [ 15.,315], [ 18.,318], [ 21.,321], 
                           [ 24.,324], [ 27.,327], [ 30.,330], [ 33.,333], [ 36.,336], [ 39.,339], 
                           [ 42.,342]]],
        [[[ 51.,351], [ 54.,354], [ 57.,357], [ 60.,360], [ 63.,363], [ 66.,366], 
          [ 69.,369], [ 72.,372], [75.,375], [ 78.,378], [ 81.,381], [ 84.,384], 
          [ 87.,387]]]], 
            dtype=np.float32),
        layer_activations[0])
    
    assert equal_without_nans(np.array(
        [[[[  6., 306], [ 12.,312], [ 18.,318], [ 24.,324], [ 30.,330], [ 36.,336], [ 42.,342]]],
       [[[ 51., 351], [ 57., 357], [ 63., 363], [ 69., 369], [ 75., 375], [ 81., 381], [ 87., 387]]],
       [[[  9., 309], [ 15., 315], [ 21., 321], [ 27., 327], [ 33., 333], [ 39., 339], [ np.nan, np.nan]]],
       [[[ 54., 354], [ 60., 360], [ 66., 366], [ 72., 372], [ 78., 378], [ 84., 384], [ np.nan, np.nan]]]],
       dtype=np.float32),
       layer_activations[1])
    assert equal_without_nans(np.array([[[[  18.,618.], [  30., 630], [  42., 642], [  54., 654], [  66., 666], [  78., 678]]],
       [[[ 108., 708], [ 120., 720], [ 132., 732], [ 144., 744], [ 156., 756], [ 168., 768]]],
       [[[  24., 624], [  36., 636], [  48., 648], [  60., 660], [  72., 672], [  np.nan, np.nan]]],
       [[[ 114., 714], [ 126., 726], [ 138., 738], [ 150., 750], [ 162., 762], [  np.nan, np.nan]]]],
       dtype=np.float32),
       layer_activations[2])
    assert equal_without_nans(np.array([[[[  18., 618], [  42.,642], [  66.,666]]],
        [[[ 108.,708], [ 132.,732], [ 156.,756]]],
        [[[  24.,624], [  48.,648], [  72.,672]]],
        [[[ 114.,714], [ 138.,738], [ 162.,762]]],
        [[[  30.,630], [  54.,654], [  78.,678]]],
        [[[ 120.,720], [ 144.,744], [ 168.,768]]],
        [[[  36.,636], [  60.,660], [  np.nan, np.nan]]],
        [[[ 126., 726], [ 150., 750], [  np.nan, np.nan]]]],
        dtype=np.float32),
        layer_activations[3])
    assert allclose_without_nans(np.array(
        [[[[  60.        ,1260], [ 108.        , 1308]], [[ -60.        , -1260], [-108.        ,-1308]],
        [[   6.00000048, 126], [  10.80000019, 130.80000305]], [[  -6.00000048, -126], [ -10.80000019, -130.80000305]]],
          [[[  240.        ,  1440.        ], [  288.        ,  1488.        ]],   [[ -240.        , -1440.        ],
             [ -288.        , -1488.        ]], [[   24.        ,   144.        ],    [   28.79999924,   148.80000305]],
            [[  -24.        ,  -144.        ],  [  -28.79999924,  -148.80000305]]],
         [[[   72.        ,  1272.        ],  [  120.        ,  1320.        ]],  [[  -72.        , -1272.        ],
             [ -120.        , -1320.        ]], [[    7.20000029,   127.20000458],  [   12.        ,   132.        ]],
            [[   -7.20000029,  -127.20000458],   [  -12.        ,  -132.        ]]],
           [[[  252.        ,  1452.        ], [  300.        ,  1500.        ]], [[ -252.        , -1452.        ],
             [ -300.        , -1500.        ]],   [[   25.20000076,   145.19999695], [   30.        ,   150.        ]],
            [[  -25.20000076,  -145.19999695],  [  -30.        ,  -150.        ]]],
           [[[   84.        ,  1284.        ], [  132.        ,  1332.        ]],  [[  -84.        , -1284.        ],
             [ -132.        , -1332.        ]], [[    8.39999962,   128.3999939 ], [   13.19999981,   133.19999695]],
            [[   -8.39999962,  -128.3999939 ], [  -13.19999981,  -133.19999695]]],
           [[[  264.        ,  1464.        ], [  312.        ,  1512.        ]], [[ -264.        , -1464.        ],
             [ -312.        , -1512.        ]], [[   26.39999962,   146.3999939 ], [   31.20000076,   151.19999695]],
            [[  -26.39999962,  -146.3999939 ],  [  -31.20000076,  -151.19999695]]],
           [[[   96.        ,  1296.        ],  [np.nan, np.nan]], [[  -96.        , -1296.        ],
             [np.nan,  np.nan]],
            [[    9.60000038,   129.6000061 ], [np.nan, np.nan]], [[   -9.60000038,  -129.6000061 ],
             [np.nan, np.nan]]],
           [[[  276.        ,  1476.        ], [np.nan, np.nan]],
            [[ -276.        , -1476.        ], [np.nan, np.nan]],
            [[   27.60000038,   147.6000061 ], [np.nan, np.nan]],
            [[  -27.60000038,  -147.6000061 ], [np.nan, np.nan]]]], dtype=np.float32),
        layer_activations[4])
    expected_5 = [np.sum(layer_activations[4][:,[0,2]] * 1 + layer_activations[4][:,[1,3]] * 0.5, axis=(1,3)),
             np.sum(layer_activations[4][:,[0,2]] * (-1), axis=(1,3)) + 
                  np.sum(layer_activations[4][:,[1]] * (-0.5), axis=(1,3)),
             layer_activations[4][:,2,:,0] - layer_activations[4][:,2,:,1]]
    expected_5 = np.array(expected_5).swapaxes(0,1)[:,:,:,np.newaxis]
    assert allclose_without_nans(expected_5, layer_activations[5])
    assert allclose_without_nans(layer_activations[6][:,0],
    np.concatenate((np.linspace(726,805.2, 7), [np.nan], np.linspace(924, 1003.2, 7), [np.nan])))
    assert allclose_without_nans(layer_activations[6][:,1],
        np.concatenate((np.linspace(-792,-878.4, 7), [np.nan], np.linspace(-1008, -1094.4, 7), [np.nan])))
    assert allclose_without_nans(layer_activations[6][:,2], 
                                 np.concatenate(([-120] * 7, [np.nan], [-120] * 7, [np.nan],)))
    
    for elem in layer_activations[6].flatten():
        assert np.isnan(elem) or elem in layer_activations[5]
    assert np.sum(np.isnan(layer_activations[5])) == np.sum(np.isnan(layer_activations[6]))
    
def test_stride_reshape_layer_with_padding():
    """Testing with padding... actually should never be any problem, as
    Stide reshape layer is independent of padding before..."""
    input_var = T.tensor4('inputs').astype(theano.config.floatX)
    network = lasagne.layers.InputLayer(shape=[None,1,15,1], input_var=input_var)
    network = lasagne.layers.Conv2DLayer(network, num_filters=1, pad='same', filter_size=[3, 1],
                                         W=lasagne.init.Constant(1), stride=(1,1),
                                    )
    network = StrideReshapeLayer(network, n_stride=2, invalid_fill_value=np.nan)
    
    preds_cnt = lasagne.layers.get_output(lasagne.layers.get_all_layers(network)[1:])
    pred_cnt_func = theano.function([input_var], preds_cnt)
    layer_activations = pred_cnt_func(to_4d_time_array([range(1,16), range(16,31)]))
    assert equal_without_nans(np.array([[[[  3.], [  6.], [  9.], [ 12.], [ 15.], [ 18.], [ 21.], 
                           [ 24.], [ 27.], [ 30.], [ 33.], [ 36.], [ 39.], 
                           [ 42.], [ 29.]]],
        [[[ 33.], [ 51.], [ 54.], [ 57.], [ 60.], [ 63.], [ 66.], 
          [ 69.], [ 72.], [75.], [ 78.], [ 81.], [ 84.], 
          [ 87.], [59. ]]]], 
            dtype=np.float32),
        layer_activations[0])
        
    assert equal_without_nans(np.array(
        [[[[3.], [  9.], [ 15.], [ 21.], [ 27.], [ 33.], [ 39.], [29.]]],
       [[[33.], [ 54.], [ 60.], [ 66.], [ 72.], [ 78.], [ 84.], [59.]]],
       [[[  6.], [ 12.], [ 18.], [ 24.], [ 30.], [ 36.], [ 42.], [ np.nan]]],
       [[[ 51.], [ 57.], [ 63.], [ 69.], [ 75.], [ 81.], [ 87.], [ np.nan]]]],
       dtype=np.float32),
       layer_activations[1])




    
    
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
