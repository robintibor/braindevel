import numpy as np
import theano
import theano.tensor as T
from braindecode.analysis.patterns import transform_weights_to_patterns,\
    add_time_delayed_channels, upsample_pool

def test_patterns_square_case():
    # source and noise are chosen so that
    # noise is uncorrelated with source
    source1 = np.array([1,3,1,4,8,1])
    source2 = np.array([1,5,2,4,-2,3])
    source = np.vstack((source1, source2))
    # Mix source to signal plus some noise 
    noise1 = [0.4,0.3,-0.5,-0.2,0.2,0.5]
    noise2 = [-0.3,-0.5,+0.5,+0.4,-0.2,-0.5]
    signal1 = 0.2 * source1 + 0.5 * source2 + noise1
    signal2 = 0.8 * source1 - 0.2 * source2 + noise2
    signal = np.vstack((signal1, signal2))
    
    W, _, _, _  = np.linalg.lstsq(signal.T, source.T)
    
    reconstructed = np.dot(W.T, signal)
    patterns = transform_weights_to_patterns(W.T, signal, reconstructed)
    # Note columns are  close to given mixing 0.2/0.5, 0.8/-0.2
    assert np.allclose([[ 0.20550224,  0.80095521],
                       [ 0.54812122, -0.25528973]],
               patterns)
    # Also compare to solution by inversion
    pattern_by_inversion = np.linalg.inv(W)
    assert np.allclose(pattern_by_inversion, patterns)

def test_patterns_reduced_case():
    """ Less sources than signals. """
    # source and noise are chosen so that
    # noise is uncorrelated with source
    source1 = np.array([1,3,1,4,8,1])
    source2 = np.array([1,5,2,4,-2,3])
    source = np.vstack((source1, source2))
    noise1 = [0.4,0.3,-0.5,-0.2,0.2,0.5]
    noise2 = [-0.3,-0.5,+0.5,+0.4,-0.2,-0.5]
    noise3 = [-0.2,-0.5,+0.5,+0.4,-0.2,-0.5]
    # Mix source to signal plus some noise 
    signal1 = 0.2 * source1 + 0.5 * source2 + noise1
    signal2 = 0.8 * source1 - 0.2 * source2 + noise2
    signal3 = 0.5 * source1 - 0.3 * source2 + noise3
    signal = np.vstack((signal1, signal2, signal3))
    W, _, _, _  = np.linalg.lstsq(signal.T, source.T)
    reconstructed = np.dot(W.T, signal)
    patterns = transform_weights_to_patterns(W.T, signal, reconstructed)
    # Note columns are  close to given mixing 0.2/0.5, 0.8/-0.2, 0.5/-0.3
    assert np.allclose([[ 0.18594579,  0.80136242,  0.49412571],
                    [ 0.52567645, -0.25482237, -0.36055654]],
           patterns)
    # Also compare to solution by inversion
    pattern_by_inversion = np.linalg.pinv(W)
    assert not np.allclose(pattern_by_inversion, patterns), ("In reduced case "
        "inversion does not have to lead to same result as patterns and "
        "in this case it shouldn't.")
    
def test_add_time_delayed_channels():
    channel1 = [1,2,3,4,5,6,7]
    channel2 = [-1,-2,-3,-4,-5,-6,-7]
    topo = np.array([channel1, channel2])[np.newaxis,:,:,np.newaxis]
    transformed = add_time_delayed_channels(topo, kernel_length=3)
    expected = [[[1,2,3,4,5], [2,3,4,5,6], [3,4,5,6,7],
               [-1,-2,-3,-4,-5], [-2,-3,-4,-5,-6], [-3,-4,-5,-6,-7]]]
    expected = np.array(expected)[:,:,:,np.newaxis]
    assert np.array_equal(expected, transformed)

def test_upsample_pool():
    pool_size = (1,2)
    pool_stride = (1,2)
    out = T.ftensor4()
    inputs = T.ftensor4()
    
    actual_in = upsample_pool(out, inputs, pool_size, pool_stride)
    upsample_pool_fn = theano.function([out, inputs], actual_in)
    # needs pool size, stride= 1,2
    output = np.float32([[[[5,4,1]]]])
    inputs = np.float32([[[[3,0,8,4,5,6]]]])
    upsampled = upsample_pool_fn(output, inputs)
    assert np.allclose([[[[ 5.,  0.,  4.,  0.,  0.,  1.]]]],
                       upsampled)
    
    # Test for pooling across several channels
    pool_size = (1,2)
    pool_stride = (1,2)
    out = T.ftensor4()
    inputs = T.ftensor4()
    
    actual_in = upsample_pool(out, inputs, pool_size, pool_stride)
    upsample_pool_fn = theano.function([out, inputs], actual_in)
    # needs pool size, stride= 1,2
    output = np.float32([[[[-3,5,2]],[[5,4,1]]]])
    inputs = np.float32([[[[2,1,3,4,1,-7]], [[3,0,8,4,5,6]]]])
    upsampled = upsample_pool_fn(output, inputs)
    assert np.allclose([[[[ -3.,  0.,  0.,  5.,  2.,  0.]],
                         [[ 5.,  0.,  4.,  0.,  0.,  1.]]]],
                       upsampled)