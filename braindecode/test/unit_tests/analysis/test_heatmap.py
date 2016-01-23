import numpy as np
from braindecode.analysis.heatmap import (back_relevance_conv,
    back_relevance_dense_layer, back_relevance_pool)

def test_conv_w_sqr():
    out_relevances = np.array([[[1,0,0]]]).astype(np.float32)
    conv_weights = np.array([[[[1,0]]]])[:,:,::-1,::-1]
    in_relevances = back_relevance_conv(out_relevances, np.zeros((1,1,4)), conv_weights, 'w_sqr')
    assert np.array_equal([[[ 1.,  0.,  0.,  0.]]], in_relevances)
    out_relevances = np.array([[[1,0,0]]]).astype(np.float32)
    conv_weights = np.array([[[[-1,0]]]])[:,:,::-1,::-1]
    in_relevances = back_relevance_conv(out_relevances, np.zeros((1,1,4)), conv_weights, 'w_sqr')
    assert np.array_equal([[[ 1.,  0.,  0.,  0.]]], in_relevances)
    out_relevances = np.array([[[1,0,0]]]).astype(np.float32)
    conv_weights = np.array([[[[1,1]]]])[:,:,::-1,::-1]
    in_relevances = back_relevance_conv(out_relevances, np.zeros((1,1,4)), conv_weights, 'w_sqr')
    assert np.array_equal([[[ 0.5,  0.5,  0.,  0.]]], in_relevances)
    out_relevances = np.array([[[1,0,0]]]).astype(np.float32)
    conv_weights = np.array([[[[1,2]]]])[:,:,::-1,::-1]
    in_relevances = back_relevance_conv(out_relevances, np.zeros((1,1,4)), conv_weights, 'w_sqr')
    assert np.allclose([[[ 0.2,  0.8,  0.,  0.]]], in_relevances)
    # Test NaN case
    out_relevances = np.array([[[1,0,0]]]).astype(np.float32)
    conv_weights = np.array([[[[0,0]]]])[:,:,::-1,::-1]
    in_relevances = back_relevance_conv(out_relevances, np.zeros((1,1,4)), conv_weights, 'w_sqr')
    assert np.array_equal([[[ 0.5,  0.5,  0.,  0.]]], in_relevances)
    
    assert np.allclose([ 1.57243109,  4.92130327,  0.50626564],
                   back_relevance_dense_layer(np.array([2,1,0,3,1]),
                           np.array([1,0,3]),
                           np.array([[0,1,2,3,4], [9,3,4,-5,1],[0,0,5,-2,2]]), 
                          'w_sqr'))
    assert np.allclose([1.65684497,  4.97152281,  1.37163186], 
        back_relevance_dense_layer(np.array([2,1,1,3,1]),
                           np.array([1,4,3]),
                           np.array([[1,1,2,3,4], [9,3,4,-5,1],[1,2,5,-2,2]]), 
                          'w_sqr'))
    
def test_conv_z_plus():
    out_relevances = np.array([[[1,0,0]]]).astype(np.float32)
    conv_weights = np.array([[[[1,2]]]])[:,:,::-1,::-1]
    in_relevances = back_relevance_conv(out_relevances, np.ones((1,1,4)), conv_weights, 'z_plus')
    assert np.allclose([[[ 1/3.0,  2/3.0,  0.,  0.]]], in_relevances)
    out_relevances = np.array([[[1,0,0]]]).astype(np.float32)
    conv_weights = np.array([[[[1,2]]]])[:,:,::-1,::-1]
    in_relevances = back_relevance_conv(out_relevances, np.zeros((1,1,4)), conv_weights, 'z_plus')
    assert np.array_equal([[[ 0.5,  0.5,  0.,  0.]]], in_relevances)
    # negative weights ignored
    out_relevances = np.array([[[1,0,0]]]).astype(np.float32)
    conv_weights = np.array([[[[1,-2]]]])[:,:,::-1,::-1]
    in_relevances = back_relevance_conv(out_relevances, np.ones((1,1,4)), conv_weights, 'z_plus')
    assert np.allclose([[[ 1,0,  0.,  0.]]], in_relevances)
    in_activations = [[[1,2,6]]]
    out_relevances = [[[3,4]]]
    weights = np.array([[[[2,3]]]])[:,:,::-1,::-1]
    in_relevances = back_relevance_conv(out_relevances, in_activations,weights, rule='z_plus')
    assert np.allclose([[[3/4.0, 131/44.0, 36/11.0]]], in_relevances)
    
def test_dense_z_plus():
    out_relevances = np.array([0,2,1], dtype=np.float32)
    weights = np.array([[1,1,1], [2,2.25,0]], dtype=np.float32)
    in_activations = np.array([1,4], dtype=np.float32)
    in_relevances = back_relevance_dense_layer(out_relevances,in_activations,weights,'z_plus')
    assert np.allclose([1.2,1.8], in_relevances)
    assert np.allclose([ 3.44895196,  3.20214176,  1.3489064 ],
                   back_relevance_dense_layer(np.array([2,1,1,3,1]),
                           np.array([1,4,3]),
                           np.array([[1,1,2,3,4], [9,3,4,-5,1],[1,2,5,-2,2]]), 
                          'z_plus'))
    assert np.allclose([ 5.06666708,  0.66666669,  1.26666665], back_relevance_dense_layer(np.array([2,1,0,3,1]),
                           np.array([1,0,3]),
                           np.array([[0,1,2,3,4], [9,3,4,-5,1],[0,0,5,-2,2]]), 
                          'z_plus'))

def test_pool():
    out_rel = [[[1,3,2]]]
    in_act = [[[1, 4, 3, 0.3, 4, 6]]]
    pool_size =(1,2)
    pool_stride = (1,2)
    in_relevance = back_relevance_pool(out_rel, in_act, pool_size, pool_stride)
    assert np.allclose(in_relevance, [[[0.2,0.8, 3*3/3.3, 3*0.3/3.3, 0.8,1.2]]])
    
    # Case with stride!=size
    out_rel = [[[1,4]]]
    in_act = [[[1, 2, 6, 4, 5]]]
    pool_size =(1,3)
    pool_stride = (1,2)
    in_relevance = back_relevance_pool(out_rel, in_act, pool_size, pool_stride)
    assert np.allclose(in_relevance, [[[1/9.0,2/9.0, 204/90.0, 
        16/15.0, 20/15.0]]])