import numpy as np
import theano
import theano.tensor as T
from braindecode.analysis.heatmap import (back_relevance_conv,
    back_relevance_dense_layer, back_relevance_pool, relevance_pool,
    create_back_conv_z_b_fn, create_back_conv_w_sqr_fn, create_back_dense_fn)

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
    
def test_conv_w_sqr_theano():
    conv_w_sqr_fun = create_back_conv_w_sqr_fn()
    out_relevances = np.array([[[1,0,0]]], dtype=np.float32)
    conv_weights = np.array([[[[1,0]]]],dtype=np.float32)[:,:,::-1,::-1]
    in_relevances = conv_w_sqr_fun(out_relevances, conv_weights)
    assert np.array_equal([[[ 1.,  0.,  0.,  0.]]], in_relevances)
    
    out_relevances = np.array([[[1,0,0]]], dtype=np.float32)
    conv_weights = np.array([[[[-1,0]]]], dtype=np.float32)[:,:,::-1,::-1]
    in_relevances = conv_w_sqr_fun(out_relevances, conv_weights)
    assert np.array_equal([[[ 1.,  0.,  0.,  0.]]], in_relevances)
    
    out_relevances = np.array([[[1,0,0]]], dtype=np.float32)
    conv_weights = np.array([[[[1,1]]]], dtype=np.float32)[:,:,::-1,::-1]
    in_relevances = conv_w_sqr_fun(out_relevances, conv_weights)
    assert np.array_equal([[[ 0.5,  0.5,  0.,  0.]]], in_relevances)
    
    out_relevances = np.array([[[1,0,0]]], dtype=np.float32)
    conv_weights = np.array([[[[1,2]]]], dtype=np.float32)[:,:,::-1,::-1]
    in_relevances = conv_w_sqr_fun(out_relevances, conv_weights)
    assert np.allclose([[[ 0.2,  0.8,  0.,  0.]]], in_relevances)
    
    out_relevances = np.array([[[1,0,0]]], dtype=np.float32)
    conv_weights = np.array([[[[0,0]]]], dtype=np.float32)[:,:,::-1,::-1]
    in_relevances = conv_w_sqr_fun(out_relevances, conv_weights)
    assert np.array_equal([[[ 0.,  0.,  0.,  0.]]], in_relevances)
    
    
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
    
def test_conv_z_b():
    out_rel = [[[3,4]]]
    in_act = [[[1,-2,3]]]
    weights = np.array([[[[-2,3]]]])[:,:,::-1,::-1]
    in_relevances = back_relevance_conv(out_rel,in_act,weights,
        'z_b', min_in=-2, max_in=4)
    assert np.allclose([[[3,48/27.0,60/27.0]]], in_relevances)

def test_conv_z_b_theano():
    conv_z_b_fn = create_back_conv_z_b_fn(-2,4)

    out_rel = np.array([[[3,4]]], dtype=np.float32)
    in_act = np.array([[[1,-2,3]]], dtype=np.float32)
    weights = np.array([[[[-2,3]]]], dtype=np.float32)[:,:,::-1,::-1]
    in_relevances = conv_z_b_fn(out_rel,in_act,weights)
    assert np.allclose([[[3,48/27.0,60/27.0]]], in_relevances)
        
def test_dense_w_sqr():
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

def test_dense_w_sqr_theano():
    back_dense_fn = create_back_dense_fn('w_sqr')
    assert np.allclose([ 1.57243109,  4.92130327,  0.50626564],
                       back_dense_fn(np.array([2,1,0,3,1], dtype=np.float32),
                   np.array([[0,1,2,3,4], [9,3,4,-5,1],[0,0,5,-2,2]],
                       dtype=np.float32)))
    
    assert np.allclose([1.65684497,  4.97152281,  1.37163186],
                       back_dense_fn(np.array([2,1,1,3,1], dtype=np.float32),
                   np.array([[1,1,2,3,4], [9,3,4,-5,1],[1,2,5,-2,2]],
                       dtype=np.float32)))

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
    assert np.allclose([ 5.06666708,  0.66666669,  1.26666665],
        back_relevance_dense_layer(np.array([2,1,0,3,1]),
                           np.array([1,0,3]),
                           np.array([[0,1,2,3,4], [9,3,4,-5,1],[0,0,5,-2,2]]), 
                          'z_plus'))

def test_dense_z_plus_theano():
    back_dense_fn = create_back_dense_fn('z_plus')
    out_relevances = np.array([0,2,1], dtype=np.float32)
    weights = np.array([[1,1,1], [2,2.25,0]], dtype=np.float32)
    in_activations = np.array([1,4], dtype=np.float32)
    in_relevances = back_dense_fn(out_relevances,in_activations,weights)
    assert np.allclose([1.2,1.8], in_relevances)
    assert np.allclose([ 3.44895196,  3.20214176,  1.3489064 ],
                   back_dense_fn(np.array([2,1,1,3,1], dtype=np.float32),
                           np.array([1,4,3], dtype=np.float32),
                           np.array([[1,1,2,3,4], [9,3,4,-5,1],[1,2,5,-2,2]],
                               dtype=np.float32))) 
    # different from numpy due to NaNs being eplaced by zeros instead of
    # being replaced by 1 / #input units
    assert np.allclose([ 4.4000001 ,  0.        ,  0.60000002], 
                       back_dense_fn(np.array([2,1,0,3,1], dtype=np.float32),
                           np.array([1,0,3], dtype=np.float32),
                           np.array([[0,1,2,3,4], [9,3,4,-5,1],[0,0,5,-2,2]],
                               dtype=np.float32)))

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
    
def test_pool_theano():
    inputs_var = T.ftensor3()
    out_rel_var = T.ftensor3()
    pool_size =(1,2)
    pool_stride = (1,2)
    
    in_relevances_var = relevance_pool(out_rel_var, inputs_var, pool_size, pool_stride)
    pool_relevance_fn = theano.function([out_rel_var,inputs_var], in_relevances_var)
    out_rel = [[[1,3,2]]]
    in_act = [[[1, 4, 3, 0.3, 4, 6]]]
    in_relevance = pool_relevance_fn(np.array(out_rel, dtype=np.float32), np.array(in_act, dtype=np.float32))
    assert np.allclose(in_relevance, [[[0.2,0.8, 3*3/3.3, 3*0.3/3.3, 0.8,1.2]]])
    
    inputs_var = T.ftensor3()
    out_rel_var = T.ftensor3()
    pool_size =(1,3)
    pool_stride = (1,2)
    
    in_relevances_var = relevance_pool(out_rel_var, inputs_var, pool_size, pool_stride)
    pool_relevance_fn = theano.function([out_rel_var,inputs_var], in_relevances_var)
    out_rel = [[[1,4]]]
    in_act = [[[1, 2, 6, 4, 5]]]
    in_relevance = pool_relevance_fn(out_rel, in_act)
    assert np.allclose(in_relevance, [[[1/9.0,2/9.0, 204/90.0, 16/15.0, 20/15.0]]])
        