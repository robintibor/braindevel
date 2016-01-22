import numpy as np
from braindecode.analysis.heatmap import back_relevance_conv

def test_w_sqr_conv():
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
    
def test_z_plus():
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
