import numpy as np
from braindecode.analysis.kaggle import transform_to_time_activations,\
    transform_to_cnt_activations
from braindecode.test.util import equal_without_nans

def test_transform_time_cnt_act():
    acts = [[[[  6.], [ 12.], [ 18.], [ 24.], [ 30.], [ 36.], [ 42.]]],
       [[[ 51.], [ 57.], [ 63.], [ 69.], [ 75.], [ 81.], [ 87.]]],
       [[[  9.], [ 15.], [ 21.], [ 27.], [ 33.], [ 39.], [ np.nan]]],
       [[[ 54.], [ 60.], [ 66.], [ 72.], [ 78.], [ 84.], [ np.nan]]]]


    time_act = transform_to_time_activations(np.array([acts]), [2])
    
    assert len(time_act) == 1
    assert equal_without_nans(np.array([[[  6.,   9.,  12.,  15.,  18., 
         21.,  24.,  27.,  30.,  33.,  36.,
               39.,  42.,  np.nan]],
     
            [[ 51.,  54.,  57.,  60.,  63.,  66.,  69.,  72.,  75.,  78.,  81.,
               84.,  87.,  np.nan]]]),time_act[0])
    
    
    cnt_act = transform_to_cnt_activations(time_act, n_sample_preds=7, n_samples=14)

    assert np.array_equal([[ 24.,  27.,  30.,  33.,  36.,  39.,  42.,  69.,  72.,  75.,  78.,
             81.,  84.,  87.]], cnt_act)
    cnt_act = transform_to_cnt_activations(time_act, n_sample_preds=9, n_samples=18)
    
    assert np.array_equal([[ 18.,21, 24.,  27.,  30.,  33.,  36.,  39.,  42., 
             63, 66, 69.,  72.,  75.,  78.,
             81.,  84.,  87.]], cnt_act)
    cnt_act = transform_to_cnt_activations(time_act, n_sample_preds=13, n_samples=26)
    
    assert np.array_equal([[ 6, 9, 12, 15, 18.,21, 24.,  27.,  30., 
                            33.,  36.,  39.,  42., 
             51, 54 ,57, 60, 63, 66, 69.,  72.,  75.,  78.,
             81.,  84.,  87.]], cnt_act)
    

def test_transform_time_cnt_act_multiple_chans():
    before = np.array([[[[1,4,7], [-1,-4,-7]], 
                   [[11,14,17], [-11,-14,-17]],
                   [[2,5,8], [-2,-5,-8]],
                   [[12,15,18], [-12,-15,-18]],
                   [[3,6,9], [-3,-6,-9]],
                   [[13,16,19], [-13,-16,-19]]],
                   [[[21,24,27], [-21,-24,-27]], 
                   [[22,25,28], [-22,-25,-28]],
                   [[23,26,29], [-23,-26,-29]]]])


    before[0] = np.array(before[0])[:,:,:, np.newaxis]
    before[1] = np.array(before[1])[:,:,:, np.newaxis]
    time_act = transform_to_time_activations(before, [2,1])
        
        
    assert np.array_equal(np.array([[[  1,   2,   3,   4,   5,   6,   7,   8,   9],
         [ -1,  -2,  -3,  -4,  -5,  -6,  -7,  -8,  -9]],
 
        [[ 11,  12,  13,  14,  15,  16,  17,  18,  19],
         [-11, -12, -13, -14, -15, -16, -17, -18, -19]]]),
     time_act[0])
    assert np.array_equal(np.array([[[ 21,  22,  23,  24,  25,  26,  27,  28,  29],
             [-21, -22, -23, -24, -25, -26, -27, -28, -29]]]),
         time_act[1])
    cnt_act = transform_to_cnt_activations(time_act,n_sample_preds=5, n_samples=15)

    assert np.array_equal(np.array([[  5,   6,   7,   8,   9,  15,  16,  17,  18,  19,  25,  26,  27,
             28,  29],
           [ -5,  -6,  -7,  -8,  -9, -15, -16, -17, -18, -19, -25, -26, -27,
            -28, -29]]),
           cnt_act)
    
    cnt_act = transform_to_cnt_activations(time_act,n_sample_preds=9, n_samples=27)
    
    assert np.array_equal(np.array([[ 1, 2, 3, 4 , 5,   6,   7,   8,   9,  
                                     11, 12, 13, 14, 15,  16,  17,  18,  19,  
                                     21, 22, 23, 24, 25,  26,  27, 28,  29],
           [ -1, -2, -3, -4, -5,  -6,  -7,  -8,  -9, 
            -11, -12, -13, -14, -15, -16, -17, -18, -19, 
            -21, -22, -23, -24, -25, -26, -27, -28, -29]]),
       cnt_act)