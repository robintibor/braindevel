from braindecode.mywyrm.processing import lda_train_scaled
import numpy as np

def test_lda_train_scaled():
    ## these values were checked with matlab
    # matlab had almost same values (<0.01 differences on weights, 
    # < 0.1 on bias)
    featurevector = lambda: None
    featurevector.data = np.array([[8.2, 3.5, 5.6], [9.1, 1.2, 2.4], 
        [8.8, 3.5, 5.6], [7.1, 1.5, 2.9]])
    featurevector.axes = [[1,0,1,0], []]
    w,b = lda_train_scaled(featurevector, shrink=True)
    assert np.allclose([ 0.13243638,  0.31727594,  0.42877362], w)
    assert np.allclose(-3.6373072863307785, b)