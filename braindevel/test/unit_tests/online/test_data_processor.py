from numpy.random import RandomState
import numpy as np
from braindevel.datahandling.preprocessing import exponential_running_standardize
from braindevel.online.data_processor import StandardizeProcessor


def test_data_processor():
    """Compare standardized data to data standardized online
    always giving blocks of 10 samples"""
    rng = RandomState(3904890384)
    n_samples_in_buffer = 1000
    dataset = rng.rand(n_samples_in_buffer*2,5)
    
    factor_new=0.001
    n_stride = 10
    standardized = exponential_running_standardize(dataset,
        factor_new=factor_new, init_block_size=n_stride)
    
    processor = StandardizeProcessor(factor_new=factor_new,
        n_samples_in_buffer=n_samples_in_buffer)
    
    processor.initialize(n_chans=dataset.shape[1])
    
    for i_start_sample in xrange(0,dataset.shape[0]-n_stride+1,n_stride):
        processor.process_samples(dataset[i_start_sample:i_start_sample+n_stride])
        # compare all so far processed samples
        assert np.allclose(standardized[:i_start_sample+n_stride][-n_samples_in_buffer:],
             processor.get_samples(-i_start_sample-n_stride,None), rtol=1e-3, atol=1e-5)  
    
