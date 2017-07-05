import numpy as np
from numpy.random import RandomState
from pylearn2.format.target_format import OneHotFormatter
from scipy.interpolate import interp1d

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

def spread_to_sensors(topo_view, mixing_matrix):
    sensor_spread_topo_view = np.tensordot(topo_view, mixing_matrix, axes=(1,1)).transpose(0,3,1,2)
    return sensor_spread_topo_view

def add_timecourse(topo_view,y, fixed_points=8, change_magnitude=0.5):
    # topo view shd be bc01
    rng = RandomState(np.uint64(hash('decreasetimecourse')))
    trials = topo_view.shape[0]
    samples = topo_view.shape[2]
    fixed_points_x = np.linspace(0,samples,fixed_points)
    fixed_points_y = [rng.uniform(1-change_magnitude,1+change_magnitude,
                                  fixed_points) for _ in xrange(trials)]
    interp_fun = interp1d(fixed_points_x, fixed_points_y, 'linear')
    timecourses = interp_fun(np.arange(600)) # now has shape trialsxsamples
    timecourses = timecourses[:,np.newaxis,:,np.newaxis]
    
    topo_view = topo_view * timecourses
    return topo_view
    
def randomly_shifted_sines(number, samples, freq, sampling_freq, rng=None):
    if rng is None:
        rng = RandomState(np.uint64(hash('randomly_shifted_sines')))
    random_shifts = rng.rand(number) * 2 * np.pi
    return np.array([create_sine_signal(samples, freq, sampling_freq, shift=shift) for shift in random_shifts])

def create_sine_signal(samples, freq, sampling_freq, shift=0):
    x = (np.arange(0,samples,1) * 2 * np.pi * freq / float(sampling_freq)) + shift
    return np.sin(x)

class RandomDataset(DenseDesignMatrix):
    def __init__(self, cleaner=None, filenames=None): # cleaner filenames always given to dataset, but ignoreable
        pass
    
    def load(self):
        rng = RandomState(np.uint64(hash("RandomDataset")))
        input_shape = [500,3,600,1]
        
        y = rng.random_integers(0,1,size=input_shape[0])
        y = OneHotFormatter(2).format(y)
        topo_view = rng.rand(*input_shape)
        super(RandomDataset, self).__init__(topo_view=topo_view, y=y, 
                                              axes=('b', 'c', 0, 1))
        
        

class SpreadedSinesDataset(DenseDesignMatrix):
    def __init__(self, weight_old_data=0.08,
        cleaner=None, filenames=None): # cleaner filenames always given to dataset, but ignoreable
        self.__dict__.update(locals())
        del self.self
        
        
    def load(self):
        input_shape=(1000,1,600,1)      

        lower_freq_topo_view, y = create_shifted_sines(input_shape, 
            RandomState(np.uint64(hash("pooled_pipeline"))), 
            sampling_freq=150.0)
        
        topo_view, y = pipeline(lower_freq_topo_view, y,
            lambda x,y: spread_to_sensors(x, [[0.8], [0.5]]), #mix
            lambda x,y: put_noise(x, weight_old_data=self.weight_old_data, 
                same_for_chans=True)
        )

        super(SpreadedSinesDataset, self).__init__(topo_view=topo_view, y=y, 
                                              axes=('b', 'c', 0, 1))

def create_shifted_sines(input_shape, rng, sampling_freq=None, freq=11):
    """ Creates sines for class 1 and just flat zeroline for class 0 """
    if sampling_freq is None:
        sampling_freq = input_shape[2] / 4
    topo_view = np.zeros(input_shape, dtype=np.float32)
    y = OneHotFormatter(2).format(rng.random_integers(0,1,size=input_shape[0]))
    shifted_sines_for_class_1 = randomly_shifted_sines(
            number=sum(np.argmax(y, axis=1)),
               samples=input_shape[2], freq=freq,
               sampling_freq=sampling_freq, rng=rng)
    topo_view[np.argmax(y, axis=1) == 1] += shifted_sines_for_class_1[:,np.newaxis,:,np.newaxis]
    return topo_view, y

def put_noise(topo_view, weight_old_data, rng=None, same_for_chans=True):
    rng = rng or RandomState(np.uint64(hash('noise_topo')))
    topo_view = topo_view * weight_old_data
    # make same noise on all sensors for now
    random_shape = list(topo_view.shape)
    if (same_for_chans):
        # Make sensor dim 1 for random numbers to enforce all sensors to get same 
        # noise, broadcasting should work appropriately
        random_shape[1] = 1
    topo_view += rng.rand(*random_shape)
    return topo_view

def convolve_with_weight(topo_view, weight):
    output =  []
    for trial_i in range(topo_view.shape[0]):
        trial = topo_view[trial_i]
        assert trial.ndim == 3
        assert trial.shape[2] == 1, "should not have any height"
        trial = np.atleast_2d(np.squeeze(trial))
        output_trial = np.array([np.convolve(chan, weight, mode='valid')
            for chan in trial])
        output.append(output_trial)
    topo_view = np.array(output)[:,:,:,np.newaxis]
    return topo_view

def max_pool_topo_view(topo_view, pool_shape):
    # first cut out the last parts which do not fit in the pooling (we only do valid pooling)
    cut_topo_view = topo_view[:,:,:topo_view.shape[2] - (topo_view.shape[2] % pool_shape),:]
    cut_topo_view = np.reshape(cut_topo_view, 
       (cut_topo_view.shape[0], cut_topo_view.shape[1], cut_topo_view.shape[2] / pool_shape, -1))

    pooled = np.max(cut_topo_view, axis=3,keepdims=True)
    return pooled
def log_sum_pool(topo_view, pool_shape):
    # first cut out the last parts which do not fit in the pooling (we only do valid pooling)
    cut_topo_view = topo_view[:,:,:topo_view.shape[2] - (topo_view.shape[2] % pool_shape),:]
    cut_topo_view = np.reshape(cut_topo_view, 
       (cut_topo_view.shape[0], cut_topo_view.shape[1], cut_topo_view.shape[2] / pool_shape, -1))

    pooled = np.log(np.sum(cut_topo_view, axis=3,keepdims=True))
    return pooled


def pipeline(topo_view, y, *functions):
    for func in functions:
        topo_view = func(topo_view, y)
    return topo_view,y
