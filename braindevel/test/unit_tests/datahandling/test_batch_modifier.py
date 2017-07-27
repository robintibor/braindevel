import numpy as np
from numpy.random import RandomState
from braindevel.datahandling.batch_modifier import BandpowerMeaner

def test_phase_equal_after_bandpower_mean():
    rng = RandomState(3098284)
    inputs = rng.randn(50,20,1001,1)
    targets = rng.choice(4, size=50)
    target_arr = np.zeros((50,4))
    target_arr[:,0] = targets == 0
    target_arr[:,1] = targets == 1
    target_arr[:,2] = targets == 2
    target_arr[:,3] = targets == 3
    mod_inputs, mod_targets = BandpowerMeaner().process(inputs, target_arr)
    assert np.allclose(np.angle(np.fft.rfft(inputs, axis=2)),
            np.angle(np.fft.rfft(mod_inputs, axis=2)), rtol=1e-4, atol=1e-5)
    assert np.array_equal(target_arr, mod_targets)                          