import numpy as np
from braindevel.datahandling.preprocessing import (exponential_running_mean,
                                                    exponential_running_var_from_demeaned,
                                                   exponential_running_standardize)

def test_exponential_preprocessings():
    data = np.array([ 1, 3, 5, 9, 7, -2])
    exp_run_mean = np.array([ 0.200000, 0.160000, 0.328000, 0.262400, 0.409920, 0.327936])
    exp_run_var = np.array([ 0.928000, 1.745920, 3.697869, 9.952428, 9.922424, 12.347100])
    exp_standardized = np.array([ 0.830455, 1.695258, 1.763925, 1.874509, 0.993934, -1.336228])
    
    run_mean = exponential_running_mean(np.array(data), factor_new=0.2,start_mean=0)
    demeaned = data - run_mean
    run_var = exponential_running_var_from_demeaned(demeaned, factor_new=0.2, start_var=1)
    standardized = exponential_running_standardize(data, factor_new=0.2, start_mean=0, start_var=1)
    assert np.allclose(exp_run_mean, run_mean)
    assert np.allclose(exp_run_var, run_var)
    assert np.allclose(exp_standardized, standardized)

    data = np.array([ 2, 0, -5, -3, 0, 4])
    exp_run_mean = np.array([ 0.400000, 0.320000, -0.744000, -1.195200, -0.956160, 0.035072])
    exp_run_var = np.array([ 1.312000, 1.070080, 4.478771, 4.234478, 3.570431, 6.000475])
    exp_standardized = np.array([ 1.396861, -0.309344, -2.011047, -0.877060, 0.506023, 1.618611])
    
    run_mean = exponential_running_mean(np.array(data), factor_new=0.2,start_mean=0)
    demeaned = data - run_mean
    run_var = exponential_running_var_from_demeaned(demeaned, factor_new=0.2, start_var=1)
    standardized = exponential_running_standardize(data, factor_new=0.2, start_mean=0, start_var=1)
    assert np.allclose(exp_run_mean, run_mean)
    assert np.allclose(exp_run_var, run_var)
    assert np.allclose(exp_standardized, standardized)
    
    data = np.array([ -3, 5, 8, 7, 4, -2])
    exp_run_mean = np.array(np.array([ -0.600000, 0.520000, 2.016000, 3.012800, 3.210240, 2.168192]))
    exp_run_var = np.array([ 1.952000, 5.575680, 11.622195, 12.477309, 10.106591, 11.560038])
    exp_standardized = np.array([ -1.717795, 1.897270, 1.755284, 1.128775, 0.248424, -1.225937])
    
    run_mean = exponential_running_mean(np.array(data), factor_new=0.2,start_mean=0)
    demeaned = data - run_mean
    run_var = exponential_running_var_from_demeaned(demeaned, factor_new=0.2, start_var=1)
    standardized = exponential_running_standardize(data, factor_new=0.2, start_mean=0, start_var=1)
    
    assert np.allclose(exp_run_mean, run_mean)
    assert np.allclose(exp_run_var, run_var)
    assert np.allclose(exp_standardized, standardized)
    
    data = np.array([ 1, 0, 1, 0, 1, 0])
    exp_run_mean = np.array([ 0.200000, 0.160000, 0.328000, 0.262400, 0.409920, 0.327936])
    exp_run_var = np.array([ 0.928000, 0.747520, 0.688333, 0.564437, 0.521188, 0.438459])
    exp_standardized = np.array([ 0.830455, -0.185058, 0.809972, -0.349266, 0.817360, -0.495250])
    
    run_mean = exponential_running_mean(np.array(data), factor_new=0.2,start_mean=0)
    demeaned = data - run_mean
    run_var = exponential_running_var_from_demeaned(demeaned, factor_new=0.2, start_var=1)
    standardized = exponential_running_standardize(data, factor_new=0.2, start_mean=0, start_var=1)
    assert np.allclose(exp_run_mean, run_mean)
    assert np.allclose(exp_run_var, run_var)
    assert np.allclose(exp_standardized, standardized)

def test_exponential_multidimensional():
    data = np.array([[[ 1, 3, 5, 9, 7, -2],
        [ 2, 0, -5, -3, 0, 4]],
        [[ -3, 5, 8, 7, 4, -2],
        [ 1, 0, 1, 0, 1, 0]]])
    exp_standardized = np.array([[[ 0.830455, 1.695258, 1.763925, 1.874509, 0.993934, -1.336228],
        [ 1.396861, -0.309344, -2.011047, -0.877060, 0.506023, 1.618611]],
        [[ -1.717795, 1.897270, 1.755284, 1.128775, 0.248424, -1.225937],
        [ 0.830455, -0.185058, 0.809972, -0.349266, 0.817360, -0.495250]]])
    standardized = exponential_running_standardize(data.swapaxes(0,2), factor_new=0.2, start_mean=0, start_var=1)
    assert np.allclose(standardized.swapaxes(0,2), exp_standardized)
