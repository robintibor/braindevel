import numpy as np
from braindevel.datasets.trial_segmenter import (RestrictTrialRange,
    RestrictTrialClasses, AddTrialBreaks)

def test_trial_segmenters():
    y = np.array([[0,1,1,0,1,1,0,1,1,1,0]]).T
    
    y_out, _ = RestrictTrialRange(start_trial=1,stop_trial=None).segment(None,y,None)
    
    assert np.array_equal(y_out, np.array([[0,0,0,0,1,1,0,1,1,1,0]]).T)
    
    y_out, _ = RestrictTrialRange(start_trial=None,stop_trial=None).segment(None,y,None)
    assert np.array_equal(y_out, y)
    y_out, _ = RestrictTrialRange(start_trial=None,stop_trial=1).segment(None,y,None)
    assert np.array_equal(y_out,np.array([[0,1,1,0,0,0,0,0,0,0,0]]).T)
    
    
    y = np.array([[0,0,0,1,0],[0,0,1,0,0], [0,1,0,0,0], [1,0,0,0,0]]).T
    class_names = ['1','2','3','4']
    y_out, new_class_names = RestrictTrialClasses(['2','4']).segment(None,y,class_names)
    assert np.array_equal(new_class_names, ['2', '4'])
    assert np.array_equal(y_out, np.array([[0,0,1,0,0],[1,0,0,0,0]]).T)
    y_out, new_class_names = RestrictTrialClasses(['3','1']).segment(None,y,class_names)
    assert np.array_equal(new_class_names, ['3', '1'])
    assert np.array_equal(y_out, np.array([[0,1,0,0,0],[0,0,0,1,0]]).T)
    
    
    y = np.array([[0,0,1,0,0,0,1,0,0,0,0,0,1,0]]).T
    
    cnt = lambda: None
    cnt.fs = 1000
    y_expected = np.array([[0,0,1,0,0,0,1,0,0,0,0,0,1,0],
                           [0,0,0,0,1,0,0,0,1,1,1,0,0,0]]).T
    y_out, new_class_names = AddTrialBreaks(0,20,1,-1).segment(cnt,y,["1"])
    assert np.array_equal(new_class_names, ['1', 'TrialBreak'])
    assert np.array_equal(y_expected, y_out)
    
    y_expected = np.array([[0,0,1,0,0,0,1,0,0,0,0,0,1,0],
                           [0,0,0,1,1,1,0,1,1,1,1,1,0,0]]).T
    y_out, new_class_names = AddTrialBreaks(0,20,0,0).segment(cnt,y,["1"])
    assert np.array_equal(new_class_names, ['1', 'TrialBreak'])
    assert np.array_equal(y_expected, y_out)
    
    
    cnt.fs = 250
    y = np.array([[0,1,0,0,0,0,0,1,0]]).T
    y_expected = np.array([[0,1,0,0,0,0,0,1,0],
                           [0,0,0,1,1,1,0,0,0]]).T
    y_out, new_class_names = AddTrialBreaks(0,20,4,-4).segment(cnt,y,["1"])
    assert np.array_equal(new_class_names, ['1', 'TrialBreak'])
    assert np.array_equal(y_expected, y_out)
    
    
    y = np.array([[0,1,0,0,0,0,0,1,0]]).T
    y_expected = np.array([[0,1,0,0,0,0,0,1,0],
                           [0,0,0,0,0,0,0,0,0]]).T
    y_out, new_class_names = AddTrialBreaks(0,19,4,-4).segment(cnt,y,["1"])
    assert np.array_equal(new_class_names, ['1', 'TrialBreak'])
    assert np.array_equal(y_expected, y_out)
    
    y = np.array([[0,1,0,0,0,0,0,1,0]]).T
    y_expected = np.array([[0,1,0,0,0,0,0,1,0],
                           [0,0,0,0,0,0,0,0,0]]).T
    y_out, new_class_names = AddTrialBreaks(21,50,4,-4).segment(cnt,y,["1"])
    assert np.array_equal(new_class_names, ['1', 'TrialBreak'])
    assert np.array_equal(y_expected, y_out)
