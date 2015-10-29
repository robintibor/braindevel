import numpy as np
from braindecode.datahandling.splitters import (
    SingleFoldSplitter, PreprocessedSplitter, FixedTrialSplitter)
from braindecode.datasets.pylearn import DenseDesignMatrixWrapper

def to_4d_array(arr):
    arr = np.array(arr)
    assert arr.ndim == 1
    return arr[:, np.newaxis, np.newaxis, np.newaxis]

def test_last_fold():
    data = np.arange(10)
    dataset = DenseDesignMatrixWrapper(topo_view=to_4d_array(data), y=np.zeros(10))
    splitter = SingleFoldSplitter(n_folds=10, 
        i_test_fold=9)
    datasets= splitter.split_into_train_valid_test(dataset)
    
    assert np.array_equal(to_4d_array(np.arange(8)), 
                   datasets['train'].get_topological_view() )
    assert np.array_equal(to_4d_array([8]),
         datasets['valid'].get_topological_view() )
    assert np.array_equal(to_4d_array([9]), 
                   datasets['test'].get_topological_view() )

def test_first_fold():
    data = np.arange(10)
    dataset = DenseDesignMatrixWrapper(topo_view=to_4d_array(data), y=np.zeros(10))
    splitter = SingleFoldSplitter(n_folds=10, 
        i_test_fold=0)
    datasets= splitter.split_into_train_valid_test(dataset)
    
    assert np.array_equal(to_4d_array(np.arange(1,9)), 
                   datasets['train'].get_topological_view() )
    assert np.array_equal(to_4d_array([9]), 
                   datasets['valid'].get_topological_view() )
    assert np.array_equal(to_4d_array([0]), 
                   datasets['test'].get_topological_view() )

def test_preprocessed_splitter():
    class DemeanPreproc():
        """Just for tests :)"""
        def apply(self, dataset, can_fit=False):
            topo_view = dataset.get_topological_view()
            if can_fit:
                self.mean = np.mean(topo_view)
            dataset.set_topological_view(topo_view - self.mean)


    data = np.arange(10)
    dataset = DenseDesignMatrixWrapper(topo_view=to_4d_array(data), y=np.zeros(10))
    splitter = SingleFoldSplitter(n_folds=10, i_test_fold=9)
    preproc_splitter = PreprocessedSplitter(dataset_splitter=splitter,
        preprocessor=DemeanPreproc())

    first_round_sets = preproc_splitter.get_train_valid_test(dataset)
    
    train_topo = first_round_sets['train'].get_topological_view()
    valid_topo = first_round_sets['valid'].get_topological_view()
    test_topo = first_round_sets['test'].get_topological_view()
    assert np.array_equal(train_topo, 
                          to_4d_array([-3.5, -2.5,-1.5,-0.5,0.5,1.5,2.5,3.5]))
    assert np.array_equal(valid_topo, to_4d_array([4.5]))
    assert np.array_equal(test_topo, to_4d_array([5.5]))
    
    second_round_set = preproc_splitter.get_train_merged_valid_test(dataset)
    
    train_topo = second_round_set['train'].get_topological_view()
    valid_topo = second_round_set['valid'].get_topological_view()
    test_topo = second_round_set['test'].get_topological_view()
    assert np.array_equal(train_topo, to_4d_array([-4,-3,-2,-1,0,1,2,3,4]))
    assert np.array_equal(valid_topo, to_4d_array([4]))
    assert np.array_equal(test_topo, to_4d_array([5]))
    
def test_repeated_calls_with_shuffle():
    """Repeated calls should always lead to same split"""
    data = np.arange(100)
    dataset = DenseDesignMatrixWrapper(topo_view=to_4d_array(data), 
        y=np.zeros(100))
    splitter = SingleFoldSplitter(n_folds=10, 
        i_test_fold=9, shuffle=True)
    reference_datasets = splitter.split_into_train_valid_test(dataset)
    
    # 20 attemptsat splitting should all lead to same datasets!
    for _ in range(20):
        new_datasets = splitter.split_into_train_valid_test(dataset)
        for key in reference_datasets:
            assert np.array_equal(reference_datasets[key].get_topological_view(),
                new_datasets[key].get_topological_view())
            
def test_fixed_trial():
    dataset = DenseDesignMatrixWrapper(topo_view=to_4d_array(range(12)),
         y=np.zeros(12))
    splitter = FixedTrialSplitter(n_train_trials=10, 
        valid_set_fraction=0.2)
    sets = splitter.split_into_train_valid_test(dataset)
    assert np.array_equal(sets['train'].get_topological_view().squeeze(),
                          range(8))
    assert np.array_equal(sets['valid'].get_topological_view().squeeze(), 
                          [8,9])
    assert np.array_equal(sets['test'].get_topological_view().squeeze(), 
                          range(10,12))

def test_fixed_trial_with_rounding():
    dataset = DenseDesignMatrixWrapper(topo_view = to_4d_array(range(12)), 
        y= np.zeros(12))
    splitter = FixedTrialSplitter(n_train_trials=9,
        valid_set_fraction=0.2)
    sets = splitter.split_into_train_valid_test(dataset)
    assert np.array_equal(sets['train'].get_topological_view().squeeze(), 
                          range(8))
    assert sets['valid'].get_topological_view().squeeze() == 8
    assert np.array_equal(sets['test'].get_topological_view().squeeze(), 
                          range(9,12))
