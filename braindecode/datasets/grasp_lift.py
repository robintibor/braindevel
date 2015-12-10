import os
from scikits.samplerate import resample
import pandas as pd
import numpy as np
import logging
from copy import deepcopy
from braindecode.datasets.pylearn import DenseDesignMatrixWrapper
import lasagne
import theano
from zipfile import ZipFile
from zipfile import ZIP_DEFLATED
import StringIO
from braindecode.veganlasagne.layers import get_n_sample_preds
from braindecode.veganlasagne.monitors import get_reshaped_cnt_preds
from braindecode.datahandling.preprocessing import exponential_running_mean,\
    exponential_running_var_from_demeaned
log = logging.getLogger(__name__)

def load_train(train_folder, i_subject, i_series):
    data_filename = 'subj{:d}_series{:d}_data.csv'.format(
        i_subject, i_series)
    data_file_path = os.path.join(train_folder, data_filename)
    data = pd.read_csv(data_file_path)
    # events file
    events_file_path = data_file_path.replace('_data','_events')
    # read event file
    labels= pd.read_csv(events_file_path)
    clean = data.drop(['id' ], axis=1)#remove id
    labels = labels.drop(['id' ], axis=1)#remove id
    return clean, labels

def load_test(test_folder, i_subject, i_series):
    data_filename = 'subj{:d}_series{:d}_data.csv'.format(
        i_subject, i_series)
    data_file_path = os.path.join(test_folder, data_filename)
    data = pd.read_csv(data_file_path)
    clean = data.drop(['id' ], axis=1)#remove id
    return clean

class KaggleGraspLiftSet(object):
    """ Dataset from the kaggle grasp lift competition.
    resample_half true means resampling to 250 Hz (from original 500 Hz)
    """
    reloadable=False
    def __init__(self, data_folder, i_subject, resample_half,
            standardize=False):
        self.data_folder = data_folder
        self.i_subject = i_subject
        self.resample_half = resample_half
        self.standardize = standardize
        
    def ensure_is_loaded(self):
        if not hasattr(self, 'train_X_series'):
            self.load()
    
    def load(self):
        log.info("Loading data...")
        self.load_data()
        if self.resample_half:
            log.info("Resampling data...")
            self.resample_data()
        if self.standardize:
            log.info("Standardizing data...")
            self.standardize_data()
        log.info("..Done.")
        # hack to allow experiment class to know targets will have two dimensions
        self.y = np.ones((1,1)) * np.nan

    def load_data(self):
        # First just load the data
        self.train_X_series = []
        self.train_y_series = []
        train_folder = os.path.join(self.data_folder, 'train/')
        for i_series in xrange(1,9):
            X_series, y_series = load_train(train_folder, self.i_subject, i_series)
            # all sensor names should be the same :)
            # so just set it here directly
            if not hasattr(self, 'sensor_names'):
                self.sensor_names = X_series.keys()
            else:
                assert np.array_equal(self.sensor_names, X_series.keys())
            self.train_X_series.append(np.array(X_series).astype(np.float32))
            self.train_y_series.append(np.array(y_series).astype(np.int32))
            
        assert len(self.train_X_series) == 8, "Should be 8 train series for each subject"

    def resample_data(self):
        for i_series in xrange(8):
            X_series = np.array(self.train_X_series[i_series]).astype(np.float32)
            X_series = resample(X_series, 250.0/500.0, 'sinc_fastest')
            self.train_X_series[i_series] = X_series
            y_series = np.array(self.train_y_series[i_series]).astype(np.int32)
            # take later predictions ->
            # shift all predictions backwards compared to data.
            # this ensures you are not using data from the future to make a prediciton
            # rather in a bad case maybe you do not even have all data up to the sample
            # to make the prediction
            y_series = y_series[1::2]
            # maybe some border effects remove predictions
            y_series = y_series[-len(X_series):]
            self.train_y_series[i_series] = y_series

    def standardize_data(self):
        factor_new = 0.01
        for i_series in xrange(8):
            X_series = self.train_X_series[i_series]
            if i_series == 0:
                init_block_size=8000
                means = exponential_running_mean(X_series, factor_new=factor_new,
                    init_block_size=init_block_size, axis=None)
                demeaned = X_series - means
                stds = np.sqrt(exponential_running_var_from_demeaned(
                    demeaned, factor_new, init_block_size=init_block_size, axis=None))
            else:
                start_mean = means[-1]
                start_var = stds[-1] * stds[-1]
                means = exponential_running_mean(X_series, factor_new=factor_new,
                    start_mean=start_mean, axis=None)
                demeaned = X_series - means
                stds = np.sqrt(exponential_running_var_from_demeaned(
                    demeaned, factor_new, start_var=start_var, axis=None))
            eps = 1e-6
            standardized = demeaned / np.maximum(stds, eps)
            self.train_X_series[i_series] = standardized
        # for later test standardizing
        self.final_std = stds[-1]
        self.final_mean = means[-1]
            

    def load_test(self):
        """Refers to test set from evaluation(without labels)"""
        log.info("Loading test data...")
        self.load_test_data()
        if self.resample_half:
            log.info("Resampling test data...")
            self.resample_test_data()
        if self.standardize:
            log.info("Standardizing test data...")
            self.standardize_test_data()
        log.info("..Done.")

    def load_test_data(self):
        test_folder = os.path.join(self.data_folder, 'test/')
        self.test_X_series = []
        for i_series in xrange(9,11):
            X_series = load_test(test_folder, self.i_subject, i_series)
            self.test_X_series.append(np.array(X_series).astype(np.float32))
        assert len(self.test_X_series) == 2, "Should be 2 test series for each subject"

    def resample_test_data(self):
        for i_series in xrange(2):
            X_series = np.array(self.test_X_series[i_series]).astype(np.float32)
            X_series = resample(X_series, 250.0/500.0, 'sinc_fastest')
            self.test_X_series[i_series] = X_series
            
    def standardize_test_data(self):
        factor_new = 0.01
        for i_series in xrange(2):
            X_series = self.test_X_series[i_series]
            start_mean = self.final_mean
            start_var = self.final_std * self.final_std
            means = exponential_running_mean(X_series, factor_new=factor_new,
                start_mean=start_mean, axis=None)
            demeaned = X_series - means
            stds = np.sqrt(exponential_running_var_from_demeaned(
                demeaned, factor_new, start_var=start_var, axis=None))
            eps = 1e-6
            standardized = demeaned / np.maximum(stds, eps)
            self.test_X_series[i_series] = standardized

class AllSubjectsKaggleGraspLiftSet(object):
    """ Kaggle grasp lift set loading the data for all subjects """
    reloadable=False

    def __init__(self, data_folder, resample_half, standardize=False,
            last_subject=12):
        self.data_folder = data_folder
        self.resample_half = resample_half
        self.standardize = standardize
        self.last_subject = last_subject
        
    def ensure_is_loaded(self):
        if not hasattr(self, 'kaggle_sets'):
            self.load()
    
    def load(self):
        self.create_kaggle_sets()
        self.load_kaggle_sets()
        # hack to allow experiment class to know targets will have two dimensions
        self.y = np.ones((1,1)) * np.nan
       
     
    def create_kaggle_sets(self):
        self.kaggle_sets = [
            KaggleGraspLiftSet(self.data_folder, i_sub, self.resample_half,
                self.standardize) 
            for i_sub in range(1,self.last_subject+1)]
        
    def load_kaggle_sets(self):
        for i_set, kaggle_set in enumerate(self.kaggle_sets):
            log.info("Loading Subject {:d}...".format(i_set + 1))
            kaggle_set.load()
            
    def standardize_train_data(self):
        for i_set, kaggle_set in enumerate(self.kaggle_sets):
            log.info("Standardizing Train Subject {:d}...".format(i_set + 1))
            kaggle_set.standardize_data()
    
    def load_test(self):
        for i_set, kaggle_set in enumerate(self.kaggle_sets):
            log.info("Loading Test Subject {:d}...".format(i_set + 1))
            kaggle_set.load_test()

    def load_test_data(self):
        for i_set, kaggle_set in enumerate(self.kaggle_sets):
            log.info("Loading Test Data Subject {:d}...".format(i_set + 1))
            kaggle_set.load_test_data()

    def resample_test_data(self):
        for i_set, kaggle_set in enumerate(self.kaggle_sets):
            log.info("Resample Test Subject {:d}...".format(i_set + 1))
            kaggle_set.resample_test_data()

    def standardize_test_data(self):
        for i_set, kaggle_set in enumerate(self.kaggle_sets):
            log.info("Standardizing Test Subject {:d}...".format(i_set + 1))
            kaggle_set.standardize_test_data()
        

def create_submission_csv_for_one_subject(folder_name, kaggle_set, iterator, preprocessor,
        final_layer, submission_id):
    ### Load and preprocess data
    kaggle_set.load()
    # remember test series lengths before and after resampling to more accurately pad predictions
    # later (padding due to the lost samples)
    kaggle_set.load_test_data()
    test_series_lengths = [len(series) for series in kaggle_set.test_X_series] 
    kaggle_set.resample_test_data()
    test_series_lengths_resampled = [len(series) for series in kaggle_set.test_X_series] 
    X_train = deepcopy(np.concatenate(kaggle_set.train_X_series)[:,:,np.newaxis,np.newaxis])
    X_test_0 = deepcopy(kaggle_set.test_X_series[0][:,:,np.newaxis,np.newaxis])
    X_test_1 = deepcopy(kaggle_set.test_X_series[1][:,:,np.newaxis,np.newaxis])

    # create dense design matrix sets
    train_set = DenseDesignMatrixWrapper(
        topo_view=X_train, 
        y=None, axes=('b','c',0,1))
    fake_test_y = np.ones((len(X_test_0), 6))
    test_set_0 = DenseDesignMatrixWrapper(
        topo_view=X_test_0, 
        y=fake_test_y)
    fake_test_y = np.ones((len(X_test_1), 6))
    test_set_1 = DenseDesignMatrixWrapper(
        topo_view=X_test_1, 
        y=fake_test_y)
    log.info("Preprocessing data...")
    preprocessor.apply(train_set, can_fit=True)
    preprocessor.apply(test_set_0, can_fit=False)
    preprocessor.apply(test_set_1, can_fit=False)
    
    ### Create prediction function and create predictions
    log.info("Create prediction functions...")
    input_var = lasagne.layers.get_all_layers(final_layer)[0].input_var
    predictions = lasagne.layers.get_output(final_layer, deterministic=True)
    pred_fn = theano.function([input_var], predictions)
    log.info("Make predictions...")
    batch_gen_0 = iterator.get_batches(test_set_0, shuffle=False)
    all_preds_0 = [pred_fn(batch[0]) for batch in batch_gen_0]
    batch_gen_1 = iterator.get_batches(test_set_1, shuffle=False)
    all_preds_1 = [pred_fn(batch[0]) for batch in batch_gen_1]
    
    ### Pad and reshape predictions
    n_sample_preds = get_n_sample_preds(final_layer)
    input_time_length = lasagne.layers.get_all_layers(final_layer)[0].shape[2]
    
    n_samples_0 = test_set_0.get_topological_view().shape[0]
    preds_arr_0 = get_reshaped_cnt_preds(all_preds_0, n_samples_0, 
        input_time_length, n_sample_preds)
    n_samples_1 = test_set_1.get_topological_view().shape[0]
    preds_arr_1 = get_reshaped_cnt_preds(all_preds_1, n_samples_1, 
        input_time_length, n_sample_preds)

    series_preds = [preds_arr_0, preds_arr_1]
    assert len(series_preds[0]) == test_series_lengths_resampled[0]
    assert len(series_preds[1]) == test_series_lengths_resampled[1]
    assert False, ("TODO: here only duplicate if resample half is true for the dataset.. "
        "also take care how to create submission cv if trained on all subjects")
    series_preds_duplicated = [np.repeat(preds, 2,axis=0) for preds in series_preds]
    n_classes = preds_arr_0.shape[1]
    # pad missing ones with zeros
    missing_0 = test_series_lengths[0] - len(series_preds_duplicated[0])
    full_preds_0 = np.append(np.zeros((missing_0, n_classes), dtype=np.float32), 
                             series_preds_duplicated[0], axis=0)
    missing_1 = test_series_lengths[1] - len(series_preds_duplicated[1])
    full_preds_1 = np.append(np.zeros((missing_1, n_classes), dtype=np.float32),
                             series_preds_duplicated[1], axis=0)
    assert len(full_preds_0) == test_series_lengths[0]
    assert len(full_preds_1) == test_series_lengths[1]

    full_series_preds = [full_preds_0, full_preds_1]
    assert sum([len(a) for a in full_series_preds]) == np.sum(test_series_lengths)
    
    ### Create csv 

    log.info("Create csv...")
    csv_filename =  "{:02d}".format(submission_id) + '.csv'
    csv_filename = os.path.join(folder_name, csv_filename)
    cols = ['HandStart','FirstDigitTouch',
        'BothStartLoadPhase','LiftOff',
        'Replace','BothReleased']

    # collect ids
    all_ids = []
    all_preds = []
    for i_series in (9,10):
        id_prefix = "subj{:d}_series{:d}_".format(kaggle_set.i_subject, i_series)
        this_preds = full_series_preds[i_series-9] # respect offsets
        all_preds.extend(this_preds)
        this_ids = [id_prefix + str(i_sample) for i_sample in range(this_preds.shape[0])]
        all_ids.extend(this_ids)
    all_ids = np.array(all_ids)
    all_preds = np.array(all_preds)
    submission = pd.DataFrame(index=all_ids,
                              columns=cols,
                              data=all_preds)

    submission.to_csv(csv_filename, index_label='id',float_format='%.3f')
    log.info("Done")

def create_submission_csv_for_all_subjects(folder):
    all_lines = []
    for i in xrange(1,13,1):
        content = open(os.path.join(folder, '{:02d}.csv'.format(i)), 'r').readlines()
        if i == 1:
            all_lines.append(content[0])
        all_lines.extend(content[1:])
    csv_str = "".join(all_lines)
    submission_zip_file = ZipFile(os.path.join(folder,'all_submission.zip'), 'w', ZIP_DEFLATED)
    submission_zip_file.writestr("submission.csv", csv_str)
    submission_zip_file.close()

def create_submission_csv_for_all_subject_model(folder_name, 
        all_sub_kaggle_set, dataset_provider, iterator, final_layer,
        submission_id):
    all_sub_kaggle_set.load()
    assert all_sub_kaggle_set.resample_half == False, ("Not implemented for "
        "resample half")
    all_sub_kaggle_set.load_test()
    # following line will just do the preprocessing already on the train set...
    dataset_provider.get_train_merged_valid_test(all_sub_kaggle_set)
    test_sets_per_subj = []
    for i_subject in range(12):
        kaggle_set = all_sub_kaggle_set.kaggle_sets[i_subject]
        this_sets = []
        for i_test_series in range(2):
            # Get input
            X_test = kaggle_set.test_X_series[i_test_series][:,:,np.newaxis,np.newaxis]
            fake_test_y = np.ones((len(X_test), 6))
            test_set = DenseDesignMatrixWrapper(
                topo_view=X_test, 
                y=fake_test_y)
            if dataset_provider.preprocessor is not None:
                dataset_provider.preprocessor.apply(test_set, can_fit=False)
            this_sets.append(test_set)
        assert len(this_sets) == 2
        test_sets_per_subj.append(this_sets)
    
    ### Create prediction function and create predictions
    log.info("Create prediction functions...")
    input_var = lasagne.layers.get_all_layers(final_layer)[0].input_var
    predictions = lasagne.layers.get_output(final_layer, deterministic=True)
    pred_fn = theano.function([input_var], predictions)
    log.info("Setup iterator...")
    n_sample_preds = get_n_sample_preds(final_layer)
    iterator.n_sample_preds = n_sample_preds
    log.info("Make predictions...")
    preds_per_subject = []
    for i_subject in range(12):
        log.info("Predictions for Subject {:d}...".format(i_subject + 1))
        test_sets_subj = test_sets_per_subj[i_subject]
        preds = get_y_for_subject(pred_fn, test_sets_subj[0], test_sets_subj[1],
                         iterator, final_layer)
        preds_per_subject.append(preds)
    log.info("Done")
    log.info("Create csv...")
    cols = ['HandStart','FirstDigitTouch',
    'BothStartLoadPhase','LiftOff',
    'Replace','BothReleased']
    # collect ids
    all_ids = []
    all_preds = []
    for i_subject in range(12):
        pred_subj_per_series = preds_per_subject[i_subject]
        for i_series in (9,10):
            id_prefix = "subj{:d}_series{:d}_".format(i_subject+1, i_series)
            this_preds = pred_subj_per_series[i_series-9] # respect offsets
            all_preds.extend(this_preds)
            this_ids = [id_prefix + str(i_sample) for i_sample in range(this_preds.shape[0])]
            all_ids.extend(this_ids)
            
    all_ids = np.array(all_ids)
    all_preds = np.array(all_preds)
    assert all_ids.shape == (3144171,)
    assert all_preds.shape == (3144171,6)
    submission = pd.DataFrame(index=all_ids,
                              columns=cols,
                              data=all_preds)
    
    
    
    csv_output = StringIO.StringIO()
    submission.to_csv(csv_output, index_label='id',float_format='%.3f')
    csv_str = csv_output.getvalue()
    
    log.info("Create zip...")
    zip_file_name = os.path.join(folder_name, "{:d}.zip".format(submission_id))
    submission_zip_file = ZipFile(zip_file_name, 'w', ZIP_DEFLATED)
    submission_zip_file.writestr("submission.csv", csv_str)
    submission_zip_file.close()
    log.info("Done")

def get_y_for_subject(pred_fn, test_set_0, test_set_1, iterator, final_layer):
    """Assumes there was no resampling!!"""
    batch_gen_0 = iterator.get_batches(test_set_0, shuffle=False)
    all_preds_0 = [pred_fn(batch[0]) for batch in batch_gen_0]
    batch_gen_1 = iterator.get_batches(test_set_1, shuffle=False)
    all_preds_1 = [pred_fn(batch[0]) for batch in batch_gen_1]
    n_sample_preds = get_n_sample_preds(final_layer)
    input_time_length = lasagne.layers.get_all_layers(final_layer)[0].shape[2]
    
    n_samples_0 = test_set_0.get_topological_view().shape[0]
    preds_arr_0 = get_reshaped_cnt_preds(all_preds_0, n_samples_0, 
        input_time_length, n_sample_preds)
    n_samples_1 = test_set_1.get_topological_view().shape[0]
    preds_arr_1 = get_reshaped_cnt_preds(all_preds_1, n_samples_1, 
        input_time_length, n_sample_preds)
    series_preds = [preds_arr_0, preds_arr_1]
    return series_preds