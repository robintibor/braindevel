import os
from scikits.samplerate import resample
import pandas as pd
import numpy as np
import logging
from copy import deepcopy
from braindecode.datasets.pylearn import DenseDesignMatrixWrapper
import lasagne
import theano
from braindecode.veganlasagne.layers import get_n_sample_preds
from braindecode.veganlasagne.monitors import get_reshaped_cnt_preds
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
    return clean,labels

def load_test(test_folder, i_subject, i_series):
    data_filename = 'subj{:d}_series{:d}_data.csv'.format(
        i_subject, i_series)
    data_file_path = os.path.join(test_folder, data_filename)
    data = pd.read_csv(data_file_path)
    clean = data.drop(['id' ], axis=1)#remove id
    return clean

class KaggleGraspLiftSet(object):
    reloadable=False
    def __init__(self, data_folder, i_subject):
        self.data_folder = data_folder
        self.i_subject = i_subject
        
    def ensure_is_loaded(self):
        if not hasattr(self, 'train_X_series'):
            self.load()
    
    def load(self):
        log.info("Loading data...")
        self.load_data()
        log.info("Resampling data...")
        self.resample_data()
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
            self.train_X_series.append(X_series)
            self.train_y_series.append(y_series)
            
        assert len(self.train_X_series) == 8, "Should be 8 train series for each subject"

       
        # all sensor names should be the same :)
        # so just take first part
        self.sensor_names = self.train_X_series[0].keys()
    
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

    def load_test(self):
        """Refers to test set from evaluation(without labels)"""
        log.info("Loading test data...")
        self.load_test_data()
        log.info("Resampling test data...")
        self.resample_test_data()
        log.info("..Done.")

    def load_test_data(self):
        test_folder = os.path.join(self.data_folder, 'test/')
        self.test_X_series = []
        for i_series in xrange(9,11):
            X_series = load_test(test_folder, self.i_subject, i_series)
            self.test_X_series.append(X_series)
        assert len(self.test_X_series) == 2, "Should be 2 test series for each subject"

    def resample_test_data(self):
        for i_series in xrange(2):
            X_series = np.array(self.test_X_series[i_series]).astype(np.float32)
            X_series = resample(X_series, 250.0/500.0, 'sinc_fastest')
            self.test_X_series[i_series] = X_series


def create_submission_csv(folder_name, kaggle_set, iterator, preprocessor,
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
    log.info("Create prediciton functions...")
    input_var = lasagne.layers.get_all_layers(final_layer)[0].input_var
    predictions = lasagne.layers.get_output(final_layer, deterministic=True)
    log.info("Make predictions...")
    pred_fn = theano.function([input_var], predictions)
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