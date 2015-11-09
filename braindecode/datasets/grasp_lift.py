import os
from scikits.samplerate import resample
import pandas as pd
import numpy as np
import logging
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

    