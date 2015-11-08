import os
from scikits.samplerate import resample
import pandas as pd
import numpy as np
from braindecode.datasets.pylearn import DenseDesignMatrixWrapper
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

def load_test(i_subject, i_series):
    train_folder = 'data/kaggle-grasp-lift/test/'
    data_filename = 'subj{:d}_series{:d}_data.csv'.format(
        i_subject, i_series)
    data_file_path = os.path.join(train_folder, data_filename)
    data = pd.read_csv(data_file_path)
    clean = data.drop(['id' ], axis=1)#remove id
    return clean

class KaggleGraspLiftSet(object):
    reloadable=False
    def __init__(self, data_folder, i_subject):
        self.data_folder = data_folder
        self.i_subject = i_subject
        
    def ensure_is_loaded(self):
        if not hasattr(self, 'train_set'):
            self.load()
    
    def load(self):
        log.info("Loading data...")
        self.load_data()
        log.info("Resampling data...")
        self.resample_data()
        self.create_dense_design_matrices()
        log.info("..Done.")
        # hack to allow experiment to know targets will have two dimensions
        self.y = np.ones((1,1)) * np.nan

    def load_data(self):
        # First just load the data
        X_parts = []
        y_parts = []
        for i_series in xrange(1,9):
            X_series,y_series = load_train(self.data_folder, self.i_subject, i_series)
            X_parts.append(X_series)
            y_parts.append(y_series)
            
        assert len(X_parts) == 8, "Should be 8 series for each subject"

        # all sensor names should be the same :)
        # so just take first part
        self.sensor_names = X_parts[0].keys()
        self.X_train = np.array(pd.concat(X_parts[:6]), dtype=np.float32)
        self.X_valid = np.array(X_parts[6], dtype=np.float32)
        self.X_test = np.array(X_parts[7], dtype=np.float32)
        
        self.y_train = np.array(pd.concat(y_parts[:6]),dtype=np.int32)
        self.y_valid = np.array(y_parts[6], dtype=np.int32)
        self.y_test = np.array(y_parts[7], dtype=np.int32)
    
    def resample_data(self):
        # Now resample
        self.X_train = resample(np.array(self.X_train), 250.0/500.0, 'sinc_fastest')
        # shift all predictions backwards compared to data.
        # this ensures you are not using data from the future to make a prediciton
        # rather in a bad case maybe you do not even have all data up to the sample
        # to make the prediction
        self.y_train = self.y_train[1::2]
        # maybe some border effects remove predictions
        self.y_train = self.y_train[-len(self.X_train):]

        self.X_valid = resample(np.array(self.X_valid), 250.0/500.0, 'sinc_fastest')
        self.y_valid = self.y_valid[1::2]
        self.y_valid = self.y_valid[-len(self.X_valid):]

        self.X_test = resample(np.array(self.X_test), 250.0/500.0, 'sinc_fastest')
        self.y_test = self.y_test[1::2]
        self.y_test = self.y_test[-len(self.X_test):]
        
    def create_dense_design_matrices(self):
        # create dense design matrix sets
        self.train_set = DenseDesignMatrixWrapper(
            topo_view=self.X_train[:,:,np.newaxis,np.newaxis], 
            y=self.y_train, axes=('b','c',0,1))
        self.valid_set = DenseDesignMatrixWrapper(
            topo_view=self.X_valid[:,:,np.newaxis,np.newaxis],
            y=self.y_valid, axes=('b','c',0,1))
        self.test_set = DenseDesignMatrixWrapper(
             topo_view=self.X_test[:,:,np.newaxis,np.newaxis],
             y=self.y_test, axes=('b','c',0,1))
        # free memory now
        del self.X_train, self.X_valid, self.X_test
        del self.y_train, self.y_valid, self.y_test
        
        

    