from wyrm.processing import select_channels, append_cnt
from braindecode.mywyrm.processing import (
    resample_cnt, common_average_reference_cnt, exponential_standardize_cnt,
    set_channel_to_zero)
from braindecode.mywyrm.clean import (NoCleaner, clean_train_test_cnt)
import itertools
from sklearn.cross_validation import KFold
from braindecode.csp.pipeline import (BinaryCSP, FilterbankCSP,
    MultiClassWeightedVoting)
import numpy as np
from copy import deepcopy
from pylearn2.utils import serial
from pylearn2.config import yaml_parse
from numpy.random import RandomState
from braindecode.datasets.sensor_positions import sort_topologically
from braindecode.datasets.generate_filterbank import generate_filterbank,\
    filterbank_is_stable
import logging
log = logging.getLogger(__name__)

def create_csp_experiment(yaml_filename):
    """Utility function to create csp experiment from yaml file"""
    # used to be called standardize, now called standardize_epo
    content = open(yaml_filename, 'r').read()
    content = content.replace('standardize:', 'standardize_epo:')

    train_dict = yaml_parse.load(content)
    return train_dict['csp_train']

class CSPExperiment(object):
    """
        A Filter Bank Common Spatial Patterns Experiment.

        Parameters
        ----------
        set_loader : Dataset loader
            An object which loads the dataset. 
            Should have a load method which returns a wyrm data object with 
            the continuuous signal.
        cleaner : Cleaning object
            Should have a clean method which accepts the continuuous signal as 
            a wyrm object and returns the rejected chans, rejected trials and
            the clean trials.
        sensor_names : string array
             Names of sensors to use. 
             None implies all sensors.
        resample_fs : float
            The resampling frequency. None means no resampling.
        min_freq : int
            The minimum frequency of the filterbank.
        max_freq : int
            The maximum frequency of the filterbank.
        last_low_freq : int
            The last frequency with the low width frequency of the filterbank.
        low_width : int
            The width of the filterbands in the lower frequencies.
        low_overlap : int
            The overlap of the filterbands in the lower frequencies.
        high_width : int
            The width of the filterbands in the higher frequencies.
        high_overlap : int
            The overlap of the filterbands in the higher frequencies.
        filt_order : int
            The filter order of the butterworth filter which computes the filterbands.
        standardize_filt_cnt: bool
            Whether to do (channel-wise) exponential moving standardization of the
            continuous signal
        segment_ival : sequence of 2 floats
            The start and end of the trial in milliseconds with respect to the markers.
        standardize_epo : bool
            Whether to standardize the features of the filterbank before training.
            Will do online standardization, i.e., will compute means and standard
            deviations on the training fold and then compute running means and
            standard deviations going trial by trial through the test fold.
        n_folds : int
            How many folds. Also determines size of the test fold, e.g.
            5 folds imply the test fold has 20% of the original data.
        n_top_bottom_csp_filters : int
            Number of top and bottom CSP filters to select from all computed filters.
            Top and bottom refers to CSP filters sorted by their eigenvalues.
            So a value of 3 here will lead to 6(!) filters.
            None means all filters.
        n_selected_filterbands : int
            Number of filterbands to select for the filterbank.
            Will be selected by the highest training accuracies.
            None means all filterbands.
        n_selected_features : int
            Number of features to select for the filterbank.
            Will be selected by an internal cross validation across feature
            subsets.
            None means all features.
        forward_steps : int
            Number of forward steps to make in the feature selection,
            before the next backward step.
        backward_steps : int
            Number of backward steps to make in the feature selection,
            before the next forward step.
        stop_when_no_improvement: bool
            Whether to stop the feature selection if the interal cross validation
            accuracy could not be improved after an epoch finished
            (epoch=given number of forward and backward steps).
            False implies always run until wanted number of features.
        only_last_fold: bool
            Whether to train only on the last fold. 
            True implies a train-test split, where the n_folds parameter
            determines the size of the test fold.
            Test fold will always be at the end of the data (timewise).
        restricted_n_trials: int
            Take only a restricted number of the clean trials.
            None implies all clean trials.
        common_average_reference: bool
            Whether to run a common average reference on the signal (before filtering).
        ival_optimizer: IvalOptimizer object
            If given, optimize the ival inside the trial before CSP.
            None means use the full trial.
        shuffle: bool
            Whether to shuffle the clean trials before splitting them into folds.
            False implies folds are time-blocks, True implies folds are random
            mixes of trials of the entire file.
        marker_def: dict
            Dictionary mapping class names to marker numbers, e.g.
            {'1 - Correct': [31], '2 - Error': [32]}
    """
    def __init__(self,set_loader, 
            cleaner=None,
            sensor_names=None,
            resample_fs=None,
            standardize_cnt=False,
            min_freq=0,
            max_freq=48,
            last_low_freq=48,
            low_width=4,
            low_overlap=0,
            high_width=4,
            high_overlap=0,
            filt_order=3,
            standardize_filt_cnt=False,
            segment_ival=[0,4000], 
            standardize_epo=True,
            n_folds=5,
            n_top_bottom_csp_filters=None,
            n_selected_filterbands=None,
            n_selected_features=None,
            forward_steps=4,
            backward_steps=2,
            stop_when_no_improvement=False,
            only_last_fold=False,
            restricted_n_trials=None,
            common_average_reference=False,
            ival_optimizer=None,
            shuffle=False,
            marker_def=None,
            set_cz_to_zero=False,
            low_bound=0.2):
        local_vars = locals()
        del local_vars['self']
        self.__dict__.update(local_vars)
        # remember params for later result printing etc
        # maybe delete this again?
        self.original_params = deepcopy(local_vars)
        if self.original_params['cleaner'] is not None:
            self.original_params['cleaner'] = self.original_params['cleaner'].__class__.__name__
        if self.original_params['set_loader'] is not None:
            self.original_params['set_loader'] = self.original_params['set_loader'].__class__.__name__
        if self.original_params['ival_optimizer'] is not None:
            self.original_params['ival_optimizer'] = self.original_params['ival_optimizer'].__class__.__name__
        # Default marker def is form our EEG 3-4 sec motor imagery dataset
        if self.marker_def is None:
            self.marker_def = {'1 - Right Hand': [1], '2 - Left Hand': [2], 
                    '3 - Rest': [3], '4 - Feet': [4]}
        if self.cleaner is None:
            self.cleaner = NoCleaner(segment_ival=self.segment_ival,
                marker_def=self.marker_def)

    def run(self):
        log.info("Loading set...")
        self.load_bbci_set()
        log.info("Cleaning set...")
        self.clean_set()
        log.info("Preprocessing set...")
        self.preprocess_set()
        self.remember_sensor_names()
        self.init_training_vars()
        log.info("Running Training...")
        self.run_training()

    def load_bbci_set(self):
        self.cnt = self.set_loader.load()

    def clean_set(self):
        clean_result = self.cleaner.clean(self.cnt)
        self.rejected_chan_names = np.array(clean_result.rejected_chan_names)
        self.rejected_trials = np.array(clean_result.rejected_trials)
        self.clean_trials = np.array(clean_result.clean_trials)
        # do not remove rejected channels yet to allow calling
        # this function several times with same result

    def preprocess_set(self):
        # only remove rejected channels now so that clean function can
        # be called multiple times without changing cleaning results
        self.cnt = select_channels(self.cnt, self.rejected_chan_names,
            invert=True)
        if self.sensor_names is not None:
            # Note this does not respect order of sensor names,
            # it selects chans form given sensor names
            # but keeps original order
            self.cnt = select_channels(self.cnt, self.sensor_names)

        if self.set_cz_to_zero is True:
            self.cnt = set_channel_to_zero(self.cnt, 'Cz')
        if self.resample_fs is not None:
            self.cnt = resample_cnt(self.cnt, newfs=self.resample_fs)
        if self.common_average_reference is True:
            self.cnt = common_average_reference_cnt(self.cnt)
        if self.standardize_cnt is True:
            self.cnt = exponential_standardize_cnt(self.cnt)

    def remember_sensor_names(self):
        """ Just to be certain have correct sensor names, take them
        from cnt signal"""
        self.sensor_names = self.cnt.axes[1]

    def init_training_vars(self):
        self.filterbands = generate_filterbank(min_freq=self.min_freq,
            max_freq=self.max_freq, last_low_freq=self.last_low_freq, 
            low_width=self.low_width, low_overlap=self.low_overlap,
            high_width=self.high_width, high_overlap=self.high_overlap,
            low_bound=self.low_bound)
        assert filterbank_is_stable(self.filterbands, self.filt_order, 
            self.cnt.fs), (
                "Expect filter bank to be stable given filter order.")
        n_classes = len(self.marker_def)
        self.class_pairs = list(itertools.combinations(range(n_classes),2))
        # use only number of clean trials to split folds
        n_clean_trials = len(self.clean_trials)
        if self.restricted_n_trials is not None:
            if self.restricted_n_trials <= 1:
                n_clean_trials = int(n_clean_trials * self.restricted_n_trials)
            else:
                n_clean_trials = min(n_clean_trials, self.restricted_n_trials)
        if not self.shuffle:
            folds = KFold(n_clean_trials, n_folds=self.n_folds, 
                shuffle=False)
        else:
            rng = RandomState(903372376)
            folds = KFold(n_clean_trials, n_folds=self.n_folds, 
                shuffle=True, random_state=rng)
            
        # remap to original indices in unclean set(!)
        self.folds = map(lambda fold: 
            {'train': self.clean_trials[fold[0]], 
             'test': self.clean_trials[fold[1]]},
            folds)
        if self.only_last_fold:
            self.folds = self.folds[-1:]

    def run_training(self):
        self.binary_csp = BinaryCSP(self.cnt, self.filterbands, 
            self.filt_order, self.folds, self.class_pairs, 
            self.segment_ival, self.n_top_bottom_csp_filters, 
            standardize_filt_cnt=self.standardize_filt_cnt,
            standardize_epo=self.standardize_epo,
            ival_optimizer=self.ival_optimizer,
            marker_def=self.marker_def)
        self.binary_csp.run()
        
        self.filterbank_csp = FilterbankCSP(self.binary_csp, 
            n_features=self.n_selected_features,
            n_filterbands=self.n_selected_filterbands,
            forward_steps=self.forward_steps,
            backward_steps=self.backward_steps,
            stop_when_no_improvement=self.stop_when_no_improvement)
        self.filterbank_csp.run()
        
        self.multi_class = MultiClassWeightedVoting(
                                    self.binary_csp.train_labels_full_fold, 
                                    self.binary_csp.test_labels_full_fold,
                                    self.filterbank_csp.train_pred_full_fold,
                                    self.filterbank_csp.test_pred_full_fold,
                                    self.class_pairs)
        self.multi_class.run()

class CSPRetrain():
    """ CSP Retraining on existing filters computed previously."""
    def __init__(self, trainer_filename, n_selected_features="asbefore",
            n_selected_filterbands="asbefore",forward_steps=2,
            backward_steps=1, stop_when_no_improvement=False):
        self.trainer_filename = trainer_filename
        self.n_selected_features = n_selected_features
        self.n_selected_filterbands = n_selected_filterbands
        self.forward_steps = forward_steps
        self.backward_steps = backward_steps
        self.stop_when_no_improvement = stop_when_no_improvement

    def run(self):
        log.info("Loading trainer...")
        self.trainer = serial.load(self.trainer_filename)
        if self.n_selected_features == "asbefore":
            self.n_selected_features = self.trainer.filterbank_csp.n_features
        if self.n_selected_filterbands == "asbefore":
            self.n_selected_filterbands = self.trainer.filterbank_csp.n_filterbands
        # For later storage, remember selected features and filterbands
        # TODELAY: solve this more cleanly during saving or sth :)
        self.trainer.original_params['n_selected_features'] = \
            self.n_selected_features
        self.trainer.original_params['n_selected_filterbands'] = \
            self.n_selected_filterbands
        recreate_filterbank(self.trainer, self.n_selected_features,
            self.n_selected_filterbands, self.forward_steps,
            self.backward_steps, self.stop_when_no_improvement)
        
        log.info("Rerunning filterbank...")
        self.trainer.filterbank_csp.run()
        recreate_multi_class(self.trainer)
        log.info("Rerunning multiclass...")
        self.trainer.multi_class.run()

class TwoFileCSPExperiment(CSPExperiment):
    def __init__(self,train_set_loader, test_set_loader, train_cleaner,
            test_cleaner, 
            sensor_names=None,
            resample_fs=None,
            standardize_cnt=False,
            min_freq=0,
            max_freq=48,
            last_low_freq=48,
            low_width=4,
            low_overlap=0,
            high_width=4,
            high_overlap=0,
            filt_order=3,
            standardize_filt_cnt=False,
            segment_ival=[0,4000], 
            standardize_epo=True,
            n_folds=5,
            n_top_bottom_csp_filters=None,
            n_selected_filterbands=None,
            n_selected_features=None,
            forward_steps=4,
            backward_steps=2,
            stop_when_no_improvement=False,
            only_last_fold=False,
            restricted_n_trials=None,
            common_average_reference=False,
            ival_optimizer=None,
            shuffle=False,
            marker_def=None,
            set_cz_to_zero=False,
            low_bound=0.2):
        self.test_set_loader = test_set_loader
        self.test_cleaner = test_cleaner
        super(TwoFileCSPExperiment, self).__init__(train_set_loader, 
            cleaner=train_cleaner,sensor_names=sensor_names,
            resample_fs=resample_fs, standardize_cnt=standardize_cnt,
            min_freq=min_freq,max_freq=max_freq,
            last_low_freq=last_low_freq,low_width=low_width,
            low_overlap=low_overlap,high_width=high_width,
            high_overlap=high_overlap,filt_order=filt_order,
            standardize_filt_cnt=standardize_filt_cnt,
            segment_ival=segment_ival, standardize_epo=standardize_epo,
            n_folds=n_folds,n_top_bottom_csp_filters=n_top_bottom_csp_filters,
            n_selected_filterbands=n_selected_filterbands,
            n_selected_features=n_selected_features,
            forward_steps=forward_steps,backward_steps=backward_steps,
            stop_when_no_improvement=stop_when_no_improvement,
            only_last_fold=only_last_fold,restricted_n_trials=restricted_n_trials,
            common_average_reference=common_average_reference,
            ival_optimizer=ival_optimizer,shuffle=shuffle,
            marker_def=marker_def,
            set_cz_to_zero=set_cz_to_zero,
            low_bound=low_bound)

    def run(self):
        log.info("Loading train set...")
        self.load_bbci_set()
        log.info("Loading test set...")
        self.load_bbci_test_set()
        log.info("Cleaning both sets...")
        self.clean_both_sets()
        log.info("Preprocessing train set...")
        self.preprocess_set()
        log.info("Preprocessing test set...")
        self.preprocess_test_set()
        self.remember_sensor_names()
        self.init_training_vars()
        log.info("Running Training...")
        self.run_training()

    def load_bbci_test_set(self):
        self.test_cnt = self.test_set_loader.load()

    def clean_both_sets(self):
        # Clean by directy changing the cnt variables...
        clean_train_cnt, clean_test_cnt = clean_train_test_cnt(
            self.cnt, self.test_cnt, self.cleaner,
            self.test_cleaner)
        self.cnt = clean_train_cnt
        self.test_cnt = clean_test_cnt
        self.rejected_chan_names = [] # necessary since will be used
        # in preprocess set...
        self.rejected_trials = 'unknown'
        self.clean_trials = 'unknown'

    def preprocess_test_set(self):
        if self.sensor_names is not None:
            self.sensor_names = sort_topologically(self.sensor_names)
            self.test_cnt = select_channels(self.test_cnt, self.sensor_names)
        if self.set_cz_to_zero is True:
            self.test_cnt = set_channel_to_zero(self.test_cnt, 'Cz')
        if self.resample_fs is not None:
            self.test_cnt = resample_cnt(self.test_cnt, newfs=self.resample_fs)
        if self.common_average_reference is True:
            self.test_cnt = common_average_reference_cnt(self.test_cnt)
        if self.standardize_cnt is True:
            self.test_cnt = exponential_standardize_cnt(self.test_cnt)

    def init_training_vars(self):
        assert self.n_folds is None, "Cannot use folds on train test split"
        assert self.restricted_n_trials is None, "Not implemented yet"
        self.filterbands = generate_filterbank(min_freq=self.min_freq,
            max_freq=self.max_freq, last_low_freq=self.last_low_freq, 
            low_width=self.low_width, low_overlap=self.low_overlap,
            high_width=self.high_width, high_overlap=self.high_overlap,
            low_bound=self.low_bound)
        assert filterbank_is_stable(self.filterbands, self.filt_order, 
            self.cnt.fs), (
                "Expect filter bank to be stable given filter order.")
        n_classes = len(self.marker_def)
        self.class_pairs = list(itertools.combinations(range(n_classes),2))
        train_fold = range(len(self.cnt.markers))
        test_fold = np.arange(len(self.test_cnt.markers)) + len(train_fold)
        self.folds = [{'train': train_fold, 'test': test_fold}]
        assert np.intersect1d(self.folds[0]['test'], 
            self.folds[0]['train']).size == 0

        # merge cnts!!
        self.cnt = append_cnt(self.cnt, self.test_cnt)

def recreate_filterbank(train_csp_obj, n_features, n_filterbands,
        forward_steps, backward_steps, stop_when_no_improvement):
    train_csp_obj.filterbank_csp = FilterbankCSP(train_csp_obj.binary_csp,
        n_features, n_filterbands, 
            forward_steps=forward_steps,
            backward_steps=backward_steps,
            stop_when_no_improvement=stop_when_no_improvement)

def recreate_multi_class(train_csp_obj):
    """ Assumes filterbank + possibly binary csp was rerun and
    recreates multi class weighted voting object 
    with new labels + predictions. """
    train_csp_obj.multi_class = MultiClassWeightedVoting(
        train_csp_obj.binary_csp.train_labels_full_fold, 
        train_csp_obj.binary_csp.test_labels_full_fold,
        train_csp_obj.filterbank_csp.train_pred_full_fold,
        train_csp_obj.filterbank_csp.test_pred_full_fold,
        train_csp_obj.class_pairs)