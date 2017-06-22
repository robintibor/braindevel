import logging
from braindecode2.trial_segment import segment_dat
from braindecode2.mywyrm.processing import (
    lda_apply, select_classes,
    lda_train_scaled, apply_csp_var_log, bandpass_cnt,
    calculate_csp, exponential_standardize_cnt)
import numpy as np
from braindecode2.mywyrm.processing import online_standardize_epo
log = logging.getLogger(__name__)

class BinaryCSP(object):
    def __init__(self, cnt, filterbands, filt_order, folds,
            class_pairs, segment_ival, n_filters,
            ival_optimizer, standardize_filt_cnt,
            standardize_epo, marker_def):
        self.__dict__.update(locals())
        del self.self

    def run(self):
        self.init_results()
        # TODELAY: split apart collecting of features and training lda?
        # however then you would not get progress output during training
        # only at very end
        for bp_nr, filt_band in enumerate(self.filterbands):
            self.print_filter(bp_nr)
            bandpassed_cnt = bandpass_cnt(self.cnt, filt_band[0], filt_band[1],
                filt_order=self.filt_order)
            if self.standardize_filt_cnt:
                bandpassed_cnt = exponential_standardize_cnt(bandpassed_cnt)
            epo = segment_dat(bandpassed_cnt,
                marker_def=self.marker_def, 
                ival=self.segment_ival)

            for fold_nr in xrange(len(self.folds)):
                self.run_fold(epo, bp_nr, fold_nr)
    
    def run_fold(self, epo, bp_nr, fold_nr):  
        self.print_fold_nr(fold_nr) 
        train_test = self.folds[fold_nr]
        train_ind = train_test['train']
        test_ind = train_test['test']
        epo_train = epo.isel(trials=train_ind)
        epo_test = epo.isel(trials=test_ind)#select_epochs(epo, test_ind)
        if self.standardize_epo:
            epo_train, epo_test = online_standardize_epo(epo_train, epo_test)
        # TODELAY: also integrate into init and store results
        self.train_labels_full_fold[fold_nr] = epo_train.trials.data
        self.test_labels_full_fold[fold_nr] = epo_test.trials.data
        
        for pair_nr in xrange(len(self.class_pairs)):
            self.run_pair(epo_train, epo_test, bp_nr, fold_nr, pair_nr)
            
    def run_pair(self, epo_train, epo_test, bp_nr, fold_nr, pair_nr):
        class_pair = self.class_pairs[pair_nr]
        self.print_class_pair(class_pair)
        
        ### Run Training
        epo_train_pair = select_classes(epo_train, class_pair)
        epo_test_pair = select_classes(epo_test, class_pair)

        assert self.ival_optimizer is None

            
        self.train_labels[fold_nr][pair_nr] = epo_train_pair.trials.data
        self.test_labels[fold_nr][pair_nr] = epo_test_pair.trials.data
        
        ## Calculate CSP
        filters, patterns, variances = calculate_csp(epo_train_pair)
        ## Apply csp, calculate features
        if self.n_filters is not None:
            # take topmost and bottommost filters, e.g.
            # for n_filters=3 0,1,2,-3,-2,-1
            columns = range(0, self.n_filters) + \
                range(-self.n_filters, 0)
        else: # take all possible filters
            columns = range(len(filters))
        train_feature = apply_csp_var_log(epo_train_pair, filters, 
            columns)

        ## Calculate LDA
        clf = lda_train_scaled(train_feature, shrink=True)
        assert not np.any(np.isnan(clf[0]))
        assert not np.isnan(clf[1])
        ## Apply LDA to train
        train_out = lda_apply(train_feature, clf)
        true_0_1_labels_train = train_feature.trials.data == class_pair[1]
        predicted_train = train_out >= 0
        train_accuracy = np.mean(true_0_1_labels_train == predicted_train)

        ### Feature Computation and LDA Application for test
        test_feature = apply_csp_var_log(epo_test_pair, filters, 
            columns)
        test_out = lda_apply(test_feature, clf)
        true_0_1_labels_test= test_feature.trials.data == class_pair[1]
        predicted_test = test_out >= 0
        test_accuracy = np.mean(true_0_1_labels_test == predicted_test)

        ### Feature Computations for full fold (for later multiclass)
        train_feature_full_fold = apply_csp_var_log(epo_train,
             filters, columns)
        test_feature_full_fold = apply_csp_var_log(epo_test,
             filters, columns)
        ### Store results
        # only store used patterns filters variances 
        # to save memory space on disk
        self.store_results(bp_nr, fold_nr, pair_nr,
            filters[:, columns], 
            patterns[:,columns], 
            variances[columns],
            train_feature, test_feature,
            train_feature_full_fold, test_feature_full_fold, clf,
            train_accuracy, test_accuracy)
        
        self.print_results(bp_nr, fold_nr, pair_nr)  
          

    def init_results(self):
        n_filterbands = len(self.filterbands)
        n_folds = len(self.folds)
        n_class_pairs = len(self.class_pairs)
        result_shape = (n_filterbands, n_folds, n_class_pairs)
        all_varnames = ['filters', 'patterns', 'variances',
                        'train_feature', 'test_feature',
                        'train_feature_full_fold', 'test_feature_full_fold', 
                        'clf', 'train_accuracy', 'test_accuracy']
        if self.ival_optimizer is not None:
            all_varnames.append('best_ival')
        for varname in all_varnames:
            self.__dict__[varname] = np.empty(result_shape, dtype=object)

        
        # TODELAY: also integrate into init and store results
        self.train_labels_full_fold = np.empty(len(self.folds), dtype=object)
        self.test_labels_full_fold = np.empty(len(self.folds), dtype=object)
        self.train_labels = np.empty((len(self.folds), len(self.class_pairs)),
            dtype=object)
        self.test_labels = np.empty((len(self.folds), len(self.class_pairs)),
             dtype=object)

    def store_results(self, bp_nr, fold_nr, pair_nr,
                        filters, patterns, variances,
                        train_feature, test_feature,
                        train_feature_full_fold, test_feature_full_fold, clf,
                        train_accuracy, test_accuracy):
        """ Store all supplied arguments to this objects dict, at the correct
        indices for filterband/fold/class_pair."""
        local_vars = locals()
        del local_vars['self']
        del local_vars['bp_nr']
        del local_vars['fold_nr']
        del local_vars['pair_nr']
        for var in local_vars:
            self.__dict__[var][bp_nr, fold_nr, pair_nr] = local_vars[var]

    def print_filter(self, bp_nr):
        log.info("Filter {:d}/{:d}, {:4.2f} to {:4.2f} Hz".format(bp_nr + 1, 
                len(self.filterbands), *self.filterbands[bp_nr]))

    def print_fold_nr(self, fold_nr):
        log.info("Fold Nr: {:d}".format(fold_nr + 1))

    def print_class_pair(self, class_pair):
        class_pair_plus_one = (np.array(class_pair) + 1).tolist()
        log.info ("Class {:d} vs {:d}".format(*class_pair_plus_one ))

    def print_results(self, bp_nr, fold_nr, pair_nr):
        log.info("Train: {:4.2f}%".format(
            self.train_accuracy[bp_nr, fold_nr, pair_nr] * 100))
        log.info("Test:  {:4.2f}%".format(
            self.test_accuracy[bp_nr, fold_nr, pair_nr] * 100))
