from wyrm.processing import (calculate_csp, select_epochs, select_classes,
    lda_apply, append_epo, select_ival)
from braindecode.mywyrm.processing import ( 
    lda_train_scaled, segment_dat_fast, apply_csp_var_log, bandpass_cnt)
import numpy as np
from braindecode.csp.feature_selection import select_features
from copy import deepcopy
from sklearn.cross_validation import KFold
from braindecode.mywyrm.processing import online_standardize_epo
import logging 
log = logging.getLogger(__name__)

class BinaryCSP(object):
    def __init__(self, cnt, filterbands, filt_order, folds,
            class_pairs, segment_ival, n_filters,
            ival_optimizer, standardize, marker_def=None):
        self.__dict__.update(locals())
        del self.self
        # Default marker def is form our EEG 3-4 sec motor imagery dataset
        if self.marker_def is None:
            self.marker_def = {'1 - Right Hand': [1], '2 - Left Hand': [2], 
                    '3 - Rest': [3], '4 - Feet': [4]}

    def run(self):
        self.init_results()
        # TODELAY: split apart collecting of features and training lda?
        # however then you would not get progress output during training
        # only at very end
        for bp_nr, filt_band in enumerate(self.filterbands):
            self.print_filter(bp_nr)
            bandpassed_cnt = bandpass_cnt(self.cnt, filt_band[0], filt_band[1],
                filt_order=self.filt_order)
            epo = segment_dat_fast(bandpassed_cnt, 
                marker_def=self.marker_def, 
                ival=self.segment_ival)
            
            for fold_nr in xrange(len(self.folds)):
                self.run_fold(epo, bp_nr, fold_nr)
    
    def run_fold(self, epo, bp_nr, fold_nr):  
        self.print_fold_nr(fold_nr) 
        train_test = self.folds[fold_nr]
        train_ind = train_test['train']
        test_ind = train_test['test']
        epo_train = select_epochs(epo, train_ind)
        epo_test = select_epochs(epo, test_ind)
        if self.standardize:
            epo_train, epo_test = online_standardize_epo(epo_train, epo_test)
        # TODELAY: also integrate into init and store results
        self.train_labels_full_fold[fold_nr] = epo_train.axes[0]
        self.test_labels_full_fold[fold_nr] = epo_test.axes[0]
        
        for pair_nr in xrange(len(self.class_pairs)):
            self.run_pair(epo_train, epo_test, bp_nr, fold_nr, pair_nr)
            

    def run_pair(self, epo_train, epo_test, bp_nr, fold_nr, pair_nr):
        class_pair = self.class_pairs[pair_nr]
        self.print_class_pair(class_pair)
        
        ### Run Training
        epo_train_pair = select_classes(epo_train, class_pair)
        epo_test_pair = select_classes(epo_test, class_pair)
        if self.ival_optimizer is not None:
            best_segment_ival = self.ival_optimizer.optimize(epo_train_pair)
            log.info("Ival {:.0f}ms - {:.0f}ms".format(*best_segment_ival))
            epo_train_pair = select_ival(epo_train_pair, best_segment_ival)
            epo_test_pair = select_ival(epo_test_pair, best_segment_ival)
            epo_train = select_ival(epo_train, best_segment_ival)
            epo_test = select_ival(epo_test, best_segment_ival)
            
        self.train_labels[fold_nr][pair_nr] = epo_train_pair.axes[0]
        self.test_labels[fold_nr][pair_nr] = epo_test_pair.axes[0]
        
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
        correct_train = train_feature.axes[0] == class_pair[1]
        predicted_train = train_out >= 0
        train_accuracy = (sum(correct_train == predicted_train) 
            / float(len(predicted_train)))

        ### Feature Computation and LDA Application for test
        test_feature = apply_csp_var_log(epo_test_pair, filters, 
            columns)
        test_out = lda_apply(test_feature, clf)
        correct_test= test_feature.axes[0] == class_pair[1]
        predicted_test = test_out >= 0
        test_accuracy = (sum(correct_test == predicted_test) / 
            float(len(predicted_test)))
        
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
        if self.ival_optimizer is not None:
            self.best_ival[bp_nr, fold_nr, pair_nr] = best_segment_ival
        
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



class FilterbankCSP(object):
    def __init__(self, binary_csp, n_features=None, n_filterbands=None,
            forward_steps=2, backward_steps=1, stop_when_no_improvement=False):
        self.binary_csp = binary_csp
        self.n_features = n_features
        self.n_filterbands = n_filterbands
        self.forward_steps = forward_steps
        self.backward_steps = backward_steps
        self.stop_when_no_improvement = stop_when_no_improvement
        
    def run(self):
        self.select_filterbands()
        if self.n_features is not None:
            self.collect_best_features()
            #self.select_features()
        else: 
            self.collect_features()
        self.train_classifiers()
        self.predict_outputs()
        
    def select_filterbands(self):
        n_all_filterbands = len(self.binary_csp.filterbands)
        if self.n_filterbands is None:
            self.selected_filter_inds = range(n_all_filterbands)
        else:
            # Select the filterbands with the highest mean accuracy on the
            # training sets
            mean_accs = np.mean(self.binary_csp.train_accuracy, axis=(1,2))
            best_filters = np.argsort(mean_accs)[::-1][:self.n_filterbands]
            self.selected_filter_inds = best_filters
        
    def collect_features(self):
        n_folds = len(self.binary_csp.folds)
        n_class_pairs = len(self.binary_csp.class_pairs)
        result_shape = (n_folds, n_class_pairs)
        self.train_feature = np.empty(result_shape, dtype=object)
        self.train_feature_full_fold = np.empty(result_shape, dtype=object)
        self.test_feature = np.empty(result_shape, dtype=object)
        self.test_feature_full_fold = np.empty(result_shape, dtype=object)

        bincsp = self.binary_csp # just to make code shorter
        filter_inds = self.selected_filter_inds
        # merge along featureaxis: axis 1
        merge_features = lambda fv1, fv2: append_epo(fv1, fv2, classaxis=1)
        for fold_i in range(n_folds):
            for class_i in range(n_class_pairs):
                self.train_feature[fold_i, class_i] = reduce(
                    merge_features, 
                    bincsp.train_feature[filter_inds, fold_i, class_i])
                self.train_feature_full_fold[fold_i, class_i] = reduce(
                    merge_features, 
                    bincsp.train_feature_full_fold[filter_inds, fold_i, class_i])
                self.test_feature[fold_i, class_i] = reduce(
                    merge_features,
                    bincsp.test_feature[filter_inds, fold_i, class_i])
                self.test_feature_full_fold[fold_i, class_i] = reduce(
                    merge_features,
                    bincsp.test_feature_full_fold[filter_inds, fold_i, class_i])

    def collect_best_features(self):
        """ Selects features filterwise per filterband, starting with no features,
        then selecting the best filterpair from the bestfilterband (measured on internal
        train/test split)"""
        bincsp = self.binary_csp # just to make code shorter
        n_folds = len(self.binary_csp.folds)
        n_class_pairs = len(self.binary_csp.class_pairs)
        result_shape = (n_folds, n_class_pairs)
        self.train_feature = np.empty(result_shape, dtype=object)
        self.train_feature_full_fold = np.empty(result_shape, dtype=object)
        self.test_feature = np.empty(result_shape, dtype=object)
        self.test_feature_full_fold = np.empty(result_shape, dtype=object)
        self.selected_filters_per_filterband = np.empty(result_shape, dtype=object)
        for fold_i in range(n_folds):
            for class_pair_i in range(n_class_pairs):
                bin_csp_train_features = deepcopy(bincsp.train_feature[
                    self.selected_filter_inds, fold_i, class_pair_i])
                bin_csp_train_features_full_fold = deepcopy(
                    bincsp.train_feature_full_fold[
                        self.selected_filter_inds,
                        fold_i, class_pair_i])
                bin_csp_test_features = deepcopy(bincsp.test_feature[
                    self.selected_filter_inds, fold_i, class_pair_i])
                bin_csp_test_features_full_fold = deepcopy(
                    bincsp.test_feature_full_fold[
                        self.selected_filter_inds,fold_i, class_pair_i])
                selected_filters_per_filt = self.select_best_filters_best_filterbands(
                    bin_csp_train_features, max_features=self.n_features,
                    forward_steps=self.forward_steps, 
                    backward_steps=self.backward_steps,
                    stop_when_no_improvement=self.stop_when_no_improvement)
                self.train_feature[fold_i, class_pair_i] = \
                    self.collect_features_for_filter_selection(
                        bin_csp_train_features, selected_filters_per_filt)
                self.train_feature_full_fold[fold_i, class_pair_i] = \
                    self.collect_features_for_filter_selection(
                        bin_csp_train_features_full_fold, selected_filters_per_filt)
                
                self.test_feature[fold_i, class_pair_i] = \
                    self.collect_features_for_filter_selection(
                        bin_csp_test_features, selected_filters_per_filt)
                self.test_feature_full_fold[fold_i, class_pair_i] = \
                    self.collect_features_for_filter_selection(
                        bin_csp_test_features_full_fold, selected_filters_per_filt)
                    
                self.selected_filters_per_filterband[fold_i, class_pair_i] = \
                    selected_filters_per_filt

    @staticmethod
    def select_best_filters_best_filterbands(features, max_features,
            forward_steps, backward_steps, stop_when_no_improvement):
        assert max_features is not None, ("For now not dealing with the case "
            "that max features is unlimited")
        assert features[0].data.shape[1] % 2 == 0
        n_filterbands = len(features)
        n_filters_per_fb = features[0].data.shape[1] / 2
        selected_filters_per_band = [0] * n_filterbands
        best_selected_filters_per_filterband = None
        last_best_accuracy = -1
        # Run until no improvement or max features reached
        selection_finished = False
        while (not selection_finished):
            for _ in xrange(forward_steps):
                best_accuracy = -1 # lets try always taking a feature in each iteration
                for filt_i in range(n_filterbands):
                    this_filt_per_fb = deepcopy(selected_filters_per_band)
                    if (this_filt_per_fb[filt_i] == n_filters_per_fb):
                        continue
                    this_filt_per_fb[filt_i] = this_filt_per_fb[filt_i] + 1
                    all_features = FilterbankCSP.collect_features_for_filter_selection(
                        features, this_filt_per_fb)
                    # make 5 times cross validation...
                    test_accuracy = FilterbankCSP.cross_validate_lda(all_features)
                    if (test_accuracy > best_accuracy):
                        best_accuracy = test_accuracy
                        best_selected_filters_per_filterband = this_filt_per_fb
                selected_filters_per_band = best_selected_filters_per_filterband
            for _ in xrange(backward_steps):
                best_accuracy = -1 # lets try always taking a feature in each iteration
                for filt_i in range(n_filterbands):
                    this_filt_per_fb = deepcopy(selected_filters_per_band)
                    if (this_filt_per_fb[filt_i] == 0):
                        continue
                    this_filt_per_fb[filt_i] = this_filt_per_fb[filt_i] - 1
                    all_features = FilterbankCSP.collect_features_for_filter_selection(
                        features, this_filt_per_fb)
                    # make 5 times cross validation...
                    test_accuracy = FilterbankCSP.cross_validate_lda(all_features)
                    if (test_accuracy > best_accuracy):
                        best_accuracy = test_accuracy
                        best_selected_filters_per_filterband = this_filt_per_fb
                selected_filters_per_band = best_selected_filters_per_filterband
        
            selection_finished = 2 * np.sum(selected_filters_per_band) >= max_features 
            if stop_when_no_improvement:
                # there was no improvement if accuracy did not increase...
                selection_finished = (selection_finished or 
                    best_accuracy <= last_best_accuracy)
            last_best_accuracy = best_accuracy
        return selected_filters_per_band

    @staticmethod   
    def collect_features_for_filter_selection(features, filters_for_filterband):
        n_filters_per_fb = features[0].data.shape[1] / 2
        n_filterbands = len(features)
        first_features = deepcopy(features[0])
        first_n_filters = filters_for_filterband[0]
        first_features.data = first_features.data[:, range(first_n_filters) + range(-first_n_filters,0)]
    
        all_features = first_features
        for i in range(1, n_filterbands):
            this_n_filters = min(n_filters_per_fb, filters_for_filterband[i])
            if (this_n_filters > 0):
                next_features = deepcopy(features[i])
                next_features.data = next_features.data[:, range(this_n_filters) + range(-this_n_filters,0)]
                all_features = append_epo(all_features, next_features, classaxis=1)
        return all_features

    @staticmethod
    def cross_validate_lda(features):
        folds = KFold(features.data.shape[0], n_folds=5, shuffle=False)
        test_accuracies = []
        for train_inds, test_inds in folds:
                train_features = features.copy(data=features.data[train_inds], axes=[features.axes[0][train_inds]])
                test_features = features.copy(data=features.data[test_inds], axes=[features.axes[0][test_inds]])
                clf = lda_train_scaled(train_features, shrink=True)
                test_out = lda_apply(test_features, clf)
                second_class_test = test_features.axes[0] == np.max(test_features.axes[0])
                predicted_2nd_class_test = test_out >= 0
                test_accuracy = (sum(second_class_test == predicted_2nd_class_test) / 
                    float(len(predicted_2nd_class_test)))
                test_accuracies.append(test_accuracy)
        return np.mean(test_accuracies)

    def select_features(self):
        n_folds = len(self.train_feature)
        n_pairs = len(self.train_feature[0])
        n_features = self.n_features
        self.selected_features = np.ones((n_folds, n_pairs, n_features), 
            dtype=np.int) * -1
        
        # Determine best features
        for fold_nr in xrange(n_folds):
            for pair_nr in xrange(n_pairs):
                features = self.train_feature[fold_nr][pair_nr]
                this_feature_inds = select_features(features.axes[0], 
                    features.data, n_features=n_features)
                self.selected_features[fold_nr][pair_nr] = this_feature_inds
        assert np.all(self.selected_features >= 0) and np.all(self.selected_features < 
                self.train_feature[0][0].data.shape[1])
        # Only retain selected best features
        for fold_nr in xrange(n_folds):
            for pair_nr in xrange(n_pairs):
                this_feature_inds = self.selected_features[fold_nr][pair_nr]
                for feature_type in ['train_feature', 'train_feature_full_fold', 
                                    'test_feature', 'test_feature_full_fold']:
                    features = self.__dict__[feature_type][fold_nr][pair_nr]
                    features.data = features.data[:, this_feature_inds]
                    
    def train_classifiers(self):
        n_folds = len(self.binary_csp.folds)
        n_class_pairs = len(self.binary_csp.class_pairs)
        self.clf = np.empty((n_folds, n_class_pairs), 
            dtype=object)
        for fold_i in range(n_folds):
            for class_i in range(n_class_pairs):
                train_feature = self.train_feature[fold_i, class_i]
                clf = lda_train_scaled(train_feature, shrink=True)
                self.clf[fold_i, class_i] = clf
                
    def predict_outputs(self):
        n_folds = len(self.binary_csp.folds)
        n_class_pairs = len(self.binary_csp.class_pairs)
        result_shape = (n_folds, n_class_pairs)
        self.train_accuracy = np.empty(result_shape, dtype=float)
        self.test_accuracy = np.empty(result_shape, dtype=float)
        self.train_pred_full_fold = np.empty(result_shape, dtype=object)
        self.test_pred_full_fold = np.empty(result_shape, dtype=object)
        for fold_i in range(n_folds):
            log.info("Fold Nr: {:d}".format(fold_i + 1))
            for class_i, class_pair in enumerate(self.binary_csp.class_pairs):
                clf = self.clf[fold_i, class_i]
                class_pair_plus_one = (np.array(class_pair) + 1).tolist()
                log.info("Class {:d} vs {:d}".format(*class_pair_plus_one))
                train_feature = self.train_feature[fold_i, class_i]
                train_out = lda_apply(train_feature, clf)
                correct_train= train_feature.axes[0] == class_pair[1]
                predicted_train = train_out >= 0
                train_accuracy = (sum(correct_train == predicted_train) / 
                    float(len(predicted_train)))
                self.train_accuracy[fold_i, class_i] = train_accuracy
        
                test_feature = self.test_feature[fold_i, class_i]
                test_out = lda_apply(test_feature, clf)
                correct_test= test_feature.axes[0] == class_pair[1]
                predicted_test = test_out >= 0
                test_accuracy = (sum(correct_test == predicted_test) / 
                    float(len(predicted_test)))
                
                self.test_accuracy[fold_i, class_i] = test_accuracy
                
                train_feature_full_fold = self.train_feature_full_fold[fold_i,\
                     class_i]
                train_out_full_fold = lda_apply(train_feature_full_fold, clf)
                self.train_pred_full_fold[fold_i, class_i] = train_out_full_fold
                test_feature_full_fold = self.test_feature_full_fold[fold_i,\
                     class_i]
                test_out_full_fold = lda_apply(test_feature_full_fold, clf)
                self.test_pred_full_fold[fold_i, class_i] = test_out_full_fold
                
                log.info("Train: {:4.2f}%".format(train_accuracy * 100))
                log.info("Test:  {:4.2f}%".format(test_accuracy * 100))
                
class MultiClassWeightedVoting(object):
    def __init__(self, train_labels, test_labels, train_preds, test_preds,
        class_pairs):
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.train_preds = train_preds
        self.test_preds = test_preds
        self.class_pairs = class_pairs
        
    def run(self):
        n_classes = 4 # for now hardcoded
        n_folds = len(self.train_labels)
        self.train_class_sums = np.empty(n_folds, dtype=object)
        self.test_class_sums = np.empty(n_folds, dtype=object)
        self.train_predicted_labels = np.empty(n_folds, dtype=object)
        self.test_predicted_labels = np.empty(n_folds, dtype=object)
        self.train_accuracy = np.ones(n_folds) * np.nan
        self.test_accuracy = np.ones(n_folds) * np.nan
        for fold_nr in range(n_folds):
            log.info("Fold Nr: {:d}".format(fold_nr + 1))
            train_labels = self.train_labels[fold_nr]
            train_preds = self.train_preds[fold_nr]
            train_class_sums = np.zeros((len(train_labels),n_classes))
            
            test_labels = self.test_labels[fold_nr]
            test_preds = self.test_preds[fold_nr]
            test_class_sums = np.zeros((len(test_labels),n_classes))
            for pair_i, class_pair in enumerate(self.class_pairs):
                this_train_preds = train_preds[pair_i]
                assert len(this_train_preds) == len(train_labels) 
                train_class_sums[:, class_pair[0]] -= this_train_preds
                train_class_sums[:, class_pair[1]] += this_train_preds
                this_test_preds = test_preds[pair_i]
                assert len(this_test_preds) == len(test_labels)
                test_class_sums[:, class_pair[0]] -= this_test_preds
                test_class_sums[:, class_pair[1]] += this_test_preds
                
            self.train_class_sums[fold_nr] = train_class_sums
            self.test_class_sums[fold_nr] = test_class_sums
            train_predicted_labels = np.argmax(train_class_sums, axis=1)
            test_predicted_labels = np.argmax(test_class_sums, axis=1)
            self.train_predicted_labels[fold_nr] = train_predicted_labels
            self.test_predicted_labels[fold_nr] = test_predicted_labels
            train_accuracy = (np.sum(train_predicted_labels == train_labels) / 
                float(len(train_labels)))
            self.train_accuracy[fold_nr] = train_accuracy
            test_accuracy = (np.sum(test_predicted_labels == test_labels) / 
                float(len(test_labels)))
            self.test_accuracy[fold_nr] = test_accuracy 
            log.info("Train: {:4.2f}%".format(train_accuracy * 100))
            log.info("Test:  {:4.2f}%".format(test_accuracy * 100))
            