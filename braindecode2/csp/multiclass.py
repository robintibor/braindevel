import logging
import itertools
import numpy as np
log = logging.getLogger(__name__)


class MultiClassWeightedVoting(object):
    def __init__(self, train_labels, test_labels, train_preds, test_preds,
                 class_pairs):
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.train_preds = train_preds
        self.test_preds = test_preds
        self.class_pairs = class_pairs

    def run(self):
        # determine number of classes by number of unique classes
        # appearing in class pairs
        n_classes = len(np.unique(list(itertools.chain(*self.class_pairs))))
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
            train_class_sums = np.zeros((len(train_labels), n_classes))

            test_labels = self.test_labels[fold_nr]
            test_preds = self.test_preds[fold_nr]
            test_class_sums = np.zeros((len(test_labels), n_classes))
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