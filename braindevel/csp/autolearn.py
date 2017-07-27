from pylearn2.utils import serial
import numpy as np
import re
from sklearn.linear_model import LogisticRegression

def load_features_labels(features, labels):
    """ Given full fold features labels are still separated by filterband, classifier pair etc."""
    assert np.array_equal([35,1,6], features.shape), "Should have 35 filtbands, single fold, 6 classpairs."
    assert np.array_equal([1], labels.shape), "Should have just a single fold."
    y = labels[0]
    
    # Check all have features same shape
    for i_filt_band in range(35):
        for i_class_pair in range(6):
            assert np.array_equal(features[0,0,0].data.shape,
                features[i_filt_band,0,i_class_pair].data.shape)

    # initialize with nans
    X = np.nan * np.ones((features[0,0,0].data.shape[0],
                 features[0,0,0].data.shape[1] * np.prod(features.shape)),
                     dtype=np.float32)
    for i_filt_band in range(35):
        for i_class_pair in range(6):
            start_ind = i_filt_band * 6 * 10 + i_class_pair * 10
            X[:, start_ind:start_ind+10] = features[i_filt_band,0,i_class_pair].data
    # check for no nans
    assert not (np.any(np.isnan(X)))
    return X,y

def shorten_dataset_name(dataset_name):
    dataset_name = re.sub(r"(./)?data/[^/]*/", '', str(dataset_name))
    dataset_name = re.sub(r"MoSc[0-9]*S[0-9]*R[0-9]*_ds10_", '',
        dataset_name)
    dataset_name = re.sub("BBCI.mat", '', dataset_name)

    return dataset_name


if __name__ == '__main__':
    clf = LogisticRegression()
    all_old_accs = []
    all_new_accs = []
    for i_file in xrange(91,109):
        print ("Loading...")
        filename = '/home/schirrmr/motor-imagery/data/models/final-eval/csp-standardized/' + str(i_file) + '.pkl'
        csp_trainer = serial.load(filename)
        print "Training {:20s}".format(shorten_dataset_name(csp_trainer.filename))
        X_train, y_train = load_features_labels(csp_trainer.binary_csp.train_feature_full_fold,
                                           csp_trainer.binary_csp.train_labels_full_fold)
        X_test, y_test = load_features_labels(csp_trainer.binary_csp.test_feature_full_fold,
                                           csp_trainer.binary_csp.test_labels_full_fold)
        clf.fit(X_train, y_train)
        old_acc = csp_trainer.multi_class.test_accuracy[0]
        new_acc = clf.score(X_test, y_test)
        print ("Master Thesis Accuracy: {:5.2f}%".format(old_acc * 100))
        print ("New Accuracy:           {:5.2f}%".format(new_acc * 100))
        all_old_accs.append(old_acc)
        all_new_accs.append(new_acc)
        print("")
        
        
    print("Master Thesis average(std): {:5.2f}% ({:5.2f}%)".format(
            np.mean(all_old_accs) * 100, np.std(all_old_accs) * 100))
    print("New           average(std): {:5.2f}% ({:5.2f}%)".format(
            np.mean(all_new_accs) * 100, np.std(all_new_accs) * 100))