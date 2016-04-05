from gevent import socket
import signal
import numpy as np
import gevent.server
import gevent.select
import sys
from scipy import interpolate
from braindecode.datasets.loaders import BBCIDataset
from braindecode.experiments.experiment import create_experiment
from braindecode.mywyrm.processing import create_cnt_y_start_end_marker


class RememberPredictionsServer(gevent.server.StreamServer):
    def __init__(self, listener,
            handle=None, backlog=None, spawn='default', **ssl_args):
        super(RememberPredictionsServer, self).__init__(listener, 
            handle=handle, spawn=spawn)
        self.all_preds = []
        self.i_pred_samples = []
        
    def handle(self, socket, address):
        print ("new connection")
        # using a makefile because we want to use readline()
        socket_file = socket.makefile()
        while True:
            i_sample = socket_file.readline()
            preds = socket_file.readline()
            print("Number of predictions", len(self.i_pred_samples) + 1)
            print(i_sample[:-1]) # :-1 => without newline
            print(preds[:-1])
            self.all_preds.append(preds)
            self.i_pred_samples.append(i_sample)
            print("")
    
def start_remember_predictions_server():
    hostname = ''
    server = RememberPredictionsServer((hostname, 30000))
    print("Starting server")
    server.start()
    print("Started server")
    return server

def send_file_data():
    print("Loading Experiment...")
    # Use model to get cnt preprocessors
    base_name = 'data/models/online/cnt/shallow-uneven-trials/9'
    exp = create_experiment(base_name + '.yaml')

    print("Loading File...")
    offline_execution_set = BBCIDataset('data/four-sec-dry-32-sensors/cabin/'
        'MaVo2_sahara32_realMovementS001R02_ds10_1-5.BBCI.mat')
    cnt = offline_execution_set.load()
    print("Running preprocessings...")
    cnt_preprocs = exp.dataset.cnt_preprocessors
    assert cnt_preprocs[-1][0].__name__ == 'exponential_standardize_cnt'
    # Do not do standardizing as it will be done by coordinator
    for preproc, kwargs in cnt_preprocs[:-1]:
        cnt = preproc(cnt, **kwargs)
    cnt_data = cnt.data.astype(np.float32)
    assert not np.any(np.isnan(cnt_data))
    assert not np.any(np.isinf(cnt_data))
    assert not np.any(np.isneginf(cnt_data))
    print("max possible block", np.ceil(len(cnt_data) / 50.0))
    y_labels = create_y_labels(cnt).astype(np.float32)
    assert np.array_equal(np.unique(y_labels), range(5)), ("Should only have "
        "labels 0-4")
    print("Done.")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("127.0.0.1", 1234))
    
    chan_names = ['Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3',
            'AFz', 'AF4', 'AF8', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6',
            'FC1', 'FCz', 'FC2', 'C3', 'C1', 'Cz', 'C2', 'C4', 'CP3', 'CP1',
             'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz', 'marker']
    
    chan_line = " ".join(chan_names) + "\n"
    s.send(chan_line)
    n_chans = 33
    n_samples = 50
    s.send(np.array([n_chans], dtype=np.int32).tobytes())
    s.send(np.array([n_samples], dtype=np.int32).tobytes())
    print("Sending data...")
    i_block = 0 # if setting i_block to sth higher, printed results will incorrect
    max_stop_block = np.ceil(len(cnt_data) / float(n_samples))
    stop_block = 800
    assert stop_block < max_stop_block
    while i_block < stop_block:
        arr = cnt_data[i_block * n_samples:i_block*n_samples + n_samples,:].T
        this_y = y_labels[i_block * n_samples:i_block*n_samples + n_samples]
        # chan x time
        arr = np.concatenate((arr, this_y[np.newaxis, :]), axis=0).astype(np.float32)
        s.send(arr.tobytes(order='F'))
        assert arr.shape == (n_chans, n_samples)
        i_block +=1
        gevent.sleep(0.01)
    print("Done.")
    return cnt

def create_y_labels(cnt):
    classes = np.unique([m[1] for m in cnt.markers])
    if np.array_equal(range(1,5), classes):
        return create_y_labels_fixed_trial_len(cnt, trial_len=int(cnt.fs*4))
    elif np.array_equal(range(1,9), classes):
        y_signal = create_cnt_y_start_end_marker(cnt,
                start_marker_def=dict((('1',[1]), ('2', [2]), ('3',[3]), ('4', [4]))), 
                end_marker_def=dict((('1',[5]), ('2', [6]), ('3',[7]), ('4', [8]))), 
                segment_ival=(0,0), timeaxis=-2)
        y_labels = np.zeros((cnt.data.shape[0]), dtype=np.int32)
        y_labels[y_signal[:,0] == 1] = 1
        y_labels[y_signal[:,1] == 1] = 2
        y_labels[y_signal[:,2] == 1] = 3
        y_labels[y_signal[:,3] == 1] = 4
        return y_labels
    else:
        raise ValueError("Expect classes 1,2,3,4, possibly with end markers "
            "5,6,7,8, instead got {:s}".format(str(classes)))
    

def has_fixed_trial_len(cnt):
    classes = np.unique([m[1] for m in cnt.markers])
    if np.array_equal(range(1,5), classes):
        return True
    elif np.array_equal(range(1,9), classes):
        return False
    else:
        raise ValueError("Expect classes 1,2,3,4, possibly with end markers "
            "5,6,7,8, instead got {:s}".format(str(classes)))

def create_y_labels_fixed_trial_len(cnt, trial_len):
    fs = cnt.fs
    event_samples_and_classes = [(int(np.round(m[0] * fs/1000.0)), m[1]) 
        for m in cnt.markers]
    y = np.zeros((cnt.data.shape[0]), dtype=np.int32)
    for i_sample, marker in event_samples_and_classes:
        assert marker in [1,2,3,4], "Assuming 4 classes for now..."
        y[i_sample:i_sample+trial_len] = marker
    return y


def create_y_signal(cnt):
    if has_fixed_trial_len(cnt):
        return create_y_signal_fixed_trial_len(cnt, trial_len=int(cnt.fs*4))
    else:
        return create_cnt_y_start_end_marker(cnt,
            start_marker_def=dict((('1',[1]), ('2', [2]), ('3',[3]), ('4', [4]))), 
            end_marker_def=dict((('1',[5]), ('2', [6]), ('3',[7]), ('4', [8]))), 
            segment_ival=(0,0), timeaxis=-2)

def create_y_signal_fixed_trial_len(cnt, trial_len):
    fs = cnt.fs
    event_samples_and_classes = [(int(np.round(m[0] * fs/1000.0)), m[1]) for m in cnt.markers]
    return get_y_signal(cnt.data, event_samples_and_classes, trial_len)

def get_y_signal(cnt_data, event_samples_and_classes, trial_len):
        # Generate class "signals", rest always inbetween trials
    y = np.zeros((cnt_data.shape[0], 4), dtype=np.int32)
    y[:,2] = 1 # put rest class everywhere
    for i_sample, marker in event_samples_and_classes:
        i_class = marker - 1
        # set rest to zero, correct class to 1
        y[i_sample:i_sample+trial_len,2] = 0
        y[i_sample:i_sample+trial_len,i_class] = 1
    return y

if __name__ == "__main__":
    # load file as cnt 
    # send sensor NAMES
    # + MARKER as final name
    
    # number of rows + number of columns
    # send all as fast as can be 
    # also start server, should also expect a quit signal....
    # (for now just stop reading after 10 preds)
    # wait for enter press to continue
    gevent.signal(signal.SIGQUIT, gevent.kill)
    server = start_remember_predictions_server()
    cnt = send_file_data()
    
    print("Finished sending data, press enter to continue")
    enter_pressed = False
    while not enter_pressed:
        i,o,e = gevent.select.select([sys.stdin],[],[],0.1)
        for s in i:
            if s == sys.stdin:
                _ = sys.stdin.readline()
                enter_pressed = True
    
    y_signal = create_y_signal(cnt)
    i_pred_samples = [int(line[:-1]) for line in server.i_pred_samples]
    # -1 to convert from 1 to 0-based indexing
    i_pred_samples_arr = np.array(i_pred_samples) - 1
    preds = [[float(num_str) for num_str in line_str[:-1].split(' ')] 
        for line_str in server.all_preds]
    preds_arr = np.array(preds)

    input_start = 499
    input_end = i_pred_samples_arr[-1] + 1
    interpolated_classes = y_signal[input_start:input_end].T
    interpolate_fn = interpolate.interp1d(i_pred_samples_arr, preds_arr.T,
                                         bounds_error=False, fill_value=0)
    interpolated_preds = interpolate_fn(range(input_start,input_end))
    corrcoeffs = np.corrcoef(interpolated_preds, interpolated_classes)[:4,4:]
    print corrcoeffs
