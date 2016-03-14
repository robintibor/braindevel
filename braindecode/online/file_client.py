from gevent import socket
import signal
import numpy as np
import gevent.server
import gevent.select
import sys
from scipy import interpolate
from braindecode.datasets.loaders import BBCIDataset
from braindecode.experiments.experiment import create_experiment


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
            print "Number of predictions", len(self.i_pred_samples) + 1
            print i_sample[:-1]
            print preds[:-1]
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
    base_name = 'data/models/online/cnt/shallow-combined/12'
    exp = create_experiment(base_name + '.yaml')

    print("Loading File...")
    offline_execution_set = BBCIDataset('data/four-sec-dry-32-sensors/cabin/'
        'Martin_trainingS001R01_1-4.BBCI.mat')

    cnt = offline_execution_set.load()
    print("Running preprocessings...")
    cnt_preprocs = exp.dataset.cnt_preprocessors
    assert cnt_preprocs[-1][0].__name__ == 'exponential_standardize_cnt'
    # Do not do standardizing as it will be done by coordinator
    for preproc, kwargs in cnt_preprocs[:-1]:
        cnt = preproc(cnt, **kwargs)
    cnt_data = cnt.data.astype(np.float32)
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
    
    i_block = 0
    y_labels = create_y_labels(cnt, trial_len=int(cnt.fs*4)).astype(np.float32)
    
    while i_block < 150:
        arr = cnt_data[i_block * n_samples:i_block*n_samples + n_samples,:].T
        this_y = y_labels[i_block * n_samples:i_block*n_samples + n_samples]
        # chan x time
        # add fake marker
        #arr = np.concatenate((arr, np.zeros((1,arr.shape[1]))), axis=0).astype(
        #    np.float32)
        arr = np.concatenate((arr, this_y[np.newaxis, :]), axis=0)
        s.send(arr.tobytes(order='F'))
        i_block +=1
        gevent.sleep(0)
    return cnt


def create_y_labels(cnt, trial_len):
    fs = cnt.fs
    event_samples_and_classes = [(int(np.round(m[0] * fs/1000.0)), m[1]) 
        for m in cnt.markers]
    y = np.zeros((cnt.data.shape[0]), dtype=np.int32)
    for i_sample, marker in event_samples_and_classes:
        assert marker in [1,2,3,4], "Assuming 4 classes for now..."
        y[i_sample:i_sample+trial_len] = marker
    return y

def create_y_signal(cnt, trial_len):
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
    
    y_signal = create_y_signal(cnt, trial_len=int(cnt.fs*4))
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
