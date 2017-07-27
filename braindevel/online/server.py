#!/usr/bin/env python
import matplotlib
import logging
from braindevel.experiments.load import set_param_values_backwards_compatible

# Have to do this here in for choosing correct matplotlib backend before
# it is imported anywhere
def parse_command_line_arguments():
    import argparse
    parser = argparse.ArgumentParser(
        description="""Launch server for online decoding.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # see http://stackoverflow.com/a/24181138/1469195
    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument('--fs', action='store', type=int,
        help="Sampling rate of EEG signal (in Hz). Only used to convert "
        "other arguments from milliseconds to number of samples", required=True)
    required_named.add_argument('--modelfile', action='store',
        help='Basename of the modelfile')
    parser.add_argument('--inport', action='store', type=int,
        default=7987, help='Port from which to accept incoming sensor data.')
    parser.add_argument('--uihost', action='store',
        default='172.30.0.117', help='Hostname/IP of the UI server (= the '
        'server that the predictions are being sent to).')
    parser.add_argument('--uiport', action='store',
        default=30000, help='Port of the UI server')
    parser.add_argument('--paramsfile', action='store', 
        help='Use these (possibly adapted) parameters for the model. '
        'Filename should end with model_params.npy. Can also use "newest"'
        'to load the newest available  parameter file. '
        'None means to not load any new parameters, instead use '
        'originally (offline)-trained parameters.')
    parser.add_argument('--plot', action='store_true',
        help="Show plots of the sensors first.")
    parser.add_argument('--noui', action='store_true',
        help="Don't wait for UI server.")
    parser.add_argument('--noadapt', action='store_true',
        help="Don't adapt model while running online.")
    parser.add_argument('--updatesperbreak', action='store', default=5,
        type=int, help="How many updates to adapt the model during trial break.")
    parser.add_argument('--batchsize', action='store', default=45, type=int,
        help="Batch size for adaptation updates.")
    parser.add_argument('--learningrate', action='store', default=1e-4, 
        type=float, help="Learning rate for adaptation updates.")
    parser.add_argument('--mintrials', action='store', default=10, type=int,
        help="Number of trials before starting adaptation updates.")
    parser.add_argument('--trialstartoffset', action='store', default=500, type=int,
        help="Time offset for the first sample to use (within a trial, in ms) "
        "for adaptation updates.")
    parser.add_argument('--breakstartoffset', action='store', default=1000, type=int,
        help="Time offset for the first sample to use (within a break(!), in ms) "
        "for adaptation updates.")
    parser.add_argument('--breakstopoffset', action='store', default=-1000, type=int,
        help="Sample offset for the last sample to use (within a break(!), in ms) "
        "for adaptation updates.")
    parser.add_argument('--predgap', action='store', default=200, type=int,
        help="Amount of milliseconds between predictions.")
    parser.add_argument('--minbreakms', action='store', default=2000, type=int,
        help="Minimum length of a break to be used for training (in ms).")
    parser.add_argument('--mintrialms', action='store', default=0, type=int,
        help="Minimum length of a trial to be used for training (in ms).")
    parser.add_argument('--noprint', action='store_true',
        help="Don't print on terminal.")
    parser.add_argument('--nosave', action='store_true',
        help="Don't save streamed data (including markers).")
    parser.add_argument('--noolddata', action='store_true',
        help="Dont load and use old data for adaptation")
    parser.add_argument('--plotbackend', action='store',
        default='agg', help='Matplotlib backend to use for plotting.')
    parser.add_argument('--nooldadamparams', action='store_true',
        help='Do not load old adam params.')
    parser.add_argument('--inputsamples', action='store', default=None,
        type=int,
        help='Input samples (!) for the ConvNet. None means same as when trained in original experiment.')
    parser.add_argument('--nobreaktraining',action='store_true',
        help='Do not use the breaks as training examples for the rest class.')
    args = parser.parse_args()
    assert args.breakstopoffset <= 0, ("Please supply a nonpositive break stop "
        "offset, you supplied {:d}".format(args.breakstopoffset))
    return args

log = logging.getLogger(__name__)
matplotlib_backend = parse_command_line_arguments().plotbackend
try:
    matplotlib.use(matplotlib_backend)
except:
    print("Could not use {:s} backend for matplotlib".format(
        matplotlib_backend))
    
import gevent.server
import signal
import numpy as np
import lasagne
from pylearn2.config import yaml_parse
from braindevel.online.coordinator import OnlineCoordinator
from gevent import socket
from braindevel.online.model import OnlineModel
from braindevel.online.data_processor import StandardizeProcessor
import argparse
import h5py
import datetime
import os.path
import sys
import gevent.select
from scipy import interpolate
from braindevel.experiments.experiment import create_experiment
from braindevel.veganlasagne.layers import transform_to_normal_net,\
    set_input_window_length
from braindevel.online.trainer import BatchWiseCntTrainer, NoTrainer
from pylearn2.utils.logger import (CustomStreamHandler, CustomFormatter)
from glob import glob

class PredictionServer(gevent.server.StreamServer):
    def __init__(self, listener, coordinator, ui_hostname, ui_port, 
        plot_sensors, use_ui_server, save_data, model_base_name, adapt_model,
            handle=None, backlog=None, spawn='default', **ssl_args):
        """
        adapt_model only needed to know for saving
        """
        self.coordinator = coordinator
        self.ui_hostname = ui_hostname
        self.ui_port = ui_port
        self.plot_sensors = plot_sensors
        self.use_ui_server = use_ui_server
        self.save_data = save_data
        self.model_base_name = model_base_name
        self.adapt_model = adapt_model
        super(PredictionServer, self).__init__(listener, handle=handle, spawn=spawn)


    def handle(self, in_socket, address):
        log.info('New connection from {:s}!'.format(str(address)))
      
        # Connect to UI Server
        if self.use_ui_server:
            gevent.sleep(1) # hack to wait until ui server open
            ui_socket = self.connect_to_ui_server()
            log.info("Connected to UI Server")
        else:
            ui_socket=None

        # Receive Header
        chan_names, n_rows, n_cols = self.receive_header(in_socket)
        n_numbers = n_rows * n_cols
        n_bytes = n_numbers * 4 # float32
        log.info("Numbers in total:  {:d}".format(n_numbers))
        
        log.info("Before checking plot")
        # Possibly plot
        if self.plot_sensors:
            self.plot_sensors_until_enter_press(chan_names, in_socket, n_bytes,
            n_rows, n_cols)
        log.info("After checking plot")
        self.make_predictions_and_save_params(chan_names, n_rows, n_cols,
            n_bytes, in_socket, ui_socket)
        self.stop()

    def connect_to_ui_server(self):
        ui_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("UI server connected at:", self.ui_hostname, self.ui_port)
        ui_socket.connect((self.ui_hostname, self.ui_port))
        return ui_socket
        
    
    def read_until_bytes_received(self, socket, n_bytes):
        array = ''
        while len(array) < n_bytes:
            array += socket.recv(n_bytes - len(array))
        return array

    def read_until_bytes_received_or_enter_pressed(self, socket, n_bytes):
        '''
        Read bytes from socket until reaching given number of bytes, cancel
        if enter was pressed.
        
        Parameters
        ----------
        socket:
            Socket to read from.
        n_bytes: int
            Number of bytes to read.
        '''
        enter_pressed = False
        array = ''
        while len(array) < n_bytes and not enter_pressed:
            array += socket.recv(n_bytes - len(array))
            # check if enter is pressed
            i,o,e = gevent.select.select([sys.stdin],[],[],0.0001)
            for s in i:
                if s == sys.stdin:
                    _ = sys.stdin.readline()
                    enter_pressed = True
        if enter_pressed:
            return None
        else:
            assert len(array) == n_bytes
            return array
    
    def receive_header(self, in_socket):
        chan_names_line = '' + in_socket.recv(1)
        while chan_names_line[-1] != '\n':
            chan_names_line += in_socket.recv(1)
        log.info("Chan names:\n{:s}".format(chan_names_line))
        chan_names = chan_names_line.replace('\n','').split(" ")
        
        #
        assert np.array_equal(chan_names, ['Fp1', 'Fpz', 'Fp2', 'AF7',
         'AF3', 'AF4', 'AF8', 'F7',
         'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3',
         'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'M1', 'T7', 'C5', 'C3',
         'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'M2', 'TP7', 'CP5', 'CP3',
         'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1',
         'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz', 'PO4',
         'PO6', 'PO8', 'O1', 'Oz', 'O2', 'marker']
            ) or np.array_equal(chan_names,
         ['Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
         'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'FC1', 'FCz',
         'FC2', 'C3', 'C1', 'Cz', 'C2', 'C4', 'CP3', 'CP1', 'CPz',
         'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz', 'marker'])
        n_rows = self.read_until_bytes_received(in_socket, 4)
        n_rows = np.fromstring(n_rows, dtype=np.int32)[0]
        log.info("Number of rows:    {:d}".format(n_rows))
        n_cols = self.read_until_bytes_received(in_socket, 4)
        n_cols = np.fromstring(n_cols, dtype=np.int32)[0]
        log.info("Number of columns: {:d}".format(n_cols))
        assert n_rows == len(chan_names)
        return chan_names, n_rows, n_cols

    def plot_sensors_until_enter_press(self, chan_names, in_socket, n_bytes,
            n_rows, n_cols):
        log.info("Starting Plot for plot")
        from  braindevel.online.live_plot import LivePlot
        log.info("Import")
        live_plot = LivePlot(plot_freq=150)
        log.info("Liveplot created")
        live_plot.initPlots(chan_names)
        log.info("Initialized")
        enter_pressed = False
        while not enter_pressed:
            array = ''
            while len(array) < n_bytes:
                array += in_socket.recv(n_bytes - len(array))
            array = np.fromstring(array, dtype=np.float32)
            array = array.reshape(n_rows, n_cols, order='F')
            live_plot.accept_samples(array.T)
            # check if enter is pressed
            i,o,e = gevent.select.select([sys.stdin],[],[],0.001)
            for s in i:
                if s == sys.stdin:
                    _ = sys.stdin.readline()
                    enter_pressed = True

        live_plot.close()
        log.info("Plot finished")

    def make_predictions_and_save_params(self, chan_names, n_rows, n_cols, n_bytes,
        in_socket, ui_socket):
        
        self.coordinator.initialize(n_chans=n_rows - 1) # one is a marker chan(!)
        
        # this is to be able to show scores later
        all_preds =  []
        all_pred_samples = []
        all_sample_blocks = []
        while True:
            array = self.read_until_bytes_received_or_enter_pressed(in_socket,
                n_bytes)
            if array is None:
                # enter was pressed! quit! :)
                break;
            
            array = np.fromstring(array, dtype=np.float32)
            array = array.reshape(n_rows, n_cols, order='F')
            all_sample_blocks.append(array.T)
            # here now also supply y to data processor...
            self.coordinator.receive_samples(array.T)

            if self.coordinator.has_new_prediction():
                #    input_start-window_len+1+i_sample], 
                #    [np.int32(target)])
                pred, i_sample = self.coordinator.pop_last_prediction_and_sample_ind()
                log.info("Prediction for sample {:d}:\n{:s}".format(
                    i_sample, pred))
                if self.use_ui_server:
                    # +1 to convert 0-based to 1-based indexing
                    ui_socket.sendall("{:d}\n".format(i_sample + 1))
                    n_preds = len(pred[0])
                    # format all preds as floats with spaces inbetween
                    format_str = " ".join(["{:f}"] * n_preds) + "\n"
                    pred_str = format_str.format(*pred[0])
                    ui_socket.sendall(pred_str)
                all_preds.append(pred)
                all_pred_samples.append(i_sample)
        
        all_samples = np.concatenate(all_sample_blocks).astype(np.float32)
        all_preds = np.array(all_preds).squeeze()
        all_pred_samples = np.array(all_pred_samples)
        log.setLevel("INFO") # show what you are saving again even if printing
        # disabled before...
        now = datetime.datetime.now()
        now_timestring = now.strftime('%Y-%m-%d_%H-%M-%S')
        if self.adapt_model:
            # Save parameters for model
            all_layers = lasagne.layers.get_all_layers(self.coordinator.model.model)
            filename = "{:s}.{:s}.model_params.npy".format(self.model_base_name,
                now_timestring)
            log.info("Saving model params to {:s}...".format(filename))
            np.save(filename, lasagne.layers.get_all_param_values(all_layers))
            # save parameters for trainer
            # this is an ordered dict!! so should be fine to save in order
            train_params = self.coordinator.trainer.train_params
            train_param_values = [p.get_value() for p in train_params]
            filename = "{:s}.{:s}.trainer_params.npy".format(
                self.model_base_name, now_timestring)
            log.info("Saving train params to {:s}...".format(filename))
            np.save(filename, train_param_values)
        if self.save_data:
            day_string = now.strftime('%Y-%m-%d')
            data_folder = 'data/online/{:s}'.format(day_string)
            if not os.path.exists(data_folder):
                os.makedirs(data_folder)
            data_filename = os.path.join(data_folder, "{:s}.npy".format(
                now_timestring))
            log.info("Saving data to {:s}".format(data_filename))
            np.save(data_filename, all_samples)
        self.print_results(all_samples, all_preds, all_pred_samples)
            
            

    def print_results(self, all_samples, all_preds, all_pred_samples):
        # y labels i from 0 to n_classes (inclusive!), 0 representing
        # non-trial => no known marker state -> set to rest class now
        n_classes = 5
        y_labels = all_samples[:,-1]
        y_signal = np.ones((len(y_labels), n_classes)) * np.nan
        y_signal[:,0] = y_labels == 1
        y_signal[:,1] = y_labels == 2
        y_signal[:,2] = y_labels == 3
        y_signal[:,3] = y_labels == 4
        y_signal[:,4] = np.logical_or(y_labels == 0, y_labels==n_classes)
        
        assert not np.any(np.isnan(y_signal))
        
        interpolate_fn = interpolate.interp1d(all_pred_samples, all_preds.T,
                                             bounds_error=False, fill_value=0)
        interpolated_preds = interpolate_fn(range(0,len(y_labels)))
        # interpolated_preds are classes x samples (!!)
        corrcoeffs = np.corrcoef(interpolated_preds, y_signal.T)[:n_classes,
            n_classes:]

        print("Corrcoeffs")
        print corrcoeffs
        print("mean across diagonal")
        print np.mean(np.diag(corrcoeffs))
        interpolated_pred_labels = np.argmax(interpolated_preds, axis=0)
        
        # inside trials
        corrcoeffs = np.corrcoef(interpolated_preds[:,y_labels!=0], 
                                 y_signal[y_labels!=0].T)[:n_classes,n_classes:]
        print("Corrcoeffs inside trial")
        print corrcoeffs
        print("mean across diagonal inside trial")
        print np.mean(np.diag(corrcoeffs))
        
        # -1 since we have 0 as "break" "non-trial" marker
        label_pred_equal = interpolated_pred_labels == y_labels - 1
        label_pred_trial_equal = label_pred_equal[y_labels!=0]
        print("Sample accuracy inside trials")
        print np.mean(label_pred_trial_equal)
        y_label_with_breaks = np.copy(y_labels)
        # set break to rest label
        y_label_with_breaks[y_label_with_breaks == 0] = n_classes
        # from 1-based to 0-based
        y_label_with_breaks -= 1
        label_pred_equal = interpolated_pred_labels == y_label_with_breaks
        print("Sample accuracy total")  
        print np.mean(label_pred_equal)
        
        # also compute trial preds
        # compute boundarides so that boundaries give
        # indices of starts of new trials/new breaks
        trial_labels = []
        trial_pred_labels = []
        boundaries = np.flatnonzero(np.diff(y_labels) != 0) + 1
        last_bound = 0
        for i_bound in boundaries:
            # i bounds are first sample of new trial
            this_labels = y_label_with_breaks[last_bound:i_bound]
            assert len(np.unique(this_labels) == 1), (
                "Expect only one label, got {:s}".format(str(
                    np.unique(this_labels))))
            trial_labels.append(this_labels[0])
            this_preds = interpolated_preds[:,last_bound:i_bound]
            pred_label = np.argmax(np.mean(this_preds, axis=1))
            trial_pred_labels.append(pred_label)
            last_bound = i_bound
        trial_labels = np.array(trial_labels)
        trial_pred_labels = np.array(trial_pred_labels)
        print("Trialwise accuracy (mean prediction) of {:d} trials (including breaks, without offset)".format(
            len(trial_labels)))
        print(np.mean(trial_labels == trial_pred_labels))

def get_now_timestring():
    now = datetime.datetime.now()
    time_string = now.strftime('%Y-%m-%d_%H-%M-%S')
    return time_string      

def setup_logging():
    """ Set up a root logger so that other modules can use logging
    Adapted from scripts/train.py from pylearn"""

    root_logger = logging.getLogger()
    prefix = '%(asctime)s '
    formatter = CustomFormatter(prefix=prefix)
    handler = CustomStreamHandler(formatter=formatter)
    root_logger.handlers  = []
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.DEBUG)

def main(ui_hostname, ui_port, base_name, params_filename, plot_sensors,
        use_ui_server, adapt_model, save_data, n_updates_per_break, batch_size,
        learning_rate, n_min_trials, trial_start_offset, break_start_offset,
        break_stop_offset,
        pred_gap,
        incoming_port,load_old_data,use_new_adam_params,
        input_time_length,
        train_on_breaks,
        min_break_samples,
        min_trial_samples):
    setup_logging()
    assert np.little_endian, "Should be in little endian"
    train_params = None # for trainer, e.g. adam params
    if params_filename is not None:
        if params_filename == 'newest':
            # sort will already sort temporally with our time string format
            all_params_files = sorted(glob(base_name + ".*.model_params.npy"))
            assert len(all_params_files) > 0, ("Expect atleast one params file "
                "if 'newest' given as argument")
            params_filename = all_params_files[-1]
        log.info("Loading model params from {:s}".format(params_filename))
        params = np.load(params_filename)
        train_params_filename = params_filename.replace('model_params.npy',
            'trainer_params.npy')
        if os.path.isfile(train_params_filename):
            if use_new_adam_params:
                log.info("Loading trainer params from {:s}".format(train_params_filename))
                train_params = np.load(train_params_filename)
        else:
            log.warn("No train/adam params found, starting optimization params "
                "from scratch (model params will be loaded anyways).")
    else:
        params = np.load(base_name + '.npy')
    exp = create_experiment(base_name + '.yaml')
    
    # Possibly change input time length, for exmaple
    # if input time length very long during training and should be
    # shorter for online
    if input_time_length is not None:
        log.info("Change input time length to {:d}".format(input_time_length))
        set_input_window_length(exp.final_layer, input_time_length)
        # probably unnecessary, just for safety
        exp.iterator.input_time_length = input_time_length
    # Have to set for both exp final layer and actually used model
    # as exp final layer might be used for adaptation
    # maybe check this all for correctness?
    cnt_model = exp.final_layer
    set_param_values_backwards_compatible(cnt_model, params)
    prediction_model = transform_to_normal_net(cnt_model)
    set_param_values_backwards_compatible(prediction_model, params)
    
    data_processor = StandardizeProcessor(factor_new=1e-3)
    online_model = OnlineModel(prediction_model)
    if adapt_model:
        online_trainer = BatchWiseCntTrainer(exp, n_updates_per_break, 
            batch_size, learning_rate, n_min_trials, trial_start_offset,
            break_start_offset=break_start_offset,
            break_stop_offset=break_stop_offset,
            train_param_values=train_params,
            add_breaks=train_on_breaks,
            min_break_samples=min_break_samples,
            min_trial_samples=min_trial_samples)
    else:
        log.info("Not adapting model...")
        online_trainer = NoTrainer()
    coordinator = OnlineCoordinator(data_processor, online_model, online_trainer,
        pred_gap=pred_gap)
    hostname = ''
    server = PredictionServer((hostname, incoming_port), coordinator=coordinator,
        ui_hostname=ui_hostname, ui_port=ui_port, plot_sensors=plot_sensors,
        use_ui_server=use_ui_server, save_data=save_data,
        model_base_name=base_name, adapt_model=adapt_model)
    # Compilation takes some time so initialize trainer already
    # before waiting in connection in server
    online_trainer.initialize()
    if adapt_model and load_old_data:
        online_trainer.add_data_from_today(data_processor)
    log.info("Starting server on port {:d}".format(incoming_port))
    server.start()
    log.info("Started server")
    server.serve_forever()

if __name__ == '__main__':
    gevent.signal(signal.SIGQUIT, gevent.kill)
    args = parse_command_line_arguments()
    if args.noprint:
        log.setLevel("WARN")
    # factor for converting to samples
    ms_to_samples = args.fs / 1000.0
    # convert all millisecond arguments to number of samples
    main(ui_hostname=args.uihost,
        ui_port=args.uiport, 
        base_name=args.modelfile,
        params_filename=args.paramsfile,
        plot_sensors=args.plot,
        save_data=not args.nosave,
        use_ui_server=not args.noui,
        adapt_model=not args.noadapt,
        n_updates_per_break=args.updatesperbreak,
        batch_size=args.batchsize,
        learning_rate=args.learningrate,
        n_min_trials=args.mintrials, 
        trial_start_offset=int(args.trialstartoffset * ms_to_samples), 
        break_start_offset=int(args.breakstartoffset * ms_to_samples),
        break_stop_offset=int(args.breakstopoffset * ms_to_samples),
        pred_gap=int(args.predgap * ms_to_samples),
        incoming_port=args.inport,
        load_old_data=not args.noolddata,
        use_new_adam_params=not args.nooldadamparams,
        input_time_length=args.inputsamples,
        train_on_breaks=(not args.nobreaktraining),
        min_break_samples=int(args.minbreakms * ms_to_samples),
        min_trial_samples=int(args.mintrialms * ms_to_samples),
        )
    
