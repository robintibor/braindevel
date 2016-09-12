#!/usr/bin/env python
import matplotlib
import logging

def parse_command_line_arguments():
    import argparse
    parser = argparse.ArgumentParser(
        description="""Launch server for online decoding.
        Example: online/server.py --host 172.30.2.129 --port 30000"""
    )
    parser.add_argument('--inport', action='store', type=int,
        default=7987, help='Port from which to accept incoming sensor data.')
    parser.add_argument('--uihost', action='store',
        default='172.30.0.117', help='Hostname/IP of the UI server')
    parser.add_argument('--uiport', action='store',
        default=30000, help='Port of the UI server')
    parser.add_argument('--modelfile', action='store',
        default='data/models/raw-net-512/3', 
        help='Basename of the modelfile')
    parser.add_argument('--paramsfile', action='store', 
        help='Use these (possibly adapted) parameters for the model')
    parser.add_argument('--noplot', action='store_true',
        help="Don't show plots of the sensors first.")
    parser.add_argument('--nosave', action='store_true',
        help="Don't save data.")
    parser.add_argument('--noui', action='store_true',
        help="Don't wait for UI server.")
    parser.add_argument('--noadapt', action='store_true',
        help="Don't adapt model while running online.")
    parser.add_argument('--updatesperbreak', action='store', default=5,
        type=int, help="How many updates to adapt the model during trial break.")
    parser.add_argument('--batchsize', action='store', default=45, type=int,
        help="Batch size for adaptation updates.")
    parser.add_argument('--learningrate', action='store', default=1e-3, 
        type=float, help="Learning rate for adaptation updates.")
    parser.add_argument('--mintrials', action='store', default=8, type=int,
        help="Number of trials before starting adaptation updates.")
    parser.add_argument('--adaptoffset', action='store', default=500, type=int,
        help="Sample offset for the first sample to use (within a trial) "
        "for adaptation updates.")
    parser.add_argument('--predfreq', action='store', default=125, type=int,
        help="Amount of samples between predictions.")
    parser.add_argument('--noprint', action='store_true',
        help="Don't print on terminal.")
    parser.add_argument('--plotbackend', action='store',
        default='agg', help='Matplotlib backend to use for plotting.')
    args = parser.parse_args()
    return args

log = logging.getLogger(__name__)
matplotlib_backend = parse_command_line_arguments().plotbackend
try:
    matplotlib.use(matplotlib_backend)
except:
    log.warn("Could not use {:s} backend for matplotlib".format(
        matplotlib_backend))
    
import gevent.server
import signal
import numpy as np
import lasagne
from pylearn2.config import yaml_parse
from braindecode.online.coordinator import OnlineCoordinator
from gevent import socket
from braindecode.online.model import OnlineModel
from braindecode.online.data_processor import StandardizeProcessor
import argparse
import h5py
import datetime
import os.path
import sys
import gevent.select
from scipy import interpolate
from braindecode.experiments.experiment import create_experiment
from braindecode.veganlasagne.layers import transform_to_normal_net
from braindecode.online.trainer import BatchWiseCntTrainer, NoTrainer
from pylearn2.utils.logger import (CustomStreamHandler, CustomFormatter)

class PredictionServer(gevent.server.StreamServer):
    def __init__(self, listener, coordinator, ui_hostname, ui_port, 
        plot_sensors, save_data, use_ui_server, model_base_name,
            handle=None, backlog=None, spawn='default', **ssl_args):
        self.coordinator = coordinator
        self.ui_hostname = ui_hostname
        self.ui_port = ui_port
        self.plot_sensors = plot_sensors
        self.save_data = save_data
        self.use_ui_server = use_ui_server
        self.model_base_name = model_base_name
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
        

        self.make_predictions_and_save_data(chan_names, n_rows, n_cols, n_bytes,
        in_socket, ui_socket)
        self.stop()

    def connect_to_ui_server(self):
        ui_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print (self.ui_hostname, self.ui_port)
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
            
        assert np.array_equal(chan_names, ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3',
		 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'M1', 'T7', 
		 'C3', 'Cz', 'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6',
		 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'Oz', 'O2', 'AF7',
		 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FC3', 'FCz',
		 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P5',
		 'P1', 'P2', 'P6', 'PO5', 'PO3', 'PO4', 'PO6', 'FT7', 'FT8',
		 'TP7', 'TP8', 'PO7', 'PO8', 'marker']
            )
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
        from  braindecode.online.live_plot import LivePlot
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

    def make_predictions_and_save_data(self, chan_names, n_rows, n_cols, n_bytes,
        in_socket, ui_socket):
        data_saver = DataSaver(chan_names)
        self.coordinator.initialize(n_chans=n_rows - 1) # one is a marker chan(!)
        
        all_preds =  []
        all_pred_samples = []
        while True:
            array = self.read_until_bytes_received_or_enter_pressed(in_socket,
                n_bytes)
            if array is None:
                # enter was pressed! quit! :)
                break;
            
            array = np.fromstring(array, dtype=np.float32)
            array = array.reshape(n_rows, n_cols, order='F')
            data_saver.append_samples(array.T)
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
        
        all_samples = np.concatenate(data_saver.sample_blocks).astype(np.float32)
        all_preds = np.array(all_preds).squeeze()
        all_pred_samples = np.array(all_pred_samples)
        self.print_results(all_samples, all_preds, all_pred_samples)
        
        if self.save_data:
            data_saver.save()
            # Save parameters
            all_layers = lasagne.layers.get_all_layers(self.coordinator.model.model)
            filename = "{:s}.{:s}.adapted.npy".format(self.model_base_name,
                get_now_timestring())
            log.info("Saving to {:s}...".format(filename))
            np.save(filename, lasagne.layers.get_all_param_values(all_layers))

    def print_results(self, all_samples, all_preds, all_pred_samples):
        # y labels i from 0 to n_classes (inclusive!), 0 representing
		# non-trial => no known marker state
		y_labels = all_samples[:,-1]
		y_signal = np.ones((len(y_labels), 4)) * np.nan
		y_signal[:,0] = y_labels == 1
		y_signal[:,1] = y_labels == 2
		y_signal[:,2] = np.logical_or(y_labels == 0, y_labels==3)
		y_signal[:,3] = y_labels == 4
		
		assert not np.any(np.isnan(y_signal))
		
		interpolate_fn = interpolate.interp1d(all_pred_samples, all_preds.T,
						     bounds_error=False, fill_value=0)
		interpolated_preds = interpolate_fn(range(0,len(y_labels)))
		corrcoeffs = np.corrcoef(interpolated_preds, 
					 y_signal.T)[:4,4:]

		print("Corrcoeffs")
		print corrcoeffs
		print("mean across diagonal")
		print np.mean(np.diag(corrcoeffs))
		interpolated_pred_labels = np.argmax(interpolated_preds, axis=0)
		
		# inside trials
		corrcoeffs = np.corrcoef(interpolated_preds[:,y_labels!=0], 
					 y_signal[y_labels!=0].T)[:4,4:]
		print("Corrcoeffs inside trial")
		print corrcoeffs
		print("mean across diagonal inside trial")
		print np.mean(np.diag(corrcoeffs))
		
		# -1 since we have 0 as "break" "non-trial" marker
		label_pred_equal = interpolated_pred_labels == y_labels - 1
		label_pred_trial_equal = label_pred_equal[y_labels!=0]
		print("Accuracy inside trials")
		print np.sum(label_pred_trial_equal) / float(len(label_pred_trial_equal))
		

	class DataSaver(object):
	    """ Remember and save data streamed during an online session."""
	    def __init__(self, chan_names):
		self.chan_names = chan_names
		self.sample_blocks = []

	    def append_samples(self, samples):
		""" Expects timexchan"""
		self.sample_blocks.append(samples)
		
	    def save(self):
		# save with time as filename
		time_string = get_now_timestring()
		filename = os.path.join('data/online/', time_string + '.hdf5')
		log.info("Saving to {:s}...".format(filename))
		all_samples = np.concatenate(self.sample_blocks).astype(np.float32)
		with h5py.File(filename, 'w') as out_file:
		    dt = h5py.special_dtype(vlen=unicode)
		    dset = out_file.create_dataset("chan_names", (len(self.chan_names),), dtype=dt)
		    dset[:] = self.chan_names
		    out_file.create_dataset("cnt_samples", data=all_samples)
		log.info("Done.")

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

	def main(ui_hostname, ui_port, base_name, params_filename, plot_sensors, save_data,
		use_ui_server, adapt_model, n_updates_per_break, batch_size,
		learning_rate, n_min_trials, trial_start_offset, pred_freq,
		incoming_port):
	    setup_logging()
	    assert np.little_endian, "Should be in little endian"
	    if args.paramsfile is not None:
		log.info("Loading params from {:s}".format(args.paramsfile))
		params = np.load(params_filename)
	    else:
		params = np.load(base_name + '.npy')
	    exp = create_experiment(base_name + '.yaml')
	    # Have to set for both exp final layer and actually used model
	    # as exp final layer might be used for adaptation
	    # maybe check this all for correctness?
	    model = exp.final_layer
	    lasagne.layers.set_all_param_values(model, params)
	    model = transform_to_normal_net(model)
	    lasagne.layers.set_all_param_values(model, params)
	    
	    data_processor = StandardizeProcessor(factor_new=1e-3)
	    online_model = OnlineModel(model)
	    if adapt_model:
		online_trainer = BatchWiseCntTrainer(exp, n_updates_per_break, 
		    batch_size, learning_rate, n_min_trials, trial_start_offset)
	    else:
		log.info("Not adapting model...")
		online_trainer = NoTrainer()
	    coordinator = OnlineCoordinator(data_processor, online_model, online_trainer,
		pred_freq=pred_freq)
	    hostname = ''
	    server = PredictionServer((hostname, incoming_port), coordinator=coordinator,
		ui_hostname=ui_hostname, ui_port=ui_port, plot_sensors=plot_sensors,
		save_data=save_data, use_ui_server=use_ui_server, 
		model_base_name=base_name)
	    online_trainer.initialize()
	    log.info("Starting server on port {:d}".format(incoming_port))
	    server.start()
	    log.info("Started server")
	    server.serve_forever()

	if __name__ == '__main__':
	    gevent.signal(signal.SIGQUIT, gevent.kill)
	    args = parse_command_line_arguments()
	    if args.noprint:
		log.setLevel("WARN")
	    main(ui_hostname=args.uihost, ui_port=args.uiport, 
		base_name=args.modelfile, params_filename=args.paramsfile,
		plot_sensors=not args.noplot, save_data=not args.nosave,
		use_ui_server=not args.noui, adapt_model=not args.noadapt,
		n_updates_per_break=args.updatesperbreak, batch_size=args.batchsize,
		learning_rate=args.learningrate, n_min_trials=args.mintrials, 
		trial_start_offset=args.adaptoffset, pred_freq=args.predfreq,
		incoming_port=args.inport,
		)
    
