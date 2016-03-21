import numpy as np
from braindecode.online.ring_buffer import RingBuffer
import logging
log = logging.getLogger(__name__)

class OnlineCoordinator(object):
    """ Online coordinator accepts samples, preprocesses them with 
    data processor,  calls model to supply predictions
    when necessary.
    Online coordinator is mainly responsible
    for cutting out correct time windows for the model to predict on. """
    def __init__(self, data_processor, model, trainer, pred_freq):
        self.data_processor = data_processor
        self.model = model
        self.pred_freq = pred_freq
        # assuming 4 classes
        self.marker_buffer = RingBuffer(np.ones(
            data_processor.n_samples_in_buffer, 
            dtype=np.int32))
        self.trainer = trainer

    def initialize(self, n_chans):
        self.data_processor.initialize(n_chans)
        self.n_samples = 0
        self.i_last_pred = -1
        self.last_pred = None
        self.model.initialize()
        self.n_samples_pred_window = self.model.get_n_samples_pred_window()
        self.trainer.set_model(self.model.model) # lasagne model...
        self.trainer.set_data_processor(self.data_processor)
        self.trainer.set_marker_buffer(self.marker_buffer)
        self.trainer.initialize()

    def receive_samples(self, samples):
        """Expect samples in timexchan format"""
        sensor_samples = samples[:,:-1]
        markers = samples[:,-1]
        self.marker_buffer.extend(markers)
        self.data_processor.process_samples(sensor_samples)
        self.n_samples += len(samples)
        if self.should_do_next_prediction():
            self.predict()
        # important to do after marker buffer and data processor
        # have processed samples...
        self.trainer.process_samples(samples)

    def should_do_next_prediction(self):
        return (self.n_samples >= self.n_samples_pred_window and 
            self.n_samples > (self.i_last_pred + self.pred_freq))

    def predict(self):
        # Compute how many samples we already have past the
        # sample we wanted to predict
        # keep in mind: n_samples = n_samples (number of samples)
        # so how many samples are we past
        # last prediction + prediction frequency
        # -1 at the end below since python  indexing is zerobased
        n_samples_after_pred = min(self.n_samples - 
            self.n_samples_pred_window,
            self.n_samples - self.i_last_pred - self.pred_freq - 1)
        assert n_samples_after_pred < self.pred_freq, ("Other case "
            "not implemented yet")
        start = -self.n_samples_pred_window - n_samples_after_pred
        end = -n_samples_after_pred
        if end == 0:
            end = None
        topo = self.data_processor.get_samples(start, end)
        self.last_pred = self.model.predict(topo)
        # -1 since we have 0-based indexing in python
        self.i_last_pred = self.n_samples - n_samples_after_pred - 1

    
    def pop_last_prediction_and_sample_ind(self):
        last_pred = self.last_pred
        self.last_pred = None
        return last_pred, self.i_last_pred

    def has_new_prediction(self):
        return self.last_pred is not None
             
def make_predictions_with_online_predictor(predictor, cnt_data, 
    y_labels, block_len, input_start, input_end):
    predictor.initialize(n_chans=cnt_data.shape[1])
    window_len = predictor.model.get_n_samples_pred_window()
    all_preds = []
    i_pred_samples = []
    block = np.ones((block_len, cnt_data.shape[1] + 1),dtype=np.float32)
    perc_done = 0 # Logging progress
    for i_start_sample in xrange(input_start - window_len + 1, input_end+1,block_len):
        block = cnt_data[i_start_sample:i_start_sample+block_len]
        y_block = y_labels[i_start_sample:i_start_sample+block_len]
        block = np.concatenate((block, y_block[:,np.newaxis]), axis=1)
        predictor.receive_samples(block)
        if predictor.has_new_prediction():
            pred, i_sample = predictor.pop_last_prediction_and_sample_ind()
            assert ((i_sample + 1) - window_len) % predictor.pred_freq == 0
            all_preds.append(pred)
            i_pred_samples.append(i_sample)
        # Logging process
        input_len = input_end -input_start
        if (100 * (i_start_sample - input_start) / float(input_len)) > perc_done:
            log.info("{:d}% done.".format(perc_done))
            perc_done += 5
    log.info("100% done.")
    preds = np.array(all_preds).squeeze()
    i_pred_sample_arr = np.array(i_pred_samples) + input_start - window_len + 1
    #assert np.array_equal(i_pred_sample_arr, range(input_start,input_end+1,sample_stride))
    return i_pred_sample_arr, preds       
