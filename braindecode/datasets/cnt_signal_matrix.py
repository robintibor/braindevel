from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from wyrm.processing import select_channels
from braindecode.datasets.sensor_positions import sort_topologically
import numpy as np
import logging
from braindecode.mywyrm.processing import create_cnt_y, create_cnt_y_start_end_marker
log = logging.getLogger(__name__)

class CntSignalMatrix(DenseDesignMatrix):
    reloadable=False # otherwise have to deal with sensor name issue on reload
    def __init__(self, signal_processor,
        sensor_names='all',
        axes=('b', 'c', 0, 1),
        sort_topological=True,
        end_marker_def=None):
        # sort sensors topologically to allow networks to exploit topology
        if (sensor_names is not None) and (sensor_names  != 'all') and sort_topological:
            sensor_names = sort_topologically(sensor_names)
        self.__dict__.update(locals())
        del self.self

    def ensure_is_loaded(self):
        if not hasattr(self, 'X'):
            self.load()

    def load(self):
        self.load_cnt()
        self.load_from_cnt()
    
    def load_from_cnt(self):
        """ This function is split off to allow cleaner to go 
        between loading of cnt, clean markers, and then resume loading"""
        log.info("Preprocess continuous signal...")
        self.signal_processor.preprocess_continuous_signal()
        self.select_sensors()
        self.create_cnt_y()
        self.create_dense_design_matrix()
        self.remove_cnt()

    def load_cnt(self):
        log.info("Load continuous signal...")
        self.signal_processor.load_signal_and_markers()

    def select_sensors(self):
        if (self.sensor_names is not None) and (self.sensor_names is not 'all'):
            self.signal_processor.cnt = select_channels(
                self.signal_processor.cnt, 
                self.sensor_names)
        self.sensor_names = self.signal_processor.cnt.axes[-1]

    def create_cnt_y(self):
        """Create continuous target signal"""
        if self.end_marker_def is None:
            self.y = create_cnt_y(self.signal_processor.cnt,
                self.signal_processor.segment_ival,
                self.signal_processor.marker_def, timeaxis=-2)
        else:
            self.y = create_cnt_y_start_end_marker(self.signal_processor.cnt,
                self.signal_processor.marker_def, self.end_marker_def,
                self.signal_processor.segment_ival, timeaxis=-2)

    def create_dense_design_matrix(self):
        # add empty 01 (from bc01) axes ...
        topo_view = self.signal_processor.cnt.data[:,:,
            np.newaxis,np.newaxis].astype(np.float32)
        topo_view = np.ascontiguousarray(np.copy(topo_view))
        super(CntSignalMatrix, self).__init__(topo_view=topo_view, y=self.y, 
                                              axes=self.axes)

        log.info("Loaded dataset with shape: {:s}".format(
            str(self.get_topological_view().shape)))

    def remove_cnt(self):
        del self.signal_processor.cnt

    def free_memory(self):
        del self.X