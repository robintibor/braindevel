import numpy as np
from braindecode.datahandling.batch_iteration import compute_trial_start_end_samples
from braindecode.mywyrm.processing import create_cnt_y,\
    create_old_class_to_new_class
import logging
log = logging.getLogger(__name__)

def create_cnt_y_start_end_marker(cnt, start_marker_def, end_marker_def,
    segment_ival, timeaxis=-2, trial_classes=None):
    """Segment ival is : (offset to start marker, offset to end marker)"""
    start_to_end_value = dict()
    for class_name in start_marker_def:
        start_marker_vals = start_marker_def[class_name]
        end_marker_vals = end_marker_def[class_name]
        assert len(start_marker_vals) == 1
        start_to_end_value[start_marker_vals[0]] = end_marker_vals

    # Assuming start marker vals are 1 ... n_classes
    # Otherwise change code...
    all_start_marker_vals = start_to_end_value.keys()
    n_classes = np.max(all_start_marker_vals)
    
    # You might disable this if you have checked trial_classes implementation here
    assert (trial_classes is not None) or (
        np.array_equal(np.sort(all_start_marker_vals), range(1, n_classes+1))), (
        "Assume start marker values are from 1...n_classes if trial classes not given")
    all_end_marker_vals = np.concatenate(start_to_end_value.values())
    
    if trial_classes is not None:
        old_class_to_new_class = create_old_class_to_new_class(start_marker_def,
            trial_classes)
    y = np.zeros((cnt.data.shape[0], len(all_start_marker_vals)), dtype= np.int32)
    i_marker = 0
    while i_marker < len(cnt.markers):
        # first find start marker
        while ((i_marker < len(cnt.markers)) and 
              (cnt.markers[i_marker][1] not in all_start_marker_vals)):
            i_marker += 1
        if i_marker < len(cnt.markers):
            start_marker_ms = cnt.markers[i_marker][0]
            start_marker_val = cnt.markers[i_marker][1]
            # find end marker
            i_marker += 1 # advance one past start marker already
            while ((i_marker < len(cnt.markers)) and
                (cnt.markers[i_marker][1] not in all_end_marker_vals)):
                # Check if there is a new start marker already
                if cnt.markers[i_marker][1]  in all_start_marker_vals:
                    log.warn("New start marker  {:.0f} at {:.3f} sec found, "
                        "no end marker for earlier start marker {:.0f} "
                        "at {:.3f} sec found.".format(
                            cnt.markers[i_marker][1], cnt.markers[i_marker][0] / 1000.0,
                            start_marker_val, start_marker_ms / 1000.0))
                    start_marker_ms = cnt.markers[i_marker][0]
                    start_marker_val = cnt.markers[i_marker][1]
                i_marker += 1
            if i_marker == len(cnt.markers):
                log.warn(("No end marker for start marker code {:d} "
                    "at {:.3f} sec found.").format(start_marker_val, start_marker_ms /1000.0))
                break
            end_marker_ms = cnt.markers[i_marker][0]
            end_marker_val = cnt.markers[i_marker][1]
            assert end_marker_val in start_to_end_value[start_marker_val]

            first_index = np.searchsorted(cnt.axes[timeaxis], start_marker_ms + segment_ival[0])
            # +1 should be correct since last index not part... but maybe recheck?
            last_index = np.searchsorted(cnt.axes[timeaxis], end_marker_ms+segment_ival[1])
            if trial_classes is not None:
                # -1 because before is 1-based matlab-indexing(!)
                i_class = int(old_class_to_new_class[int(start_marker_val)] - 1)
            else:
                # -1 because before is 1-based matlab-indexing(!)
                i_class = int(start_marker_val - 1)
            y[first_index:last_index, i_class] = 1 
    return y

class PipelineSegmenter(object):
    def __init__(self, segmenters):
        self.segmenters = segmenters
    def segment(self, cnt):
        y, classnames = self.segmenters[0].segment(cnt)
        for segmenter in self.segmenters[1:]:
            y, classnames = segmenter.segment(cnt,y, classnames)
        return y, classnames

class MarkerSegmenter(object):
    def __init__(self, segment_ival, marker_def, trial_classes, end_marker_def=None):
        self.segment_ival = segment_ival
        self.marker_def = marker_def
        self.end_marker_def = end_marker_def
        self.trial_classes = trial_classes
        
    def segment(self, cnt, y=None, class_names=None):
        assert y is None
        assert class_names is None
        # marker segmenter, dann restrict range, dann restrict classes, dann evtl. add breaks
        assert np.all([len(labels) == 1 for labels in 
                self.marker_def.values()]), (
                "Expect only one label per class, otherwise rewrite...")
        # get class names, assume they are sorted by marker codes
        class_names = sorted(self.marker_def.keys(), 
            key= lambda k: self.marker_def[k][0])
        if self.end_marker_def is None:
            y = create_cnt_y(cnt, self.segment_ival,self.marker_def,
                trial_classes=self.trial_classes)
        else:
            y = create_cnt_y_start_end_marker(cnt,self.marker_def, self.end_marker_def,
                segment_ival=self.segment_ival,
                trial_classes=self.trial_classes)
        return y, class_names

class RestrictTrialRange(object):
    def __init__(self, start_trial, stop_trial):
        self.start_trial = start_trial
        self.stop_trial = stop_trial
        
    def segment(self, cnt, y, class_names):
        # dont modify original y
        y = np.copy(y)
        if (self.start_trial is not None) or (self.stop_trial is not None):
            trial_starts, trial_ends = compute_trial_start_end_samples(y,
                check_trial_lengths_equal=False)
            if self.start_trial is not None:
                y[:trial_starts[self.start_trial] - 1] = 0
            if self.stop_trial is not None:
                y[trial_starts[self.stop_trial] - 1:] = 0
        return y, class_names
    
class RestrictTrialClasses(object):
    def __init__(self, class_names):
        self.class_names = class_names
    
    def segment(self, cnt, y, class_names):
        # dont modify original y
        y = np.copy(y)
        indexes_to_keep = [class_names.index(name) for name in self.class_names]
        y = y[:,np.array(indexes_to_keep)]
        return y, self.class_names
    
class AddTrialBreaks(object):
    def __init__(self, min_length_ms, max_length_ms,
            start_offset_ms, stop_offset_ms,
            start_marker_def, end_marker_def=None, trial_to_break_ms=None):
        assert not((end_marker_def is None) and (trial_to_break_ms is None))
        self.start_marker_def = start_marker_def
        self.end_marker_def = end_marker_def
        self.min_length_ms = min_length_ms
        self.max_length_ms = max_length_ms
        self.start_offset_ms = start_offset_ms
        self.stop_offset_ms = stop_offset_ms
        self.trial_to_break_ms = trial_to_break_ms
        
    def segment(self, cnt, y, class_names):
        if 'Rest' not in class_names:
            # add new class vector, for now empty
            y = np.concatenate((y, y[:,0:1] * 0), axis=1)
            i_class = -1
            class_names = class_names + ['Rest']
        else:
            i_class = class_names.index('Rest')
        if self.end_marker_def is not None:
            break_start_ends_ms = compute_break_start_ends_ms(cnt.markers,
                self.start_marker_def, self.end_marker_def)
        else:
            break_start_ends_ms = compute_break_start_ends_ms_without_end_marker(
                cnt.markers, self.start_marker_def, self.trial_to_break_ms)
        start_offset = ms_to_i_sample(self.start_offset_ms, cnt.fs)
        stop_offset = ms_to_i_sample(self.stop_offset_ms, cnt.fs)
        n_breaks_added = 0
        for break_start_ms, break_end_ms in break_start_ends_ms:
            i_break_start = ms_to_i_sample(break_start_ms, cnt.fs)
            i_break_stop = ms_to_i_sample(break_end_ms, cnt.fs) + 1
            break_len_ms = break_end_ms - break_start_ms
            if (break_len_ms >= self.min_length_ms) and (break_len_ms <= self.max_length_ms):
                start = i_break_start + start_offset
                stop  = i_break_stop + stop_offset
                y[start:stop,i_class] = 1
                n_breaks_added += 1
        log.info("{:d} of {:d} possible breaks added".format(n_breaks_added,
            len(break_start_ends_ms)))
        return y, class_names

def ms_to_i_sample(ms, fs):
    return int(np.round(ms * float(fs) / 1000.0))

def compute_break_start_ends_ms(markers, start_marker_def, end_marker_def):
    '''
    Compute break start end in milliseconds as those points that lie between
    an end marker of a trial and a start marker of a trial.
    :param markers:
    :param start_marker_def:
    :param end_marker_def:
    '''
    assert np.all([len(v) == 1 for v in start_marker_def.values()])
    start_vals = [v[0] for v in start_marker_def.values()]
    end_vals = np.concatenate(end_marker_def.values())
    break_starts, break_stops = extract_break_start_stops_ms(markers, start_vals,
                                                   end_vals,)
    break_start_end = zip(break_starts, break_stops)
    return break_start_end

def extract_break_start_stops_ms(markers, all_start_marker_vals,
                                 all_end_marker_vals,):
    break_starts = []
    break_stops = []
    i_marker = 0
    while i_marker < len(markers):
        # first find start marker
        while ((i_marker < len(markers)) and
                   (markers[i_marker][1] not in all_end_marker_vals)):
            i_marker += 1

        if i_marker < len(markers):
            end_marker_ms = markers[i_marker][0]
            end_marker_val = markers[i_marker][1]
            # find start marker
            i_marker += 1  # advance one past end marker already
            while ((i_marker < len(markers)) and
                       (markers[i_marker][1] not in all_start_marker_vals)):
                # Check if there is a new start marker already
                if markers[i_marker][1] in all_end_marker_vals:
                    log.warn("New end marker  {:.0f} at {:.3f} sec found, "
                             "no start marker for earlier end marker {:.0f} "
                             "at {:.3f} sec found.".format(
                        markers[i_marker][1],
                        markers[i_marker][0] / 1000.0,
                        end_marker_val, end_marker_ms / 1000.0))
                    end_marker_ms = markers[i_marker][0]
                    end_marker_val = markers[i_marker][1]
                i_marker += 1
            if i_marker == len(markers):
                log.warn(("No start marker for end marker code {:.0f} "
                          "at {:.3f} sec found.").format(end_marker_val,
                                                         end_marker_ms / 1000.0))
                break
            start_marker_ms = markers[i_marker][0]
            start_marker_val = markers[i_marker][1]
            assert start_marker_val in all_start_marker_vals
            # + window_stride should only create maximum one extra window at the end
            #  to account for fact there may be extra data which does not fill a whole window
            # at the end
            break_starts.append(end_marker_ms)
            break_stops.append(start_marker_ms)
    return break_starts, break_stops

def compute_break_start_ends_ms_without_end_marker(markers, start_marker_def,
    trial_to_break_ms):
    assert np.all([len(v) == 1 for v in start_marker_def.values()])
    start_vals = [v[0] for v in start_marker_def.values()]
    start_mrk_ms = [m[0] for m in markers if m[1] in start_vals]
    start_mrk_ms = np.array(start_mrk_ms)
    end_trial_ms = start_mrk_ms + trial_to_break_ms
    assert np.all(start_mrk_ms[1:] > end_trial_ms[:-1])
    return zip(end_trial_ms[:-1], start_mrk_ms[1:])


class FilterTrialLength(object):
    """Never used, except once in notebook, but why not :)"""
    def __init__(self, min_length_ms):
        self.min_length_ms = min_length_ms

    def segment(self, cnt, y, class_names):
        print("fs", cnt.fs)
        n_min_samples = int(self.min_length_ms * cnt.fs / 1000.0)
        starts, ends = compute_trial_start_end_samples(y, check_trial_lengths_equal=False,)
        n_removed_trials = 0
        for i_sample_start, i_sample_end in zip(starts, ends):
            n_samples_in_trial = i_sample_end - i_sample_start + 1
            if  n_samples_in_trial < n_min_samples:
                y[i_sample_start:i_sample_end+1] = 0
                n_removed_trials += 1
        return y, class_names
    