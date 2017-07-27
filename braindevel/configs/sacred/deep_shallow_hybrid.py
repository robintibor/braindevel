from braindevel.models.merged_net import MergedNet
from hyperoptim.parse import cartesian_dict_of_lists_product,\
    product_of_list_of_lists_of_dicts
from numpy.random import RandomState
import lasagne
from braindevel.veganlasagne.nonlinearities import square, safe_log
from lasagne.nonlinearities import elu,softmax,identity
from braindevel.models.shallow_fbcsp import ShallowFBCSPNet
from braindevel.models.deep5 import Deep5Net
from braindevel.veganlasagne.clip import ClipLayer


def create_deep_net(in_chans, input_time_length, final_dense_length, n_classes,
                    filter_time_length=10, filter_length_4=10):
    d5net = create_deep_model(in_chans, input_time_length, final_dense_length,
                              n_classes, filter_time_length=filter_time_length,
                              filter_length_4=filter_length_4)
    final_layer = d5net.get_layers()[-1]
    final_layer = ClipLayer(final_layer, 1e-4, 1 - 1e-4)
    return final_layer

def create_deep_model(in_chans, input_time_length, final_dense_length, n_classes,
                      filter_time_length=10, filter_length_4=10):
    # receptive field size is determined by model architecture
    num_filters_time = 25
    num_filters_spat = 25
    pool_time_length = 3
    pool_time_stride = 3
    num_filters_2 = 50
    filter_length_2 = 10
    num_filters_3 = 100
    filter_length_3 = 10
    num_filters_4 = 200
    filter_length_4 = filter_length_4
    final_nonlin = softmax
    first_nonlin = elu
    first_pool_mode = 'max'
    first_pool_nonlin = identity
    later_nonlin = elu
    later_pool_mode = 'max'
    later_pool_nonlin = identity
    drop_in_prob = 0.0
    drop_prob = 0.5
    batch_norm_alpha = 0.1
    double_time_convs = False
    split_first_layer = True
    batch_norm = True

    d5net = Deep5Net(in_chans=in_chans, input_time_length=input_time_length,
                     num_filters_time=num_filters_time,
                     filter_time_length=filter_time_length,
                     num_filters_spat=num_filters_spat,
                     pool_time_length=pool_time_length,
                     pool_time_stride=pool_time_stride,
                     num_filters_2=num_filters_2,
                     filter_length_2=filter_length_2,
                     num_filters_3=num_filters_3,
                     filter_length_3=filter_length_3,
                     num_filters_4=num_filters_4,
                     filter_length_4=filter_length_4,
                     final_dense_length=final_dense_length, n_classes=n_classes,
                     final_nonlin=final_nonlin, first_nonlin=first_nonlin,
                     first_pool_mode=first_pool_mode,
                     first_pool_nonlin=first_pool_nonlin,
                     later_nonlin=later_nonlin, later_pool_mode=later_pool_mode,
                     later_pool_nonlin=later_pool_nonlin,
                     drop_in_prob=drop_in_prob, drop_prob=drop_prob,
                     batch_norm_alpha=batch_norm_alpha,
                     double_time_convs=double_time_convs,
                     split_first_layer=split_first_layer, batch_norm=batch_norm)
    return d5net


def create_shallow_net(in_chans, input_time_length, final_dense_length,
                       n_classes):
    shallow_net = create_shallow_model(in_chans, input_time_length,
                                       final_dense_length,
                                       n_classes)
    final_layer = shallow_net.get_layers()[-1]
    final_layer = ClipLayer(final_layer, 1e-4, 1 - 1e-4)
    return final_layer


def create_shallow_model(in_chans, input_time_length, final_dense_length,
                       n_classes):
    shallow_net = ShallowFBCSPNet(in_chans, input_time_length, n_classes,
                                  n_filters_time=40,
                                  filter_time_length=25,
                                  n_filters_spat=40,
                                  pool_time_length=75,
                                  pool_time_stride=15,
                                  final_dense_length=final_dense_length,
                                  conv_nonlin=square,
                                  pool_mode='average_exc_pad',
                                  pool_nonlin=safe_log,
                                  split_first_layer=True,
                                  batch_norm=True,
                                  batch_norm_alpha=0.1,
                                  drop_prob=0.5)
    return shallow_net



def create_merged_net(in_chans, input_time_length, final_dense_length_deep,
                      final_dense_length_shallow,
                      n_classes, filter_time_length_deep=13):
    n_deep_features = 60
    n_shallow_features = 40
    batch_norm_before_merge = True
    nonlin_before_merge = elu
    deep_net = create_deep_model(in_chans, input_time_length,
                               final_dense_length_deep, n_classes,
                                 filter_time_length=filter_time_length_deep)
    shallow_net = create_shallow_model(in_chans, input_time_length,
                                     final_dense_length_shallow, n_classes)
    merged_net = MergedNet(
        [deep_net, shallow_net,], [n_deep_features,n_shallow_features],
        n_classes, batch_norm_before_merge=batch_norm_before_merge,
        nonlin_before_merge=nonlin_before_merge)
    final_layer = merged_net.get_layers()[-1]
    final_layer = ClipLayer(final_layer, 1e-4, 1 - 1e-4)
    return final_layer


def get_templates():
    return  {}

def get_grid_param_list():
    return [{}]

def sample_config_params(rng, params):
    return params