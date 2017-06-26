from torch import nn
from braindecode2.modules.expression import Expression
from braindecode2.torchext.functions import safe_log, square


class ShallowFBCSPNet(object):
    # TODO: auto final dense length for shallow
    def __init__(self, in_chans,
                 n_classes,
                 n_filters_time=40,
                 filter_time_length=25,
                 n_filters_spat=40,
                 pool_time_length=75,
                 pool_time_stride=15,
                 final_dense_length=30,
                 conv_nonlin=square,
                 pool_mode='mean',
                 pool_nonlin=safe_log,
                 split_first_layer=True,
                 batch_norm=True,
                 batch_norm_alpha=0.1,
                 drop_prob=0.5):
        self.__dict__.update(locals())
        del self.self

    def create_network(self):
        # todo: check if dropout or dropout2d is better
        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]
        model = nn.Sequential()
        if self.split_first_layer:
            model.add_module('dimshuffle',
                             Expression(lambda x: x.permute(0, 3, 2, 1)))
            model.add_module('conv_time', nn.Conv2d(1, self.n_filters_time,
                                                    (
                                                    self.filter_time_length, 1),
                                                    stride=1, ))
            model.add_module('conv_spat',
                             nn.Conv2d(self.n_filters_time, self.n_filters_spat,
                                       (1, self.in_chans), stride=1,
                                       bias=not self.batch_norm))
            n_filters_conv = self.n_filters_spat
        else:
            model.add_module('conv_time',
                             nn.Conv2d(self.in_chans, self.n_filters_time,
                                       (self.filter_time_length, 1),
                                       stride=1,
                                       bias=not self.batch_norm))
            n_filters_conv = self.n_filters_time
        if self.batch_norm:
            model.add_module('bnorm',
                             nn.BatchNorm2d(n_filters_conv,
                                            momentum=self.batch_norm_alpha,
                                            affine=True),)
        model.add_module('conv_nonlin', Expression(self.conv_nonlin))
        model.add_module('pool',
                         pool_class(kernel_size=(self.pool_time_length, 1),
                                    stride=(self.pool_time_stride, 1)))
        model.add_module('pool_nonlin', Expression(self.pool_nonlin))
        model.add_module('drop', nn.Dropout2d(p=self.drop_prob))
        model.add_module('conv_classifier',
                         nn.Conv2d(n_filters_conv, self.n_classes,
                                   (self.final_dense_length, 1)), )
        model.add_module('softmax', nn.LogSoftmax())
        return model
