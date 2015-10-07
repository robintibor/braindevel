
from lasagne.layers.pool import Pool2DLayer

from theano.tensor.signal import downsample
import numpy as np
class SumPool2dLayer(Pool2DLayer):
    def get_output_for(self, input, **kwargs):
        pooled = downsample.max_pool_2d(input,
                                        ds=self.pool_size,
                                        st=self.stride,
                                        ignore_border=self.ignore_border,
                                        padding=self.pad,
                                        mode=self.mode,
                                        )
        # cast size to float32 to prevent upcast to float64 of entire data
        return pooled * np.float32(np.prod(self.pool_size))