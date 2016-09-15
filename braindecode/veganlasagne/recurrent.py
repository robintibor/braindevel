import lasagne.init
from lasagne.layers import  MergeLayer
import theano

class BandpassLayer(MergeLayer):
    def __init__(self, incomings, n_filt_order, W_b=lasagne.init.Normal(std=0.01), W_a=lasagne.init.Normal(),
                 truncate_gradient=-1,
                 **kwargs):
        super(BandpassLayer, self).__init__(incomings, **kwargs)
        self.n_filt_order = n_filt_order
        self.truncate_gradient = truncate_gradient
        in_band_layer = incomings[1]
        n_filters = in_band_layer.output_shape[-1]
        self.W_b = self.add_param(W_b, (n_filters, self.n_filt_order), name='W_b',
                                            regularizable=True, trainable=True)
        self.W_a = self.add_param(W_a, (n_filters, self.n_filt_order), name='W_a',
                                            regularizable=True, trainable=True)
        
    def get_output_shape_for(self, input_shapes):
        """ input_shapes[0] should be examples x time x chans
            input_shapes[1] should be examples x time x chans x filters
        """
        out_shape = list(input_shapes[1])
        if out_shape[1] is not None:
            out_shape[1] - self.n_filt_order + 1
        return tuple(out_shape)

    def get_output_for(self, inputs, **kwargs):
        in_raw = inputs[0]
        in_filtered = inputs[1]
        # have to swap example axes into second axes
        # have to transpose W_a/W_b to filt_order x filters
        out = multi_row_recurrent_bandpass(in_raw.swapaxes(0,1), in_filtered.swapaxes(0,1), self.W_b.T, self.W_a.T,
                                          truncate_gradient=self.truncate_gradient)
        # swap examples axes back into first axes
        out = out.swapaxes(0,1)
        return out
        
def oneStep(x_tm6, x_tm5, x_tm4, x_tm3, x_tm2, x_tm1, x_t,
            y_tm6, y_tm5, y_tm4, y_tm3, y_tm2, y_tm1,
           b,a,y0):
    new_out = (x_tm6 * b[6] + x_tm5 * b[5] + x_tm4 * b[4] +
               x_tm3 * b[3] + x_tm2 * b[2] + x_tm1 * b[1] + x_t * b[0])
    new_out -= (y_tm6 * a[6] + y_tm5 * a[5] + y_tm4 * a[4] + 
                y_tm3 * a[3] + y_tm2 * a[2]+ y_tm1 * a[1])
    
    return new_out

def multi_row_recurrent_bandpass(x0, y0, b_sym, a_sym, truncate_gradient=-1):
    a_sym = a_sym.dimshuffle(0,'x', 'x',1)
    b_sym = b_sym.dimshuffle(0, 'x', 'x',1)
    x0 = x0.dimshuffle(0,1,2,'x')
    (x_vals, _) = theano.scan(fn=oneStep,
        sequences = [dict(input=x0, taps=range(-6,1))],
        outputs_info=[dict(initial=y0, taps=range(-6,0))],
        non_sequences = [b_sym, a_sym,y0],
        strict=True,
        truncate_gradient=truncate_gradient)
    return x_vals
    

