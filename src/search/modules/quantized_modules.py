import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from utils.ste import *


class QLinear(torch.nn.Linear):
    def __init__(self, *args, num_bits=None, min_x=-1.0, max_x=1.0, **kwargs):
        super(QLinear, self).__init__(*args, **kwargs)
        self.num_bits = num_bits
        self.quantized_weight = Parameter(torch.zeros_like(self.weight))
        if self.num_bits:
            self.min_x = min_x
            self.max_x = max_x
            self.quantized_weight = Parameter(LinearQuantizerDorefa.apply(
                self.weight, self.num_bits, self.min_x, self.max_x))

    def forward(self, input):
        if self.num_bits:
            quantized_weight = LinearQuantizerDorefa.apply(self.weight, self.num_bits, self.min_x, self.max_x)
            self.quantized_weight = Parameter(quantized_weight)
            return F.linear(input, quantized_weight)
        else:
            return F.linear(input, self.weight)

    def extra_repr(self):
        return super(QLinear, self).extra_repr() + ', num_bits={}'.format(self.num_bits)


class QConv2d(torch.nn.Conv2d):
    def __init__(self, *args, num_bits=None, min_x=-1.0, max_x=1.0, **kwargs):
        super(QConv2d, self).__init__(*args, **kwargs)
        self.num_bits = num_bits
        self.quantized_weight = Parameter(torch.zeros_like(self.weight))
        if self.num_bits:
            self.min_x = min_x
            self.max_x = max_x
            self.quantized_weight = Parameter(LinearQuantizerDorefa.apply(
                self.weight, self.num_bits, self.min_x, self.max_x))

    def forward(self, input):
        if self.num_bits:
            quantized_weight = LinearQuantizerDorefa.apply(self.weight, self.num_bits, self.min_x, self.max_x)
            self.quantized_weight = Parameter(quantized_weight)
            return self._conv_forward(input, quantized_weight)
        else:
            return self._conv_forward(input, self.weight)

    def extra_repr(self):
        return super(QConv2d, self).extra_repr() + ', num_bits={}'.format(self.num_bits)