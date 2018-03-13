from enum import Enum, auto
import numpy as np
from torch import nn


class ObjectRecognitionCNN(nn.Module):

    def __init__(self, input_dim, input_channels, convSpecs):
        super().__init__()
        dim = np.array(input_dim, dtype=np.int)
        layers = []
        prev_channels = 3
        for spec in layerspecs:
            layer = nn.Conv2d(prev_channels, spec.channels, spec.size, spec.stride)
            layers.append(layer)
            prev_channels = spec.channels
            dim = update_size(dim, spec.size, spec.stride, 0)
            if spec.pooling !+ PoolingTypes.NONE
                poolingType = None
                if spec.pooling = PoolingTypes.AVERAGE:
                    poolingType = nn.AveragePool2d
                elif spec.pooling = PoolingTypes.MAX:
                    poolingType = nn.MaxPool2d
                else:
                    raise ValueError('Invalid pooling type: ' + str(poolingType))
                layers.append(poolingType(spec.poolsize, spec.poolstride))
                dim = update_size(dim, spec.poolsize, spec.poolstride, 0)
        self.output_dim = dim

    def forward(x):
        for layer in self.layers:
            x = layer(x)
        return x


def update_size(dim, size, stride, padding):
    return (dim + padding + 1 - size) // stride


LayerSpec = collections.namedtuple('LayerSpec', ['channels', 'size', 'stride', 'pooling', 'poolsize', 'poolstride'])


class PoolingTypes(Enum):
    MAX = auto()
    AVERAGE = auto()
    NONE = auto()
