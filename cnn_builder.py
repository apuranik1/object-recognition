import collections
from enum import Enum, auto
from functools import reduce
import torch
from torch import nn
import torch.nn.functional as F


class CNNStack(nn.Module):

    def __init__(self, input_dim, input_channels, convSpecs):
        super().__init__()
        dim = torch.LongTensor(input_dim)
        layers = []
        prev_channels = input_channels
        for spec in convSpecs:
            layer = nn.Conv2d(prev_channels, spec.channels, spec.size, spec.stride)
            layers.append(layer)
            layers.append(nn.LeakyReLU())
            prev_channels = spec.channels
            dim = update_size(dim, spec.size, spec.stride, 0)
            if spec.pooling != PoolingTypes.NONE:
                poolingType = None
                if spec.pooling == PoolingTypes.AVERAGE:
                    poolingType = nn.AveragePool2d
                elif spec.pooling == PoolingTypes.MAX:
                    poolingType = nn.MaxPool2d
                else:
                    raise ValueError('Invalid pooling type: ' + str(poolingType))
                layers.append(poolingType(spec.poolsize, spec.poolstride))
                dim = update_size(dim, spec.poolsize, spec.poolstride, 0)
        self.layers = nn.Sequential(*layers)
        self.channels = prev_channels
        self.output_dim = tuple(dim)

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):

    def __init__(self, channels, categories):
        super().__init__()
        self.decoder = nn.Conv2d(channels, categories, 1)
        self.out_channels = categories

    def forward(self, x):
        n = x.size(0)
        x = F.leaky_relu(self.decoder(x))
        x = x.view(n, self.out_channels, -1).mean(dim=2)  # flatten input
        return x


class ObjectRecognitionCNN(nn.Module):

    def __init__(self, input_dim, input_channels, convSpecs, categories):
        super().__init__()
        self.cnn = CNNStack(input_dim, input_channels, convSpecs)
        self.decoder = Decoder(self.cnn.channels, categories)

    def forward(self, x):
        return self.decoder(self.cnn(x))


def update_size(dim, size, stride, padding):
    return (dim + padding + 1 - size) / stride


LayerSpec = collections.namedtuple('LayerSpec', ['channels', 'size', 'stride', 'pooling', 'poolsize', 'poolstride'])


class PoolingTypes(Enum):
    MAX = auto()
    AVERAGE = auto()
    NONE = auto()
