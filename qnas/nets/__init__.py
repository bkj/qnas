from __future__ import absolute_import
from .rnet import RNet
from .mnist_net import MNISTNet
from .two_layer_net import TwoLayerNet
from .optnet import OptNetSmall

net_constructors = {
    "RNet" : RNet,
    "MNISTNet" : MNISTNet,
    "TwoLayerNet" : TwoLayerNet,
    "OptNetSmall" : OptNetSmall,
}