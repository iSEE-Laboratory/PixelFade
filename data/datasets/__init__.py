# encoding: utf-8

from .cuhk03 import CUHK03
from .market1501 import Market1501
from .msmt17 import MSMT17
from .dataset_loader import ImageDataset

__factory = {
    'market1501': Market1501,
    'cuhk03': CUHK03,
    'msmt17': MSMT17,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
