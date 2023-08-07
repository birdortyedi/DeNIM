from .awb import CubeWB, RenderedWB
from .transforms import *


__all__ = [
    "CubeWB", 
    "RenderedWB"
]


def dataset_getter(name: str):
    if name == "RenderedWB":
        return RenderedWB
    elif name == "CubeWB":
        return CubeWB
    else:
        raise NotImplementedError(f"Not implemented for dataset named {name}")


def transform_getter(name: str, im_sz: int):
    if name == "RenderedWB":
        return Compose([
            ResizeInstances(max(384, im_sz)),
            RandomCropInstances(im_sz),
            RandomHorizontalFlipInstances(),
            ToTensors()
        ])
    elif name == "CubeWB":
        return Compose([
            ToTensors()
        ])
    else:
        raise NotImplementedError(f"Not implemented for dataset named {name}")
