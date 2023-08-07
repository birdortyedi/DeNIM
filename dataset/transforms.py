import torch
import numbers
from typing import Tuple, Sequence
from torchvision.transforms import functional as TF
from torchvision import transforms
from PIL import Image


class RandomCropInstances:
    @staticmethod
    def get_params(img: torch.Tensor, output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image or Tensor): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
            )

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
        return i, j, th, tw

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__()

        self.size = tuple(self._setup_size(
            size, error_msg="Please provide only two dimensions (h, w) for size."
        ))

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, *imgs):
        imgs = list(imgs)
        if self.padding is not None:
            for i in range(len(imgs)):
                imgs[i] = TF.pad(imgs[i], self.padding, self.fill, self.padding_mode)

        width, height = imgs[0].size
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            for i in range(len(imgs)):
                imgs[i] = TF.pad(imgs[i], padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            for i in range(len(imgs)):
                imgs[i] = TF.pad(imgs[i], padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(imgs[0], self.size)
        
        for i in range(len(imgs)):
            imgs[i] = TF.crop(imgs[i], i, j, h, w)

        return imgs

    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, padding={1})".format(self.size, self.padding)

    def _setup_size(self, size, error_msg):
        if isinstance(size, numbers.Number):
            return int(size), int(size)

        if isinstance(size, Sequence) and len(size) == 1:
            return size[0], size[0]

        if len(size) != 2:
            raise ValueError(error_msg)

        return size


class ResizeInstances(torch.nn.Module):
    def __init__(self, size, interpolation=transforms.InterpolationMode.BILINEAR):
        super().__init__()
        if not isinstance(size, (int, Sequence)):
            raise TypeError("Size should be int or sequence. Got {}".format(type(size)))
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")
        self.size = size
        self.interpolation = interpolation

    def forward(self, *imgs):
        return [TF.resize(imgs[i], self.size, self.interpolation) for i in range(len(imgs))]

    def __repr__(self):
        _pil_interpolation_to_str = {
            Image.NEAREST: 'PIL.Image.NEAREST',
            Image.BILINEAR: 'PIL.Image.BILINEAR',
            Image.BICUBIC: 'PIL.Image.BICUBIC',
            Image.LANCZOS: 'PIL.Image.LANCZOS',
            Image.HAMMING: 'PIL.Image.HAMMING',
            Image.BOX: 'PIL.Image.BOX',
        }
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)
    

class RandomHorizontalFlipInstances:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, *imgs):
        if torch.rand(1) < self.p:
            return [TF.hflip(imgs[i]) for i in range(len(imgs))]
        return list(imgs)

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    

class ToTensors:
    def __call__(self, *imgs):
        return [TF.to_tensor(imgs[i]) for i in range(len(imgs))]

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *imgs):
        for t in self.transforms:
            imgs = t(*imgs)
        return list(imgs)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
