from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset


class RenderedWB(Dataset):
    def __init__(self, root, transform=None) -> None:
        super().__init__()
        self.data = list(set([
            ("_".join(str(p).split("_")[:-2]+['G_AS.png']), "_".join(str(p).split("_")[:-2]+['{}_CS.png']))
            for p in list((Path(root) / "imgs").iterdir())
        ]))
        self.data = [(g, i) for g, i in self.data if self.check_file(g, i)]
        self.transform = transform

    def check_file(self, g, p):
        return Path(g).is_file() and Path(p.format("D")).is_file() and Path(p.format("T")).is_file() and \
            Path(p.format("F")).is_file() and Path(p.format("C")).is_file() and Path(p.format("S")).is_file()

    def __getitem__(self, index):
        gt_fname, tmp_img_fname = self.data[index]
        gt = Image.open(gt_fname).convert("RGB")
        D = Image.open(tmp_img_fname.format("D")).convert("RGB")
        T = Image.open(tmp_img_fname.format("T")).convert("RGB")
        F = Image.open(tmp_img_fname.format("F")).convert("RGB")
        C = Image.open(tmp_img_fname.format("C")).convert("RGB")
        S = Image.open(tmp_img_fname.format("S")).convert("RGB")
        if self.transform:
            gt, D, T, F, C, S = self.transform(gt, D, T, F, C, S)
        return gt, D, T, F, C, S
    
    def __len__(self):
        return len(self.data)


class CubeWB(Dataset):
    def __init__(self, root, transform=None) -> None:
        super().__init__()
        self.GTs = list((Path(root) / "GTs").iterdir())
        self.transform = transform
    
    def __getitem__(self, index):
        gt_fname = self.GTs[index]
        tmp_img_fname = str(gt_fname.parent.parent / "cube-wb" / (f"{str(gt_fname.stem)}_"+"{}.png"))
        gt = Image.open(gt_fname).convert("RGB")
        D = Image.open(tmp_img_fname.format("D")).convert("RGB")
        T = Image.open(tmp_img_fname.format("T")).convert("RGB")
        F = Image.open(tmp_img_fname.format("F")).convert("RGB")
        C = Image.open(tmp_img_fname.format("C")).convert("RGB")
        S = Image.open(tmp_img_fname.format("S")).convert("RGB")
        if self.transform:
            gt, D, T, F, C, S = self.transform(gt, D, T, F, C, S)
        return gt, D, T, F, C, S
    
    def __len__(self):
        return len(self.GTs)
    

if __name__ == "__main__":
    from transforms import *
    dataset = RenderedWB(
        root="/media/birdortyedi/e5042b8f-ca5e-4a22-ac68-7e69ff648bc4/RenderedWB",
        transform=Compose([
            ResizeInstances(384),
            RandomCropInstances(256),
            RandomHorizontalFlipInstances(),
            ToTensors()
        ])
    )
    for i in range(len(dataset)):
        gt, D, T, F, C, S = dataset.__getitem__(i)
        print(gt.shape, T.shape)