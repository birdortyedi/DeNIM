import yaml
import time
import cv2
import torch
import glog as log
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils import data
from colour.difference import delta_E

from dataset import dataset_getter, transform_getter
from modeling.denim import DeNIM_to_Canon, DeNIM_wo_Fusion, AWBEncoder


class AWBCorrectionTester:
    def __init__(self, cfg) -> None:
        self._setup(cfg)
        self._init_parameters()

        self.transform = transform_getter(self.DATASET_NAME, self.IMG_SIZE)
        self.dataset = dataset_getter(self.DATASET_NAME)(root=self.DATASET_ROOT, transform=self.transform)
        self.image_loader = data.DataLoader(dataset=self.dataset, batch_size=1, shuffle=False)

        self.dncm_to_canon = DeNIM_to_Canon(self.k, self.ch)
        self.dncm_wo_fusion = DeNIM_wo_Fusion(self.k)
        self.awb_encoder = AWBEncoder(self.sz, self.k, self.ch, self.ps, str(self.BACKBONE_WEIGHTS_PATH), self.BACKBONE_TYPE)

        if self.INIT_FROM is None:
            log.info("Checkpoints file not found. Random initialization applied.")
        else:
            self.load_checkpoints(self.INIT_FROM)
        self.to_cuda()

    def _setup(self, cfg):
        with open(cfg, 'r') as stream:
            self.cfg = yaml.safe_load(stream)

    def _init_parameters(self):
        self.k = int(self.cfg["k"])
        self.sz = int(self.cfg["sz"])
        self.ch = int(self.cfg["ch"])
        self.ps = int(self.cfg["ps"])
        self.IMG_SIZE = int(self.cfg["IMG_SIZE"])
        self.DATASET_NAME = str(self.cfg["DATASET_NAME"])
        self.DATASET_ROOT = Path(self.cfg["DATASET_ROOT"])
        self.INIT_FROM = self.cfg["INIT_FROM"]
        self.BACKBONE_WEIGHTS_PATH = Path(self.cfg["BACKBONE_WEIGHTS_PATH"])
        self.BACKBONE_TYPE = str(self.cfg["BACKBONE_TYPE"])
    
    def __call__(self):
        self.run()
    
    def run(self):
        time_lst = list()
        mse_lst, mae_lst, deltaE2000_lst = list(), list(), list()
        with torch.no_grad():
            for GT, D, T, F, C, S in tqdm(self.image_loader, total=len(self.image_loader)):
                GT = GT.float()
                GT_np = GT.squeeze(0).permute(1, 2, 0).numpy()
                D = D.float().cuda()
                T = T.float().cuda()
                S = S.float().cuda()

                if self.ch == 9:
                    I = torch.cat([D,T,S], dim=1)
                else:
                    F = F.float().cuda()
                    C = C.float().cuda()
                    I = torch.cat([D,T,F,C,S], dim=1)

                tick = time.time()
                d = self.awb_encoder(I)
                canon = self.dncm_to_canon(I, d)
                out = self.dncm_wo_fusion(canon)
                out_np = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
                tock = time.time()
                time_lst.append(tock-tick)

                mae = self.mean_angular_error(out_np.reshape(-1, 3) * 255., GT_np.reshape(-1, 3) * 255.)
                mae_lst.append(mae)
                deltae = delta_E(cv2.cvtColor(GT_np, cv2.COLOR_RGB2Lab), cv2.cvtColor(out_np, cv2.COLOR_RGB2Lab)).mean()
                deltaE2000_lst.append(deltae)
                mse = (((GT_np - out_np) * 255.) ** 2).mean()
                mse_lst.append(mse)
                log.info(
                    f"MSE: {mse}, MAE: {mae}, DELTA_E: {deltae}"
                )
                log.info(
                    "Average:\n"
                    f"\nMSE: {round(np.mean(mse_lst), 4)}, Q1: {round(np.quantile(mse_lst, 0.25), 4)}, Q2: {round(np.quantile(mse_lst, 0.5), 4)}, Q3: {round(np.quantile(mse_lst, 0.75), 4)}"
                    f"\nMAE: {round(np.mean(mae_lst), 4)}, Q1: {round(np.quantile(mae_lst, 0.25), 4)}, Q2: {round(np.quantile(mae_lst, 0.5), 4)}, Q3: {round(np.quantile(mae_lst, 0.75), 4)}"
                    f"\nDELTA_E: {round(np.mean(deltaE2000_lst), 7)}, Q1: {round(np.quantile(deltaE2000_lst, 0.25), 7)}, Q2: {round(np.quantile(deltaE2000_lst, 0.5), 7)}, Q3: {round(np.quantile(deltaE2000_lst, 0.75), 7)}"
                )
        log.info(f"Average time elapsed: {np.mean(time_lst)}")


    def mean_angular_error(self, a, b):
        """Calculate mean angular error (via cosine similarity)."""
        def angular_error(a, b):
            radians_to_degrees = 180.0 / np.pi

            ab = np.sum(np.multiply(a, b), axis=1)
            a_norm = np.linalg.norm(a, axis=1)
            b_norm = np.linalg.norm(b, axis=1)

            # Avoid zero-values (to avoid NaNs)
            a_norm = np.clip(a_norm, a_min=1e-7, a_max=None)
            b_norm = np.clip(b_norm, a_min=1e-7, a_max=None)

            similarity = np.divide(ab, np.multiply(a_norm, b_norm))

            return np.arccos(similarity) * radians_to_degrees
        return np.nanmean(angular_error(a, b))

    def load_checkpoints(self, ckpt_path):
        checkpoints = torch.load(ckpt_path)
        self.dncm_to_canon.load_state_dict(checkpoints["dncm_to_canon"])
        self.dncm_wo_fusion.load_state_dict(checkpoints["dncm_wo_fusion"])
        self.awb_encoder.load_state_dict(checkpoints["awb_encoder"])
    
    def to_cuda(self):
        log.info(f"GPU ID: {torch.cuda.current_device()}")
        self.dncm_to_canon = self.dncm_to_canon.cuda()
        self.dncm_wo_fusion = self.dncm_wo_fusion.cuda()
        self.awb_encoder = self.awb_encoder.cuda()