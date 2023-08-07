import os
import yaml
import torch
import glog as log
import wandb
from pathlib import Path
from tqdm import tqdm
from torch.utils import data
from torchvision import transforms

from modeling.denim import DeNIM_to_Canon, DeNIM_wo_Fusion, AWBEncoder
from dataset import dataset_getter, transform_getter


class AWBCorrectionTrainer:
    def __init__(self, cfg) -> None:
        self._setup(cfg)
        self._init_parameters()
        
        self.wandb = wandb
        self.wandb.init(
            project=self.PROJECT_NAME,
            resume=self.INIT_FROM is not None, 
            notes=str(self.LOG_DIR), 
            config=self.cfg, 
            entity=self.ENTITY
        )
        
        self.transform = transform_getter(self.DATASET_NAME, self.IMG_SIZE)
        self.dataset = dataset_getter(self.DATASET_NAME)(root=self.DATASET_ROOT, transform=self.transform)
        self.image_loader = data.DataLoader(dataset=self.dataset, batch_size=self.BATCH_SIZE, shuffle=self.SHUFFLE, num_workers=self.NUM_WORKERS)
        self.to_pil = transforms.ToPILImage()

        self.dncm_to_canon = DeNIM_to_Canon(self.k, self.ch).cuda()
        self.dncm_wo_fusion = DeNIM_wo_Fusion(self.k).cuda()
        self.awb_encoder = AWBEncoder(self.sz, self.k, self.ch, self.ps, str(self.BACKBONE_WEIGHTS_PATH), self.BACKBONE_TYPE).cuda()

        self.optimizer = torch.optim.AdamW(
            list(self.dncm_to_canon.parameters()) + list(self.dncm_wo_fusion.parameters()) + list(self.awb_encoder.parameters()),
            lr=self.LR, betas=self.BETAS  #, weight_decay=self.WEIGHT_DECAY
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.SCHEDULER_STEP, gamma=self.SCHEDULER_GAMMA)
       
        self.init_epoch = 0
        self.current_epoch = 0
        if self.INIT_FROM is not None and self.INIT_FROM != "":
            log.info("Checkpoints loading from ckpt file...")
            self.load_checkpoints(self.INIT_FROM)
        self.check_and_use_multi_gpu()
        self.l2_loss = torch.nn.MSELoss().cuda()
    
    def _setup(self, cfg):
        with open(cfg, 'r') as stream:
            self.cfg = yaml.safe_load(stream)
    
    def _init_parameters(self):
        self.k = int(self.cfg["k"])
        self.sz = int(self.cfg["sz"])
        self.ch = int(self.cfg["ch"])
        self.ps = int(self.cfg["ps"])
        self.LR = float(self.cfg["LR"])
        self.BETAS = (self.cfg["BETA1"], self.cfg["BETA2"])
        self.WEIGHT_DECAY = float(self.cfg["WEIGHT_DECAY"])
        self.NUM_GPU = int(self.cfg["NUM_GPU"])
        self.DATASET_NAME = str(self.cfg["DATASET_NAME"])
        self.DATASET_ROOT = Path(self.cfg["DATASET_ROOT"])
        self.BACKBONE_WEIGHTS_PATH = Path(self.cfg["BACKBONE_WEIGHTS_PATH"])
        self.BACKBONE_TYPE = str(self.cfg["BACKBONE_TYPE"])
        self.IMG_SIZE = int(self.cfg["IMG_SIZE"])
        self.BATCH_SIZE = int(self.cfg["BATCH_SIZE"])
        self.EPOCHS = int(self.cfg["EPOCHS"])
        self.LAMBDA = float(self.cfg["LAMBDA"])
        self.SCHEDULER_STEP = int(self.cfg["SCHEDULER_STEP"])
        self.SCHEDULER_GAMMA = float(self.cfg["SCHEDULER_GAMMA"])
        self.VISUALIZE_STEP = int(self.cfg["VISUALIZE_STEP"])
        self.SHUFFLE = bool(self.cfg["SHUFFLE"])
        self.NUM_WORKERS = int(self.cfg["NUM_WORKERS"])
        self.CKPT_DIR = Path(self.cfg["CKPT_DIR"])
        self.INIT_FROM = self.cfg["INIT_FROM"]
        self.PROJECT_NAME = self.cfg["PROJECT_NAME"]
        self.LOG_DIR = Path(self.cfg["LOG_DIR"])
        self.ENTITY = self.cfg["ENTITY"]
                
    def __call__(self):
        self.run()
    
    def run(self):
        for e in range(self.init_epoch, self.EPOCHS):
            self.current_epoch = e
            log.info(f"Epoch {self.current_epoch+1}/{self.EPOCHS}")
            for step, (GT, D, T, F, C, S) in enumerate(tqdm(self.image_loader, total=len(self.image_loader))):
                self.optimizer.zero_grad()
                
                GT = GT.float().cuda()
                D = D.float().cuda()
                T = T.float().cuda()
                S = S.float().cuda()
                if self.ch == 9:
                    I = torch.cat([D,S,T], dim=1)
                else:
                    F = F.float().cuda()
                    C = C.float().cuda()
                    I = torch.cat([D,T,F,C,S], dim=1)
                
                d = self.awb_encoder(I)
                canon = self.dncm_to_canon(I, d)
                out = self.dncm_wo_fusion(canon)
                
                loss = self.l2_loss(out, GT)
                loss.backward()
                self.optimizer.step()
                self.wandb.log({
                    "l2_loss": loss.item()
                }, commit=False)
                if step % self.VISUALIZE_STEP == 0 and step != 0:
                    self.visualize(GT, canon, out)
                else:
                    self.wandb.log({})
            self.scheduler.step()
            self.do_checkpoint()
    
    def visualize(self, GT, canon, out):
        amplify = lambda im, val: torch.clip(im * val, min=0., max=1.)
        idx = 0
        self.wandb.log({"examples": [
            self.wandb.Image(self.to_pil(GT[idx].cpu()), caption="GT"),
            self.wandb.Image(self.to_pil(torch.clamp(out, min=0., max=1.)[idx].cpu()), caption="AWB Corrected"),
            self.wandb.Image(self.to_pil(torch.clamp(amplify(canon, 5.), min=0., max=1.)[idx].cpu()), caption="Canonical")
        ]}, commit=False)

    def check_and_use_multi_gpu(self):
        if torch.cuda.device_count() > 1 and self.NUM_GPU > 1:
            log.info(f"Using {torch.cuda.device_count()} GPUs...")
            self.dncm_to_canon = torch.nn.DataParallel(self.dncm_to_canon).cuda()
            self.dncm_wo_fusion = torch.nn.DataParallel(self.dncm_wo_fusion).cuda()
            self.awb_encoder = torch.nn.DataParallel(self.awb_encoder).cuda()
        else:
            log.info(f"GPU ID: {torch.cuda.current_device()}")
            self.dncm_to_canon = self.dncm_to_canon.cuda()
            self.dncm_wo_fusion = self.dncm_wo_fusion.cuda()
            self.awb_encoder = self.awb_encoder.cuda()

    def do_checkpoint(self):
        os.makedirs(str(self.CKPT_DIR), exist_ok=True)
        checkpoint = {
            'epoch': self.current_epoch,
            'dncm_to_canon': self.dncm_to_canon.module.state_dict() if isinstance(self.dncm_to_canon, torch.nn.DataParallel) else self.dncm_to_canon.state_dict(),
            'dncm_wo_fusion': self.dncm_wo_fusion.module.state_dict() if isinstance(self.dncm_wo_fusion, torch.nn.DataParallel) else self.dncm_wo_fusion.state_dict(),
            'awb_encoder': self.awb_encoder.module.state_dict() if isinstance(self.awb_encoder, torch.nn.DataParallel) else self.awb_encoder.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(checkpoint, str(self.CKPT_DIR / f"{self.BACKBONE_TYPE}_{self.ps}_{self.ch}_latest.pth"))
    
    def load_checkpoints(self, ckpt_path):
        checkpoints = torch.load(ckpt_path)
        self.dncm_to_canon.load_state_dict(checkpoints["dncm_to_canon"])
        self.dncm_wo_fusion.load_state_dict(checkpoints["dncm_wo_fusion"])
        self.awb_encoder.load_state_dict(checkpoints["awb_encoder"])
        self.optimizer.load_state_dict(checkpoints["optimizer"])
        self.init_epoch = checkpoints["epoch"]
        self.optimizers_to_cuda()

    def optimizers_to_cuda(self):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
