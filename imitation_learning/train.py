import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torchvision import transforms as tf
import wandb
import numpy as np

from dataset import KITTIDataset
import custom_transforms as ctf
from model import E2EModel
from utils import batch_tensor_to_array, display_path


class KITTIDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        train_img_augs = tf.Compose([
            tf.Resize(self.cfg["image_size"]),
            tf.GaussianBlur(self.cfg["kernel_size"], self.cfg["sigma"]),
            tf.ColorJitter(self.cfg["brightness"], self.cfg["contrast"], self.cfg["saturation"], self.cfg["hue"]),
            tf.Normalize(mean=self.cfg["norm_mean"], std=self.cfg["norm_std"])
        ])
        train_full_augs = tf.Compose([
            ctf.RandomHorizontalFlip(0.5)
        ])
        val_img_augs = tf.Compose([
            tf.Resize(self.cfg["image_size"]),
            tf.Normalize(mean=self.cfg["norm_mean"], std=self.cfg["norm_std"])
        ])

        self.train_set = KITTIDataset(
            self.cfg["root_dir"],
            "train",
            self.cfg["horizon"],
            self.cfg["num_frames"],
            img_transforms=train_img_augs,
            full_transforms=train_full_augs
        )
        self.val_set = KITTIDataset(
            self.cfg["root_dir"],
            "val",
            self.cfg["horizon"],
            self.cfg["num_frames"],
            img_transforms=val_img_augs
        )
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.cfg["batch_size"],
            num_workers=self.cfg["num_workers"],
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
        
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_set,
            batch_size=self.cfg["batch_size"],
            num_workers=self.cfg["num_workers"],
            shuffle=True,
            pin_memory=True
        )


class LitE2EModel(pl.LightningModule):
    def __init__(self, cfg, steps_per_epoch):
        super().__init__()
        self.cfg = cfg
        self.steps_per_epoch = steps_per_epoch
        self.model = E2EModel(cfg)

    def training_step(self, sample_batched, batch_idx):
        frames = sample_batched['frames']
        label_path = sample_batched['label_path']

        model_output = self.model(frames) # (B, horizon, 2)
        loss = F.l1_loss(label_path, model_output)

        self.log('train_loss', loss, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, sample_batched, batch_idx):
        if self.global_step == 0: 
            wandb.define_metric('val_loss', summary='min')

        frames = sample_batched['frames']
        label_path = sample_batched['label_path']

        model_output = self.model(frames) # (B, horizon, 2)
        loss = F.l1_loss(label_path, model_output)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True)

        # Log images of the predictions once per epoch
        if batch_idx == 0:
            num_imgs = self.cfg['num_log_imgs']
            img_tensor = frames[:num_imgs, 3:].detach().cpu()
            img_tensor = ctf.Denormalize()(img_tensor)
            img_np = batch_tensor_to_array(img_tensor)
            label_path_np = label_path[:num_imgs].detach().cpu().numpy()
            pred_path_np = model_output[:num_imgs].detach().cpu().numpy()

            viz_imgs = []
            for i in range(num_imgs):
                # Display label in green and pred in red
                viz_img = display_path(img_np[i], label_path_np[i], color=(0, 255, 0))
                viz_img = display_path(viz_img, pred_path_np[i], color=(255, 0, 0))
                viz_imgs.append(viz_img)

            self.logger.log_image(key='val_img_viz', images=viz_imgs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            self.cfg['lr'],
            self.cfg['betas'],
            self.cfg['eps'],
            self.cfg['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.cfg['max_lr'],
            epochs=self.cfg['epochs'],
            steps_per_epoch=self.steps_per_epoch
        )
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        # Lowers memory use and improves performance
        optimizer.zero_grad(set_to_none=True)


def main(cfg):
    pl.seed_everything(0)
    
    # Prep logger and callbacks
    wandb_logger = pl.loggers.WandbLogger(project="cs-885-imitation-learning", entity="methier")
    wandb_logger.experiment.config.update(cfg)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step", log_momentum=True)

    # Create trainer
    dm = KITTIDataModule(cfg)
    steps_per_epoch = int(len(dm.train_set)/cfg['batch_size'])
    model = LitE2EModel(cfg, steps_per_epoch)
    trainer = pl.Trainer(
        devices=[1],
        accelerator="gpu",
        precision=16,
        amp_backend="native",
        benchmark=True,
        max_epochs=cfg["epochs"],
        log_every_n_steps=cfg["log_steps"],
        logger=wandb_logger,
        callbacks=[lr_monitor]
    )
    trainer.fit(model, datamodule=dm)

if __name__=="__main__":
    cfg = {
        # Dataset
        "root_dir": "~/repos/neural-scene-graphs/data/kitti",
        "horizon": 50,
        "num_frames": 2,
        "image_size": [256, 256],

        # Training
        "epochs": 15,
        "log_steps": 20,
        "num_log_imgs": 4,
        "batch_size": 64,
        "num_workers": 12,

        # Augmentations
        "norm_mean": [0.485, 0.456, 0.406],
        "norm_std": [0.229, 0.224, 0.225],
        "kernel_size": [3, 5],
        "sigma": [0.01, 2],
        "brightness": 0.8,
        "contrast": 0.7,
        "saturation": 0.8,
        "hue": 0.3,

        # Model
        "name": "regnety_032",
        "timm_feat_size": 1512, # Depends on model
        "fc_size": 128,
        "output_size": 50*2,
        "dropout_prob": 0.3,
        
        # AdamW parameters
        "lr": 1e-3,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 8e-3,

        # OneCycle parameters
        "max_lr": 1e-2,
    }
    main(cfg)