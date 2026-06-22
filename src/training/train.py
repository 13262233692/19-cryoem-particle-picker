import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import time
from src.utils.logging import get_logger
from src.utils.config import load_config, get_config
from src.utils.metrics import compute_dice_coefficient, compute_precision_recall
from src.modeling.segmentation import LightResUNet_MSA
from src.training.dataset import MRCDataModule
from src.training.loss import DiceCELoss, DeepSupervisionLoss

logger = get_logger("training.train")

class Trainer:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = load_config(config_path)
        self.device = torch.device(self.config["system"]["device"] if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.model = LightResUNet_MSA(
            in_channels=self.config["model"]["in_channels"],
            out_channels=self.config["model"]["out_channels"],
            base_channels=self.config["model"]["base_channels"],
            num_res_blocks=self.config["model"]["num_res_blocks"],
            attention_heads=self.config["model"]["attention_heads"],
            attention_channels=self.config["model"]["attention_channels"],
            dropout_rate=self.config["model"]["dropout_rate"],
            use_deep_supervision=self.config["model"]["use_deep_supervision"]
        ).to(self.device)
        base_loss = DiceCELoss(
            dice_weight=self.config["training"]["loss"]["dice_weight"],
            ce_weight=self.config["training"]["loss"]["ce_weight"]
        )
        if self.config["model"]["use_deep_supervision"]:
            self.criterion = DeepSupervisionLoss(base_loss)
        else:
            self.criterion = base_loss
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"]
        )
        warmup_epochs = self.config["training"]["warmup_epochs"]
        num_epochs = self.config["training"]["num_epochs"]
        warmup_scheduler = LinearLR(
            self.optimizer, start_factor=0.01, end_factor=1.0,
            total_iters=warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-6
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
        self.checkpoint_dir = self.config["training"]["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.writer = SummaryWriter(os.path.join(self.config["system"]["log_dir"], "tensorboard"))
        self.best_dice = 0.0
        self.start_epoch = 0
        self.datamodule: Optional[MRCDataModule] = None

    def setup_data(self, data_dir: str) -> None:
        preproc_config = self.config["preprocessing"]
        self.datamodule = MRCDataModule(
            data_dir=data_dir,
            batch_size=self.config["training"]["batch_size"],
            num_workers=self.config["system"]["num_workers"],
            patch_size=self.config["preprocessing"]["patch_size"],
            val_split=self.config["training"]["val_split"],
            preprocessing_config={
                "clahe_clip_limit": preproc_config["clahe"]["clip_limit"],
                "clahe_tile_grid": tuple(preproc_config["clahe"]["tile_grid_size"]),
                "bandpass_low_sigma": preproc_config["bandpass_filter"]["low_sigma"],
                "bandpass_high_sigma": preproc_config["bandpass_filter"]["high_sigma"],
                "percentile_low": preproc_config["normalize"]["percentile_low"],
                "percentile_high": preproc_config["normalize"]["percentile_high"],
            }
        )

    def train_epoch(self, epoch: int, train_loader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_dice = 0.0
        total_precision = 0.0
        total_recall = 0.0
        num_batches = len(train_loader)
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            if isinstance(outputs, list):
                outputs = outputs[-1]
            if outputs.shape[1] > 1:
                probs = torch.softmax(outputs, dim=1)[:, 1, :, :]
            else:
                probs = torch.sigmoid(outputs).squeeze(1)
            preds_np = (probs > 0.5).cpu().numpy()
            labels_np = labels.cpu().numpy()
            batch_dice = 0.0
            batch_precision = 0.0
            batch_recall = 0.0
            for i in range(preds_np.shape[0]):
                dice = compute_dice_coefficient(preds_np[i], labels_np[i])
                prec, rec, _ = compute_precision_recall(preds_np[i], labels_np[i])
                batch_dice += dice
                batch_precision += prec
                batch_recall += rec
            batch_dice /= preds_np.shape[0]
            batch_precision /= preds_np.shape[0]
            batch_recall /= preds_np.shape[0]
            total_loss += loss.item()
            total_dice += batch_dice
            total_precision += batch_precision
            total_recall += batch_recall
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "dice": f"{batch_dice:.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            global_step = (epoch - 1) * num_batches + batch_idx
            if global_step % self.config["training"]["log_interval"] == 0:
                self.writer.add_scalar("Train/Loss", loss.item(), global_step)
                self.writer.add_scalar("Train/Dice", batch_dice, global_step)
                self.writer.add_scalar("Train/LR", self.optimizer.param_groups[0]["lr"], global_step)
        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        avg_precision = total_precision / num_batches
        avg_recall = total_recall / num_batches
        self.writer.add_scalar("Train/Avg_Loss", avg_loss, epoch)
        self.writer.add_scalar("Train/Avg_Dice", avg_dice, epoch)
        return {
            "loss": avg_loss,
            "dice": avg_dice,
            "precision": avg_precision,
            "recall": avg_recall
        }

    @torch.no_grad()
    def validate(self, epoch: int, val_loader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        total_precision = 0.0
        total_recall = 0.0
        num_batches = len(val_loader)
        pbar = tqdm(val_loader, desc=f"Val Epoch {epoch}")
        for batch in pbar:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            if isinstance(outputs, list):
                outputs = outputs[-1]
            if outputs.shape[1] > 1:
                probs = torch.softmax(outputs, dim=1)[:, 1, :, :]
            else:
                probs = torch.sigmoid(outputs).squeeze(1)
            preds_np = (probs > 0.5).cpu().numpy()
            labels_np = labels.cpu().numpy()
            batch_dice = 0.0
            batch_precision = 0.0
            batch_recall = 0.0
            for i in range(preds_np.shape[0]):
                dice = compute_dice_coefficient(preds_np[i], labels_np[i])
                prec, rec, _ = compute_precision_recall(preds_np[i], labels_np[i])
                batch_dice += dice
                batch_precision += prec
                batch_recall += rec
            batch_dice /= preds_np.shape[0]
            batch_precision /= preds_np.shape[0]
            batch_recall /= preds_np.shape[0]
            total_loss += loss.item()
            total_dice += batch_dice
            total_precision += batch_precision
            total_recall += batch_recall
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "dice": f"{batch_dice:.4f}"
            })
        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        avg_precision = total_precision / num_batches
        avg_recall = total_recall / num_batches
        self.writer.add_scalar("Val/Avg_Loss", avg_loss, epoch)
        self.writer.add_scalar("Val/Avg_Dice", avg_dice, epoch)
        self.writer.add_scalar("Val/Avg_Precision", avg_precision, epoch)
        self.writer.add_scalar("Val/Avg_Recall", avg_recall, epoch)
        return {
            "loss": avg_loss,
            "dice": avg_dice,
            "precision": avg_precision,
            "recall": avg_recall
        }

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float],
                        is_best: bool = False) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "best_dice": self.best_dice,
            "config": self.config
        }
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path} (Dice: {metrics['dice']:.4f})")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "epoch" in checkpoint:
            self.start_epoch = checkpoint["epoch"] + 1
        if "best_dice" in checkpoint:
            self.best_dice = checkpoint["best_dice"]
        logger.info(f"Loaded checkpoint: {checkpoint_path} (epoch {self.start_epoch - 1})")

    def train(self, data_dir: str, num_epochs: Optional[int] = None,
              resume_checkpoint: Optional[str] = None) -> None:
        if num_epochs is None:
            num_epochs = self.config["training"]["num_epochs"]
        if resume_checkpoint:
            self.load_checkpoint(resume_checkpoint)
        self.setup_data(data_dir)
        train_loader = self.datamodule.train_dataloader()
        val_loader = self.datamodule.val_dataloader()
        logger.info(f"Starting training from epoch {self.start_epoch} to {num_epochs}")
        logger.info(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
        for epoch in range(self.start_epoch, num_epochs + 1):
            epoch_start = time.time()
            train_metrics = self.train_epoch(epoch, train_loader)
            val_metrics = self.validate(epoch, val_loader)
            self.scheduler.step()
            epoch_time = time.time() - epoch_start
            logger.info(f"Epoch {epoch}/{num_epochs} | "
                       f"Train Loss: {train_metrics['loss']:.4f} | "
                       f"Train Dice: {train_metrics['dice']:.4f} | "
                       f"Val Loss: {val_metrics['loss']:.4f} | "
                       f"Val Dice: {val_metrics['dice']:.4f} | "
                       f"Time: {epoch_time:.2f}s")
            is_best = val_metrics["dice"] > self.best_dice
            if is_best:
                self.best_dice = val_metrics["dice"]
            if epoch % self.config["training"]["save_interval"] == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics, is_best)
        logger.info(f"Training complete! Best Dice: {self.best_dice:.4f}")
        self.writer.close()

def main():
    parser = argparse.ArgumentParser(description="Train Cryo-EM Particle Picker")
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Path to data directory")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to config file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of epochs")
    args = parser.parse_args()
    trainer = Trainer(config_path=args.config)
    trainer.train(args.data_dir, num_epochs=args.epochs,
                  resume_checkpoint=args.resume)

if __name__ == "__main__":
    main()
