import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf


class PLWrapper(pl.LightningModule):
    def __init__(self, config: OmegaConf) -> None:
        super().__init__()
        self.config = config

    def step_loss(self, batch_data):
        raise NotImplementedError("super class method")

    def training_step(self, batch_data, batch_idx):
        # print(batch_idx)
        total_loss, log_dict = self.step_loss(batch_data)
        self.logging_metric("train", log_dict)
        return total_loss

    def validation_step(self, batch_data, batch_idx):
        total_loss, log_dict = self.step_loss(batch_data)
        self.logging_metric("val", log_dict)
        return total_loss

    def test_step(self, batch_data, batch_idx):
        total_loss, log_dict = self.step_loss(batch_data)
        self.logging_metric("test", log_dict)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.config.every_n_epoch, gamma=self.config.gamma
        )
        return [optimizer], [scheduler]

    def logging_metric(self, phase: str, metric_dict: dict):
        for key, value in metric_dict.items():
            value = value.item() if isinstance(value, torch.Tensor) else value
            self.log(
                f"{phase}/{key}",
                value,
                on_epoch=True,
                batch_size=self.config.batch_size,
            )
