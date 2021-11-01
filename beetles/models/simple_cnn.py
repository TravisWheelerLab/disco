import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import torchmetrics
import numpy as np

MASK_CHARACTER = -1

__all__ = ["MASK_CHARACTER", "SimpleCNN"]


class SimpleCNN(pl.LightningModule):
    def __init__(
        self,
        in_channels,
        learning_rate,
        mel,
        apply_log,
        n_fft,
        vertical_trim,
        mask_beginning_and_end,
        begin_mask,
        end_mask,
        train_files,
        val_files,
    ):

        super(SimpleCNN, self).__init__()
        self._setup_layers()
        self.accuracy = torchmetrics.Accuracy()
        self.learning_rate = learning_rate
        self.mel = mel
        self.apply_log = apply_log
        self.n_fft = n_fft
        self.vertical_trim = vertical_trim
        self.in_channels = in_channels
        self.mask_beginning_and_end = mask_beginning_and_end
        self.begin_mask = begin_mask
        self.end_mask = end_mask
        self.train_files = list(train_files)
        self.val_files = list(val_files)

        self.save_hyperparameters()

    def _setup_layers(self):
        self.conv1 = torch.nn.Conv1d(
            in_channels=in_channels, out_channels=256, kernel_size=3, padding=1
        )
        self.conv2 = torch.nn.Conv1d(256, 512, 3, padding=1)
        self.conv3 = torch.nn.Conv1d(512, 512, 3, padding=1)
        self.conv4 = torch.nn.Conv1d(512, 1024, 3, padding=1)
        self.conv5 = torch.nn.Conv1d(1024, 512, 1, padding=0)
        self.conv6 = torch.nn.Conv1d(512, 512, 3, padding=1)
        self.conv7 = torch.nn.Conv1d(512, 256, 3, padding=1)
        self.conv8 = torch.nn.Conv1d(
            in_channels=256, out_channels=32, kernel_size=1, padding=0
        )
        self.conv9 = torch.nn.Conv1d(
            in_channels=32, out_channels=3, kernel_size=1, padding=0
        )

    def _masked_forward(self, x, x_mask):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.conv3(x)
        x = torch.nn.functional.relu(x)
        x = self.conv4(x)
        x = torch.nn.functional.relu(x)
        x = self.conv5(x)
        x = torch.nn.functional.relu(x)
        x = self.conv6(x)
        x = torch.nn.functional.relu(x)
        x = self.conv7(x)
        x = torch.nn.functional.relu(x)
        x = self.conv8(x)
        x = torch.nn.functional.relu(x)
        x = self.conv9(x)
        x[x_mask.expand(-1, 3, -1)] = 0
        output = torch.nn.functional.log_softmax(x, dim=1)
        return output

    def _forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.conv3(x)
        x = torch.nn.functional.relu(x)
        x = self.conv4(x)
        x = torch.nn.functional.relu(x)
        x = self.conv5(x)
        x = torch.nn.functional.relu(x)
        x = self.conv6(x)
        x = torch.nn.functional.relu(x)
        x = self.conv7(x)
        x = torch.nn.functional.relu(x)
        x = self.conv8(x)
        x = torch.nn.functional.relu(x)
        x = self.conv9(x)
        output = torch.nn.functional.log_softmax(x, dim=1)
        return output

    def forward(self, x, mask=None):
        if mask is not None:
            logits = self._masked_forward(x, mask)
        else:
            logits = self._forward(x)

        return logits

    def _shared_step(self, batch):
        if len(batch) == 3:
            x, x_mask, y = batch
            logits = self._masked_forward(x, x_mask)
            loss = F.nll_loss(logits, y, ignore_index=MASK_CHARACTER)
            preds = logits.argmax(dim=1)
            preds = preds[y != MASK_CHARACTER]
            labels = y[y != MASK_CHARACTER]
            acc = self.accuracy(preds, labels)
        else:
            x, y = batch
            logits = self.forward(x)
            loss = F.nll_loss(logits, y, ignore_index=MASK_CHARACTER)
            preds = logits.argmax(dim=1)
            acc = self.accuracy(preds, y)

        return loss, acc

    def training_step(self, batch, batch_nb):
        loss, acc = self._shared_step(batch)
        return {"loss": loss, "train_acc": acc}

    def validation_step(self, batch, batch_nb):
        loss, acc = self._shared_step(batch)
        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_epoch_end(self, outputs):
        train_loss = self.all_gather([x["loss"] for x in outputs])
        train_acc = self.all_gather([x["train_acc"] for x in outputs])
        loss = torch.mean(torch.stack(train_loss))
        acc = torch.mean(torch.stack(train_acc))
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        self.log("learning_rate", self.learning_rate)

    def on_train_start(self):
        self.log("hp_metric", self.learning_rate + self.n_fft)

    def validation_epoch_end(self, outputs):
        val_loss = self.all_gather([x["val_loss"] for x in outputs])
        val_acc = self.all_gather([x["val_acc"] for x in outputs])
        val_loss = torch.mean(torch.stack(val_loss))
        val_acc = torch.mean(torch.stack(val_acc))
        self.log("val_loss", val_loss)
        self.log("val_acc", val_acc)
