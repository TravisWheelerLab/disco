import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, filter_width):
        super(ConvBlock, self).__init__()

        if filter_width % 2 == 1:
            pad_width = (filter_width - 1) // 2
        else:
            pad_width = filter_width // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=filter_width, padding=pad_width
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=filter_width, padding=pad_width
        )

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x


class UNet1D(pl.LightningModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        learning_rate,
        mask_character,
        divisible_by=16,
    ):

        super(UNet1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.learning_rate = learning_rate
        self.filter_width = 3
        self._setup_layers()
        self.divisible_by = divisible_by
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=out_channels, top_k=1
        )
        self.mask_character = mask_character
        self.final_activation = torch.nn.functional.log_softmax

        self.save_hyperparameters()

    def _setup_layers(self):
        base = 2
        power = 5
        self.conv1 = ConvBlock(self.in_channels, base**power, self.filter_width)
        self.conv2 = ConvBlock(
            base ** (power + 0), base ** (power + 1), self.filter_width
        )
        self.conv3 = ConvBlock(
            base ** (power + 1), base ** (power + 2), self.filter_width
        )
        self.conv4 = ConvBlock(
            base ** (power + 2), base ** (power + 3), self.filter_width
        )
        self.conv5 = ConvBlock(
            base ** (power + 3), base ** (power + 3), self.filter_width
        )
        self.conv6 = ConvBlock(
            base ** (power + 3), base ** (power + 2), self.filter_width
        )
        self.conv7 = ConvBlock(
            base ** (power + 2), base ** (power + 1), self.filter_width
        )
        self.conv8 = ConvBlock(
            base ** (power + 1), base ** (power + 0), self.filter_width
        )
        self.conv9 = ConvBlock(
            base ** (power + 0), base ** (power + 0), self.filter_width
        )

        self.conv_out = nn.Conv1d(
            base**power, self.out_channels, kernel_size=1, padding=0
        )
        self.act = nn.ReLU()
        self.downsample = nn.MaxPool2d(kernel_size=(1, 2))
        self.upsample = nn.Upsample(scale_factor=2)

    def _masked_forward(self, x, x_mask):
        x1 = self.conv1(x)
        d1 = self.downsample(x1)

        x2 = self.conv2(d1)
        d2 = self.downsample(x2)

        x3 = self.conv3(d2)
        d3 = self.downsample(x3)

        x4 = self.conv4(d3)
        d4 = self.downsample(x4)

        x5 = self.conv5(d4)
        u1 = self.upsample(x5) + x4

        x6 = self.conv6(u1)
        u2 = self.upsample(x6) + x3

        x7 = self.conv7(u2)
        u3 = self.upsample(x7) + x2

        x8 = self.conv8(u3)
        u4 = self.upsample(x8) + x1

        x9 = self.conv9(u4)
        x = self.conv_out(x9)
        x[x_mask.expand(-1, self.out_channels, -1)] = 0
        return x

    def _forward(self, x):
        x1 = self.conv1(x)
        d1 = self.downsample(x1)

        x2 = self.conv2(d1)
        d2 = self.downsample(x2)

        x3 = self.conv3(d2)
        d3 = self.downsample(x3)

        x4 = self.conv4(d3)
        d4 = self.downsample(x4)

        x5 = self.conv5(d4)
        u1 = self.upsample(x5) + x4

        x6 = self.conv6(u1)
        u2 = self.upsample(x6) + x3

        x7 = self.conv7(u2)
        u3 = self.upsample(x7) + x2

        x8 = self.conv8(u3)
        u4 = self.upsample(x8) + x1

        x9 = self.conv9(u4)
        x = self.conv_out(x9)
        return x

    def forward(self, x, mask=None):
        x, pad_len = self._pad_batch(x)
        if mask is not None:
            mask, _ = self._pad_batch(mask)
            logits = self._masked_forward(x, mask)
        else:
            logits = self._forward(x)

        if pad_len != 0:
            logits = logits[:, :, :-pad_len]

        return logits

    def _pad_batch(self, x):
        pad_len = (self.divisible_by - x.shape[-1]) % self.divisible_by
        if pad_len == 0:
            return x, 0
        else:
            x_out = torch.zeros((x.shape[0], x.shape[1], x.shape[2] + pad_len))
            x_out = x_out.type_as(x)
            x_out[:, :, :-pad_len] = x

            return x_out, pad_len

    def _shared_step(self, batch):

        if len(batch) == 3:
            x, x_mask, y = batch
            logits = self.forward(x, x_mask)
        else:
            x, y = batch
            logits = self.forward(x)

        logits = self.final_activation(logits, dim=1)

        loss = torch.nn.functional.nll_loss(logits, y, ignore_index=self.mask_character)
        preds = logits.argmax(dim=1)
        preds = preds[y != self.mask_character]
        labels = y[y != self.mask_character]
        acc = self.accuracy(preds, labels)
        return loss, acc

    def training_step(self, batch, batch_nb):
        loss, acc = self._shared_step(batch)
        return {"loss": loss, "train_acc": acc}

    def validation_step(self, batch, batch_nb):
        loss, acc = self._shared_step(batch)
        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=150, gamma=0.7
            ),
        }

    def training_epoch_end(self, outputs):
        train_loss = self.all_gather([x["loss"] for x in outputs])
        train_acc = self.all_gather([x["train_acc"] for x in outputs])
        loss = torch.mean(torch.stack(train_loss))
        acc = torch.mean(torch.stack(train_acc))
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        self.log("learning_rate", self.learning_rate)

    def on_train_start(self):
        self.log("hp_metric", self.learning_rate)

    def validation_epoch_end(self, outputs):
        val_loss = self.all_gather([x["val_loss"] for x in outputs])
        val_acc = self.all_gather([x["val_acc"] for x in outputs])
        val_loss = torch.mean(torch.stack(val_loss))
        val_acc = torch.mean(torch.stack(val_acc))
        self.log("val_loss", val_loss)
        self.log("val_acc", val_acc)


class WhaleUNet(UNet1D):
    def _shared_step(self, batch):

        if len(batch) == 3:
            x, x_mask, y = batch
            logits = self.forward(x, x_mask)
        else:
            x, y = batch
            logits = self.forward(x)

        # duplicate labels so we can do fully-convolutional predictions
        y = y.unsqueeze(-1).repeat(1, logits.shape[-1])
        loss = torch.nn.functional.binary_cross_entropy(
            torch.sigmoid(logits.ravel()).float(), y.ravel().float()
        )

        acc = self.accuracy(torch.round(torch.sigmoid(logits)).ravel(), y.ravel())

        if self.global_step % 500 == 0:
            with torch.no_grad():
                fig = plt.figure(figsize=(10, 10))
                plt.imshow(x[0].detach().to("cpu").numpy())
                plt.title(y[0][0].detach().to("cpu").numpy())
                plt.colorbar()
                self.logger.experiment.add_figure(
                    f"image", plt.gcf(), global_step=self.global_step
                )

        return loss, acc
