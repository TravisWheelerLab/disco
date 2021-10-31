import torch
import torchmetrics
import pytorch_lightning as pl
from torch import nn

__all__ = ["UNet1DAttn"]
# TODO: add mask character to something global
MASK_CHARACTER = -1


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, filter_width):
        super(ConvBlock, self).__init__()

        if filter_width % 2 == 1:
            pad_width = (filter_width - 1) // 2
        else:
            pad_width = filter_width // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=filter_width,
            padding=1 if filter_width == 3 else pad_width,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=filter_width,
            padding=1 if filter_width == 3 else pad_width,
        )

        self.bn2 = nn.BatchNorm1d(out_channels)

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.bn2(self.conv1(x)))
        x = self.act(self.bn1(self.conv2(x) + x))
        return x


class UNet1DAttn(pl.LightningModule):
    """
    Since we're convolving over the time dimension, the application of attention is somewhat
    redundant. Attention is usually applied to get the model to look at long-range dependencies,
    and we get that (albeit in a less dense way) through the successive application of convolutions and
    downsampling. I think this might shine when applied to temporal stacks of images; each channel of
    activation maps post 2D conv would be considered separate tokens.
    """

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
        divisible_by=16,
    ):

        super(UNet1DAttn, self).__init__()
        self.in_channels = in_channels
        self.filter_width = 3
        self._setup_layers()
        self.divisible_by = divisible_by
        self.accuracy = torchmetrics.Accuracy()
        self.learning_rate = learning_rate
        self.mel = mel
        self.apply_log = apply_log
        self.n_fft = n_fft
        self.vertical_trim = vertical_trim
        self.mask_beginning_and_end = mask_beginning_and_end
        self.begin_mask = begin_mask
        self.end_mask = end_mask
        self.train_files = list(train_files)
        self.val_files = list(val_files)
        self.initial_power = 5

        self.save_hyperparameters()

    def _setup_layers(self):
        base = 2
        power = 5
        self.conv1 = ConvBlock(self.in_channels, base ** power, self.filter_width)
        self.a1 = nn.TransformerEncoderLayer(
            base ** power, 1, dim_feedforward=base ** (power + 1)
        )

        self.conv2 = ConvBlock(base ** power, base ** (power + 1), self.filter_width)
        self.a2 = nn.TransformerEncoderLayer(
            base ** (power + 1), 1, dim_feedforward=base ** (power + 1)
        )

        self.conv3 = ConvBlock(
            base ** (power + 1), base ** (power + 2), self.filter_width
        )
        self.a3 = nn.TransformerEncoderLayer(
            base ** (power + 2), 1, dim_feedforward=base ** (power + 1)
        )

        self.conv4 = ConvBlock(
            base ** (power + 2), base ** (power + 3), self.filter_width
        )
        self.a4 = nn.TransformerEncoderLayer(
            base ** (power + 3), 1, dim_feedforward=base ** (power + 1)
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
        self.conv8 = ConvBlock(base ** (power + 1), base ** power, self.filter_width)
        self.conv9 = ConvBlock(base ** power, base ** power, self.filter_width)

        self.conv_out = nn.Conv1d(base ** power, 3, kernel_size=1, padding=0)
        self.act = nn.ReLU()
        self.downsample = nn.MaxPool2d(kernel_size=(1, 2))
        self.upsample = nn.Upsample(scale_factor=2)

    def _masked_forward(self, x, x_mask):
        x1 = self.conv1(x)
        d1 = self.downsample(x1)

        d1 = d1.transpose(-1, -2)
        d1 = self.a1(d1) + d1
        d1 = d1.transpose(-1, -2)

        x2 = self.conv2(d1)
        d2 = self.downsample(x2)

        d2 = d2.transpose(-1, -2)
        d2 = self.a2(d2) + d2
        d2 = d2.transpose(-1, -2)

        x3 = self.conv3(d2)
        d3 = self.downsample(x3)

        d3 = d3.transpose(-1, -2)
        d3 = self.a3(d3) + d3
        d3 = d3.transpose(-1, -2)

        x4 = self.conv4(d3)
        d4 = self.downsample(x4)

        d4 = d4.transpose(-1, -2)
        d4 = self.a4(d4) + d4
        d4 = d4.transpose(-1, -2)

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
        x[x_mask.expand(-1, 3, -1)] = 0
        return torch.nn.functional.log_softmax(x)

    def _forward(self, x):
        x1 = self.conv1(x)
        d1 = self.downsample(x1)

        d1 = d1.transpose(-1, -2)
        d1 = self.a1(d1) + d1
        d1 = d1.transpose(-1, -2)

        x2 = self.conv2(d1)
        d2 = self.downsample(x2)

        d2 = d2.transpose(-1, -2)
        d2 = self.a2(d2) + d2
        d2 = d2.transpose(-1, -2)

        x3 = self.conv3(d2)
        d3 = self.downsample(x3)

        d3 = d3.transpose(-1, -2)
        d3 = self.a3(d3) + d3
        d3 = d3.transpose(-1, -2)

        x4 = self.conv4(d3)
        d4 = self.downsample(x4)

        d4 = d4.transpose(-1, -2)
        d4 = self.a4(d4) + d4
        d4 = d4.transpose(-1, -2)

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
        return torch.nn.functional.log_softmax(x)

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

        loss = torch.nn.functional.nll_loss(logits, y, ignore_index=MASK_CHARACTER)
        preds = logits.argmax(dim=1)
        preds = preds[y != MASK_CHARACTER]
        labels = y[y != MASK_CHARACTER]
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
                optimizer, step_size=150, gamma=0.5
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
        self.log("hp_metric", self.learning_rate + self.n_fft)

    def validation_epoch_end(self, outputs):
        val_loss = self.all_gather([x["val_loss"] for x in outputs])
        val_acc = self.all_gather([x["val_acc"] for x in outputs])
        val_loss = torch.mean(torch.stack(val_loss))
        val_acc = torch.mean(torch.stack(val_acc))
        self.log("val_loss", val_loss)
        self.log("val_acc", val_acc)


if __name__ == "__main__":
    # batch size by channels (feature depth) by sequence length
    model = UNet1DAttn(
        in_channels=128,
        learning_rate=None,
        mel=None,
        apply_log=None,
        n_fft=None,
        vertical_trim=None,
        mask_beginning_and_end=None,
        begin_mask=None,
        end_mask=None,
        train_files=[1],
        val_files=[1],
    )

    for i in range(48, 500, 1):
        random_data = torch.rand((16, 128, i))
        random_labels = torch.rand((16, i))
        random_labels[random_labels < 0.3] = 0
        random_labels[random_labels != 0] = 1
        a = model._shared_step([random_data, random_labels.long()])
        break
