import pytorch_lightning as pl


class BaseModel(pl.LightningModule):
    def __init__(self, args):

        for k, v in args.items():
            setattr(self, k, v)

        self._setup_layers()

    def _setup_layers(self):
        raise NotImplementedError(
            "Classes that inherit from BaseModel"
            "must implement the _setup_layers()"
            "method"
        )

    def _forward(self):
        raise NotImplementedError()

    def _masked_forward(self):
        raise NotImplementedError()

    def forward(self, x, mask=None):
        raise NotImplementedError()

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

    def train_dataloader(self):
        train_dataset = SpectrogramDatasetMultiLabel(
            self.train_files,
            apply_log=self.log,
            vertical_trim=self.vertical_trim,
            bootstrap_sample=self.bootstrap,
            mask_beginning_and_end=self.mask_beginning_and_end,
            begin_mask=self.begin_mask,
            end_mask=self.end_mask,
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=None if self.batch_size == 1 else pad_batch,
        )
        return train_loader

    def val_dataloader(self):
        val_dataset = SpectrogramDatasetMultiLabel(
            self.val_files,
            apply_log=self.log,
            vertical_trim=self.vertical_trim,
            bootstrap_sample=False,
            mask_beginning_and_end=False,
            begin_mask=None,
            end_mask=None,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=None if self.batch_size == 1 else pad_batch,
        )
        return val_loader
