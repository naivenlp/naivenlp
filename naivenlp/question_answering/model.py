import pytorch_lightning as pl
import torch
import torchmetrics
from transformers import BertForQuestionAnswering


def unpack_data(inputs):
    if len(inputs) == 1:
        return inputs, None
    if len(inputs) == 2:
        return inputs[0], inputs[1]
    return inputs[0], None


class LitBertForQuestionAnswering(pl.LightningModule):
    """ """

    def __init__(self, model_name_or_path, lr=3e-5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.model = BertForQuestionAnswering.from_pretrained(model_name_or_path)
        self.train_start_accuracy = torchmetrics.Accuracy()
        self.train_end_accuracy = torchmetrics.Accuracy()
        self.valid_start_accuracy = torchmetrics.Accuracy()
        self.valid_end_accuracy = torchmetrics.Accuracy()

    @property
    def example_input_array(self):
        inputs = {
            "input_ids": torch.range(0, 127).view([1, 128]).long(),
            "segment_ids": torch.zeros([1, 128]).long(),
            "attention_mask": torch.ones([1, 128]).long(),
        }
        return inputs

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def forward(self, input_ids, segment_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            token_type_ids=segment_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        start, end = outputs["start_logits"], outputs["end_logits"]
        return {"start": torch.argmax(start, dim=-1), "end": torch.argmax(end, dim=-1)}

    def training_step(self, batch, batch_index):
        x, y = unpack_data(batch)

        outputs = self.model(
            input_ids=x["input_ids"],
            token_type_ids=x["segment_ids"],
            attention_mask=x["attention_mask"],
            start_positions=y["start_positions"],
            end_positions=y["end_positions"],
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        start_logits = torch.argmax(outputs["start_logits"], dim=-1)
        end_logits = torch.argmax(outputs["end_logits"], dim=-1)
        train_start_acc = self.train_start_accuracy(start_logits, y["start_positions"])
        train_end_acc = self.train_end_accuracy(end_logits, y["end_positions"])
        self.log("start_acc", train_start_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("end_acc", train_end_acc, on_step=True, on_epoch=True, prog_bar=True)
        return outputs["loss"]

    def validation_step(self, batch, batch_index):
        x, y = unpack_data(batch)

        outputs = self.model(
            input_ids=x["input_ids"],
            token_type_ids=x["segment_ids"],
            attention_mask=x["attention_mask"],
            start_positions=y["start_positions"],
            end_positions=y["end_positions"],
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        start_logits = torch.argmax(outputs["start_logits"], dim=-1)
        end_logits = torch.argmax(outputs["end_logits"], dim=-1)
        valid_start_acc = self.valid_start_accuracy(start_logits, y["start_positions"])
        valid_end_acc = self.valid_end_accuracy(end_logits, y["end_positions"])
        self.log("valid_start_acc", valid_start_acc, on_step=True, on_epoch=True, prog_bar=False)
        self.log("valid_end_acc", valid_end_acc, on_step=True, on_epoch=True, prog_bar=False)
        return outputs["loss"]

    def validation_epoch_end(self, outputs) -> None:
        self.log("start_acc", self.valid_start_accuracy.compute(), prog_bar=True)
        self.log("end_acc", self.valid_end_accuracy.compute(), prog_bar=True)
