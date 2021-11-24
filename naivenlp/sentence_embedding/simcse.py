import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class LitAbstractSimCSE(pl.LightningModule):
    """Abstract SimCSE model"""

    def __init__(self, model_name_or_path, lr=3e-5, temperature=0.05, negative_weight=0.2, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.bert = BertModel.from_pretrained(model_name_or_path)
        self.loss = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def _step(self, input_ids, segment_ids, attention_mask):
        embedding = self.bert(
            input_ids=input_ids,
            token_type_ids=segment_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        ).pooler_output
        return embedding

    def forward(self, x):
        return self._step(x["input_ids"], x["segment_ids"], x["attention_mask"])

    def training_step(self, batch, batch_index):
        raise NotImplementedError()

    def validation_step(self, batch, batch_index):
        raise NotImplementedError()


class LitUnsupSimCSE(LitAbstractSimCSE):
    """Unsupervised SimCSE with cls pooling strategy."""

    def _compute_loss(self, a_embedding, b_embedding, labels):
        a_embedding = F.normalize(a_embedding)
        b_embedding = F.normalize(b_embedding)
        cosine = torch.matmul(a_embedding, b_embedding.T) / self.hparams.temperature
        loss = self.loss(cosine, labels)
        return loss

    def training_step(self, batch, batch_index):
        x, _ = batch
        input_ids, segment_ids, attention_mask = x["input_ids"], x["segment_ids"], x["attention_mask"]
        a_embedding = self._step(input_ids, segment_ids, attention_mask)
        b_embedding = self._step(input_ids, segment_ids, attention_mask)
        labels = torch.range(0, input_ids.size(0))
        loss = self._compute_loss(a_embedding, b_embedding, labels)
        return loss


    def validation_step(self, batch, batch_index):
        x, _ = batch
        input_ids, segment_ids, attention_mask = x["input_ids"], x["segment_ids"], x["attention_mask"]
        a_embedding = self._step(input_ids, segment_ids, attention_mask)
        b_embedding = self._step(input_ids, segment_ids, attention_mask)
        labels = torch.range(0, input_ids.size(0))
        loss = self._compute_loss(a_embedding, b_embedding, labels)
        return loss


class LitSupervisedSimCSE(LitAbstractSimCSE):
    """Supervised SimCSE with cls pooling strategy"""

    def training_step(self, batch, batch_index):
        x, _ = batch
        a_embedding = self._step(x["input_ids"], x["segment_ids"], x["attention_mask"])
        b_embedding = self._step(x["pos_input_ids"], x["pos_segment_ids"], x["pos_attention_mask"])
        labels = torch.range(0, a_embedding.size(0))
        loss = self._compute_loss(a_embedding, b_embedding, labels)
        return loss

    def validation_step(self, batch, batch_index):
        x, _ = batch
        a_embedding = self._step(x["input_ids"], x["segment_ids"], x["attention_mask"])
        b_embedding = self._step(x["pos_input_ids"], x["pos_segment_ids"], x["pos_attention_mask"])
        labels = torch.range(0, a_embedding.size(0))
        loss = self._compute_loss(a_embedding, b_embedding, labels)
        return loss


class LitHardNegativeSimCSE(LitAbstractSimCSE):
    """Hard negative SimCSE model with cls pooling strategy"""

    def _compute_contrastive_loss(self, seq_embedding, pos_embedding, neg_embedding, labels):
        seq_embedding = F.normalize(seq_embedding)
        pos_embedding = F.normalize(pos_embedding)
        neg_embedding = F.normalize(neg_embedding)
        # cosine similarity: [batch_size, 2 * batch_size]
        pos_cosine = torch.matmul(seq_embedding, pos_embedding.T)
        neg_cosine = torch.matmul(seq_embedding, neg_embedding.T)
        cat_cosine = torch.cat([pos_cosine, neg_cosine], dim=1)
        # cosine weights: [batch_size, 2 * batch_size]
        pos_weight = torch.zeros_like(pos_cosine)
        neg_weight = torch.eye(seq_embedding.size(0)) * self.hparams.negative_weight
        cat_weight = torch.cat([pos_weight, neg_weight], dim=1)
        # final logits
        logits = (cat_cosine + cat_weight) / self.hparams.temperature
        return self.loss(logits, labels)
        
    def training_step(self, batch, batch_index):
        x, _ = batch
        seq_embedding = self._step(x["input_ids"], x["segment_ids"], x["attention_mask"])
        pos_embedding = self._step(x["pos_input_ids"], x["pos_segment_ids"], x["pos_attention_mask"])
        neg_embedding = self._step(x["neg_input_ids"], x["neg_segment_ids"], x["neg_attention_mask"])
        labels = torch.range(0, seq_embedding.size(0))
        loss = self._compute_contrastive_loss(seq_embedding, pos_embedding, neg_embedding, labels)
        return loss

    def validation_step(self, batch, batch_index):
        x, _ = batch
        a_embedding = self._step(x["input_ids"], x["segment_ids"], x["attention_mask"])
        b_embedding = self._step(x["pos_input_ids"], x["pos_segment_ids"], x["pos_attention_mask"])
        labels = torch.range(0, a_embedding.size(0))
        loss = self._compute_loss(a_embedding, b_embedding, labels)
        return loss
