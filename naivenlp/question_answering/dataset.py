import logging
from typing import List

import pytorch_lightning as pl
import torch
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import DataLoader, Dataset

from . import readers
from .example import ExampleForQuestionAnswering
from .parsers import InstanceParserForQuestionAnswering


class DatasetForQuestionAnswering(Dataset):
    """Dataset for question answering"""

    @classmethod
    def from_dureader_robust(cls, input_files, tokenizer: BertWordPieceTokenizer = None, vocab_file=None, **kwargs):
        instances = readers.read_dureader_robust(input_files, **kwargs)
        return cls.from_instances(instances, tokenizer=tokenizer, vocab_file=vocab_file, **kwargs)

    @classmethod
    def from_dureader_checklist(cls, input_files, tokenizer: BertWordPieceTokenizer = None, vocab_file=None, **kwargs):
        instances = readers.read_dureader_checklist(input_files, **kwargs)
        return cls.from_instances(instances, tokenizer=tokenizer, vocab_file=vocab_file, **kwargs)

    @classmethod
    def from_jsonl_files(cls, input_files, tokenizer: BertWordPieceTokenizer = None, vocab_file=None, **kwargs):
        instances = readers.read_jsonl_files(input_files, **kwargs)
        return cls.from_instances(instances, tokenizer=tokenizer, vocab_file=vocab_file, **kwargs)

    @classmethod
    def from_instances(
        cls,
        instances,
        tokenizer: BertWordPieceTokenizer = None,
        vocab_file=None,
        max_sequence_length=512,
        do_lower_case=True,
        **kwargs
    ):
        assert tokenizer or vocab_file, "`tokenizer` or `vocab_file` must be provided!"
        tokenizer = tokenizer or BertWordPieceTokenizer.from_file(vocab_file, lowercase=do_lower_case)
        parser = InstanceParserForQuestionAnswering.from_tokenizer(tokenizer, **kwargs)
        examples = []
        for instance in instances:
            if not instance:
                continue
            e = parser.parse(instance, **kwargs)
            if not e:
                continue
            if max_sequence_length < len(e.input_ids):
                continue
            examples.append(e)
        logging.info("Load %d examples.", len(examples))
        return cls(examples=examples, max_sequence_length=max_sequence_length, **kwargs)

    def __init__(
        self, examples: List[ExampleForQuestionAnswering], max_sequence_length=512, pad_id=0, **kwargs
    ) -> None:
        super().__init__()
        self.examples = examples
        self.max_sequence_length = max_sequence_length
        self.pad_id = pad_id

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        def _to_tensor(x):
            x = x + [self.pad_id] * (self.max_sequence_length - len(x))
            return torch.tensor(x).long()

        features = {
            "input_ids": _to_tensor(example.input_ids),
            "segment_ids": _to_tensor(example.segment_ids),
            "attention_mask": _to_tensor(example.attention_mask),
        }
        labels = {
            "start_positions": torch.tensor(example.start).long(),
            "end_positions": torch.tensor(example.end).long(),
        }
        return features, labels


class DataModuleForQuestionAnswering(pl.LightningDataModule):
    """Data module for question answering"""

    @classmethod
    def from_dureader_robust(
        cls, train_input_files, valid_input_files, tokenizer: BertWordPieceTokenizer = None, vocab_file=None, **kwargs
    ):
        train_dataset = DatasetForQuestionAnswering.from_dureader_robust(
            train_input_files, tokenizer=tokenizer, vocab_file=vocab_file, **kwargs
        )
        valid_dataset = DatasetForQuestionAnswering.from_dureader_robust(
            valid_input_files, tokenizer=tokenizer, vocab_file=vocab_file, **kwargs
        )
        return cls(train_dataset=train_dataset, valid_dataset=valid_dataset, **kwargs)

    @classmethod
    def from_jsonl_files(
        cls, train_input_files, valid_input_files, tokenizer: BertWordPieceTokenizer = None, vocab_file=None, **kwargs
    ):
        train_dataset = DatasetForQuestionAnswering.from_jsonl_files(
            train_input_files, tokenizer=tokenizer, vocab_file=vocab_file, **kwargs
        )
        valid_dataset = DatasetForQuestionAnswering.from_jsonl_files(
            valid_input_files, tokenizer=tokenizer, vocab_file=vocab_file, **kwargs
        )
        return cls(train_dataset=train_dataset, valid_dataset=valid_dataset, **kwargs)

    def __init__(self, train_dataset: Dataset, valid_dataset: Dataset, test_dataset: Dataset = None, **kwargs):
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.train_batch_size = kwargs.get("train_batch_size", 32)
        self.valid_batch_size = kwargs.get("valid_batch_size", 32)
        self.test_batch_size = kwargs.get("test_batch_size", 32)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.valid_batch_size, shuffle=True, num_workers=1)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size)
