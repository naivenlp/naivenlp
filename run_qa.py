import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from naivenlp.question_answering.dataset import DataModuleForQuestionAnswering
from naivenlp.question_answering.model import LitBertForQuestionAnswering

model = LitBertForQuestionAnswering("hfl/chinese-roberta-wwm-ext")
dm = DataModuleForQuestionAnswering.from_dureader_robust(
    train_input_files=os.path.join(os.environ["DUREADER_ROBUST_PATH"], "train.json"),
    valid_input_files=os.path.join(os.environ["DUREADER_ROBUST_PATH"], "dev.json"),
    vocab_file=os.environ["BERT_VOCAB_PATH"],
    train_batch_size=64,
    valid_batch_size=64,
    max_sequence_length=256,
)

dm.setup("fit")
trainer = pl.Trainer(
    gpus=1,
    callbacks=[
        EarlyStopping(monitor="valid_start_acc", min_delta=0.001, patience=3, verbose=False, mode="max"),
        ModelCheckpoint(dirpath="models/dureader-qa", filename="bert4qa-{epoch:04d}"),
    ],
)
trainer.fit(model, dm)
