import os
import unittest

from naivenlp.question_answering.dataset import DataModuleForQuestionAnswering, DatasetForQuestionAnswering

VOCAB_PATH = os.environ["BERT_VOCAB_PATH"]


class DatasetTest(unittest.TestCase):
    """ """

    def test_dataset_from_jsonl(self):
        dataset = DatasetForQuestionAnswering.from_jsonl_files(
            "testdata/qa.jsonl",
            vocab_file=VOCAB_PATH,
            max_sequence_length=40,
        )
        print()
        for feature, label in dataset:
            print(feature)
            print(label)
            print()

    def test_dataset_from_dureader_robust(self):
        print()
        dataset = DatasetForQuestionAnswering.from_dureader_robust(
            input_files=os.path.join(os.environ["DUREADER_ROBUST_PATH"], "train.json"),
            vocab_file=VOCAB_PATH,
        )
        for i in range(5):
            print(dataset.examples[i])
            print(dataset[i])

    def test_dataset_from_dureader_checklist(self):
        print()
        dataset = DatasetForQuestionAnswering.from_dureader_robust(
            input_files=os.path.join(os.environ["DUREADER_CHECKLIST_PATH"], "train.json"),
            vocab_file=VOCAB_PATH,
        )

        for i in range(5):
            print(dataset.examples[i])
            print(dataset[i])


if __name__ == "__main__":
    unittest.main()
