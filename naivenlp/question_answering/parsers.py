import abc
import json
import logging
import re
from typing import Dict

from tokenizers import BertWordPieceTokenizer

from .example import ExampleForQuestionAnswering


class InstanceParserForQuestionAnswering:
    """Parse json instance to example for qa task."""

    def __init__(self, tokenizer=None, vocab_file=None, do_lower_case=True, **kwargs) -> None:
        assert tokenizer or vocab_file, "`tokenizer` or `vocab_file` must be provided!"
        self.tokenizer = tokenizer or BertWordPieceTokenizer.from_file(vocab_file, lowercase=do_lower_case)

    def from_vocab(cls, vocab_file, **kwargs):
        tokenizer = BertWordPieceTokenizer.from_file(vocab_file, lowercase=kwargs.get("do_lower_case", True))
        return cls.from_tokenizer(tokenizer, **kwargs)

    @classmethod
    def from_tokenizer(cls, tokenizer: BertWordPieceTokenizer, **kwargs):
        return cls(tokenizer=tokenizer, vocab_file=None, **kwargs)

    def _find_answer_span(self, context, answer):
        for m in re.finditer(re.escape(answer), context, re.IGNORECASE):
            start, end = m.span()
            return start, end
        return 0, 0

    def parse(self, instance: Dict, **kwargs) -> ExampleForQuestionAnswering:
        context, question, answer = instance["context"], instance["question"], instance["answer"]
        start_char_idx, end_char_idx = self._find_answer_span(context, answer)
        if end_char_idx <= start_char_idx:
            return None

        find_answer = context[start_char_idx:end_char_idx]
        if find_answer.lower() != answer.lower():
            logging.warning("Skip: find answer mismatch answer.")
            logging.warning("find answer: %s", find_answer)
            logging.warning("     answer: %s", answer)
            logging.warning("   instance: %s", json.dumps(instance, ensure_ascii=False))
            logging.warning("")
            return None

        # Mark the character indexes in context that are in answer
        is_char_in_ans = [0] * len(context)
        for idx in range(start_char_idx, end_char_idx):
            is_char_in_ans[idx] = 1
        context_encoding = self.tokenizer.encode(context, add_special_tokens=True)
        # Find tokens that were created from answer characters
        ans_token_idx = []
        for idx, (start_char_idx, end_char_idx) in enumerate(context_encoding.offsets):
            if sum(is_char_in_ans[start_char_idx:end_char_idx]) > 0:
                ans_token_idx.append(idx)
        if not ans_token_idx:
            logging.warning("Skip no answer instance: %s", json.dumps(instance, ensure_ascii=False))
            return None
        start_token_idx, end_token_idx = ans_token_idx[0], ans_token_idx[-1]

        question_encoding = self.tokenizer.encode(question, add_special_tokens=True)
        input_ids = context_encoding.ids + question_encoding.ids[1:]
        segment_ids = [0] * len(context_encoding.type_ids) + [1] * len(question_encoding.type_ids[1:])
        attention_mask = [1] * len(context_encoding.attention_mask + question_encoding.attention_mask[1:])
        assert len(input_ids) == len(segment_ids), "input_ids length:{} VS segment_ids length: {}".format(
            len(input_ids), len(segment_ids)
        )
        assert len(input_ids) == len(attention_mask), "input_ids length:{} VS attention_mask length: {}".format(
            len(input_ids), len(attention_mask)
        )
        tokens = context_encoding.tokens + question_encoding.tokens[1:]
        # answer validation:
        subtokens = "".join([x.lstrip("##") for x in tokens[start_token_idx : end_token_idx + 1]])
        answ_encoding = self.tokenizer.encode(answer, add_special_tokens=False)
        anstokens = "".join([x.lstrip("##") for x in answ_encoding.tokens])
        if anstokens != subtokens:
            logging.warning("Skip: find answer mismatch answer.")
            logging.warning("     tokens: %s", tokens[start_token_idx : end_token_idx + 1])
            logging.warning("find answer: %s", subtokens)
            logging.warning("     answer: %s", anstokens)
            logging.warning("   instance: %s", json.dumps(instance, ensure_ascii=False))
            logging.warning("")
            return None
        example = ExampleForQuestionAnswering(
            tokens=tokens,
            input_ids=input_ids,
            segment_ids=segment_ids,
            attention_mask=attention_mask,
            start=start_token_idx,
            end=end_token_idx,
        )
        return example
