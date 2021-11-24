from collections import namedtuple

ExampleForQuestionAnswering = namedtuple(
    "ExampleForQuestionAnswering",
    ["tokens", "input_ids", "segment_ids", "attention_mask", "start", "end"],
)
