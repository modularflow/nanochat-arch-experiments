"""
Python code instructions dataset (18K examples).
https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca

Each example has a natural-language instruction and a Python code solution,
already in instruction/output format -- ideal for self-supervised training
on code structure.
"""

from datasets import load_dataset
from tasks.common import Task


class CodeStack(Task):
    """
    Python code instructions (18.6K examples).
    Fields: instruction, input, output (code).
    """

    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train"], "CodeStack only has a train split"
        self.ds = load_dataset(
            "iamtarun/python_code_instructions_18k_alpaca",
            split="train",
        ).shuffle(seed=42)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        instruction = row["instruction"].strip()
        extra_input = (row.get("input") or "").strip()
        code = (row.get("output") or "").strip()

        if extra_input:
            user_content = f"{instruction}\n\n{extra_input}"
        else:
            user_content = instruction

        if not code:
            code = "# No solution provided\npass"

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": code},
        ]
        return {"messages": messages}
