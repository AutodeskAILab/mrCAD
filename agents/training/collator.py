from typing import List, Optional, Union
import itertools
import torch
from transformers import BatchFeature
from trl import DataCollatorForCompletionOnlyLM


class DataCollatorForInterleavedImages(DataCollatorForCompletionOnlyLM):
    """
    Data collator that extends the functionality of DataCollatorForCompletionOnlyLM of constructing label tensors with only user turns being marked for loss computation. This supports interleaved image and text inputs.

    Args:
        processor: The processor to use for tokenization.
        response_template: Prefix string that model responses begin with
        instruction_template: Prefix string that user turns begin with
        mlm: Whether to use masked language modeling -- this is here for compatibility with the parent class.
        ignore_index: The index to ignore in loss computation.
        padding_free: Whether to use padding free collation.
    """

    def __init__(
        self,
        processor,
        response_template: Union[str, List[int]],
        instruction_template: Optional[Union[str, List[int]]] = None,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        padding_free: bool = False,
        **kwargs,
    ):
        super().__init__(
            tokenizer=processor.tokenizer,
            response_template=response_template,
            instruction_template=instruction_template,
            *args,
            mlm=mlm,
            ignore_index=ignore_index,
            padding_free=padding_free,
            **kwargs,
        )

        self.processor = processor

    def __call__(self, examples):
        collator_input = [
            {
                k: v[0]
                for k, v in x.items()
                if not k in ["pixel_values", "image_grid_thw"]
            }
            for x in examples
        ]

        batch = self.torch_call(collator_input)

        return BatchFeature(
            data={
                **batch,
                "pixel_values": torch.tensor(
                    list(
                        itertools.chain.from_iterable(
                            [x["pixel_values"] for x in examples]
                        )
                    ),
                    dtype=torch.float32,
                ),
                "image_grid_thw": torch.tensor(
                    list(
                        itertools.chain.from_iterable(
                            [x["image_grid_thw"] for x in examples]
                        )
                    ),
                    dtype=torch.int64,
                ),
            }
        )
