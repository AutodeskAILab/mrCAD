from dataclasses import dataclass
from typing import List, Union, Literal
import torch
from trl import SFTTrainer, get_peft_config, SFTConfig, ModelConfig
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
)
import base64
from io import BytesIO
from datasets import load_dataset, concatenate_datasets
import itertools
from PIL import Image
from mrcad import Design, Instruction
from mrcad.editing_actions import EditExecution
from agents.vlm_agents import QwenDesignEditorAgent
from agents.training.collator import DataCollatorForInterleavedImages


@dataclass
class mrCADArguments:
    training_splits: Union[str, List[str]]
    validation_splits: Union[str, List[str]]
    image_size: int = 256
    agent_outputs: Literal["design", "actions"] = "design"


def get_image_from_b64_url(image_url: str):
    """
    Extracts an image from a base64 encoded URL as would be passed to an LLM API.

    Args:
        image_url (str): The base64 encoded URL.
    """
    return Image.open(
        BytesIO(
            base64.decodebytes(
                bytes(image_url[len("data:image/png;base64,") :], "utf-8")
            )
        )
    )


def prepare_game(rounds, agent, processor):
    """
    Prepares a game/trial for training by loading rounds, applying the agent's chat template, and processing the inputs.

    Args:
        rounds: A list of rounds from the mrCAD dataset.
        agent: The agent to use for processing the game.
        processor: The processor to use for processing the game.
    """
    turns = list(
        itertools.chain.from_iterable(
            [
                [
                    (
                        Design.model_validate(r["context"]).round(3),
                        Instruction.model_validate(r["instruction"]),
                    ),
                    (
                        Design.model_validate(r["context"]).round(3),
                        EditExecution.model_validate(r["edit_execution"]).round(3),
                    ),
                ]
                for r in rounds
            ]
        )
    )
    messages = agent.make_prompt(turns)

    if hasattr(agent, "chat_template"):
        chat_template = agent.chat_template
    else:
        chat_template = processor.chat_template

    text = processor.apply_chat_template(
        messages, tools=agent.tools, chat_template=chat_template
    )
    images = []
    for m in messages:
        for c in m["content"]:
            if isinstance(c, dict):
                if "image_url" in c:
                    images.append(get_image_from_b64_url(c["image_url"]["url"]))
                elif "image" in c:
                    images.append(get_image_from_b64_url(c["image"]))

    processed_inputs = processor(text=text, images=images, return_tensors="pt")

    return processed_inputs


def train(
    script_args: mrCADArguments,
    training_args: SFTConfig,
    model_config: ModelConfig,
    **kwargs,
):
    """
    Trains a model on the mrCAD dataset.

    Adapted from https://github.com/huggingface/trl/blob/dee37342a8505beb5ad7e0dd01071c8b9ac584ba/examples/scripts/sft_vlm.py
    """
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=True)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    ################
    # Model, Tokenizer & Processor
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
    )

    processor.tokenizer.padding_side = "left"

    attn_implementation = "flash_attention_2"

    model_kwargs = dict(
        revision=model_config.model_revision,
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
    )

    model = AutoModelForVision2Seq.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        **model_kwargs,
    )

    ################
    # Dataset
    ################

    agent = QwenDesignEditorAgent(None, image_size=script_args.image_size)

    dataset = load_dataset("saujasv/mrcad", trust_remote_code=True)

    dataset = dataset.map(
        lambda x: prepare_game(x["rounds"], agent, processor),
        remove_columns=[
            "trial_id",
            "target_id",
            "target",
            "dyad_id",
            "trial_num",
            "rounds",
        ],
    )

    collator = DataCollatorForInterleavedImages(
        processor=processor,
        response_template="<|im_start|>assistant",
        instruction_template="<|im_start|>user",
    )

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=concatenate_datasets(
            [dataset[x] for x in script_args.training_splits]
        ),
        eval_dataset=(
            concatenate_datasets([dataset[x] for x in script_args.validation_splits])
            if training_args.eval_strategy != "no"
            else None
        ),
        processing_class=processor.tokenizer,
        peft_config=get_peft_config(model_config),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)


def main(config_path):
    from transformers import HfArgumentParser

    parser = HfArgumentParser((mrCADArguments, SFTConfig, ModelConfig))
    mrcad_args, training_args, model_args = parser.parse_yaml_file(config_path)

    train(mrcad_args, training_args, model_args)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
