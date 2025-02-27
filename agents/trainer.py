from dataclasses import dataclass
from typing import List, Union, Optional, Literal
import torch
from unsloth import FastVisionModel, is_bf16_supported
from trl import (
    SFTTrainer,
    get_peft_config,
    DataCollatorForCompletionOnlyLM,
    ModelConfig,
    SFTConfig,
)
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from agents.vlm_agents import ChatMakerAgent, ChatEditorAgent, QwenEditorAgent
from agents.editing_actions import EditExecution
from datasets import load_dataset
from pathlib import Path
import base64
from PIL import Image
from io import BytesIO
import itertools
from transformers.feature_extraction_utils import BatchFeature
from mrcad import Design, Instruction, Execution
from pathlib import Path


@dataclass
class mrCADArguments:
    training_games: Union[str, List[str]]
    validation_games: Union[str, List[str]]
    image_size: int = 256
    agent_outputs: Literal["design", "actions"] = "design"


class DataCollatorForInterleavedImages(DataCollatorForCompletionOnlyLM):
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

    def get_image(self, image_url):
        return Image.open(
            BytesIO(
                base64.decodebytes(
                    bytes(image_url[len("data:image/png;base64,") :], "utf-8")
                )
            )
        )

    def __call__(self, examples):
        text = [e["text"] for e in examples]
        images = [list(map(self.get_image, e["images"])) for e in examples]

        processed_inputs = self.processor(
            text=text, images=images, return_tensors="pt", padding=True
        )

        collator_input = [
            {
                k: v[i]
                for k, v in processed_inputs.items()
                if not k in ["pixel_values", "image_grid_thw"]
            }
            for i in range(len(examples))
        ]

        batch = self.torch_call(collator_input)
        return BatchFeature(
            data={
                **batch,
                "pixel_values": processed_inputs["pixel_values"],
                "image_grid_thw": processed_inputs["image_grid_thw"],
            }
        )


def prepare_game(rounds, agent, processor, agent_outputs):
    turns = list(
        itertools.chain.from_iterable(
            [
                [
                    (
                        Design.model_validate(r["context"]),
                        Instruction.model_validate(r["instruction"]),
                    ),
                    (
                        Design.model_validate(r["context"]),
                        (
                            Execution.model_validate(r["execution"])
                            if agent_outputs == "design"
                            else EditExecution.model_validate(r["edit_execution"])
                        ),
                    ),
                ]
                for r in rounds
            ]
        )
    )
    messages = agent.make_prompt(turns)
    text = processor.apply_chat_template(messages)
    images = []
    for m in messages:
        for c in m["content"]:
            if isinstance(c, dict):
                if "image_url" in c:
                    images.append(c["image_url"]["url"])
                elif "image" in c:
                    images.append(c["image"])

    return {
        "text": text,
        "images": images,
    }


def run_trainer(
    mrcad_args: mrCADArguments,
    training_args: SFTConfig,
    model_config: ModelConfig,
    **kwargs,
):
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
    model, processor = FastVisionModel.from_pretrained(
        model_config.model_name_or_path,
        use_gradient_checkpointing="unsloth",
        load_in_4bit=False,
    )
    processor.image_processor.do_image_splitting = False

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        lora_dropout=model_config.lora_dropout,
        bias="none",
    )

    ################
    # Dataset
    ################
    dataset = load_dataset(
        "json",
        data_files={
            "train": mrcad_args.training_games,
            "validation": mrcad_args.validation_games,
        },
    )

    if mrcad_args.agent_outputs == "design":
        agent = ChatMakerAgent(image_size=mrcad_args.image_size)
        print("Using ChatMakerAgent")
    elif mrcad_args.agent_outputs == "actions":
        if model.config.architectures[0] == "Qwen2_5_VLForConditionalGeneration":
            agent = QwenEditorAgent(
                image_size=mrcad_args.image_size, describe_tools=False
            )
            print("Using QwenEditorAgent")
        else:
            agent = ChatEditorAgent(image_size=mrcad_args.image_size)
            print("Using ChatEditorAgent")

    dataset = dataset.map(
        lambda x: prepare_game(x["rounds"], agent, processor, mrcad_args.agent_outputs),
        remove_columns=[
            "trial_id",
            "target_id",
            "target",
            "dyad_id",
            "trial_num",
            "rounds",
        ],
    )

    ################
    # Training
    ################
    FastVisionModel.for_training(model)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForInterleavedImages(
            processor=processor,
            response_template="<|im_start|>assistant",
            instruction_template="<|im_start|>user",
        ),
        train_dataset=dataset["train"],
        eval_dataset=(
            dataset["validation"] if training_args.eval_strategy != "no" else None
        ),
        processing_class=processor.tokenizer,
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
