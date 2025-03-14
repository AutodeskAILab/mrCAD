# https://github.com/2U1/Qwen2-VL-Finetune/blob/567a625dbd49e35321fa3246c1bced08a03b5cc1/src/training/train.py
import os
import torch
from peft import LoraConfig, get_peft_model
import ast
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
    HfArgumentParser,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
)
from accelerate import Accelerator
from trl import ModelConfig
from agents.trainer import QwenTrainer
from agents.train_utils import (
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    safe_save_model_for_hf_trainer,
)
import pathlib
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl
from agents.monkey_patch_forward import (
    replace_qwen2_5_with_mixed_modality_forward,
    replace_qwen_2_with_mixed_modality_forward,
)
from trl import DataCollatorForCompletionOnlyLM
from io import BytesIO
import base64
from PIL import Image
from mrcad import Design, Instruction
from agents.editing_actions import EditExecution
from dataclasses import dataclass, field
from typing import List, Union, Optional, Literal
from datasets import load_dataset
from agents.vlm_agents import QwenDesignEditorAgent
import itertools
from transformers.feature_extraction_utils import BatchFeature

local_rank = None


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-7B-Instruct")


@dataclass
class mrCADArguments:
    training_games: Union[str, List[str]]
    validation_games: Union[str, List[str]]
    image_size: int = 256
    agent_outputs: Literal["design", "actions"] = "design"


@dataclass
class QwenTrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    tune_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)

    max_seq_length: int = field(
        default=16384,  # This is the default value of the qwen2-vl model
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    lora_namespan_exclude: str = field(
        default=None, metadata={"help": "List of namespan to exclude for LoRA"}
    )
    num_lora_modules: int = -1


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

    def __call__(self, examples):
        collator_input = [
            {
                k: v[0]
                for k, v in x.items()
                if not k in ["pixel_values", "image_grid_thw"]
            }
            for x in examples
        ]

        padded = self.processor.tokenizer.pad(collator_input, return_tensors="pt")

        batch = self.torch_call(padded)

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


def get_image_from_b64_url(image_url):
    return Image.open(
        BytesIO(
            base64.decodebytes(
                bytes(image_url[len("data:image/png;base64,") :], "utf-8")
            )
        )
    )


def prepare_game(rounds, agent, processor):
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
                        EditExecution.model_validate(r["edit_execution"]),
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


def rank0_print(*args):
    if local_rank == 0 or local_rank == "0" or local_rank is None:
        print(*args)


def find_target_linear_names(
    model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True
):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)

    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names


def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad


def configure_vision_tower(model, training_args, compute_dtype, device):
    vision_tower = model.visual
    vision_tower.to(dtype=compute_dtype, device=device)

    vision_model_params = model.visual.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)

    # Handle merger specifically
    merger_params = model.visual.merger.parameters()
    set_requires_grad(merger_params, training_args.tune_merger)


def configure_llm(model, training_args):
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_llm)

    llm_params = model.model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)


def train(config_file):
    global local_rank

    parser = HfArgumentParser((ModelArguments, mrCADArguments, QwenTrainingArguments))
    (model_args, mrcad_args, training_args) = parser.parse_yaml_file(config_file)

    if "Qwen2.5" in model_args.model_name_or_path:
        # Liger-kernel for Qwen2.5 is not supported yet.
        replace_qwen2_5_with_mixed_modality_forward()
    else:
        # It monkey patches the forward to handle mixed modality inputs.
        replace_qwen_2_with_mixed_modality_forward()
        # This is becuase mixed-modality training monkey-patches the model forward method.
        apply_liger_kernel_to_qwen2_vl(fused_linear_cross_entropy=False)

    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("If `lora_enable` is True, `freeze_llm` must also be True.")

    if not training_args.lora_enable:
        assert (
            not training_args.vision_lora
        ), "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."

    if training_args.vision_lora and not training_args.freeze_vision_tower:
        raise ValueError(
            "If `vision_lora` is True, `freeze_vision_tower` must also be True."
        )

    else:
        if training_args.lora_namespan_exclude is not None:
            training_args.lora_namespan_exclude = ast.literal_eval(
                training_args.lora_namespan_exclude
            )
        else:
            training_args.lora_namespan_exclude = []

        if not training_args.vision_lora:
            training_args.lora_namespan_exclude += ["visual"]

    local_rank = training_args.local_rank
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["visual"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,
                ),
            )
        )

    if "Qwen2.5" in model_args.model_name_or_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=compute_dtype,
            attn_implementation=(
                "flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa"
            ),
            **bnb_model_from_pretrained_args,
        )
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=compute_dtype,
            attn_implementation=(
                "flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa"
            ),
            **bnb_model_from_pretrained_args,
        )

    model.config.use_cache = False
    model_to_configure = model
    configure_llm(model_to_configure, training_args)
    configure_vision_tower(
        model_to_configure, training_args, compute_dtype, training_args.device
    )

    if training_args.bits in [4, 8]:
        model.config.torch_dtype = (
            torch.float32
            if training_args.fp16
            else (torch.bfloat16 if training_args.bf16 else torch.float32)
        )
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": True},
        )

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    if training_args.lora_enable:
        lora_namespan_exclude = training_args.lora_namespan_exclude
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(
                model,
                lora_namespan_exclude=lora_namespan_exclude,
                num_lora_modules=training_args.num_lora_modules,
            ),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA to the model...")
        model = get_peft_model(model, peft_config)

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        # The default setting is padding_side="left"
        # When training using the right-side padding is more efficient.
        padding_side="right",
    )

    # model.config.tokenizer_model_max_length = processor.tokenizer.model_max_length
    model.config.tokenizer_padding_side = processor.tokenizer.padding_side
    model.config.vision_lr = training_args.vision_lr

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)

            if "lm_head" in name or "embed_token" in name:
                if hasattr(module, "weight"):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    dataset = load_dataset(
        "json",
        data_files={
            "train": mrcad_args.training_games,
            "validation": mrcad_args.validation_games,
        },
    )

    agent = QwenDesignEditorAgent(
        model=None
    )  # No model because we are using this only for formatting the prompts

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

    trainer = QwenTrainer(
        model=model,
        processor=processor,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=DataCollatorForInterleavedImages(
            processor=processor,
            response_template="<|im_start|>assistant",
            instruction_template="<|im_start|>user",
        ),
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )

        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=False
        )

        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(
                non_lora_state_dict,
                os.path.join(training_args.output_dir, "non_lora_state_dict.bin"),
            )
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    import fire

    fire.Fire(train)
