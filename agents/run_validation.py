from agents.train import DataCollatorForInterleavedImages, prepare_game
from agents.vlm_agents import QwenDesignEditorAgent
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
)
from agents.trainer import QwenTrainer
from datasets import load_dataset
from pathlib import Path


def main(model_path, dataset_path):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="bfloat16",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_path)

    dataset = load_dataset(
        "json",
        data_files={"validation": dataset_path},
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

    processor.tokenizer.padding_side = "left"
    trainer = QwenTrainer(
        model=model,
        processor=processor,
        args=TrainingArguments(
            per_device_eval_batch_size=1,
            eval_accumulation_steps=16,
            do_train=False,
            do_eval=True,
            output_dir=f"./tmp/{Path(model_path).name}",
        ),
        train_dataset=None,
        eval_dataset=dataset["validation"],
        data_collator=DataCollatorForInterleavedImages(
            processor=processor,
            response_template="<|im_start|>assistant",
            instruction_template="<|im_start|>user",
        ),
    )

    print(trainer.evaluate())
