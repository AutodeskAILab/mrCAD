from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel


def merge_lora_weights(base_model_name_or_path, adapter_name_or_path, output_path):
    """
    Merges a base model with a trained adapter and saves the resulting model to the output path.

    This is needed because vLLM does not support serving LoRA adapters for vision-language models directly.

    Args:
        base_model_name_or_path (str): The name or path of the base model.
        adapter_name_or_path (str): The name or path of the adapter.
        output_path (str): The path to save the merged model.
    """
    base_model = AutoModelForVision2Seq.from_pretrained(base_model_name_or_path)
    processor = AutoProcessor.from_pretrained(base_model_name_or_path)

    model = PeftModel.from_pretrained(base_model, adapter_name_or_path)

    model = model.merge_and_unload()
    model.save_pretrained(output_path)
    processor.save_pretrained(output_path)
