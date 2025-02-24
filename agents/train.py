from typing import List, Union
from trl import ModelConfig, SFTConfig, TrlParser
import itertools
from dataclasses import dataclass
from trainer import mrCADArguments, train
from dotenv import load_dotenv

if __name__ == "__main__":
    import argparse

    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    args = parser.parse_args()

    parser = TrlParser((mrCADArguments, SFTConfig, ModelConfig))
    mrcad_args, training_args, model_args = parser.parse_yaml_file(args.config_file)

    train(mrcad_args, training_args, model_args)
