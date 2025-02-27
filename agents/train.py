from typing import List, Union
from trl import ModelConfig, SFTConfig, TrlParser
import itertools
from dataclasses import dataclass
from agents.trainer import mrCADArguments, run_trainer


def main(config_file):
    parser = TrlParser((mrCADArguments, SFTConfig, ModelConfig))
    mrcad_args, training_args, model_args = parser.parse_yaml_file(config_file)

    run_trainer(mrcad_args, training_args, model_args)
