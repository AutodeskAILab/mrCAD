# mrCAD environment

## Setup

```
conda create -n mrcad-env python=3.11
pip install git+https://github.com/saujasv/mrcad-env.git@faa0e981add2292b6c054508f39719d24ddc09e0
```

To run training or evaluation experiments, you would need additional dependencies installed.
```
pip install -r requirements.txt
```

## Loading the mrCAD dataset
Once you have installed the `mrcad` package, you can load the dataset using [HuggingFace Datasets](https://github.com/huggingface/datasets/).

> [!IMPORTANT]
> Make sure to install the `mrcad` package _before_ trying to load the dataset since the dataset loader uses the `mrcad` package to prepare the data.

```python
from datasets import load_dataset

mrcad_dataset = load_dataset("saujasv/mrcad", trust_remote_code=True)
```

> [!NOTE]
> You need to pass the `trust_remote_code=True` argument since the dataset loading script executes code from the `mrcad` package to process data.

## The format of a trial

The dataset is presented as a set of trials. Each trial includes:
- `trial_id`: string identifier for the trial
- `target_id`: string identifier for the target design
- `dyad_id`: string identified for the dyad completing this trial
- `trial_num`: index of the trial in the sequence of trials completed by the dyad
- `target`: JSON dump of a `Design` object that is the target to be reconstructed
- `rounds`: List of rounds in the trial
  - `round_num`: integer index of the round
  - `context`: JSON dump of a `Design` object that is the state of the design at the beginning of the round
  - `instruction`: instruction produced by the _Designer_
    - `text`: string of the text component of the instruction
    - `drawing`: array structure of coordinates used to represent drawing strokes
  - `execution`: _Maker_'s response
    - `design`: JSON dump of a `Design` object that the _Maker_ produces
  - `edit_execution`: actions taken by the _Maker_
    - `edits`: list of editing actions
    - `design`: JSON dump of a `Design` object obtained by executing the actions on the `context` design.

> [!NOTE]
> Due to minor errors with some extraneous actions recorded during data collection, there may be minor discrepancies between the `design` in the `execution` field and that in the `edit_execution` field. The one associated with the `execution` field is the actual output produced by participants, while the one in the `edit_execution` field is what is obtained by executing the recorded editing actions.

To load each of these objects, you can use the tools provided in this module.

```python
from mrcad import Design, Instruction, Execution
from mrcad.editing_actions import EditExecution

# Split used for evaluating models. You can find other splits by inspecting the dataset.
SPLIT = "eval_verified_complete"

# Example element
IDX = 0

trial = mrcad_dataset[SPLIT][IDX]
target = Design.model_validate(trial["target"])
rounds = [
    {
        "context": Design.model_validate(r["context"]),
        "instruction": Instruction.model_validate(r["instruction"]),
        "execution": Execution.model_validate(r["execution"]),
        "edit_execution": EditExecution.model_validate(r["edit_execution"]),
    }
    for r in trial["rounds"]
]
```

## Evaluating vision-language models

For experiments in the paper, we used [OpenRouter](https://openrouter.ai/) to make calls to LLM APIs so we have a unified interface with a number of LLM API providers.

```
python -m fire experiments/run_maker_evaluation.py main openrouter https://openrouter.ai/api/v1 <API_KEY> <MODEL_NAME> actions <SAVE_PATH>
```

The pipeline also supports querying models running on a vLLM server (for example, one being served at `http://localhost:$PORT/v1`).

```
python -m fire experiments/run_maker_evaluation.py main openrouter http://localhost:$PORT/v1 "dummy" <MODEL_NAME> actions <SAVE_PATH>
```

For Qwen2.5 models we evaluated, we used a different prompt that includes the tool descriptions. 

```
python -m fire experiments/run_maker_evaluation.py main qwen-vllm http://localhost:$PORT/v1 "dummy" <MODEL_NAME> actions <SAVE_PATH>
```