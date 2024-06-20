# mrCAD environment

## Setup

```
conda create -n mrcad-env python=3.11
```

## Tests
To test replay agents (agents that replay actions one-by-one from a given list of actions)
```
python -m tests.test_replay_agents <instructions_dataframe_csv> <executions_dataframe_csv> <folder_to_save_html>
```