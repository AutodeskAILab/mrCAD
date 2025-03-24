import jsonlines
from mrcad.coordinator import MakerEvaluationCoordinator
from mrcad import Design, Instruction, Execution
from mrcad.editing_actions import EditExecution
from agents.vlm_agents import (
    VLMDesignMakerAgent,
    VLMDesignEditorAgent,
    QwenDesignEditorAgent,
)
import itertools
from tqdm import tqdm
from datasets import load_dataset


def run(maker, dataset, save_path, agent_output):
    for x in tqdm(dataset):
        coordinator = MakerEvaluationCoordinator(
            Design.model_validate(x["target"]),
            maker,
            list(
                itertools.chain.from_iterable(
                    [
                        (
                            (
                                Design.model_validate(r["context"]).round(3),
                                Instruction.model_validate(r["instruction"]),
                            ),
                            (
                                Design.model_validate(r["context"]),
                                (
                                    EditExecution.model_validate(
                                        r["edit_execution"]
                                    ).round(3)
                                    if agent_output == "actions"
                                    else Execution.model_validate(r["execution"]).round(
                                        3
                                    )
                                ),
                            ),
                        )
                        for r in x["rounds"]
                    ]
                )
            ),
        )

        actions = coordinator.play()
        with jsonlines.open(save_path, mode="a") as writer:
            writer.write({**x, "model_response": [a.model_dump() for a in actions]})


def main(
    maker_type,
    base_url,
    api_key,
    model_name,
    agent_output,
    save_path,
    demonstrations_path=None,
    temperature=1,
    top_p=0.9,
    eval_split="eval_verified_complete",
    dataset_config="full",
):
    """
    Evaluate a maker agent on one split of the mrCAD dataset. All the evaluations run a VLM agent that queries a VLM through an API compatible with the OpenAI SDK.

    Args:
        maker_type: The type of maker agent to evaluate. May be "openrouter" or "qwen-vllm". Both use the OpenAI SDK to actually query the VLM, but for "qwen-vllm" the agent uses a special prompt that accounts for the different prompt used to present tools to the Qwen model.
        base_url: The base URL argument to the OpenAI client.
        api_key: The API key argument to the OpenAI client.
        model_name: The name of the VLM model to query through the OpenAI client.
        agent_output: The type of output the agent produces. May be "design" or "actions".
        save_path: The path of the JSONL file to save the results.
        demonstrations_path: The path to the demonstrations file used for few-shot prompting
        temperature: The temperature to use when querying the VLM.
        top_p: The top_p to use when querying the VLM.
        eval_split: The split of the mrCAD dataset to evaluate on.
    """
    if demonstrations_path is not None:
        with jsonlines.open(demonstrations_path) as reader:
            demonstrations = [
                list(
                    itertools.chain.from_iterable(
                        [
                            (
                                (
                                    Design.model_validate(r["context"]),
                                    Instruction.model_validate(r["instruction"]),
                                ),
                                (
                                    Design.model_validate(r["context"]),
                                    (
                                        EditExecution.model_validate(
                                            r["edit_execution"]
                                        )
                                        if agent_output == "actions"
                                        else Execution.model_validate(
                                            r["model_response"]
                                        )
                                    ),
                                ),
                            )
                            for r in x["rounds"]
                        ]
                    )
                )
                for x in reader
            ]
    else:
        demonstrations = None

    if maker_type == "openrouter":
        if agent_output == "design":
            maker = VLMDesignMakerAgent(
                model_name,
                base_url=base_url,
                api_key=api_key,
                demonstrations=demonstrations,
                temperature=temperature,
            )
        elif agent_output == "actions":
            maker = VLMDesignEditorAgent(
                model_name, base_url=base_url, api_key=api_key, temperature=temperature
            )
    elif maker_type == "qwen-vllm":
        if agent_output == "design":
            maker = VLMDesignMakerAgent(
                model_name,
                base_url=base_url,
                api_key=api_key,
                demonstrations=demonstrations,
            )
        elif agent_output == "actions":
            maker = QwenDesignEditorAgent(
                model_name,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
                top_p=top_p,
            )

    dataset = load_dataset("saujasv/mrcad", dataset_config, trust_remote_code=True)

    run(maker, dataset[eval_split], save_path, agent_output)
