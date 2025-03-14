import jsonlines
from mrcad.coordinator import MakerEvaluationCoordinator
from mrcad import Design, Instruction, Execution
from agents.editing_actions import EditExecution
from agents.vlm_agents import (
    VLMDesignMakerAgent,
    VLMDesignEditorAgent,
    QwenDesignEditorAgent,
)
import itertools
from tqdm import tqdm


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
                                Design.model_validate(r["context"]),
                                Instruction.model_validate(r["instruction"]),
                            ),
                            (
                                Design.model_validate(r["context"]),
                                (
                                    EditExecution.model_validate(r["edit_execution"])
                                    if agent_output == "actions"
                                    else Execution.model_validate(r["execution"])
                                ),
                            ),
                        )
                        for r in x["rounds"]
                    ]
                )
            ),
        )

        # try:
        actions = coordinator.play()
        with jsonlines.open(save_path, mode="a") as writer:
            writer.write({**x, "model_response": [a.model_dump() for a in actions]})
        # except Exception as e:
        #     print(e)
        #     with jsonlines.open(save_path, mode="a") as writer:
        #         writer.write({**x, "model_response": []})


def main(
    maker_type,
    base_url,
    api_key,
    model_name,
    agent_output,
    dataset_path,
    save_path,
    demonstrations_path=None,
    temperature=1,
    top_p=0.9,
):
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

    with jsonlines.open(dataset_path) as reader:
        dataset = list(reader)

    run(maker, dataset, save_path, agent_output)
