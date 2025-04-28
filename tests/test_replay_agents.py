import pandas as pd
import matplotlib.pyplot as plt
from agents.replay_agents import ReplayDesigner, ReplayMaker
from mrcad.design import Design, Line, Arc, Circle
from mrcad.coordinator import SynchronousCoordinator
from mrcad.render_utils import OutOfBoundsError, Collinear, RenderConfig
from mrcad.visualize import trajectories_to_html
from experiments.data_conversion_utils import (
    get_design_from_record,
    get_strokes_from_record,
    normalize_curves,
)


def main(
    instructions_df_file,
    executions_df_file,
    html_save_path=None,
    save_image_size=320,
    experiment_grid_size=40,
    experiment_sketchpad_size=400,
):
    experiment_render_config = RenderConfig(
        grid_size=experiment_grid_size, image_size=experiment_sketchpad_size
    )
    instructions_df = pd.read_csv(instructions_df_file)
    executions_df = pd.read_csv(executions_df_file)

    assert set(instructions_df.targetId) == set(
        executions_df.targetId
    ), "Instructions and executions must be for the same targets"

    trajectories = []
    for target_id in instructions_df.targetId.unique():
        print(f"Testing target {target_id}")

        target = get_design_from_record(
            eval(instructions_df[instructions_df.targetId == target_id].target.iloc[0])[
                "uncompressed_geometries"
            ],
            render_config=experiment_render_config,
        )
        instructions = instructions_df[
            instructions_df.targetId == target_id
        ].sort_values(by="generation", ascending=True)[
            ["generation", "text", "strokes"]
        ]
        designer = ReplayDesigner(
            [
                (
                    row.text,
                    get_strokes_from_record(
                        eval(row.strokes), experiment_render_config
                    ),
                )
                for _, row in instructions.iterrows()
            ]
        )
        executions = executions_df[executions_df.targetId == target_id].sort_values(
            by="generation", ascending=True
        )[["generation", "jsGeometries"]]

        maker = ReplayMaker(
            [
                get_design_from_record(eval(row.jsGeometries), experiment_render_config)
                for _, row in executions.iterrows()
            ]
        )
        coordinator = SynchronousCoordinator(target, designer, maker)
        try:
            trajectory = coordinator.play()
            trajectories.extend(trajectory)
        except OutOfBoundsError:
            print(f"Ran into out of bounds error for target {target_id}")
            continue
        except Collinear:
            print(f"Ran into collinear error for target {target_id}")
            continue

        print(f"Successfully played target {target_id}")

    if html_save_path is not None:
        trajectories_to_html(
            trajectories,
            html_save_path,
            render_config=RenderConfig(image_size=save_image_size),
        )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
