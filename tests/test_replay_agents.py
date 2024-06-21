import pandas as pd
from agents.replay_agents import ReplayDesigner, ReplayMaker
from mrcad.design import Design, Line, Arc, Circle
from mrcad.coordinator import SynchronousCoordinator
from mrcad.render_utils import OutOfBoundsError, Collinear, RenderConfig
from mrcad.visualize import trajectories_to_html
import matplotlib.pyplot as plt


def get_strokes_from_record(record, image_size=1280):
    return tuple(
        tuple(
            (point["x"] * image_size / 400, point["y"] * image_size / 400)
            for point in spline
        )
        for spline in record
    )


def get_design_from_record(record):
    points = {
        k: (float(v["i"]) - 20, float(v["j"]) - 20) for k, v in record["points"].items()
    }
    curves_list = []
    for curve in record["curves"]:
        if curve["name"] == "arc_3pt":
            if curve["start"] == curve["end"]:
                curves_list.append(
                    Circle((points[curve["start"]], points[curve["mid"]]))
                )
            else:
                curves_list.append(
                    Arc(
                        (
                            points[curve["start"]],
                            points[curve["mid"]],
                            points[curve["end"]],
                        )
                    )
                )
        elif curve["name"] == "line":
            curves_list.append(Line((points[curve["pt1"]], points[curve["pt2"]])))
    return Design(curves_list)


def main(instructions_df_file, executions_df_file, html_save_path, image_size=320):
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
            ]
        )
        instructions = instructions_df[
            instructions_df.targetId == target_id
        ].sort_values(by="generation", ascending=True)[
            ["generation", "text", "strokes"]
        ]
        designer = ReplayDesigner(
            [
                (row.text, get_strokes_from_record(eval(row.strokes), image_size))
                for _, row in instructions.iterrows()
            ]
        )
        executions = executions_df[executions_df.targetId == target_id].sort_values(
            by="generation", ascending=True
        )[["generation", "jsGeometries"]]

        maker = ReplayMaker(
            [
                get_design_from_record(eval(row.jsGeometries))
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

    trajectories_to_html(
        trajectories,
        html_save_path,
        render_config=RenderConfig(image_size=image_size),
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
