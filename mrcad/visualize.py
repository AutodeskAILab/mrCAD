from typing import Tuple
from mrcad.env import State
from mrcad.action import Action, Drawing
from mrcad.env_utils import Role
import mrcad.render_utils as ru
from pathlib import Path
import matplotlib.pyplot as plt


def trajectories_to_html(
    trajectories: Tuple[State, Action, float, State],
    save_path: str,
    render_config: ru.RenderConfig = None,
):
    if render_config is None:
        render_config = ru.RenderConfig()

    save_dir = Path(save_path)
    (save_dir / "images").mkdir(parents=True, exist_ok=True)

    rows = []
    for i, (state, action, reward, next_state) in enumerate(trajectories):
        if state.turn == Role.DESIGNER:
            drawing = Drawing(action.instruction[1])
            text = action.instruction[0]
            continue
        else:
            target = state.target
            current = state.current_design
            next_design = next_state.current_design

        try:
            plt.imsave(
                save_dir / "images" / f"target_{i}.png",
                target.to_image(
                    ignore_out_of_bounds=True,
                    render_config=render_config,
                ),
            )
            plt.imsave(
                save_dir / "images" / f"current_{i}.png",
                current.to_image(
                    ignore_out_of_bounds=True,
                    render_config=render_config,
                ),
            )
            plt.imsave(
                save_dir / "images" / f"drawing_{i}.png",
                drawing.to_image(
                    current.to_image(
                        ignore_out_of_bounds=True,
                        render_config=render_config,
                    ),
                    render_config=render_config,
                ),
            )
            plt.imsave(
                save_dir / "images" / f"next_{i}.png",
                next_design.to_image(
                    ignore_out_of_bounds=True,
                    render_config=render_config,
                ),
            )
        except ru.Collinear:
            continue

        rows.append(
            f"""
        <tr>
            <td><img src="./images/target_{i}.png" /></td>
            <td><img src="./images/current_{i}.png" /></td>
            <td><img src="./images/drawing_{i}.png" /></td>
            <td>{text}</td>
            <td><img src="./images/next_{i}.png" /></td>
            <td>{reward[Role.MAKER]}</td>
        </tr>
        """
        )

    with open(save_dir / "index.html", "w") as f:
        rows_str = "\n".join(rows)
        f.write(
            f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simple HTML Table</title>
    <style>
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            border: 1px solid black;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
    </style>
</head>
<body>
    <h2>Trajectories</h2>
    <table>
        <thead>
            <tr>
                <th>Target CAD</th>
                <th>Current CAD</th>
                <th>Drawing</th>
                <th>Text</th>
                <th>Next CAD</th>
                <th>Reward</th>
            </tr>
        </thead>
        <tbody>
        {rows_str}
        </tbody>
    </table>
</body>
</html>
"""
        )
