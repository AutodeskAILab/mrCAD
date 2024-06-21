from typing import Tuple, Optional
from dataclasses import dataclass
import numpy as np
from scipy import interpolate
import cv2
from mrcad.env_utils import Role
from mrcad.design import Design
import mrcad.render_utils as ru


@dataclass
class Drawing:
    splines: Tuple[Tuple[float, float]]

    def to_image(
        self, image: np.ndarray = None, render_config: Optional[ru.RenderConfig] = None
    ):
        if render_config is None:
            render_config = ru.RenderConfig()

        if image is None:
            # create a blank RBG image
            image = np.ones(
                (
                    render_config.image_size,
                    render_config.image_size,
                    3,
                )
                * np.array(render_config.get_background_color())
            )

        for spline in self.splines:
            deduplicated_spline = list()
            for pt in spline:
                if pt not in deduplicated_spline:
                    deduplicated_spline.append(pt)

            if len(deduplicated_spline) <= 3:
                # need more than 3 points to interpolate a spline, so render as a line instead
                spline_array = np.array(deduplicated_spline)
                image = cv2.polylines(
                    image,
                    [spline_array.astype(np.int32)],
                    False,
                    render_config.get_drawing_color(),
                    thickness=render_config.drawing_thickness,
                )
            else:
                spline_array = np.array(deduplicated_spline)
                x = spline_array[:, 0]
                y = spline_array[:, 1]
                tck, u = interpolate.splprep([x, y], s=render_config.spline_smoothing)
                x_shape, y_shape = interpolate.splev(
                    np.linspace(0, 1, render_config.spline_resolution), tck, der=0
                )
                spline_shape = np.column_stack(
                    (x_shape.astype(np.int32), y_shape.astype(np.int32))
                )
                image = cv2.polylines(
                    image,
                    [spline_shape],
                    False,
                    render_config.get_drawing_color(),
                    thickness=render_config.drawing_thickness,
                )

        return image


@dataclass
class Action:
    role: Role
    instruction: tuple[str, Drawing]
    design: Design
