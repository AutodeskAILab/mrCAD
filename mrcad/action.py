from typing import Tuple, Optional
from dataclasses import dataclass
import numpy as np
from scipy import interpolate
import cv2
from mrcad.env_utils import Role
from mrcad.design import Design
import mrcad.render_utils as ru
from PIL import Image
from io import BytesIO
import base64


@dataclass
class Drawing:
    splines: Tuple[Tuple[float, float]]

    def render(
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

                spline_shape = np.array(
                    [
                        render_config.transform(x, y)
                        for x, y in np.column_stack((x_shape, y_shape)).tolist()
                    ],
                    dtype=np.int32,
                )

                image = cv2.polylines(
                    image,
                    [spline_shape],
                    False,
                    render_config.get_drawing_color(),
                    thickness=render_config.drawing_thickness,
                )

        return image

    def to_image(
        self,
        return_image_type: str = "PIL.Image",
        image=None,  # Render onto an existing image if provided else initialize a blank image
        render_config: ru.RenderConfig = None,
    ):
        assert return_image_type in [
            "PIL.Image",
            "numpy.ndarray",
            "base64",
        ], f"Unknown return_image_type: {return_image_type}"

        canvas = None
        if image is not None:
            if isinstance(image, np.ndarray):
                if (
                    image.dtype == np.float64
                    and image.max() <= 1.0
                    and image.min() >= 0.0
                ):
                    canvas = image
                elif (
                    image.dtype == np.uint8 and image.max() <= 255 and image.min() >= 0
                ):
                    canvas = image / 255.0
                else:
                    raise ValueError("Invalid image type")
            elif isinstance(image, bytes):
                canvas = np.array(Image.open(BytesIO(image))) / 255.0
            elif isinstance(image, Image.Image):
                canvas = np.array(image) / 255.0
            else:
                raise ValueError("Invalid image type provided as input")

        rendered = self.render(image=canvas, render_config=render_config)

        if return_image_type == "PIL.Image":
            return Image.fromarray((rendered * 255).astype(np.uint8))
        elif return_image_type == "numpy.ndarray":
            return rendered
        elif return_image_type == "base64":
            img = Image.fromarray((rendered * 255).astype(np.uint8))
            with BytesIO() as buffer:
                img.save(buffer, format="PNG")
                img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            return img_b64


@dataclass
class Action:
    role: Role
    instruction: tuple[str, Drawing]
    design: Design
