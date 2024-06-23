from typing import Optional, List, Tuple
import openai
import os
from PIL import Image
from io import BytesIO
import base64
import json
import numpy as np
from dataclasses import dataclass
from mrcad.agents import AbstractMakerAgent
from mrcad.env import MakerObservation
from mrcad.env_utils import Role
from mrcad.design import Design
from mrcad.action import Action


class GPT4DesignMakerAgent(AbstractMakerAgent):
    def __init__(self, engine="gpt-4o", api_key=None, max_tokens=256, temperature=0.1):
        self.client = openai.OpenAI(api_key=api_key)
        self.engine = engine
        self.max_tokens = max_tokens
        self.temperature = temperature

    def make_prompt(
        self,
        observation: MakerObservation,
        demonstrations: Optional[List[Tuple[Design, Action, Design]]] = None,
    ):
        system_prompt_messages = self.make_system_prompt_messages()
        demonstration_prompt_messages = (
            self.make_demonstration_prompt_messages(demonstrations)
            if demonstrations
            else []
        )
        turn_prompt_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Current design:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": self.get_image_string(
                                observation.current_design.to_image()
                            )
                        },
                    },
                    {
                        "type": "text",
                        "text": f"JSON for current design: {json.dumps(observation.current_design.to_json())}",
                    },
                    {
                        "type": "text",
                        "text": f"Instruction: {observation.instruction[0]}",
                    },
                    {"type": "text", "text": f"Drawing:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": self.get_image_string(
                                observation.instruction[1].to_image(
                                    observation.current_design.to_image()
                                )
                            )
                        },
                    },
                ],
            }
        ]
        return [
            *system_prompt_messages,
            *demonstration_prompt_messages,
            *turn_prompt_messages,
        ]

    def make_system_prompt_messages(self):
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are playing a game called mrCAD. In this game, there is a designer and a maker. "
                        "The two players work together to iteratively create a design over a sequence of turns. In each turn "
                        "the designer provides an instruction to the maker about how to modify the design on the canvas. "
                        "The instruction may include language instructions, drawings on the canvas, or both. The drawings appear "
                        "as red strokes on the canvas. The design appears in black strokes on the canvas. The maker's goal is to "
                        "produce an updated design. The maker has to produce the entire design, including the portions already "
                        "on the current canvas. The output is in the form of a program that draws the design, which is represented "
                        "as a JSON object. The JSON object has one key, 'curves', which is a list of curves. Each curve is a dictionary "
                        "that has two keys: 'type' and 'control_points'. The 'type' key is a string that specifies the type of curve. "
                        "The only possible types of curves are 'line', 'arc', and 'circle'. The 'control_points' key is a list of points "
                        "represent the control points that define the curve. For a line, there are exactly two control points indicating "
                        "the endpoints of the line. For an arc, there are exactly three control points. The first and third indicate the "
                        "endpoints of the arc, while the second indicates some point along the arc. For a circle, there are exactly two "
                        "control points that are two points along any diameter of the circle. Each control point is a pair of floating "
                        "point numbers between -20 and 20 that represent the coordinates of the point on the canvas. "
                        "You will play the role of the maker in this game, and the user will play the role of the designer. "
                        "Here are some examples of turns of the game.",
                    }
                ],
            }
        ]

    def get_image_string(self, image: np.ndarray):
        img = Image.fromarray((image * 255).astype(np.uint8))
        with BytesIO() as buffer:
            img.save(buffer, format="PNG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_b64}"

    def make_demonstration_prompt_messages(
        self, demonstrations: List[Tuple[Design, Action, Design]]
    ):
        prompt_messages = []
        for turn in demonstrations:
            design, action, new_design = turn
            prompt_messages.extend(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Current design:"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": self.get_image_string(design.to_image())
                                },
                            },
                            {
                                "type": "text",
                                "text": f"JSON for current design: {json.dumps(design.to_json())}",
                            },
                            {
                                "type": "text",
                                "text": f"Instruction: {action.instruction[0]}",
                            },
                            {"type": "text", "text": f"Drawing:"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": self.get_image_string(
                                        action.instruction[1].to_image(
                                            design.to_image()
                                        )
                                    )
                                },
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": json.dumps(new_design.to_json())}
                        ],
                    },
                ]
            )
        return prompt_messages

    def act(self, observation: MakerObservation):
        prompt = self.make_prompt(observation)
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        return Action(
            Role.MAKER,
            None,
            Design.from_json(json.loads(response.choices[0].message.content)),
        )
