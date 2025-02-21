from typing import Optional, List, Tuple
import itertools
import openai
import os
from PIL import Image
from io import BytesIO
import base64
import json
import numpy as np
from dataclasses import dataclass
from pydantic import ValidationError
from mrcad import Design, Drawing, Instruction, Action, Role, Execution
from mrcad.agents import AbstractMakerAgent
from .prompts import DESIGN_MAKER_SYSTEM_PROMPT, DESIGN_MAKER_USER_PROMPT


class ChatAgent:
    def make_turn(self, turn: Tuple[Design, Action], round_num: int = None):
        design, action = turn
        if action.role == Role.MAKER:
            return [
                {
                    "role": "assistant",
                    "content": action.design.model_dump_json(),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"The resulting design is:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"""data:image/png;base64,{action.design.to_image(
                                    return_image_type='base64', ignore_out_of_bounds=True
                                )}"""
                            },
                        },
                    ],
                },
            ]
        else:
            if round_num:
                round_num_message = [{"type": "text", "text": f"Round {round_num}. "}]
            else:
                round_num_message = []

            drawing_in_context = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"""data:image/png;base64,{action.drawing.to_image(
                                    image=design.to_image(ignore_out_of_bounds=True), return_image_type='base64'
                                )}"""
                    },
                },
            ]

            if len(action.text) > 0:
                text_instruction = [{"type": "text", "text": action.text}]
            else:
                text_instruction = []

            return [
                {
                    "role": "user",
                    "content": [
                        *round_num_message,
                        *drawing_in_context,
                        *text_instruction,
                        {
                            "type": "text",
                            "text": DESIGN_MAKER_USER_PROMPT,
                        },
                    ],
                }
            ]

    def merge_messages(self, messages):
        if len(messages) == 0:
            return list()

        merged_messages = list()
        current_role = messages[0]["role"]
        current_message_content = list()
        for m in messages:
            if m["role"] == current_role:
                current_message_content.extend(m["content"])
            else:
                merged_messages.append(
                    {"role": current_role, "content": current_message_content}
                )
                current_role = m["role"]
                current_message_content = m["content"]

        if len(current_message_content) > 0:
            merged_messages.append(
                {"role": current_role, "content": current_message_content}
            )

        return merged_messages

    def make_prompt(
        self,
        conversation_history: List[Tuple[Design, Action]],
        demonstrations: List[List[Tuple[Design, Action]]] = None,
    ):
        system_prompt_messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": DESIGN_MAKER_SYSTEM_PROMPT,
                    }
                ],
            }
        ]
        demonstration_prompt_messages = (
            itertools.chain.from_iterable(
                [
                    [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Example game {demo_idx + 1}:",
                                }
                            ],
                        }
                    ]
                    + list(
                        itertools.chain.from_iterable(
                            [
                                self.make_turn(turn, round_num=i // 2 + 1)
                                for i, turn in enumerate(demonstration)
                            ]
                        )
                    )
                    for demo_idx, demonstration in enumerate(demonstrations)
                ]
            )
            if demonstrations
            else []
        )
        turn_prompt_messages = itertools.chain.from_iterable(
            [
                self.make_turn(turn, round_num=i // 2 + 1)
                for i, turn in enumerate(conversation_history)
            ]
        )
        return self.merge_messages(
            [
                *system_prompt_messages,
                *demonstration_prompt_messages,
                {"role": "user", "content": [{"type": "text", "text": "New game:"}]},
                *turn_prompt_messages,
            ]
        )


class VLMDesignMakerAgent(AbstractMakerAgent, ChatAgent):
    def __init__(
        self,
        model,
        demonstrations: List[List[Tuple[Design, Action]]] = None,
        base_url: str = None,
        api_key: str = None,
        max_tokens: int = 256,
        temperature: int = 0.0,
    ):
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.demonstrations = demonstrations

    def act(self, conversation_history: List[Tuple[Design, Action]]):
        prompt = self.make_prompt(
            conversation_history, demonstrations=self.demonstrations
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )

        try:
            execution = Execution(
                design=Design.model_validate_json(response.choices[0].message.content)
            )
            return execution
        except ValidationError as e:
            last_design = (
                conversation_history[-1][0]
                if conversation_history
                else Design(curves=[])
            )
            return Execution(design=last_design)
