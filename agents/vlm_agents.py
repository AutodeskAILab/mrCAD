from typing import Optional, List, Tuple
import itertools
import openai
import re
import os
from PIL import Image
from io import BytesIO
import base64
import json
import numpy as np
from dataclasses import dataclass
from pydantic import ValidationError
from mrcad import Design, Drawing, Instruction, Action, Role, Execution, RenderConfig
from mrcad.agents import AbstractMakerAgent
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    GenerationConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from peft import PeftModel
from agents.prompts import (
    DESIGN_MAKER_SYSTEM_PROMPT,
    DESIGN_MAKER_USER_PROMPT,
    DESIGN_EDITOR_SYSTEM_PROMPT,
    DESIGN_EDITOR_USER_PROMPT,
    TOOLS,
    QWEN_CUSTOM_CHAT_TEMPLATE,
)
from agents.editing_actions import (
    Edit,
    EditExecution,
    MovePoint,
    MakeCurve,
    MoveCurve,
    RemoveCurve,
    DeletePoint,
)
from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)
from tenacity import retry, stop_after_attempt, wait_random_exponential


class ChatAgent:
    def __init__(
        self,
        system_prompt,
        user_prompt,
        image_format: str = "base64",
        image_size: int = 256,
    ):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.image_size = image_size
        self.image_format = image_format

    def make_image(self, design: Design = None, drawing: Drawing = None):
        if design is None and drawing is None:
            raise ValueError("Either design or drawing must be provided")

        if drawing:
            image = drawing.to_image(
                image=design.to_image(
                    ignore_out_of_bounds=True,
                    render_config=RenderConfig(image_size=self.image_size),
                ),
                return_image_type=self.image_format,
                render_config=RenderConfig(image_size=self.image_size),
            )
        else:
            image = design.to_image(
                ignore_out_of_bounds=True,
                return_image_type=self.image_format,
                render_config=RenderConfig(image_size=self.image_size),
            )

        if self.image_format == "base64":
            return f"""data:image/png;base64,{image}"""
        else:
            return image

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
                        "text": self.system_prompt,
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
                                self.make_round(turn, round_num=i // 2 + 1)
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
                self.make_round(turn, round_num=i // 2 + 1)
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


class ChatMakerAgent(ChatAgent):
    def __init__(self, image_format="base64", image_size=256):
        super().__init__(
            system_prompt=DESIGN_MAKER_SYSTEM_PROMPT,
            user_prompt=DESIGN_MAKER_USER_PROMPT,
            image_format=image_format,
            image_size=image_size,
        )

    def make_round(self, turn: Tuple[Design, Action], round_num: int = None):
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
                                "url": self.make_image(action.design),
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
                        "url": self.make_image(design, action.drawing),
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
                            "text": self.user_prompt,
                        },
                    ],
                }
            ]


class ChatEditorAgent(ChatAgent):
    def __init__(self, image_format="base64", image_size=256):
        super().__init__(
            system_prompt=DESIGN_EDITOR_SYSTEM_PROMPT,
            user_prompt=DESIGN_EDITOR_USER_PROMPT,
            image_format=image_format,
            image_size=image_size,
        )

        self.tools = TOOLS

    def format_tool_calls(self, tool_calls):
        return json.dumps(tool_calls)

    def prep_tool(self, editing_action: Edit):
        tool = openai.pydantic_function_tool(
            editing_action, name=editing_action.model_fields["edit_type"].default
        )
        _ = tool["function"]["parameters"]["properties"].pop("edit_type")

        for p in tool["function"]["parameters"]["properties"]:
            _ = tool["function"]["parameters"]["properties"][p].pop("minItems", None)
            _ = tool["function"]["parameters"]["properties"][p].pop("maxItems", None)

        return tool

    def make_round(self, turn: Tuple[Design, Action], round_num: int = None):
        design, action = turn
        if action.role == Role.MAKER:
            return [
                {
                    "role": "assistant",
                    "content": self.format_tool_calls(
                        [edit.make_tool_call() for edit in action.edits]
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"The resulting design is:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self.make_image(action.design),
                            },
                        },
                        {"type": "text", "text": action.design.model_dump_json()},
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
                        "url": self.make_image(design, action.drawing),
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
                            "text": self.user_prompt,
                        },
                    ],
                }
            ]


class VLMDesignMakerAgent(ChatMakerAgent, AbstractMakerAgent):
    def __init__(
        self,
        model,
        demonstrations: List[List[Tuple[Design, Action]]] = None,
        base_url: str = None,
        api_key: str = None,
        max_tokens: int = 1024,
        temperature: int = 0.0,
        image_size: int = 256,
    ):
        super().__init__(image_size=image_size, image_format="base64")
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.demonstrations = demonstrations

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(1))
    def act(self, conversation_history: List[Tuple[Design, Action]]):
        prompt = self.make_prompt(
            conversation_history, demonstrations=self.demonstrations
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        extracted_response = self.extract_json_blocks(
            response.choices[0].message.content
        )
        try:
            execution = Execution(design=Design.model_validate_json(extracted_response))
            return execution
        except ValidationError as e:
            print(extracted_response)
            last_design = (
                conversation_history[-1][0]
                if conversation_history
                else Design(curves=[])
            )
            return Execution(design=last_design)


class VLMDesignEditorAgent(ChatEditorAgent, AbstractMakerAgent):
    tool_to_editing_action = {
        "move_point": MovePoint,
        "make_curve": MakeCurve,
        "move_curve": MoveCurve,
        "remove_curve": RemoveCurve,
        "delete_point": DeletePoint,
    }

    def __init__(
        self,
        model,
        demonstrations: List[List[Tuple[Design, Action]]] = None,
        base_url: str = None,
        api_key: str = None,
        max_tokens: int = 1024,
        temperature: int = 0.0,
        top_p: float = 1.0,
        image_size: int = 256,
    ):
        super().__init__(image_size=image_size, image_format="base64")
        if not (base_url is None and api_key is None):
            self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        else:
            self.client = None
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.demonstrations = demonstrations

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def act(self, conversation_history: List[Tuple[Design, Action]]):
        prompt = self.make_prompt(
            conversation_history, demonstrations=self.demonstrations
        )

        tool_choice_arg = "required" if "gpt" in self.model else "auto"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
            tools=self.tools,
            tool_choice=tool_choice_arg,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        edits = list()
        if response.choices and response.choices[0].message.tool_calls:
            for tool_call in response.choices[0].message.tool_calls:
                editing_action = self.tool_to_editing_action.get(
                    tool_call.function.name
                )
                if editing_action:
                    try:
                        edit = editing_action.model_validate_json(
                            tool_call.function.arguments
                        )
                        edits.append(edit)
                    except ValidationError as e:
                        continue

        try:
            return EditExecution.execute(conversation_history[-1][0], edits)
        except ValidationError as e:
            return EditExecution(design=conversation_history[-1][0], edits=[])


class QwenDesignEditorAgent(VLMDesignEditorAgent, AbstractMakerAgent):
    def __init__(
        self,
        model,
        demonstrations: List[List[Tuple[Design, Action]]] = None,
        base_url: str = None,
        api_key: str = None,
        max_tokens: int = 1024,
        temperature: int = 0.0,
        top_p: float = 1.0,
        image_size: int = 256,
    ):
        super().__init__(
            model=model,
            demonstrations=demonstrations,
            base_url=base_url,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            image_size=image_size,
        )

        self.chat_template = QWEN_CUSTOM_CHAT_TEMPLATE

    def format_tool_calls(self, tool_calls):
        return " ".join(
            [f"<tool_call>{tool_call}</tool_call>" for tool_call in tool_calls]
        )
