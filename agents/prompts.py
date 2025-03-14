DESIGN_MAKER_SYSTEM_PROMPT = """You are an expert CAD software user playing a game called mrCAD. In this game, there is a designer and a maker. The two players work together to iteratively create a design over a sequence of turns. You will play the role of the maker in this game, and the user will play the role of the designer. In each turn the designer provides an instruction about how to modify the design on the canvas. The instruction may include language instructions, drawings on the canvas, or both. The drawings appear as red strokes on the canvas. The design appears in black strokes on the canvas. Your goal is to follow the designer's instructions. The output is in the form of a program that draws the design, which is represented as a JSON object. The JSON object has one key, 'curves', which is a list of curves. Each curve is a dictionary that has two keys: 'type' and 'control_points'. The 'type' key is a string that specifies the type of curve. The only possible types of curves are 'line', 'arc', and 'circle'. The 'control_points' key is a list of points represent the control points that define the curve. For a line, there are exactly two control points indicating the endpoints of the line. For an arc, there are exactly three control points. The first and third indicate the endpoints of the arc, while the second indicates some point along the arc. For a circle, there are exactly two control points that are two points along any diameter of the circle. Each control point is a pair of floating point numbers between -20 and 20 that represent the coordinates of the point on the canvas."""

DESIGN_EDITOR_SYSTEM_PROMPT = """You are an expert CAD software user playing a game called mrCAD. In this game, there is a designer and a maker. The two players work together to iteratively create a design over a sequence of turns. You will play the role of the maker in this game, and the user will play the role of the designer. In each turn the designer provides an instruction about how to modify the design on the canvas. The instruction may include language instructions, drawings on the canvas, or both. The drawings appear as red strokes on the canvas. The design appears in black strokes on the canvas. Your goal is to follow the designer's instructions. You have to take actions to edit the current state of the design. Each action is taken by calling a tool that performs the action. Each control point is a pair of floating point numbers between -20 and 20 that represent the coordinates of the point on the canvas. """


DESIGN_MAKER_USER_PROMPT = "\nGenerate the JSON representation of the program that renders the design. Make sure to follow the instructions carefully. Do NOT copy the previous response. Produce the JSON within a block formated as ```json\n...\n```."

DESIGN_EDITOR_USER_PROMPT = "\nEdit the design based on the designer's instructions using the provided tools. Make sure to follow the instructions carefully."

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "move_point",
            "strict": True,
            "parameters": {
                "properties": {
                    "point": {
                        "description": "The point to move.",
                        "items": {"type": "number"},
                        "type": "array",
                    },
                    "new_point": {
                        "description": "The new location of the point.",
                        "items": {"type": "number"},
                        "type": "array",
                    },
                },
                "required": ["point", "new_point"],
                "type": "object",
                "additionalProperties": False,
            },
            "description": "Move a point to a new location. Every curve that has a control point at the given point will have the control point moved to the new location.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_curve",
            "strict": True,
            "parameters": {
                "properties": {
                    "type": {
                        "description": "The type of curve to make.",
                        "enum": ["line", "arc", "circle"],
                        "type": "string",
                    },
                    "control_points": {
                        "description": "The control points of the curve. Each control point is a list with a pair of floating point numbers that represent the coordinates of the point on the canvas. A line is represented by two control points, each corresponding to one end of the line. A circle is represented two control points that are points on either end of a diameter of the circle. An arc is represented by three control points. The first and third control points are the endpoints of the arc, while the second control point is a point along the arc.",
                        "items": {
                            "items": {"type": "number"},
                            "type": "array",
                        },
                        "type": "array",
                    },
                },
                "required": ["type", "control_points"],
                "type": "object",
                "additionalProperties": False,
            },
            "description": "Add a new curve to the design given the type of the curve and its control points.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_curve",
            "strict": True,
            "parameters": {
                "properties": {
                    "type": {
                        "description": "The type of the curve to move.",
                        "enum": ["line", "arc", "circle"],
                        "type": "string",
                    },
                    "control_points": {
                        "description": "The control points of the curve to move. Each control point is a list with a pair of floating point numbers that represent the coordinates of the point on the canvas.",
                        "items": {
                            "items": {"type": "number"},
                            "type": "array",
                        },
                        "type": "array",
                    },
                    "offset": {
                        "description": "The offset to move the curve. The offset is a pair of floating point numbers that represent how much the curve has to be moved in the x and y directions respectively.",
                        "items": {"type": "number"},
                        "type": "array",
                    },
                },
                "required": ["type", "control_points", "offset"],
                "type": "object",
                "additionalProperties": False,
            },
            "description": "Move a curve to a new location. The curve to be moved is identified by its type and control points. The curve will be moved by the given offset.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_curve",
            "strict": True,
            "parameters": {
                "properties": {
                    "type": {
                        "description": "The type of the curve to remove.",
                        "enum": ["line", "arc", "circle"],
                        "type": "string",
                    },
                    "control_points": {
                        "description": "The control points of the curve to remove.",
                        "items": {
                            "items": {"type": "number"},
                            "type": "array",
                        },
                        "type": "array",
                    },
                },
                "required": ["type", "control_points"],
                "type": "object",
                "additionalProperties": False,
            },
            "description": "Remove a curve from the design. The curve to be removed is identified by its type and control points.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_point",
            "strict": True,
            "parameters": {
                "properties": {
                    "point": {
                        "description": "The point to delete.",
                        "items": {"type": "number"},
                        "type": "array",
                    }
                },
                "required": ["point"],
                "type": "object",
                "additionalProperties": False,
            },
            "description": "Delete a point from the design. Every curve that has a control point at the given point will be removed.",
        },
    },
]

QWEN_CUSTOM_CHAT_TEMPLATE = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}\n{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'][0]['text'] }}\n    {%- else %}\n        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'][0]['text'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{% for message in messages %}\n{% if message['content'] is string %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% else %}{% if (message.role == 'user') or (message.role == 'system' and not loop.first) or (message.role == 'assistant' and not message.tool_calls) %}{{- '<|im_start|>' + message.role + '\n'}}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
