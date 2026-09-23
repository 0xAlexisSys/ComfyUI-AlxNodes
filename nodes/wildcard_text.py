import random
import re

from typing import Optional
from comfy_api.latest import io


_LEFT_BRACE_PLACEHOLDER: str = "\x00LB\x00"
_RIGHT_BRACE_PLACEHOLDER: str = "\x00RB\x00"
_COLON_PLACEHOLDER: str = "\x00C\x00"
_LEFT_CHEVRON_PLACEHOLDER: str = "\x00LC\x00"
_RIGHT_CHEVRON_PLACEHOLDER: str = "\x00RC\x00"
_ESCAPED_BRACE_PATTERN: re.Pattern = re.compile(r"\\([{}:<>])")
_NAMED_WILDCARD_DEFINITION_PATTERN: re.Pattern = re.compile(r"<(?P<id>[^:>]+?):(?:(?P<link_id>[^:>]+?):)?(?P<values>[^>]+?)>")
_NAMED_WILDCARD_PATTERN: re.Pattern = re.compile(r"\{\{(?P<id>[^{}]+?)\}\}")
_WILDCARD_PATTERN: re.Pattern = re.compile(r"(?<!\{)\{(?P<values>[^{}]*?)\}(?!\})")


class WildcardText(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WildcardText",
            display_name="Wildcard Text",
            category="AlxNodes/text",
            description=(
                "Supports {red|green|blue}, <id:a|b>, <id:link_id:a|b>, and escaping special characters "
                "with a backslash. Named wildcards are invoked with {{id}}."
            ),
            search_aliases=[
                "wildcard",
                "text wildcard",
                "text randomizer",
            ],
            inputs=[
                io.String.Input(
                    id="text",
                    multiline=True,
                    dynamic_prompts=False,
                ),
                io.Int.Input(  # ComfyUI automatically adds a "control after generate" input after a "seed" input.
                    id="seed",
                    min=0,
                    max=2 ** 31 - 1,
                    step=1,
                ),
            ],
            outputs=[
                io.String.Output("STRING"),
            ],
        )

    @classmethod
    def execute(cls, text: str, seed: int) -> io.NodeOutput:
        named_wildcard_values: dict[str, tuple[list[str], int]] = {}
        named_wildcard_selected_values: dict[str, str] = {}

        def is_wildcard_empty(values: list[str]) -> bool:
            return len(values) == 0 or all(len(value) == 0 for value in values)

        def escaped_brace_pattern_callback(match: re.Match) -> str:
            match match.group(1):
                case "{":
                    return _LEFT_BRACE_PLACEHOLDER
                case "}":
                    return _RIGHT_BRACE_PLACEHOLDER
                case ":":
                    return _COLON_PLACEHOLDER
                case "<":
                    return _LEFT_CHEVRON_PLACEHOLDER
                case ">":
                    return _RIGHT_CHEVRON_PLACEHOLDER
                case _:
                    return match.group()

        def named_wildcard_definition_pattern_callback(match: re.Match) -> str:
            values: list[str] = [value for value in match.group("values").split("|")]
            if is_wildcard_empty(values):
                return match.group()

            id: str = match.group("id")
            link_id: Optional[str] = match.group("link_id")
            if link_id is None:
                if id in named_wildcard_values:
                    raise KeyError(f"'{id}' already exists")

                value_index: int = random.randint(0, len(values) - 1)
                named_wildcard_values[id] = (values, value_index)
                named_wildcard_selected_values[id] = values[value_index]
            else:
                if link_id == id:
                    raise KeyError(f"'{link_id}' cannot link to itself")

                if link_id in named_wildcard_selected_values:
                    raise KeyError(f"'{link_id}' already exists")

                if len(values) != len(named_wildcard_values[id][0]):
                    raise ValueError(f"'{link_id}' and '{id}' have differing value list lengths")

                value_index: int = named_wildcard_values[id][1]
                named_wildcard_selected_values[link_id] = values[value_index]
            return ""

        def named_wildcard_pattern_callback(match: re.Match) -> str:
            id: str = match.group("id")
            if id not in named_wildcard_selected_values:
                raise KeyError(f"'{id}' does not exist")
            return named_wildcard_selected_values[id]

        def wildcard_pattern_callback(match: re.Match) -> str:
            values: list[str] = [value for value in match.group("values").split("|")]
            return values[random.randint(0, len(values) - 1)] if not is_wildcard_empty(values) else match.group()

        random.seed(seed)
        text = _ESCAPED_BRACE_PATTERN.sub(escaped_brace_pattern_callback, text)
        text = _NAMED_WILDCARD_DEFINITION_PATTERN.sub(named_wildcard_definition_pattern_callback, text)
        text = _NAMED_WILDCARD_PATTERN.sub(named_wildcard_pattern_callback, text)
        text = _WILDCARD_PATTERN.sub(wildcard_pattern_callback, text)
        return io.NodeOutput(text.replace(_LEFT_BRACE_PLACEHOLDER, "{").replace(_RIGHT_BRACE_PLACEHOLDER, "}").replace(_COLON_PLACEHOLDER, ":").replace(_LEFT_CHEVRON_PLACEHOLDER, "<").replace(_RIGHT_CHEVRON_PLACEHOLDER, ">"))
