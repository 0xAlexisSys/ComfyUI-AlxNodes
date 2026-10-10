from comfy_api.latest import Input, io


class ClipTextEncodePosNeg(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ClipTextEncodePosNeg",
            display_name="CLIP Text Encode (Positive/Negative)",
            category="AlxNodes/model",
            description="Encodes a positive text prompt and negative text prompt using a CLIP model.",
            search_aliases=[
                "text",
                "prompt",
                "text prompt",
                "positive prompt",
                "negative prompt",
                "encode text",
                "text encoder",
                "encode prompt",
            ],
            inputs=[
                io.Clip.Input(
                    id="clip",
                    tooltip="The CLIP model used for encoding the text.",
                ),
                io.String.Input(
                    id="positive_text",
                    tooltip="The text to encode as a positive prompt.",
                    multiline=True,
                    dynamic_prompts=True,
                ),
                io.String.Input(
                    id="negative_text",
                    tooltip="The text to encode as a negative prompt.",
                    multiline=True,
                    dynamic_prompts=True,
                ),
            ],
            outputs=[
                io.Conditioning.Output("POSITIVE"),
                io.Conditioning.Output("NEGATIVE"),
            ],
        )

    @classmethod
    def execute(cls, clip: io.Clip, positive_text: str, negative_text: str) -> io.NodeOutput:
        return io.NodeOutput(
            clip.encode_from_tokens_scheduled(clip.tokenize(positive_text)),  # type: ignore
            clip.encode_from_tokens_scheduled(clip.tokenize(negative_text)),  # type: ignore
        )
