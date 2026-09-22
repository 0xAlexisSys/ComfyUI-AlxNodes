import torch

from comfy_api.latest import io
from comfy.model_management import intermediate_device, intermediate_dtype


RESOLUTION_PRESETS: list[tuple[str, int, int]] = [
    ("512 x 512 (square)", 512, 512),
    ("768 x 768 (square)", 768, 768),
    ("1024 x 1024 (square)", 1024, 1024),
    ("1536 x 1536 (square)", 1536, 1536),
    ("896 x 512 (landscape)", 896, 512),
    ("1280 x 720 (landscape)", 1280, 720),
    ("1216 x 832 (landscape)", 1216, 832),
    ("1536 x 864 (landscape)", 1536, 864),
    ("512 x 896 (portrait)", 512, 896),
    ("720 x 1280 (portrait)", 720, 1280),
    ("832 x 1216 (portrait)", 832, 1216),
    ("864 x 1536 (portrait)", 864, 1536),
]
RESOLUTION_PRESET_CUSTOM: str = "custom"
RESOLUTION_INPUT_KWARGS: dict[str, int] = {
    "min": 16,
    "max": 16384,
    "default": 1024,
    "step": 8,
}


class EmptyLatentImageQoL(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        preset_options: list[io.DynamicCombo.Option] = [io.DynamicCombo.Option(label, []) for label, _, _ in RESOLUTION_PRESETS]
        preset_options.append(io.DynamicCombo.Option(
            RESOLUTION_PRESET_CUSTOM,
            [
                io.Int.Input(
                    id="width",
                    tooltip="The width of the latent images in pixels.",
                    **RESOLUTION_INPUT_KWARGS,
                ),
                io.Int.Input(
                    id="height",
                    tooltip="The height of the latent images in pixels.",
                    **RESOLUTION_INPUT_KWARGS,
                ),
                io.Int.Input(  # HACK: ComfyUI has no Python-native way to add buttons. It's ugly but it works.
                    id="flip",
                    min=0,
                    max=1,
                    default=0,
                    tooltip="If set to 1, width and height are flipped.",
                ),
            ],
        ))

        return io.Schema(
            node_id="EmptyLatentImageQoL",
            display_name="Empty Latent Image QoL",
            category="AlxNodes/latent",
            description="Same as the Empty Latent Image node, but with quality-of-life additions.",
            search_aliases=[
                "empty",
                "empty latent",
                "new latent",
                "create latent",
                "blank latent",
                "blank",
            ],
            inputs=[
                io.DynamicCombo.Input(
                    id="preset",
                    options=preset_options,
                ),
                io.Int.Input(
                    id="batch_size",
                    min=1,
                    max=4096,
                    default=1,
                    tooltip="The number of latent images in the batch.",
                ),
            ],
            outputs=[
                io.Latent.Output("LATENT"),
            ],
        )

    @classmethod
    def execute(cls, preset: dict[str, str | int], batch_size: int) -> io.NodeOutput:
        selected_preset = preset["preset"]

        if selected_preset == RESOLUTION_PRESET_CUSTOM:
            width, height = preset["width"], preset["height"]
            if preset["flip"] == 1:
                height, width = width, height
        else:
            for label, preset_width, preset_height in RESOLUTION_PRESETS:
                if label == selected_preset:
                    width, height = preset_width, preset_height
                    break

        latent: torch.Tensor = torch.zeros(
            [batch_size, 4, height // 8, width // 8],
            device=intermediate_device(),
            dtype=intermediate_dtype(),
        )
        return io.NodeOutput({
            "samples": latent,
            "downscale_ratio_spacial": 8,
        })
