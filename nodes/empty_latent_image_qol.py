import torch

from comfy_api.latest import io
from comfy.model_management import intermediate_device, intermediate_dtype


_RESOLUTION_PRESETS: list[tuple[str, int, int]] = [
    ("512 x 512", 512, 512),
    ("768 x 768", 768, 768),
    ("1024 x 1024", 1024, 1024),
    ("1280 x 1280", 1280, 1280),
    ("1536 x 1536", 1536, 1536),
    ("2048 x 2048", 2048, 2048),
    ("896 x 512", 896, 512),
    ("1024 x 576", 1024, 576),
    ("1280 x 720", 1280, 720),
    ("1216 x 832", 1216, 832),
    ("1536 x 864", 1536, 864),
    ("1920 x 1080", 1920, 1080),
]
_RESOLUTION_PRESET_CUSTOM: str = "custom"
_RESOLUTION_INPUT_KWARGS: dict[str, int] = {
    "min": 16,
    "max": 16384,
    "default": 1024,
    "step": 8,
}


class EmptyLatentImageQoL(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        preset_options: list[io.DynamicCombo.Option] = [io.DynamicCombo.Option(
            _RESOLUTION_PRESET_CUSTOM,
            [
                io.Int.Input(
                    id="width",
                    tooltip="The width of the latent images in pixels.",
                    **_RESOLUTION_INPUT_KWARGS,
                ),
                io.Int.Input(
                    id="height",
                    tooltip="The height of the latent images in pixels.",
                    **_RESOLUTION_INPUT_KWARGS,
                ),
            ],
        )] + [io.DynamicCombo.Option(label, []) for label, _, _ in _RESOLUTION_PRESETS]

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
                io.Combo.Input(
                    id="latent_type",
                    display_name="type",
                    tooltip="The type of the latent images.",
                    options=[
                        "default",
                        "sd3",
                        "chroma_radiance",
                        "hunyuan_image",
                        "flux2",
                        "hidream_o1",
                    ],
                ),
                io.DynamicCombo.Input(
                    id="preset",
                    options=preset_options,
                ),
                io.Boolean.Input(
                    id="swap_resolution",
                    tooltip="If true, width and height are swapped.",
                ),
                io.Int.Input(
                    id="batch_size",
                    tooltip="The number of latent images in the batch.",
                    min=1,
                    max=4096,
                    default=1,
                ),
            ],
            outputs=[
                io.Latent.Output("LATENT"),
            ],
        )

    @classmethod
    def execute(cls, latent_type: str, preset: dict[str, str | int], swap_resolution: bool, batch_size: int) -> io.NodeOutput:
        width: int = 1024
        height: int = 1024

        selected_preset: str = preset["preset"]  # type: ignore
        if selected_preset == _RESOLUTION_PRESET_CUSTOM:
            width = preset["width"]  # type: ignore
            height = preset["height"]  # type: ignore
        else:
            for label, preset_width, preset_height in _RESOLUTION_PRESETS:
                if label == selected_preset:
                    width = preset_width
                    height = preset_height
                    break

        if swap_resolution:
            width, height = height, width

        # Some latent types don't specify dtype. This is intentional as they mirror ComfyUI's
        # model-specific empty latent image nodes.
        match latent_type:
            case "default":
                return io.NodeOutput({
                    "samples": torch.zeros((batch_size, 4, height // 8, width // 8), device=intermediate_device(), dtype=intermediate_dtype()),  # type: ignore
                    "downscale_ratio_spacial": 8,
                })
            case "sd3":
                return io.NodeOutput({
                    "samples": torch.zeros((batch_size, 16, height // 8, width // 8), device=intermediate_device(), dtype=intermediate_dtype()),  # type: ignore
                })
            case "chroma_radiance":
                return io.NodeOutput({
                    "samples": torch.zeros((batch_size, 3, height, width), device=intermediate_device()),
                })
            case "hunyuan_image":
                return io.NodeOutput({
                    "samples": torch.zeros((batch_size, 64, height // 32, width // 32), device=intermediate_device()),  # type: ignore
                })
            case "flux2":
                return io.NodeOutput({
                    "samples": torch.zeros((batch_size, 128, height // 16, width // 16), device=intermediate_device()),  # type: ignore
                })
            case "hidream_o1":
                return io.NodeOutput({
                    "samples": torch.zeros((batch_size, 3, height, width), device=intermediate_device()),
                })
            case _:
                raise ValueError(f"Unknown latent type '{latent_type}'")
