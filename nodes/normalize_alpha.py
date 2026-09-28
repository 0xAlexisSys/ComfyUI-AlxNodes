import torch

from comfy_api.latest import io


class NormalizeAlpha(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NormalizeAlpha",
            display_name="Normalize Alpha",
            category="AlxNodes/image",
            description="Binarizes the alpha channel of a RGBA image.",
            search_aliases=[
                "alpha",
                "rgb",
                "rgba",
                "clean alpha",
                "binarize alpha",
                "binarise alpha",
            ],
            inputs=[
                io.Image.Input("image"),
                io.Float.Input(
                    id="threshold_black",
                    tooltip="Alpha channel values at or below this threshold become fully transparent (0).",
                    default=127.0,
                    min=0.0,
                    max=255.0,
                    step=1.0,
                ),
                io.Float.Input(
                    id="threshold_white",
                    tooltip="Alpha channel values at or above this threshold become fully opaque (255).",
                    default=128.0,
                    min=0.0,
                    max=255.0,
                    step=1.0,
                ),
                io.Boolean.Input(
                    id="fix_alpha_colors",
                    tooltip="If true, sets the RGB channels of normalized transparent pixels to zero.",
                ),
            ],
            outputs=[
                io.Image.Output("IMAGE"),
            ],
        )

    @classmethod
    def execute(cls, image: torch.Tensor, threshold_black: float, threshold_white: float, fix_alpha_colors: bool) -> io.NodeOutput:
        if image.dim() != 4 or image.shape[-1] < 4:
            return io.NodeOutput(image)

        final_image: torch.Tensor = image.clone()
        threshold_black = max(0.0, min(255.0, threshold_black)) / 255.0
        threshold_white = max(0.0, min(255.0, threshold_white)) / 255.0

        # Alpha channel is the last dimension (index -1), values in [0, 1].
        alpha: torch.Tensor = final_image[..., 3]

        # Compute midpoint between thresholds for proximity snap.
        midpoint: float = (threshold_black + threshold_white) / 2.0

        # True where alpha <= midpoint → transparent
        mask_zero: torch.Tensor = alpha <= midpoint

        final_image[..., 3] = torch.where(mask_zero, torch.zeros_like(alpha), torch.ones_like(alpha))
        if fix_alpha_colors and mask_zero.any():
            mask_expanded: torch.Tensor = mask_zero.unsqueeze(-1)  # [B, H, W, 1]
            final_image[..., :3] = torch.where(mask_expanded, torch.zeros_like(final_image[..., :3]), final_image[..., :3])
        return io.NodeOutput(final_image)
