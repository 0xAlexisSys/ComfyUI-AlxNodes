from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from .nodes import *


class AlxNodesExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            DropAlpha,
            EmptyLatentImageQoL,
        ]


async def comfy_entrypoint() -> AlxNodesExtension:
    return AlxNodesExtension()
