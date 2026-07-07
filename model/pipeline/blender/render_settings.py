"""Blender Render Configurations."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Render Engine Config."""

    computation_method: Literal[
        "CPU",  # Renders on the same CPU, with multithreading.
        "CUDA",  # Renders on a NVIDIA GPU.
        "OPTIX",  # Renders on a NVIDIA GPU, using ray-tracing.
        "METAL",  # Renders on Apple/macOS.
        "MULTI",  # Renders on multiple devices.
    ] = "OPTIX"
    device: Literal["CPU", "GPU"] = "GPU"
    name: Literal["CYCLES"] = "CYCLES"  # Ray-trace based render engine.


@dataclass(frozen=True, slots=True)
class ImageConfig:
    """Render Image Config."""

    color_mode: Literal[
        "BW",  # Black/White.
        "RGB",  # RGB.
        "RGBA",  # RGB with Alpha (falls back to RGB if Alpha unsupported).
    ] = "RGBA"
    exported_format: Literal["JPEG", "PNG"] = "PNG"
    resolution_x: int = 720
    resolution_y: int = 720
    transparent_background: bool = True


@dataclass(frozen=True, slots=True)
class RenderSettings:
    """Class for storing render configurations."""

    engine: EngineConfig = EngineConfig()
    image: ImageConfig = ImageConfig()
