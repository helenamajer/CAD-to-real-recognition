"""Forwards all Blender Events."""

from .extension_events import ExtensionEvents
from .material_events import MaterialEvents
from .object_events import ObjectEvents
from .scene_events import SceneEvents


# pylint: disable=too-few-public-methods
class Events:
    """Forwards all Blender Events."""

    Extension = ExtensionEvents
    Material = MaterialEvents
    Object = ObjectEvents
    Scene = SceneEvents
