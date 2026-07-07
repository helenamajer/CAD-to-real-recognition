"""Commonly used Blender object events."""

from typing import Literal, Sequence
from pathlib import Path
import bpy
from mathutils import Vector


class ObjectEvents:
    """Blender object events."""

    # Add more if necessary
    __native_formats = tuple(["obj"])

    @staticmethod
    def center_object(obj: bpy.types.Object):
        """Centers the object."""
        if ObjectEvents.get_active_object() is not obj:
            ObjectEvents.set_active_object(obj)
        bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_MASS", center="MEDIAN")
        obj.location = (0.0, 0.0, 0.0)

    @staticmethod
    def delete_selected_objects():
        """Deletes all selected objects."""
        bpy.ops.object.delete()
        bpy.ops.object.select_all(action="DESELECT")

    @staticmethod
    def export_objects(output_file: Path):
        """Exports the objects in a scene."""
        fmt = output_file.suffix.lstrip(".").lower()
        if fmt not in ObjectEvents.__native_formats:
            print(f"`{fmt}` is unsupported. Resorting to OBJ format!")
            fmt = ObjectEvents.__native_formats[0]
        method = getattr(bpy.ops.wm, f"{fmt}_export")
        method(filepath=str(output_file))

    @staticmethod
    def get_active_object():
        """Returns the active object in the scene."""
        obj = bpy.context.view_layer.objects.active
        return obj

    @staticmethod
    def get_object(name: str):
        """Returns a single object from the scene."""
        for obj in ObjectEvents.get_objects():
            if obj.name == name:
                return obj
        return None

    @staticmethod
    def get_objects():
        """Returns all objects in the scene."""
        return tuple(bpy.context.view_layer.objects)

    @staticmethod
    def get_bbox_size(obj: bpy.types.Object):
        """Returns the bbox of a given object."""
        # pylint: disable=line-too-long
        bbox_corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]  # type: ignore (FAKE PANIC)
        bbox_min = Vector(
            (
                min(c.x for c in bbox_corners),  # type: ignore (FAKE PANIC)
                min(c.y for c in bbox_corners),  # type: ignore (FAKE PANIC)
                min(c.z for c in bbox_corners),  # type: ignore (FAKE PANIC)
            )
        )
        bbox_max = Vector(
            (
                max(c.x for c in bbox_corners),  # type: ignore (FAKE PANIC)
                max(c.y for c in bbox_corners),  # type: ignore (FAKE PANIC)
                max(c.z for c in bbox_corners),  # type: ignore (FAKE PANIC)
            )
        )
        return (bbox_max - bbox_min).length

    @staticmethod
    def import_native_object(input_file: Path):
        """Imports a native object to the scene."""
        fmt = input_file.suffix.lstrip(".").lower()
        method = getattr(bpy.ops.wm, f"{fmt}_import")
        method(filepath=str(input_file))

    @staticmethod
    def import_dxf_object(input_file: Path):
        """Imports a non-native object to the scene."""
        bpy.ops.import_scene.dxf(  # type: ignore (FAKE PANIC)
            files=[{"name": str(input_file)}], merge_lines=True
        )

    @staticmethod
    def join_selected():
        """Joins all selected objects in the scene.

        :NOTE: joined objects update the active object (see Blender's API)."""
        bpy.ops.object.join()

    @staticmethod
    def rename_object(obj: bpy.types.Object, name: str):
        """Renames an object in Blender."""
        obj.name = name

    @staticmethod
    def select_objects(
        obj_type: Sequence[Literal["MESH", "LIGHT", "CAMERA"]] | None = None,
    ):
        """Selects all objects in a scene. If a specific type is requested,
        it will grab objects of that type."""
        objs = ObjectEvents.get_objects()
        for obj in objs:
            if (obj_type is None) or (obj.type in obj_type):
                obj.select_set(True)

    @staticmethod
    def set_active_object(obj: bpy.types.Object | None = None):
        """Sets the active object in the scene."""
        if obj is None:
            obj = bpy.context.view_layer.objects[0]
        bpy.context.view_layer.objects.active = obj

    @staticmethod
    def set_light_color(light: bpy.types.Object, rgb: Vector):
        """Sets the color of a light.
        TODO: Cast Object to Light."""
        light.data.color = rgb  # type: ignore

    @staticmethod
    def set_light_energy(light: bpy.types.Object, energy: float):
        """Sets the color of a light.
        TODO: Cast Object to Light."""
        light.data.energy = energy  # type: ignore

    @staticmethod
    def set_object_location(obj: bpy.types.Object, loc: Vector):
        """Sets the location of an object."""
        obj.location = loc
