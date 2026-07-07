"""Blender scene events."""

import bpy
from .object_events import ObjectEvents
from .render_settings import RenderSettings


class SceneEvents:
    """Blender scene events."""

    @staticmethod
    def add_camera(name: str):
        """Adds a camera to the scene."""
        data = bpy.data.cameras.new(name=name)
        cam = bpy.data.objects.new(name=name, object_data=data)
        bpy.context.collection.objects.link(cam)
        bpy.context.scene.camera = cam
        bpy.context.view_layer.update()

    @staticmethod
    def add_light(name: str):
        """Adds a light to the scene."""
        data = bpy.data.lights.new(name=name, type="AREA")
        light = bpy.data.objects.new(name=name, object_data=data)
        bpy.context.collection.objects.link(light)
        bpy.context.view_layer.update()

    @staticmethod
    def clear_scene(remove_light_and_camera: bool = True):
        """Removes all objects from the scene.
        Includes the light and camera by default."""
        if remove_light_and_camera:
            ObjectEvents.select_objects()
        else:
            ObjectEvents.select_objects(["MESH"])
        ObjectEvents.delete_selected_objects()

    @staticmethod
    def point_camera(cam: bpy.types.Object, target: bpy.types.Object):
        """Points the camera towards a target."""
        direction = target.location - cam.location
        rotation_quat = direction.to_track_quat("-Z", "Y")
        cam.rotation_euler = rotation_quat.to_euler()

    @staticmethod
    def write_render():
        """Writes the render output."""
        bpy.ops.render.render(write_still=True)

    @staticmethod
    def set_render_settings(settings: RenderSettings):
        """Sets the render settings of the scene. Preconfigured by default."""
        scene = bpy.context.scene

        scene.render.image_settings.file_format = settings.image.exported_format
        scene.render.resolution_x = settings.image.resolution_x
        scene.render.resolution_y = settings.image.resolution_y
        scene.render.film_transparent = settings.image.transparent_background
        scene.render.image_settings.color_mode = settings.image.color_mode

        scene.render.engine = settings.engine.name  # type: ignore (FAKE PANIC)

        prefs = bpy.context.preferences.addons[settings.engine.name.lower()].preferences
        prefs.compute_device_type = settings.engine.computation_method  # type: ignore (FAKE PANIC)
        prefs.get_devices()  # type: ignore (FAKE PANIC)
        for device in prefs.devices:  # type: ignore (FAKE PANIC)
            if device.type in settings.engine.computation_method:
                device.use = True
            else:
                device.use = False
        scene.cycles.device = settings.engine.device  # type: ignore (FAKE PANIC)

    @staticmethod
    def set_render_filepath(filepath: str):
        """Sets the filepath of the render output."""
        bpy.context.scene.render.filepath = filepath

    @staticmethod
    def set_cycle_sample_count(num_samples: int):
        """Sets the number of samples that cycles will produce."""
        bpy.context.scene.cycles.samples = num_samples  # type: ignore
