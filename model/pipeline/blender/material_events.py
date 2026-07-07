"""Blender material events."""

from typing import cast
import bpy


class MaterialEvents:
    """Blender material events."""

    __mask_material_name: str = "MaskMaterial"

    @staticmethod
    def __create_segmentation_mask_material():
        """Creates and returns the segmentation mask material."""
        material = bpy.data.materials.new(name=MaterialEvents.__mask_material_name)

        # Clear Blender's default nodes
        material.use_nodes = True
        nodes = material.node_tree.nodes  # type: ignore
        nodes.clear()

        # Add an emission shader node
        emission = nodes.new("ShaderNodeEmission")
        emission.inputs["Color"].default_value = (1, 1, 1, 1)  # type: ignore
        emission.inputs["Strength"].default_value = 1.0  # type: ignore
        output = nodes.new("ShaderNodeOutputMaterial")

        # Create new links
        links = material.node_tree.links  # type: ignore
        links.new(emission.outputs["Emission"], output.inputs["Surface"])
        return material

    @staticmethod
    def get_object_materials(obj: bpy.types.Object):
        """Gets the materials of an object."""
        return tuple(slot.material for slot in obj.material_slots)

    @staticmethod
    def get_principled_bsdf(obj: bpy.types.Object):
        """Returns the Principled BSDF (shading node)."""
        if len(obj.material_slots) == 0:
            material = bpy.data.materials.new(name="PartMaterial")
            obj.data.materials.append(material)  # type: ignore
        material = obj.material_slots[0].material
        assert material is not None

        material.use_nodes = True
        nodes = material.node_tree.nodes  # type: ignore
        links = material.node_tree.links  # type: ignore
        nodes.clear()

        principled = nodes.new("ShaderNodeBsdfPrincipled")
        out_node = nodes.new("ShaderNodeOutputMaterial")
        links.new(principled.outputs["BSDF"], out_node.inputs["Surface"])
        return principled

    @staticmethod
    def get_segmentation_mask_material():
        """Returns the all-white material necessary for segmentation masks."""
        material = bpy.data.materials.get(MaterialEvents.__mask_material_name)
        if material is None:
            material = MaterialEvents.__create_segmentation_mask_material()
        else:
            material = cast(bpy.types.Material, material)
        return material
