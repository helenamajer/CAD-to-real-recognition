"""Blender extension events."""

import bpy


class ExtensionEvents:
    """Blender extension events."""

    __blender_extension_prefix: str = "bl_ext.blender_org"

    @staticmethod
    def ensure_extension(name: str):
        """Ensures that Blender contains the desired extension."""
        alt_name: str = f"{ExtensionEvents.__blender_extension_prefix}.{name}"
        name_variants = tuple([name, alt_name])
        bpy.ops.wm.read_userpref()
        if not any(
            key in bpy.context.preferences.addons.keys() for key in name_variants
        ):
            print(f"Installing `{name}`...")
            bpy.ops.extensions.userpref_allow_online()
            bpy.ops.extensions.package_install(
                repo_index=0, pkg_id=name, enable_on_install=True
            )
            bpy.ops.wm.save_userpref()
        else:
            print(f"`{name}` already exists, continuing...")
