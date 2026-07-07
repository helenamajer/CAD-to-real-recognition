"""Converts DXF format to OBJ. This is important for preparing
the synthetic data for the CV pipeline."""

import argparse
from os import environ
from pathlib import Path
from blender.events import Events
from dotenv import load_dotenv

EXTENSION = "import_autocad_dxf_format_dxf"


def load_env_vars():
    """Loads a .env (if there is one) from the pipeline directory, before
    checking if the necessary environment variables are set."""
    load_dotenv()

    root_dir = environ.get("MODEL_DATA_DIR")
    if root_dir is None:
        raise FileNotFoundError("`MODEL_DATA_DIR` is not set.")
    root_dir = Path(root_dir)
    return root_dir / "dxf", root_dir / "obj"


DXF_DIR, OBJ_DIR = load_env_vars()


def ensure_dirs():
    """Ensures the DXF and OBJ directories are on the system."""
    DXF_DIR.mkdir(parents=True, exist_ok=True)
    OBJ_DIR.mkdir(parents=True, exist_ok=True)


def handle_obj_conversion(input_file: Path, output_file: Path):
    """Converts and exports the DXF file in OBJ format."""
    Events.Object.import_dxf_object(input_file)
    Events.Object.select_objects()
    Events.Object.set_active_object()
    Events.Object.join_selected()

    obj = Events.Object.get_active_object()
    assert obj

    Events.Object.center_object(obj)
    Events.Object.rename_object(obj, input_file.with_suffix("").name)
    Events.Object.export_objects(output_file)


def main(filename: str | None):
    """Runs the script."""
    Events.Extension.ensure_extension(EXTENSION)
    ensure_dirs()

    if filename:
        file = DXF_DIR / filename
        if file.suffix.lower() != ".dxf":
            file = file.with_suffix(".dxf")

        files = [file]
        if not list(DXF_DIR.glob(f"{filename}*")):
            print(f"`{DXF_DIR}` does not contain `{filename}`")
            return
    else:
        files = list(DXF_DIR.glob("*.dxf"))
        if not files:
            print(f"`{DXF_DIR}` is empty!")
            return

    for file in files:
        Events.Scene.clear_scene()
        output_file = Path(OBJ_DIR / file.name).with_suffix(".obj")
        handle_obj_conversion(file, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename",
        type=str,
        nargs="?",
        default = None,
        help = "Enter the filename with or without the .dxf extension."
    )
    args = parser.parse_args()
    main(args.filename)
