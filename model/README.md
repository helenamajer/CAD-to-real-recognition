# Model Pipeline Handbook
## Installing Dependencies
From the root directory:
- pip install -e model

From the model directory:
- pip install -e .

## Converting DXF to OBJ
Environment Variables:
- MODEL_DATA_DIR - The root of the model's data directory. Parent of the dxf and obj directories.

Run:
- python \<root-dir>/model/pipeline/convert_dxf_to_obj.py

- python \<root-dir>/model/pipeline/convert_dxf_to_obj.py <model.dfx>

## Generating a Synthetic Dataset
Environment Variables:
- MODEL_DATA_DIR - The root of the model's data directory. Parent of the obj and raw directories.

- NUM_RENDERS (OPTIONAL) - Sets the number of renders of each synthetic model. Resorts to 200 renders if unprovided.

Run:
- python \<root-dir>/model/pipeline/generate_synthetic_dataset.py

