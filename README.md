# ITCS6169-A1
CV assignment 1 Train and optimize a CNN

Environment
Python 3.12.10
PyTorch 2.12.0 (dev build, cu128) — note: a nightly build was required for RTX 50-series (Blackwell) GPU support; a stable PyTorch release may behave slightly differently
Torchvision 0.27.0 (dev build)
Seed: SEED = 0 (set via set_random_seed(), applied before each training run)

Install (adjust CUDA version to match your driver via nvidia-smi):

bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy matplotlib

for normal cuda version 

bash
pip install torch torchvision
or
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu<x>
x - your driver version

dataset/
  train/
    Bedroom/
    Coast/
    ...
    Mountain/   <-- currently missing locally, must be present
    ...
  test/
    Bedroom/
    ...

Dataset folder should be named data as per code requirments
data folder should be placed in the same directory as the python notebook.

Model checkpoints are note comitted as per git ignore, to do so make changes to the code or .gitignore file
