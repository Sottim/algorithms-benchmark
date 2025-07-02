## HoVer-Net Codebase Modified Guide: Python 3.9 & GPU Setup

This guide provides a clean, end-to-end procedure for setting up the HoVer-Net project to run with Python 3.9 and utilize a modern NVIDIA GPU (like RTX A5000 with CUDA 12.6), including environment configuration and essential code modifications.
### 1. Environment Setup

The original HoVer-Net environment was based on Python 3.6, which is incompatible with the PyTorch versions required for modern GPUs. We will create a new Python 3.9 environment.
#### 1.1. Clean Up Existing Environment (If Applicable)

If you have an old hovernet conda environment, it's best to remove it for a clean start.

conda deactivate # Deactivate if currently active
conda env remove -n hovernet # Replace 'hovernet' with your old env name

#### 1.2. Create a New Python 3.9 Environment

Create a fresh conda environment with Python 3.9 and pip.

conda create -n hovernet_py39 python=3.9 pip
conda activate hovernet_py39

#### 1.3. Install PyTorch for GPU Support

Your NVIDIA RTX A5000 GPU has CUDA capability sm_86 and your system has CUDA Toolkit 12.6. PyTorch requires a specific build to leverage this. We will install PyTorch compiled for CUDA 12.1, which is compatible with your 12.6 toolkit.

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Verification: Confirm PyTorch is correctly installed and detects your GPU:

python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.__version__)"
# Expected Output:
# True
# 12.1 (or similar 12.x version)
# 2.x.x+cu121 (or similar)

#### 1.4. Install openslide Dependency

openslide is a low-level C library. It's often best installed via conda-forge before its Python binding.

conda install -c conda-forge openslide

#### 1.5. Prepare and Install Remaining Python Dependencies

The original requirements.txt pins very old versions of packages that are incompatible with Python 3.9+. We need to relax these pins so pip installs versions compatible with Python 3.9 and the newer NumPy.

    Create/Modify requirements.txt:
    Ensure your requirements.txt file (e.g., in /home/KutumLabGPU/Documents/santosh/TNBC-project/hover_net/) is updated to remove specific version pins for most packages, especially those identified during debugging (like numpy, matplotlib, imgaug, etc.).

    Example requirements.txt (adjusted to be more flexible):

    docopt==0.6.2
    future==0.18.2
    # imgaug has been removed entirely from the code, so remove this line
    # imgaug

    matplotlib
    numpy
    opencv-python
    pandas
    pillow
    psutil
    scikit-image
    scikit-learn
    scipy
    tensorboard==2.3.0 # Consider unpinning if issues arise
    tensorboardx==2.1  # Consider unpinning if issues arise
    termcolor==1.1.0
    tqdm==4.48.0
    openslide-python==1.1.2 # Or `openslide-python` if 1.1.2 fails

    Install from the modified requirements.txt:

    pip install -r /path/to/your/hovernet/requirements.txt

### 2. Code Modifications

Due to updated library APIs and deprecations, minor modifications are needed in the HoVer-Net codebase.
#### 2.1. Create type_info.json

This file is necessary for mapping nuclei type IDs to names and colors for visualization, especially for QuPath output.

    Create the file with the following content:

    {
        "0": {
            "name": "Neoplastic",
            "color": [0, 0, 255]
        },
        "1": {
            "name": "Inflammatory",
            "color": [0, 255, 0]
        },
        "2": {
            "name": "Connective",
            "color": [255, 0, 0]
        },
        "3": {
            "name": "Dead",
            "color": [0, 255, 255]
        },
        "4": {
            "name": "Epithelial",
            "color": [255, 255, 0]
        }
    }

    Save the file to: /home/KutumLabGPU/Documents/santosh/TNBC-project/models/hovernet/hover_net/type_info.json

#### 2.2. Modify run_utils/utils.py (Remove imgaug dependency)

The imgaug library (v0.4.0) is incompatible with newer NumPy versions. Fortunately, its use in utils.py is minimal.

    Open /home/KutumLabGPU/Documents/santosh/TNBC-project/models/hovernet/hover_net/run_utils/utils.py

    Remove the import statement:

    # Locate and delete this line:
    from imgaug import imgaug as ia

    Remove the seeding line:

    # Within the `check_manual_seed` function, locate and delete this line:
    # ia.random.seed(seed)

    Save the modified utils.py.

#### 2.3. Modify infer/tile.py (Update NumPy pad function call)

Newer NumPy versions (2.x+) have moved the pad function from np.lib.pad to np.pad.

    Open /home/KutumLabGPU/Documents/santosh/TNBC-project/models/hovernet/hover_net/infer/tile.py

    Locate line 76 (or similar, depending on exact file content).

    Change np.lib.pad to np.pad:

    # Original (old):
    # img = np.lib.pad(img, ((padt, padb), (padl, padr), (0, 0)), "reflect")

    # Modified (new):
    img = np.pad(img, ((padt, padb), (padl, padr), (0, 0)), "reflect")

    Save the modified tile.py.

### 3. Final Inference Command

After completing all the above steps (environment setup, dependency installation, and code modifications), you can run the HoVer-Net inference script with the following command:

python run_infer.py \
    --model_path=/home/KutumLabGPU/Documents/santosh/TNBC-project/models/hovernet/save-model-weight/hovernet_fast_pannuke_type_tf2pytorch.tar \
    --model_mode=fast \
    --nr_types=6 \
    --type_info_path=/home/KutumLabGPU/Documents/santosh/TNBC-project/models/hovernet/hover_net/type_info.json \
    tile \
    --input_dir=/home/KutumLabGPU/Documents/santosh/TNBC-project/input-dir/pannuke/patches \
    --output_dir=/home/KutumLabGPU/Documents/santosh/TNBC-project/prediction-results/hovernet \
    --save_qupath \
    --save_raw_map

This comprehensive setup should allow you to successfully perform nuclei segmentation and classification with HoVer-Net on your system.