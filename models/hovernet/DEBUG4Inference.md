## HoVer-Net Inference Troubleshooting Guide

This guide documents the end-to-end process of resolving dependency and compatibility issues encountered while setting up and running HoVer-Net inference, particularly when migrating an older codebase to a modern GPU environment.
Initial Setup & Command Structure Issues

Problem: The initial attempts to run run_infer.py failed due to incorrect command arguments and the script not being found.

Initial Command Attempt:

hover_net python run_infer.py --input_dir=... --output_dir=... --save_qupath --save_raw_map
# (Also tried: run_infer.py --input_dir=...)

Reasoning:

    The run_infer.py script uses docopt, which requires a <command> argument (tile or wsi) to specify the inference mode. This was missing.

    run_infer.py was not directly executable or in PATH, requiring python prefix.

    Global arguments (like --model_path, --model_mode, --nr_types, --gpu) must come before the command (tile or wsi), while command-specific arguments (--input_dir, --output_dir, etc.) come after the command.

    The script explicitly requires --model_path to be provided.

### Solution 1: Correcting Command Structure and Adding Essential Arguments

The correct general structure for tile inference is:

python run_infer.py [GLOBAL_OPTIONS] <command> [COMMAND_OPTIONS]

For your specific case, with the fast model and PanNuke's 5 nuclei types + background (total 6 classes for the model's output layer), the command became:

python run_infer.py \
    --model_path=/home/KutumLabGPU/Documents/santosh/TNBC-project/models/hovernet/save-model-weight/hovernet_fast_pannuke_type_tf2pytorch.tar \
    --model_mode=fast \
    --nr_types=6 \
    tile \
    --input_dir=/home/KutumLabGPU/Documents/santosh/TNBC-project/input-dir/pannuke/patches \
    --output_dir=/home/KutumLabGPU/Documents/santosh/TNBC-project/prediction-results/hovernet \
    --save_qupath \
    --save_raw_map

Dependency & Compatibility Errors (Iterative Debugging)

After fixing the command structure, a series of more technical errors emerged, primarily due to mismatches between the old Python 3.6 environment/dependencies and the requirements of a modern GPU (RTX A5000, CUDA 12.6).
Error 2: RuntimeError: Error(s) in loading state_dict for HoVerNet: size mismatch for decoder.tp.u0.conv.weight...

Cause: The model checkpoint (hovernet_fast_pannuke_type_tf2pytorch.tar) was trained to output 6 channels for the type prediction branch (5 nuclei types + 1 background), but the nr_types argument was initially interpreted as 5 (for just the nuclei types), leading to a mismatch in the model's final layer definition.

### Solution 2: Correcting --nr_types
The nr_types parameter should reflect the total number of output classes for the classification head, which is usually (number of nuclei types) + 1 (for background). For PanNuke (5 nuclei types), this means 5 + 1 = 6.

We updated the command to include --nr_types=6.
Error 3: FileNotFoundError: [Errno 2] No such file or directory: "''"

Cause: The run_infer.py script attempts to load a type_info.json file for visualizing nuclei types, but the --type_info_path argument was either not provided or was an empty string, causing a FileNotFoundError.

### Solution 3: Creating and Specifying type_info.json

    Created type_info.json file with the following content (specifying names and RGB colors for the 5 PanNuke nuclei types):

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

    Saved the file to: /home/KutumLabGPU/Documents/santosh/TNBC-project/models/hovernet/hover_net/type_info.json

    Added --type_info_path argument to the command:

    # ... (previous arguments) ...
    --type_info_path=/home/KutumLabGPU/Documents/santosh/TNBC-project/models/hovernet/hover_net/type_info.json \
    # ... (remaining arguments) ...

### Error 4: UserWarning: NVIDIA RTX A5000 with CUDA capability sm_86 is not compatible... and RuntimeError: CUDA error: no kernel image is available for execution on the device

Cause: This was the primary GPU compatibility issue. Your RTX A5000 GPU (Ampere architecture, sm_86) requires CUDA 11+ (preferably 12.x). However, the existing PyTorch installation in your hovernet conda environment was either CPU-only or compiled for an older CUDA version (e.g., CUDA 10.2 or 11.0, which doesn't support sm_86).

Solution 4: Reinstalling PyTorch for GPU Compatibility

    Identified System CUDA Version: Ran nvcc --version, which showed CUDA Toolkit 12.6.

    Identified Python Version Conflict: The latest PyTorch builds (required for CUDA 12.6 support) require Python 3.9 or later, while your original environment was python=3.6.12.

    Removed Old Conda Environment: Since a Python version upgrade within an existing environment can be problematic, the old environment was removed to ensure a clean slate.

    conda deactivate
    conda env remove -n hovernet

    Created New Conda Environment with Python 3.9:

    conda create -n hovernet_py39 python=3.9 pip
    conda activate hovernet_py39

    Installed PyTorch with CUDA 12.1 (compatible with 12.6) using pip and explicit index URL: This is crucial to get the GPU-enabled build.

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    Verified PyTorch Installation:

    python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.__version__)"

    This successfully showed True for CUDA availability and 12.1 for torch.version.cuda, indicating proper GPU setup.

#### Error 5: error: subprocess-exited-with-error during numpy==1.19.1 installation (after Python 3.9 upgrade)

Cause: Older pinned versions of libraries (like numpy==1.19.1) are often incompatible with newer Python versions (3.9+) because they rely on deprecated modules (like distutils) or their source code cannot be built with the newer Python interpreters.

#### Solution 5: Unpinning Outdated Dependencies

    Modified requirements.txt: Removed specific version pins for core libraries that are known to evolve with Python versions (numpy, matplotlib, opencv-python, pandas, pillow, scikit-image, scikit-learn, scipy). This allows pip to install the latest versions compatible with Python 3.9.

    Example requirements.txt after this step (actual may vary slightly):

    docopt==0.6.2
    future==0.18.2
    imgaug==0.4.0 # Still pinned here
    matplotlib
    numpy
    opencv-python
    pandas
    pillow
    psutil
    scikit-image
    scikit-learn
    scipy
    tensorboard==2.3.0
    tensorboardx==2.1
    termcolor==1.1.0
    tqdm==4.48.0

    Installed openslide via conda: Since openslide is a C library dependency, installing it via conda first often ensures proper linking.

    conda install -c conda-forge openslide

    Re-installed openslide-python and other dependencies:

    pip install openslide-python==1.1.2 # If this fails, try `pip install openslide-python`
    pip install -r requirements.txt # Using the updated file

### Error 6: AttributeError: 'numpy.lib' has no attribute 'pad'

Cause: After installing newer versions of common libraries (like NumPy 2.x), the imgaug==0.4.0 library (which was still pinned) was using an old NumPy API (np.sctypes) that had been removed. This then led to the AttributeError: 'numpy.lib' has no attribute 'pad' error. It means the imgaug library itself was too old.

#### Solution 6: Removing Obsolete imgaug Dependency and Direct Code Fix

    Inspected run_utils/utils.py: Found that imgaug was only used for ia.random.seed(seed).

    Modified run_utils/utils.py:

        Removed from imgaug import imgaug as ia.

        Removed the line # ia.random.seed(seed).

    Modified requirements.txt: Removed imgaug entirely from the list.

    Uninstalled imgaug:

    pip uninstall imgaug

    Modified infer/tile.py to use np.pad: The error AttributeError: module 'numpy.lib' has no attribute 'pad' was caused by a call to np.lib.pad in infer/tile.py. This function was replaced by np.pad in newer NumPy versions.

        Opened /home/KutumLabGPU/Documents/santosh/TNBC-project/models/hovernet/hover_net/infer/tile.py

        Changed np.lib.pad to np.pad on line 76:

        # Old: img = np.lib.pad(img, ((padt, padb), (padl, padr), (0, 0)), "reflect")
        # New:
        img = np.pad(img, ((padt, padb), (padl, padr), (0, 0)), "reflect")

    Saved all modified files.

    Re-installed remaining requirements (if any were unpinned and needed re-install, though most should be okay from previous step):

    pip install -r requirements.txt

Final Successful Inference Command

After all these steps, the HoVer-Net inference script finally ran without errors, fully utilizing the GPU:

python run_infer.py \
    --model_path=/home/KutumLabGPU/Documents/santosh/TNBC-project/models/hovernet/hover_net/save-model-weight/hovernet_fast_pannuke_type_tf2pytorch.tar \
    --model_mode=fast \
    --nr_types=6 \
    --type_info_path=/home/KutumLabGPU/Documents/santosh/TNBC-project/models/hovernet/hover_net/type_info.json \
    tile \
    --input_dir=/home/KutumLabGPU/Documents/santosh/TNBC-project/input-dir/pannuke/patches \
    --output_dir=/home/KutumLabGPU/Documents/santosh/TNBC-project/prediction-results/hovernet \
    --save_qupath \
    --save_raw_map
