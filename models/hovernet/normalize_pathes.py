import os
import glob
import cv2
import numpy as np
import tqdm
import sys


# --- Stain Normalization Class (Vahadane Method) ---
# This class implements a common stain normalization technique
# which adapts to different stainings by learning a reference stain matrix.
class StainNorm(object):
    """
    Performs stain normalization using the Vahadane method.
    Adapted from: https://github.com/mit-han-lab/HoverNet/blob/master/misc/utils.py
    (This is a common implementation for HoVer-Net style stain normalization)
    """

    def __init__(self, ref_img):
        # H_ref is the stain matrix for H&E. Rows are color channels (R,G,B), columns are stain components (H, E).
        # This is a common empirical matrix for H&E.
        self.H_ref = np.array(
            [
                [0.65, 0.70, 0.29],  # R channel contribution from H, E, and residual
                [
                    0.21,
                    0.27,
                    0.95,
                ],  # G channel contribution from H, E, and residual (simplified)
            ]
        )
        # Note: The original HoVerNet code for H_ref in `misc/utils.py` is often transposed or used differently
        # For Vahadane, H_ref represents the absorbance of each stain at each wavelength (RGB channels).
        # It's usually (3, N_stains). Since we have 2 stains (H&E), it should be (3, 2).
        # Let's use the standard interpretation where rows are colors and columns are stains.
        # This H_ref will be used to deconvolve stains from OD values.

        # The initial H_ref in the previous version was incorrect after transpose as (3,2) not (2,3)
        # H_ref should be (Number of channels, Number of stains) i.e. (3, 2) for H&E
        self.H_ref = np.array(
            [
                [0.65, 0.21],  # Red channel contribution for Hematoxylin, Eosin
                [0.70, 0.27],  # Green channel contribution for Hematoxylin, Eosin
                [0.29, 0.95],  # Blue channel contribution for Hematoxylin, Eosin
            ]
        )

        self.ref_img_norm(ref_img)

    def ref_img_norm(self, ref_img):
        """
        Calculates reference stain concentrations (maxCRef) from the reference image.
        This is based on the Vahadane method's principle of normalizing concentrations.
        """
        # Convert to optical density (OD) space, avoiding log(0)
        # Adding 1 to pixel values before division to prevent log(0) if any pixel is 0.
        OD_ref = -np.log(
            ((ref_img.astype(np.float32) + 1) / 255.0) + 1e-8
        )  # Add small epsilon for robustness

        # Flatten OD_ref for SVD and reshape for matrix multiplication (N_pixels, 3) -> (3, N_pixels)
        OD_ref_flat = OD_ref.reshape((-1, 3)).T  # Shape (3, N_pixels)

        # Estimate stain concentrations (C_ref) for the reference image
        # C_ref = inv(H_ref.T @ H_ref) @ H_ref.T @ OD_ref_flat
        # Using numpy.linalg.lstsq for a robust solution (solves H_ref @ C_ref = OD_ref_flat)
        # C_ref will have shape (N_stains, N_pixels)
        C_ref = np.linalg.lstsq(self.H_ref, OD_ref_flat, rcond=None)[0]

        # Calculate maximum concentration values for each of the 2 stain components (H and E)
        self.maxCRef = np.array(
            [
                np.percentile(
                    C_ref[0, :], 99
                ),  # 99th percentile for Hematoxylin concentrations
                np.percentile(
                    C_ref[1, :], 99
                ),  # 99th percentile for Eosin concentrations
            ]
        )

    def transform(self, img):
        """
        Normalizes the input image to the reference stain concentrations.
        """
        # Convert input image to optical density (OD) space
        OD = -np.log(((img.astype(np.float32) + 1) / 255.0) + 1e-8)

        # Flatten OD to (N_pixels, 3) and transpose to (3, N_pixels) for multiplication
        OD_flat = OD.reshape((-1, 3)).T

        # Estimate stain concentrations (C) for the current image
        # C will have shape (N_stains, N_pixels)
        C = np.linalg.lstsq(self.H_ref, OD_flat, rcond=None)[0]

        # Calculate max concentrations for the current image's stains
        maxC_curr = np.array(
            [
                np.percentile(C[0, :], 99),  # Hematoxylin max concentration
                np.percentile(C[1, :], 99),  # Eosin max concentration
            ]
        )

        # Calculate scaling factors based on reference max concentrations
        scaling_factor = self.maxCRef / maxC_curr

        # Apply scaling to current concentrations to normalize them
        Cn = (
            C * scaling_factor[:, np.newaxis]
        )  # Element-wise multiplication, broadcasting scaling_factor

        # Convert normalized concentrations (Cn) back to optical density (ODn)
        ODn = np.dot(self.H_ref, Cn)
        ODn = ODn.T.reshape(img.shape)  # Reshape back to image dimensions (H, W, 3)

        # Convert normalized OD back to RGB image (pixel values 0-255)
        # np.exp(-ODn * np.log(10)) reverses the log10
        # 255.0 is the max value, adjusted if `OD` was normalized differently.
        normalized_img = (255.0 * np.exp(-ODn)).astype(np.uint8)

        return normalized_img


# --- Configuration ---
# Directory containing your original PanNuke patches
INPUT_PATCH_DIR = (
    "/home/KutumLabGPU/Documents/santosh/TNBC-project/input-dir/pannuke/patches"
)
# Directory where normalized patches will be saved
OUTPUT_NORMALIZED_DIR = "/home/KutumLabGPU/Documents/santosh/TNBC-project/input-dir/pannuke/pannuke_normalized_patches"
# Path to a reference image for stain normalization.
# This should be a representative H&E image that the model was trained against.
REFERENCE_IMAGE_PATH = "/home/KutumLabGPU/Documents/santosh/TNBC-project/models/hovernet/reference-img.png"  # <--- VERIFY THIS PATH!

# --- Ensure output directory exists ---
os.makedirs(OUTPUT_NORMALIZED_DIR, exist_ok=True)

# --- Load the reference image ---
print(f"Loading reference image from: {REFERENCE_IMAGE_PATH}")
try:
    # OpenCV loads images in BGR format by default
    ref_image = cv2.imread(REFERENCE_IMAGE_PATH)
    if ref_image is None:
        raise FileNotFoundError(f"Reference image not found at: {REFERENCE_IMAGE_PATH}")
    ref_image = cv2.cvtColor(
        ref_image, cv2.COLOR_BGR2RGB
    )  # Convert to RGB as deep learning models usually expect RGB
    print("Reference image loaded successfully.")
except Exception as e:
    print(
        f"ERROR: Could not load reference image. Please check the path and file integrity."
    )
    print(f"Details: {e}")
    sys.exit(1)  # Exit if reference image can't be loaded

# --- Initialize the stain normalizer ---
stain_normalizer = StainNorm(ref_image)
print("Stain normalizer initialized.")

# --- Process each patch ---
# Supported image extensions
image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]
patch_files = []
for ext in image_extensions:
    patch_files.extend(glob.glob(os.path.join(INPUT_PATCH_DIR, ext)))
patch_files.sort()  # Ensure consistent order

if not patch_files:
    print(f"No image files found in: {INPUT_PATCH_DIR}")
    sys.exit(0)

print(f"Found {len(patch_files)} patches to normalize in {INPUT_PATCH_DIR}.")

# Use tqdm for a progress bar
for i, patch_path in enumerate(tqdm.tqdm(patch_files, desc="Normalizing Patches")):
    # Load the patch
    img = cv2.imread(patch_path)
    if img is None:
        print(f"Warning: Could not load {patch_path}. Skipping.")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Normalize the patch
    normalized_img = stain_normalizer.transform(img)

    # Convert back to BGR for saving with OpenCV (cv2.imwrite expects BGR or grayscale)
    normalized_img_bgr = cv2.cvtColor(normalized_img, cv2.COLOR_RGB2BGR)

    # Define output path
    output_path = os.path.join(OUTPUT_NORMALIZED_DIR, os.path.basename(patch_path))

    # Save the normalized patch
    cv2.imwrite(output_path, normalized_img_bgr)

print("\nAll patches normalized successfully!")
print(f"Normalized patches saved to: {OUTPUT_NORMALIZED_DIR}")
