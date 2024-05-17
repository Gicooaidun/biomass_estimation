import os
import numpy as np
import rasterio

# Get the path of the current script
script_path = os.path.dirname(os.path.abspath(__file__))

# Define the filename of the TIFF file
tif_filename = "cropped_mosaic/50TPT_cropped_mosaic.tif" #has nan (water)
# tif_filename = "cropped_mosaic/50TPS_cropped_mosaic.tif" #doesn't have nan (no water)


# Construct the full path to the TIFF file
tif_path = os.path.join(script_path, tif_filename)

# Open the TIFF file using rasterio
with rasterio.open(tif_path) as src:
    # Read the image as a NumPy array
    tif_array = src.read(1)

# Check if any values in the array are NaN
has_nan = np.isnan(tif_array).any()

print(tif_array)

if has_nan:
    print("The TIFF file contains NaN values.")
    # Get the indices of the NaN values
    nan_indices = np.argwhere(np.isnan(tif_array))

    # Define the path and filename for the smaller TIFF file
    small_tif_filename = "small_with_nans.tif"
    small_tif_path = os.path.join(script_path, small_tif_filename)

    # Create a new rasterio dataset for the smaller TIFF file
    with rasterio.open(
        small_tif_path,
        'w',
        driver='GTiff',
        height=tif_array.shape[0],
        width=tif_array.shape[1],
        count=1,
        dtype=tif_array.dtype,
        crs=src.crs,
        transform=src.transform
    ) as dst:
        # Set the NaN values in the smaller TIFF array
        tif_array_with_nans = np.copy(tif_array)
        tif_array_with_nans[np.isnan(tif_array_with_nans)] = -9999

        # Write the array to the smaller TIFF file
        dst.write(tif_array_with_nans, 1)

    print("The smaller TIFF file with NaN values has been saved.")
    print("Indices of NaN values:", nan_indices)
else:
    print("The TIFF file does not contain any NaN values.")
