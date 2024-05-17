import os
import glob
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

def convert_crs(input_folder, output_folder):
    # Get all tif files in the input folder
    tif_files = glob.glob(os.path.join(input_folder, '*.tiff'))

    for tif_file in tif_files:
        # Open the tif file
        with rasterio.open(tif_file) as src:
            # Get the spatial reference of the tif file
            source_crs = src.crs

            # Define the target spatial reference as EPSG:32652
            target_crs = 'EPSG:4326'

            # Calculate the transform and output dimensions
            transform, width, height = calculate_default_transform(source_crs, target_crs, src.width, src.height, *src.bounds)

            # Create a new file name for the converted tif file
            output_file = os.path.join(output_folder, os.path.basename(tif_file))

            # Create the output dataset
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': target_crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            with rasterio.open(output_file, 'w', **kwargs) as dst:
                 for i in range(1, src.count + 1):
                    # Reproject each band of the source dataset to the target dataset
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=source_crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.nearest
                    )

        print(f"Converted {tif_file} to {output_file}")

# Specify the input and output folders
input_folder = 'data'
output_folder = 'reprojected data'

# Call the convert_crs function
convert_crs(input_folder, output_folder)