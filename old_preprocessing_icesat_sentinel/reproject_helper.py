import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject


# Open the original raster dataset
def custom_reproject(src_path, output_path, dst_crs):
    # Open the original raster dataset
    with rasterio.open(src_path) as src:
        
        # Calculate the transformation parameters and the width and height of the output
        # in the new CRS
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        
        # Copy the metadata from the source dataset
        kwargs = src.meta.copy()
        # Update the metadata with the new CRS, transformation, and dimensions
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        # Open a new raster file for the reprojected data
        with rasterio.open(output_path, 'w', **kwargs) as dst:
            
            # Reproject each band in the raster dataset
            for i in range(1, src.count + 1):
                reproject(
                    # Source raster band
                    source=rasterio.band(src, i),
                    # Destination raster band
                    destination=rasterio.band(dst, i),
                    # Source transformation and CRS
                    src_transform=src.transform,
                    src_crs=src.crs,
                    # Destination transformation and CRS
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    # Resampling method
                    resampling=Resampling.nearest)

