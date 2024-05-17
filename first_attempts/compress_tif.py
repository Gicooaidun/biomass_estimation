from PIL import Image
import rasterio
def compress_tif(input_file, output_file, width = 1000, height = 1000):
    # Open the TIFF file using rasterio
    with rasterio.open(input_file) as src:
        # Read the first band of the image
        band = src.read(1)

    # Convert the band array to an Image object
    image = Image.fromarray(band)

    # Compress the image using the desired compression format (e.g., JPEG)
    # You can adjust the quality parameter for JPEG compression
    image = image.resize((width, height))
    image = image.convert("RGB")
    image.save(output_file, format="JPEG", compression="JPEG", quality=75)
    return output_file
