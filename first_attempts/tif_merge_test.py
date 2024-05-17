import rasterio
from rasterio.merge import merge
from tif_file_analysis import plot_bands
from tif_file_analysis import plot_multiple
import os
from compress_tif import compress_tif

# # Open the first dataset
# # dataset1 = rasterio.open('downloads firefox/boreal_agb_202302061675663867_3062.tif')
# dataset1 = rasterio.open('downloads firefox/boreal_agb_202302061675670196_2975.tif')
# # Open the second dataset
# dataset2 = rasterio.open('downloads firefox/boreal_agb_202302061675669371_3148.tif')

output_path = 'merge_test_output.tif'
# Get the path to the downloads folder
downloads_folder = 'tomerge'
# Get a list of all files in the downloads folder
files = os.listdir(downloads_folder)
# Filter the list to include only TIF files
tif_files = [file for file in files if file.endswith('.tiff')]
# Create an array with the full paths to the TIF files
tif_file_paths = [os.path.join(downloads_folder, file) for file in tif_files]

# Open all the TIF datasets
datasets = [rasterio.open(file_path) for file_path in tif_file_paths]

# Merge the datasets
# datasets_to_merge = [dataset1, dataset2]
datasets_to_merge = datasets
merged_dataset, merged_transform = merge(datasets_to_merge)
output_meta = datasets[0].meta.copy()
output_meta.update({'driver': 'GTiff',
                    'height': merged_dataset.shape[1],
                    'width': merged_dataset.shape[2],
                    'transform': merged_transform})

# plot_bands(dataset1.read(1), dataset1.read(2))
# plot_bands(dataset2.read(1), dataset2.read(2))

# Save the merged dataset
with rasterio.open(output_path, 'w', **output_meta) as dst:
    dst.write(merged_dataset)

# Open the merged dataset
merged_dataset = rasterio.open(output_path)
compress_tif(output_path, "merged_compressed.jpg", 2000, 4000)

files = [merged_dataset]
print("about to plot")
plot_multiple(files)
print("plotted")