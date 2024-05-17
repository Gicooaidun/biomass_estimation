
import geopandas as gpd
from os.path import join, basename, exists
import os

# #FOR GETTING ALL THE PATCH NAMES OF ALL S2 TILES

# # Absolute path to `S2_tiles_Siberia_all.geojson` file
# path_shp = join('/scratch2', 'biomass_estimation', 'code', 'notebook', 'S2_tiles_Siberia_polybox', 'S2_tiles_Siberia_all.geojson')

# # Read the Sentinel-2 grid shapefile
# grid_df = gpd.read_file(path_shp, engine = 'pyogrio')

# # Create a list to store the patch names
# patch_names = []

# # Iterate over each row in the grid dataframe
# for index, row in grid_df.iterrows():
#     # Get the patch name from the 'name' column
#     patch_name = row['Name']
#     # Add the patch name to the list
#     patch_names.append(patch_name)

# # Define the output file path
# output_file = 'tile_names.txt'

# # Write the patch names to the output file
# with open(output_file, 'w') as file:
#     for patch_name in patch_names:
#         file.write(patch_name + '\n')

mosaic_folder = '/scratch2/biomass_estimation/code/notebook/cropped_mosaic'

# Get all file names in the folder
file_names = os.listdir(mosaic_folder)

# Define the output file path
output_file = 'tile_names.txt'

# Write the file names to the output file
with open(output_file, 'w') as file:
    for file_name in file_names:
        file.write(file_name[:5] + '\n')