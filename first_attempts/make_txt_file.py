import os
import shapefile

def create_txt_file_from_folder(folder_path):
    # Get the list of files in the folder
    files = os.listdir(folder_path)

    # Create a new .txt file
    with open('code/file_list.txt', 'w') as file:
        # Write each file name to the .txt file
        for file_name in files:
            file.write(file_name + '\n')

    print('File list created successfully.')


def extract_tile_names_shp(shp_file_path, txt_file_path):
    # Open the shapefile
    sf = shapefile.Reader(shp_file_path)
    
    # Get the field names
    field_names = [field[0] for field in sf.fields[1:]]
    
    # Get the tile names from the attribute records
    tile_names = [record[0] for record in sf.records()]
    
    # Create a new .txt file
    with open(txt_file_path, 'w') as file:
        # Write each tile name to the .txt file
        for tile_name in tile_names:
            file.write(tile_name + '\n')
    
    print('Tile names extracted and saved to file successfully.')

# Example usage
shp_file_path = 'Sentinel-2 geodata/Siberia_S2_tiles-polygon.shp'
txt_file_path = 'code/tile_names.txt'
extract_tile_names_shp(shp_file_path, txt_file_path)
