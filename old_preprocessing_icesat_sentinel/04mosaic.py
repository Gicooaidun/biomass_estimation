import geopandas as gpd
import rasterio as rs
from rasterio.merge import merge
import pandas as pd



# Specify the path to the CSV file
csv_file = 'code/newer/overlapping_tiles.csv'

# Read the CSV file as a GeoDataFrame
df = pd.read_csv(csv_file)

# Get the unique values in the "S2 Tile Name" column
unique_tiles = df['S2 Tile Name'].unique()

s2_df = gpd.read_file('S2_tiles_Siberia_polybox/S2_tiles_Siberia_all.geojson', driver='GeoJSON', index = False)
# Iterate over the unique tiles
for tile in unique_tiles:
    print(f'Processing tile {tile}...')
    entry = s2_df[s2_df['Name']==tile]
    icesat_tiles = []
    for index, row in df[df['S2 Tile Name']==tile].iterrows():
        icesat_tiles.append(row['GeoTIFF'])
    
    to_merge = []
    # Construct the path to the corresponding tif file
    for icesat_tile in icesat_tiles:
        temp_tiff = rs.open(f'code/newer/new_data/{icesat_tile}')
        to_merge.append(temp_tiff)
    
    assert len(to_merge)!=0, "No corresponding ICESat-2 tiles found"
    # Check if the CRS of all the to_merge are the same
    crs_check = all(to_merge[0].crs == t.crs for t in to_merge)
    if not crs_check:
        raise ValueError("CRS of the to_merge tiles are not the same")
  
    # Merge the tiles
    mosaic, mosaic_trans = merge(to_merge)
    mosaic_crs = to_merge[0].crs

    height, width = mosaic.shape[1], mosaic.shape[2]
    output_meta = to_merge[0].meta.copy()
    output_meta.update({"driver": "GTiff",
                        "height": height,
                        "width": width,
                        "transform": mosaic_trans,
                        "count": mosaic.shape[0]})
    output_path = f'code/newer/merged_mosaic/{tile}_mosaic.tif'

    with rs.open(output_path, "w", **output_meta) as dest:
        dest.write(mosaic)


    # Close the tiles' files
    for src in to_merge : src.close()
