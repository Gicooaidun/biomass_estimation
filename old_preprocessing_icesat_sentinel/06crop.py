import os
import rasterio as rs
from rasterio.mask import mask
import geopandas as gpd
import pandas as pd


def get_CRS_from_S2_tilename(tname) :
    """
    Get the CRS of the Sentinel-2 tile from its name. The tiles are named as DDCCC (where D is a digit and C a character).
    MGRS tiles are in UTM projection, which means the CRS will be EPSG=326xx in the Northern Hemisphere, and 327xx in the
    Southern. The first character of the tile name gives you the hemisphere (C to M is South, N to X is North); and the
    two digits give you the UTM zone number.

    Args:
    - tname: str, name of the Sentinel-2 tile

    Returns:
    - rasterio.crs.CRS, the CRS of the Sentinel-2 tile
    """

    tile_code, hemisphere = tname[:2], tname[2]

    if 'C' <= hemisphere <= 'M':
        crs = f'EPSG:327{tile_code}'
    elif 'N' <= hemisphere <= 'X':
        crs = f'EPSG:326{tile_code}'
    else:
        raise ValueError(f'Invalid hemisphere code: {hemisphere}')
    
    return rs.crs.CRS.from_string(crs)



# Specify the directory path
directory = 'code/newer/reprojected_mosaic'

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
    # Get the file paths of all the files in the directory
    file_path = os.path.join(directory, tile + "_reprojected_mosaic.tif")
    entry = s2_df[s2_df['Name']==tile]

    
    
    #reproject the tiles to the same CRS
    dest_crs = get_CRS_from_S2_tilename(tile)
    # reprojected_S2 = entry.to_crs(dest_crs, inplace=False)
    # print("entry.crs:  " + str(entry.crs))
    reprojected_entry = entry.to_crs(dest_crs, inplace=False)
    # print(reprojected_entry.crs.to_epsg())


    with rs.open(file_path) as dataset:
        # Access the data or perform any required operations
        data = dataset.read()
        # Do something with the data
        out_image, out_transform = mask(dataset, reprojected_entry.geometry, crop=True)
        out_meta = dataset.meta.copy()
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})
        output_path = f'code/newer/cropped_mosaic/{tile}_cropped_mosaic.tif'   
        with rs.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)
