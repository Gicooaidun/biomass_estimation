import geopandas as gpd
import rasterio as rs
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
import pandas as pd
import numpy as np
from reproject_helper import custom_reproject
from rasterio.mask import mask


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

    dest_crs = get_CRS_from_S2_tilename(tile)

    # Read in the (tile)_mosaic.tif file
    mosaic_file = f'code/newer/merged_mosaic/{tile}_mosaic.tif'
    custom_reproject(mosaic_file, f'code/newer/reprojected_mosaic/{tile}_reprojected_mosaic.tif', dest_crs)
