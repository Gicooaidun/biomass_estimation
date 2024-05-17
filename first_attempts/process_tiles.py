"""

This script processes the ICESat-2 tiles. It takes as input a list of Sentinel-2 tiles, and for each of them, it lists the ICESat-2
tiles that intersect it, mosaics them, crops the resulting tile to the bounds of the Sentinel-2 tile, and saves the resulting
tile to a GeoTIFF file.

Execution:
    python process_tiles.py --tilenames /path/to/tile_names.txt 
                            --output_path /path/to/output_folder
                            --path_shp /path/to/Sentinel-2_index_shapefile.shp
                            --year 2020
                            --i 0
                            --N 10
"""

############################################################################################################################
# IMPORTS

from os.path import join
import geopandas as gpd
import numpy as np
from list_tiles import get_tiles_from_coordinates, get_true_bounds
from rasterio.merge import merge
from rasterio.crs import CRS
from rasterio.mask import mask
import rasterio as rs
import argparse
from os.path import exists
from zipfile import ZipFile
from os import remove

local_path_shp = join('/scratch2', 'biomass estimation', 'Sentinel-2 geodata', 'Siberia_S2_tiles-polygon.shp')

############################################################################################################################
# Helper functions

def setup_parser():
    """ 
    Set up the parser for the command line arguments.
    """

    parser = argparse.ArgumentParser(description = 'Download ICESat-2 data.')

    # Paths arguments
    parser.add_argument("--tilenames", type = str, required = True, 
                   help = "Path to the .txt file listing the tiles to consider.") 
    parser.add_argument('--path_shp', help = 'Path to the Sentinel-2 index shapefile.', default = local_path_shp)
    parser.add_argument("--output_path", type = str, required = True, 
                   help = "Path to the folder where the tiles will be downloaded.")

    # Arguments for the procedure
    parser.add_argument("--year", type = int, required = True, help = "Year for which to download the tiles.")
    parser.add_argument('--i', help = 'Process split i/N.', type = int, default = 0)
    parser.add_argument('--N', help = 'Total number of splits.', type = int, default = 1)
    
    args = parser.parse_args()

    # Check that the tilenames argument is a .txt file
    if args.tilenames is not None: 
        if not args.tilenames.endswith('.txt'): 
            raise ValueError('Please provide a .txt file for the --tilenames argument.')

    return args.tilenames, args.output_path, args.year, args.path_shp, args.i, args.N


def list_s2_tiles(tilenames, grid_df) :
    """
    This function performs two tasks: 1) return the list of Sentinel-2 tile names for which we want to extract patches (this
    is done either reading the tiles listed in a .txt file); and 2) return the geometries of those tiles, from the Sentinel-2
    grid shapefile.

    Args:
    - tilenames: string, path to a .txt file listing the Sentinel-2 tiles to consider.
    - grid_df: geopandas dataframe, Sentinel-2 grid shapefile.

    Returns:
    - tile_names: list of strings, names of the Sentinel-2 tiles to consider.
    - tile_geoms: list of shapely geometries, geometries of the Sentinel-2 tiles to consider.
    """
    
    # List the tiles from the .txt file
    with open(tilenames) as f: 
        tile_names = [tile_name.strip() for tile_name in f.readlines()]
    
    # Get the geometries from the Sentinel-2 grid shapefile
    tile_geoms = [grid_df[grid_df['Name'] == tile_name]['geometry'].values[0] for tile_name in tile_names]

    return tile_names, tile_geoms


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
    
    return CRS.from_string(crs)


def mosaic_tiles(tiles, band, year, path_ICESat_data) :
    """
    Mosaic the ICESat-2 tiles.

    Args:
    - tiles: list of str, names of the ICESat-2 tiles

    Returns:
    - numpy.ndarray, the mosaic of the ICESat-2 tiles
    """
    
    # Open the tiles' files (they need to be open for the merge operation)
    to_mosaic = []
    #TODO: figure out what is going on with year
    for tile in tiles :
        if year == 2022 : src = rs.open(join(path_ICESat_data, tile, f'{tile}_{year}_sl_{band}_F02DAR.tif'))
        else: src = rs.open(join(path_ICESat_data, tile, f'{tile}_{str(year)[-2:]}_sl_{band}_F02DAR.tif'))
        to_mosaic.append(src)
        mosaic_crs = src.crs
    
    # Merge the tiles
    mosaic, mosaic_trans = merge(to_mosaic)

    # Close the tiles' files
    for src in to_mosaic : src.close()
        
    return mosaic, mosaic_trans, mosaic_crs


def save_tile(s2_tname, year, bands_data, output_path) :
    """
    Save the ICESat-2 tile to a GeoTIFF file.

    Args:
    - bands_data: dict, keys are the names of the bands, and values are the data of the bands
    - output_path: str, path to the folder where the processed tiles will be stored

    Returns:
    - None
    """
    
    # Extract the bands' data
    bands_values = list(bands_data.values())

    # Check that the bands have the same shape, transform and CRS
    shapes = [val['mosaic'].shape for val in bands_values]
    assert len(set(shapes)) == 1, "The shapes of the bands are not the same"
    transforms = [val['transform'] for val in bands_values]
    assert len(set(transforms)) == 1, "The transforms of the bands are not the same"
    crs = [val['crs'] for val in bands_values]
    assert len(set(crs)) == 1, "The CRS of the bands are not the same"


    ICESat_meta = {'driver': 'GTiff', 'height': shapes[0][1], 'width': shapes[0][2], \
                'transform': transforms[0], 'crs': crs[0], 'count' : 2, 'nodata' : 1.0,
                'dtype': 'uint16'} 
    
    # Write the bands to the file
    fname = f'ICESat2_{s2_tname}_{str(year)[-2:]}.tif'
    with rs.open(join(output_path, fname), 'w', **ICESat_meta) as dst:
        for band_id, (band_name, band_data) in enumerate(bands_data.items()):
            dst.write(band_data['mosaic'][0, :, :], band_id + 1)
            dst.set_band_description(band_id + 1, band_name)


def unzip(ICESat_tiles, path_ICESat_data, year) :
    """
    Unzip the ICESat tiles (if necessary). The zipped tiles are stored in /path_ICESat_data/<tile>_<year>.zip, and the unzipped
    files will be stored in /path_ICESat_data/<tile>/. 
    
    FROM GHJULIAS VERSION:
    (((Note that the following files are extracted:
        . <tile>_<year>_date_F02DAR.tif  
        . <tile>_<year>_F02DAR.xml
        . <tile>_<year>_linci_F02DAR.tif
        . <tile>_<year>_mask_F02DAR.tif
        . <tile>_<year>_sl_HH_F02DAR.tif
        . <tile>_<year>_sl_HV_F02DAR.tif)))

    Args:
    - ICESat_tiles: list of str, names of the ICESat tiles
    - path_ICESat_data: str, path to the folder where the ICESat data is stored
    - year: int, year for which to process the ICESat tiles

    Returns:
    - existing_tiles: list of str, ICESat tiles that were already unzipped or successfully unzipped
    """

    existing_tiles = []
    for tile in ICESat_tiles :
        if year == 2022 : fname = f'{tile}_{year}_sl_HH_F02DAR.tif'
        else: fname = f'{tile}_{str(year)[-2:]}_sl_HH_F02DAR.tif'
        if not exists(join(path_ICESat_data, tile, fname)) :
            if exists(join(path_ICESat_data, f'{tile}_{year}.zip')) :
                try:
                    # Unzip the file
                    with ZipFile(join(path_ICESat_data, f'{tile}_{year}.zip'), 'r') as zip_ref:
                        zip_ref.extractall(join(path_ICESat_data, tile))
                    # And delete the .zip file
                    remove(join(path_ICESat_data, f'{tile}_{year}.zip'))
                except Exception as e: raise e
                existing_tiles.append(tile)
        else: existing_tiles.append(tile)
    
    return existing_tiles

############################################################################################################################
# Main function

def process_ICESat_tiles(s2_tname, s2_geom, year, path_ICESat_data) :
    """
    For a given Sentinel-2 tile, list the ICESat tiles that intersect it, mosaic them, and crop the resulting tile to the
    bounds of the Sentinel-2 tile. Then save the resulting tile to a GeoTIFF file.

    Args:
    - args: tuple, containing the following elements:
        - s2_tile_data: tuple, (int, pandas.core.series.Series), the index and the geometry of the Sentinel-2 tile
        - year: int, year for which to process the ICESat tiles
        - path_ICESat_data: str, path to the folder where the ICESat data is stored

    Returns:
    - bool, whether the ICESat tile was successfully processed
    """
    
    try:

        # Get the corresponding ICESat tiles
        lon_min, lat_min, lon_max, lat_max = s2_geom.bounds
        if abs(lon_min - lon_max) > 180 :
            meridian_flag = True
            # TODO that's not gonna work, country is not defined, and get_true_bounds won't take that as an argument
            lon_min, lat_min, lon_max, lat_max = get_true_bounds(s2_geom)
        else: meridian_flag = False
        ICESat_tiles = get_tiles_from_coordinates(lat_min, lat_max, lon_min, lon_max, meridian_flag)

        print(f'>> Got {len(ICESat_tiles)} ICESat tiles.')

        # Unzip the ICESat tiles if necessary
        ICESat_tiles = unzip(ICESat_tiles, path_ICESat_data, year)

        # Iterate over the ICESat bands of interest
        bands_data = {}
        #TODO: probably change these bands
        for band in ['HH', 'HV'] :

            # Mosaic the ICESat tiles
            mosaic, mosaic_trans, mosaic_crs = mosaic_tiles(ICESat_tiles, band, year, path_ICESat_data)

            # Crop to the Sentinel-2 tile (use a Memory File because `mask` requires a dataset open in 'r' mode)
            with rs.MemoryFile() as memfile:
                with memfile.open(driver = 'GTiff', height = mosaic.shape[1], width = mosaic.shape[2], count = 1, 
                                    dtype = mosaic.dtype, crs = mosaic_crs, transform = mosaic_trans) as mosaic_vrt:
                    mosaic_vrt.write(np.squeeze(mosaic, axis = 0), 1)
                with memfile.open() as src:
                    mosaic, mosaic_trans = mask(src, shapes = [s2_geom], crop = True)
            
            # Store the data
            bands_data[band] = {'mosaic': mosaic, 'transform': mosaic_trans, 'crs': mosaic_crs}
        
        # Save the ICESat tile
        save_tile(s2_tname, year, bands_data, path_ICESat_data)

    except Exception as e:
        print('Error', e)
        return f'Error processing {s2_tname}: {e}'


############################################################################################################################
# Execute

if __name__ == "__main__":

    tilenames, path_ICESat_data, year, path_shp, i, N = setup_parser()

    # Read the Sentinel-2 grid shapefile
    grid_df = gpd.read_file(path_shp, engine = 'pyogrio')

    # List all S2 tiles and their geometries
    tile_names, tile_geoms = list_s2_tiles(tilenames, grid_df)
    assert len(tile_names) == len(tile_geoms), "The number of tile names and geometries is not the same."

    # Split into N, and process the i-th split
    tile_names = np.array_split(tile_names, N)[i]
    tile_geoms = np.array_split(tile_geoms, N)[i]
    assert len(tile_names) == len(tile_geoms), "The number of tile names and geometries is not the same."

    tiles_num = len(tile_names)
    print(f'Processing {tiles_num} tile(s)...')

    for tile_idx, (tile_name, tile_geom) in enumerate(zip(tile_names, tile_geoms)) :
        print(f'({tile_idx + 1}/{tiles_num}) Extracting for tile {tile_name}...')
        process_ICESat_tiles(tile_name, tile_geom, year, path_ICESat_data)


"""
python process_tiles.py --tilenames /scratch2/gsialelli/BiomassDatasetCreation/Data/download_Sentinel/Sentinel_California_Cuba_Paraguay_UnitedRepublicofTanzania_Ghana_Austria_Greece_Nepal_ShaanxiProvince_NewZealand_FrenchGuiana.txt --output_path /scratch2/gsialelli/ICESat/ --year 2020

N = 10
for year in [2018, 2019, 2020, 2021, 2022] :
    for i in range(N) :
        print(f'nohup python process_tiles.py --tilenames /scratch2/gsialelli/BiomassDatasetCreation/Data/download_Sentinel/Sentinel_California_Cuba_Paraguay_UnitedRepublicofTanzania_Ghana_Austria_Greece_Nepal_ShaanxiProvince_NewZealand_FrenchGuiana.txt --output_path /scratch2/gsialelli/ICESat/ --year {year} --i {i} --N 10 > logs/mosaicing-{year}-{i}-{N}.txt 2>&1 &')
"""