"""

This script extracts patches of Sentinel-2 L2A data, Sentinel-1 L1C data, and GEDI data, for the purpose of training a neural network to
predict biomass. 


Execution:
    python create_patches.py    --tilenames /path/to/tile_names.txt
                                --year 2019
                                --patch_size 15 15 
                                --chunk_size 1
                                --path_shp /path/to/sentinel_2_index_shapefile.shp 
                                --path_gedi /path/to/GEDI 
                                --path_s2 /path/to/S2_L2A 
                                --path_s1 /path/to/S1
                                --path_alos /path/to/ALOS
                                --path_ch /path/to/CH
                                --path_lc /path/to/LC
                                --output_path /path/to/patches
                                --output_fname (optional)
                                --S1
                                --ALOS
                                --CH
                                --LC
                                --DEM
                                --i
                                --N

"""

############################################################################################################################
# IMPORTS

import sys
import h5py
import glob
import argparse
import traceback
import numpy as np
import pandas as pd
import rasterio as rs
import datetime as dt
import geopandas as gpd
from shutil import rmtree
from zipfile import ZipFile
from rasterio.crs import CRS
from shapely.geometry import box
import xml.etree.ElementTree as ET
from skimage.transform import resize
from os.path import join, basename, exists
from rasterio.transform import rowcol, AffineTransformer

sys.path.append('../Sentinel')
from Sentinel_settings import GEDI_START_MISSION

# Absolute path to `sentinel_2_index_shapefile.shp` file
local_path_shp = join('/scratch2', 'gsialelli', 'BiomassDatasetCreation', 'Data', 'download_Sentinel', 'sentinel_2_index_shapefile.shp')

# Sentinel-2 L2A bands that we want to use
S2_L2A_BANDS = {'10m' : ['B02', 'B03', 'B04', 'B08'],
                '20m' : ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'SCL'],
                '60m' : ['B01', 'B09']}

# GEDI attributes and their corresponding data types
GEDI_attrs = {
    # GEDI inherent attributes
    'pft_class': np.uint8, 'region_cla': np.uint8, 'leaf_off_f': np.uint8, 'urban_prop': np.uint8, 'agbd': np.float32,
       'agbd_se': np.float32, 'elev_lowes': np.float32, 'selected_a': np.uint8, 'shot_numbe': np.uint64, 
       'sensitivit': np.float32, 'solar_elev': np.float32, 'rh98': np.float32, 'date': np.uint16, 
    # GEDI derived attributes   
    'granule_num': np.uint8, 'track_num': np.uint32, 'ppds': np.uint8, 'version_num': np.uint16, 'lat_offset': np.uint8, 
        'lat_decimal': np.float32, 'lon_offset': np.uint8, 'lon_decimal': np.float32
    }

# Sentinel-2 attributes and their corresponding data types (we differentiate between 2d and 1d attributes)
S2_attrs = {'bands' : {'B01': np.uint16, 'B02': np.uint16, 'B03': np.uint16, 'B04': np.uint16, 'B05': np.uint16, 'B06': np.uint16, 
                        'B07': np.uint16, 'B08': np.uint16, 'B8A': np.uint16, 'B09': np.uint16, 'B11': np.uint16, 'B12': np.uint16, 
                        'SCL': np.uint8},
            'metadata' : {'vegetation_score': np.uint8, 'date' : np.int16, 'pbn' : np.uint16, 'ron' : np.uint8, 'boa_offset': np.uint8}
            }

# Sentinel-1 attributes and their corresponding data types (TODO)
S1_attrs = {'bands' : {}, 'metadata' : {'date' : np.uint16}}

# ALOS PALSAR-2 attributes and their corresponding data types
ALOS_attrs = {'HH': np.uint16, 'HV': np.uint16}

# Canopy Height (CH) attributes and their corresponding data types
CH_attrs = {'ch': np.uint8, 'std': np.uint8}

# Land Cover (LC) attributes and their corresponding data types
LC_attrs = {'lc': np.uint8, 'prob' : np.uint8}

# ALOS DEM attributes and their corresponding data types
DEM_attrs = {'dem': np.int16}

NODATAVALS = {'S2' : 0, 'CH': 255, 'ALOS': 1, 'LC': 255, 'DEM': -9999}

############################################################################################################################
# Helper functions

def setup_parser() :
    """ 
    Setup the parser for the command line arguments.
    """
    parser = argparse.ArgumentParser()
    
    # Arguments for the patches extraction procedure
    parser.add_argument('--year', help = 'Year of the data to process product.', type = str, required = True)
    parser.add_argument('--patch_size', help = 'Size of the patches to extract, in pixels.', nargs = 2, type = int, default = [15, 15])
    parser.add_argument('--chunk_size', help = 'Number of patches to write to file at once.', type = int, default = 1)
    parser.add_argument('--i', help = 'Process split i/N.', type = int, required = True)
    parser.add_argument('--N', help = 'Total number of splits.', type = int, required = True)

    # Paths arguments
    parser.add_argument('--tilenames', help = 'Path to a .txt file listing the S2 tiles to consider.')
    parser.add_argument('--path_shp', help = 'Path to the Sentinel-2 index shapefile.', default = local_path_shp)
    parser.add_argument('--path_gedi', help = 'Path to the GEDI data directory.', default = '/scratch2/gsialelli/GEDI')
    parser.add_argument('--path_s2', help = 'Path to the Sentinel-2 data directory.', default = '/scratch2/gsialelli/S2_L2A')
    parser.add_argument('--path_s1', help = 'Path to the Sentinel-1 data directory.', default = '/scratch2/gsialelli/S1')
    parser.add_argument('--path_alos', help = 'Path to the ALOS data directory.', default = '/scratch2/gsialelli/ALOS')
    parser.add_argument('--path_ch', help = 'Path to the CH data.', default = '/scratch3/gsialelli/CH')
    parser.add_argument('--path_lc', help = 'Path to the LC data.', default = '/scratch2/gsialelli/LC')
    parser.add_argument('--path_dem', help = 'Path to the DEM data.', default = '/scratch2/gsialelli/ALOS')
    parser.add_argument('--output_path', help = 'Path to the output directory.', default = '/scratch2/gsialelli/patches')
    parser.add_argument('--output_fname', help = 'Name of the output file.', default = '')

    # Flags for the data to extract
    parser.add_argument('--S1', help = 'Whether to extract Sentinel-1 patches.', action = 'store_true')
    parser.add_argument('--ALOS', help = 'Whether to extract ALOS patches.', action = 'store_true')
    parser.add_argument('--CH', help = 'Whether to extract Canopy Height patches.', action = 'store_true')
    parser.add_argument('--LC', help = 'Whether to extract Land Cover patches.', action = 'store_true')
    parser.add_argument('--DEM', help = 'Whether to extract DEM patches.', action = 'store_true')

    # Parse the arguments
    args = parser.parse_args()

    # Check that the tilenames argument is a .txt file
    if args.tilenames is not None: 
        if not args.tilenames.endswith('.txt'): 
            raise ValueError('Please provide a .txt file for the --tilenames argument.')
    
    # Check that the patch_size is odd
    if args.patch_size[0] % 2 == 0 or args.patch_size[1] % 2 == 0 :
        raise ValueError('patch_size must be odd.')

    return args.tilenames, args.year, args.patch_size, args.chunk_size, args.path_shp, args.path_gedi, args.path_s2, args.path_s1, args.path_alos, args.path_ch, args.path_lc, args.path_dem, args.output_path, args.output_fname, args.i, args.N, args.S1, args.ALOS, args.CH, args.LC, args.DEM


def list_s2_tiles(tilenames, grid_df, path_s2) :
    """
    This function performs two tasks: 1) return the list of Sentinel-2 tile names for which we want to extract patches (this
    is done either by listing the files in the Sentinel-2 data directory, or by reading a .txt file if one is provided); and
    2) return the geometries of those tiles, from the Sentinel-2 grid shapefile.

    Args:
    - tilenames: string, path to a .txt file listing the Sentinel-2 tiles to consider.
    - grid_df: geopandas dataframe, Sentinel-2 grid shapefile.

    Returns:
    - tile_names: list of strings, names of the Sentinel-2 tiles to consider.
    - tile_geoms: list of shapely geometries, geometries of the Sentinel-2 tiles to consider.
    """
    
    # Option 1 : list them from the folder of downloaded tiles
    if tilenames is None: 
        all_files = glob.glob(join(path_s2, f'*MSI*.zip'))
        tile_names = [basename(f).strip('.zip') for f in all_files]
    
    # Option 2 : list them from the .txt file
    else:
        with open(tilenames) as f: 
            tile_names = [tile_name.strip().strip('.zip') for tile_name in f.readlines()]
    
    # Get the geometries from the Sentinel-2 grid shapefile
    tile_geoms = [grid_df[grid_df['Name'] == tile_name]['geometry'].values[0] for tile_name in tile_names]

    return tile_names, tile_geoms


def explode_pattern(pattern) :
    """
    `pattern` (string) : the `date`, `orbit_number`, `granule_number`, `track_number`, `ppds_type` and `version_number`
            of the input file; parsed as explained in daac.ornl.gov/GEDI/guides/GEDI_L4A_AGB_Density_V2_1.html
    cf. process_h5_filename() in GEDI/h5_to_csv_to_shp.py

    The date (YYYYDDDHHMMSS) don't keep because we have date, don't need H/M/S
    The orbit number don't keep because is in shot_number

    """
    _, _, granule_number, track_number, ppds_type, version_number = pattern.split('_')
    track_number, version_number = track_number.lstrip('T'), version_number.lstrip('V')
    return int(granule_number), int(track_number), int(ppds_type), int(version_number)


def explode_fp(x) :
    """
    For a floating point (double precision) number `x`, we first split it into its fractional and decimal parts. Then,
    we convert the fractional part to a single precision floating point number; and the decimal part to an unsigned 8-bit
    integer.

    Args:
    - x: float64, floating point number.

    Returns:
    - fractional: float32, fractional part of `x`.
    - decimal: uint8, decimal part of `x`.
    """
    fractional, decimal = np.modf(x)
    return fractional.astype(np.float32), np.abs(decimal).astype(np.uint8)


def modify_GEDI_data(GEDI) :
    """
    This function implements the following changes to the GEDI data:
        1) drop the following columns: 'doy_cos', 'doy_sin', 'lat_cos', 'lat_sin', 'lon_cos', 'lon_sin', 'beam';
        2) explode the `pattern` column into the following columns: 'granule_number', 'track_number', 'ppds_type', 
            'version_number'; and drop the `pattern` column;
        3) for the `lat_lowest` and `lon_lowest` columns, split them into `lat_offset`, `lat_decimal`, `lon_offset`,
            and `lon_decimal`, and drop the original columns.
    
    Args:
    - GEDI: geopandas dataframe, GEDI data.

    Returns:
    - GEDI: geopandas dataframe, modified GEDI data.
    """

    # 1) Remove columns
    columns_to_remove = ['doy_cos', 'doy_sin', 'lat_cos', 'lat_sin', 'lon_cos', 'lon_sin', 'beam']

    # 2) Explode the `pattern` column
    GEDI[['granule_num', 'track_num', 'ppds','version_num']] = GEDI['pattern'].apply(lambda x: pd.Series(explode_pattern(x)))
    columns_to_remove.append('pattern')

    # 3) Split the `lat_lowest` and `lon_lowest` columns
    GEDI[['lat_decimal', 'lat_offset']] = GEDI['lat_lowest'].apply(lambda x: pd.Series(explode_fp(x)))
    GEDI[['lon_decimal', 'lon_offset']] = GEDI['lon_lowest'].apply(lambda x: pd.Series(explode_fp(x)))

    # Specify the data types of the new columns
    GEDI = GEDI.astype(GEDI_attrs)
    
    return GEDI.drop(columns = columns_to_remove)


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


def load_GEDI_data(path_gedi, tile_geom, tile_name) :
    """
    This function loads the GEDI footprints whose geometry intersect the provided geometry.

    Args:
    - path_gedi: string, path to the GEDI data directory.
    - tile_geom: shapely geometry, geometry of the Sentinel-2 tile (in the same CRS as the GEDI data, WGS84)

    Returns:
    - GEDI: geopandas dataframe, GEDI data.
    """

    # Load the data contained in the bounding box of the tile
    # TODO uncomment GEDI = gpd.read_file(path_gedi, engine = 'pyogrio', bbox = tile_geom.bounds)
    GEDI = gpd.read_file('/scratch2/gsialelli/GEDI/L4A_30NXM.shp', engine = 'pyogrio', bbox = tile_geom.bounds)

    # And further filter, to have only the data that intersects the tile
    GEDI = GEDI[GEDI.intersects(tile_geom)]

    # And apply necessary changes
    if not GEDI.empty: GEDI = modify_GEDI_data(GEDI)

    # And reproject the GEDI data to the Sentinel-2 tile's local CRS
    crs = get_CRS_from_S2_tilename(tile_name)
    GEDI = GEDI.to_crs(crs)

    return GEDI, crs


def reproject_gedi_data(GEDI, crs, tile_bounds) :
    """
    This function reprojects the GEDI data to the same CRS as the Sentinel-2 tile, and further filters it to have only the data
    that intersects the tile.

    Args:
    - GEDI: geopandas dataframe, GEDI data.
    - crs: string, CRS of the Sentinel-2 tile.
    - tile_bounds: tuple of floats, bounds of the Sentinel-2 tile (in the tile's CRS).

    Returns:
    - GEDI: geopandas dataframe, GEDI data.
    """

    # Reproject the GEDI data to the same CRS as the Sentinel-2 tile
    GEDI = GEDI.to_crs(crs)

   # And further filter, to have only the data that intersects the tile
    bbox = box(*tile_bounds)
    GEDI = GEDI[GEDI.intersects(bbox)]

    return GEDI


def scl_quality_criteria(scl_patch) :
    """
    This function returns a quality score for the scene classification mask of a Sentinel-2 L2A product.
    The score is between 0 and 100, and is computed as the fraction of pixels that are vegetated.
    If the central pixel is not vegetated, the score is 0, independently of the other pixels.

    Args:
    - scl_patch: 2d array of the scene classification mask.

    Returns:
    - quality_score: int between 0 and 100.
    """

    # Check if the center pixel is vegetated (i.e. not obstructed in any way)
    center_pixel = scl_patch[scl_patch.shape[0] // 2, scl_patch.shape[1] // 2]
    if center_pixel != 4 : return 0

    # Return the fraction of pixels that is vegetated
    fraction = (np.count_nonzero(scl_patch == 4) / (scl_patch.shape[0] * scl_patch.shape[1]))
    
    return int(fraction * 100)


def get_s1_product(s2_prod, path_s1) :
    """
    TODO
    """
    pass


def get_gedi_data(footprint) :
    """
    Return the GEDI footprint data. We drop the `Index`, `geometry`, and `s2_product` keys.

    Args:
    - footprint: pandas.core.frame.Pandas, GEDI footprint.

    Returns:
    - gedi_data : dictionary, with the GEDI attributes as keys, and the corresponding values as values.
    """
    
    # Turn the pandas object into a dictionary
    gedi_data = footprint._asdict()
    
    # Remove the `Index`, `geometry`, and `s2_product` keys
    del gedi_data['Index'], gedi_data['geometry'], gedi_data['s2_product']

    # Remove the lat_lowest and lon_lowest keys
    del gedi_data['lat_lowest'], gedi_data['lon_lowest']
    
    return gedi_data


def get_sentinel1_patch(s1_prod, footprint) :
    """
    TODO

    https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-1-sar/naming-conventions
    """

    # ...
    # patch_data = ...

    # Get metadata like in Sentinel-2

    # Do the same as for Sentinel-2, with `bands` vs `metadata`

    pass


def get_sentinel2_1d_attributes(s2_footprint_data, s2_prod, boa_offset) :
    """
    This function extracts the attributes of the Sentinel-2 L2A product at hand, and returns them.
    The attributes are: the number of days between the start of the GEDI mission and the date of
    acquisition; and identifiers for the product, namely the Processing Baseline Number (PBN), and
    the Relative Orbit Number (RON). Details of the naming convention of Sentinel-2 products can be
    found at https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/naming-convention.

    Args:
    - s2_footprint_data: dictionary, with the Sentinel-2 data.
    - s2_prod: string, name of the Sentinel-2 L2A product.
    - boa_offset: int, 1 if the product was acquired after January 25th, 2022; 0 otherwise.

    Returns:
    - s2_footprint_data: dictionary, with the Sentinel-2 data, plus the attributes.
    """

    # Parse the product name
    _, _, date, pbn, ron, _, _ = s2_prod.split('_')
    
    # Get the date of acquisition and day of year
    date_obj = dt.datetime.strptime(date, "%Y%m%dT%H%M%S")
    start = dt.datetime.strptime(GEDI_START_MISSION, '%Y-%m-%d')

    # Save the # of days since the start of the GEDI mission
    difference = (date_obj - start).days
    s2_footprint_data['date'] = difference

    # Also save information to identify the product
    s2_footprint_data['pbn'] = int(pbn.lstrip('N'))
    s2_footprint_data['ron'] = int(ron.lstrip('R'))

    # And save whether to apply the BOA offset
    s2_footprint_data['boa_offset'] = boa_offset

    return s2_footprint_data


def get_sentinel2_patch(transform, processed_bands, footprint, patch_size, s2_prod, boa_offset) :
    """
    This function extracts a patch of `patch_size` pixels from the Sentinel-2 L2A product (`processed_bands`),
    centered around the GEDI `footprint`. It then checks the quality of the patch, and if it is good enough
    (in terms of being vegetated), extracts all the bands' information. Otherwise, it returns None. Once the
    patch is extracted, it also extracts 1d attributes from the Sentinel-2 L2A product name (`s2_prod`), namely
    the number of days between the start of the GEDI mission and the date of acquisition; the Julian day of year
    (encoded with cos and sin); the tile name; and the product name.

    Args:
    - processed_bands: dictionary, with the band names as keys, and the corresponding 2d arrays as values.
    - footprint: geopandas Series, GEDI footprint.
    - patch_size: tuple of ints, size of the patch to extract.
    - s2_prod: string, name of the Sentinel-2 L2A product.

    Returns:
    - s2_footprint_data: dictionary, with 'bands' and 'metadata' as keys, and the corresponding data as values.
            or None, if the patch is not good enough.
    """

    # Initialize the data
    patch_bands, patch_data = {}, {}

    # Get the row and column corresponding to the footprint center
    pt_x, pt_y = footprint.geometry.x, footprint.geometry.y
    x, y = rowcol(transform, pt_x, pt_y)
    
    # Get the size of the window to extract
    x_offset, y_offset = (patch_size[0] - 1) // 2, (patch_size[1] - 1) // 2

    # Check that the patch fits in the tile, otherwise skip this patch
    data = processed_bands['B02']
    if (x - x_offset < 0) or (x + x_offset + 1 > data.shape[0]) \
        or (y - y_offset < 0) or (y + y_offset + 1 > data.shape[1]) :
            return None

    # Check the SCL band to see if the patch is worth keeping
    patch = processed_bands['SCL'][x - x_offset : x + x_offset + 1, y - y_offset : y + y_offset + 1]
    vegetation_score = scl_quality_criteria(patch)
    if vegetation_score < 20 :
        return None
    else:
        patch_data['vegetation_score'] = vegetation_score

    # Iterate over the rest of the bands
    for band, data in processed_bands.items() :
        patch = data[x - x_offset : x + x_offset + 1, y - y_offset : y + y_offset + 1]
        patch_bands[band] = patch
    
    # Add the 1d attributes to the result dictionary
    patch_data = get_sentinel2_1d_attributes(patch_data, s2_prod, boa_offset)

    # We add the stacked bands to the result dictionary
    patch_data['bands'] = np.array([patch_bands[attr] for attr in S2_attrs['bands'].keys()])

    # TODO FOR DEBUGGING PURPOSES, REMOVE
    west, north = rs.transform.xy(transform, x - x_offset, y - y_offset)
    east, south = rs.transform.xy(transform, x + x_offset, y + y_offset)
    patch_transform = rs.transform.from_bounds(west, south, east, north, patch_size[0], patch_size[1])
    patch_data['patch_transform'] = patch_transform

    return patch_data


def crop_and_pad_arrays(alos_data, ul_row, lr_row, ul_col, lr_col):
    """
    This function crops (and pads if necessary) the ALOS data to match the shape provided by
    the upper left and lower right indices.

    Args:
    - alos_data: 2d array, ALOS data.
    - (ul_row, ul_col) : tuple of ints, indices of the ALOS pixel corresponding to the upper
        left corner of the Sentinel-2 tile.
    - (lr_row, lr_col) : tuple of ints, indices of the ALOS pixel corresponding to the lower
        right corner of the Sentinel-2 tile.
    
    Returns:
    - alos_data: 2d array, cropped (and padded) ALOS HH data.
    """

    # Get the dimensions of the arrays
    alos_height, alos_width = alos_data.shape

    # If any of the slicing indices are out of bounds, pad with zeros
    if ul_row < 0 or lr_row >= alos_height or ul_col < 0 or lr_col >= alos_width:

        print('The Sentinel-2 tile is not contained in the ALOS tile.')

        # Calculate the new shape after padding
        new_height = lr_row - ul_row + 1
        new_width = lr_col - ul_col + 1

        # Create new arrays to store the padded data
        padded_alos_data = np.zeros((new_height, new_width), dtype = alos_data.dtype)

        # Compute the region of interest in the new padded arrays
        start_row = max(0, -ul_row)
        end_row = min(alos_height - ul_row, lr_row - ul_row + 1)
        start_col = max(0, -ul_col)
        end_col = min(alos_width - ul_col, lr_col - ul_col + 1)

        # Copy the original data to the new padded arrays
        padded_alos_data[start_row : end_row, start_col : end_col] = alos_data[max(0, ul_row) : min(alos_height, lr_row + 1), max(0, ul_col) : min(alos_width, lr_col + 1)]

        # Update the variables to point to the new padded arrays
        alos_data = padded_alos_data

    # Otherwise, simply perform the slicing operation
    else: alos_data = alos_data[ul_row : lr_row + 1, ul_col : lr_col + 1]

    return alos_data


def get_tile(data, s2_transform, upsampling_shape, data_source, data_attrs) :
    """
    This function extracts the data for the Sentinel-2 L2A product at hand, crops it so as to perfectly match
    the Sentinel-2 tile, resamples it to 10m resolution when necessary, and returns it.

    Args:
    - data: 
    - s2_transform: affine.Affine, transform of the Sentinel-2 L2A product.
    - upsampling_shape: tuple of ints, shape of the Sentinel-2 L2A product, at 10m resolution.
    - footprint: geopandas Series, GEDI footprint.

    Returns:
    - res: dict, with the attributes as keys and the corresponding 2d arrays as values.
    """

    if data == {} : return None

    # Get the transforms
    s2_transformer, data_transformer = AffineTransformer(s2_transform), AffineTransformer(data['transform'])    

    # Upper left corner
    ul_x, ul_y = s2_transformer.xy(0, 0)
    ul_row, ul_col = data_transformer.rowcol(ul_x, ul_y)

    # Lower right corner
    lr_x, lr_y = s2_transformer.xy(upsampling_shape[0] - 1, upsampling_shape[1] - 1)
    lr_row, lr_col = data_transformer.rowcol(lr_x, lr_y)

    # Crop the data to the same bounds, padding the data if necessary
    res = {}
    for data_attr in data_attrs.keys() :
        res[data_attr] = crop_and_pad_arrays(data[data_attr], ul_row, lr_row, ul_col, lr_col)

        # Resample to 10m resolution if necessary
        if data_source in ['ALOS', 'DEM'] :
            res[data_attr] = resize(res[data_attr], upsampling_shape, order = 3, preserve_range = True).astype(data_attrs[data_attr])
        elif data_source == 'LC' :
            res[data_attr] = resize(res[data_attr], upsampling_shape, order = 1, preserve_range = True).astype(data_attrs[data_attr])
        
        assert res[data_attr].shape == upsampling_shape, f'{data_source} | {data_attr} | {data[data_attr].shape} | {res[data_attr].shape} | {upsampling_shape} | {ul_row} | {lr_row} | {ul_col} | {lr_col}'

    return res


def get_patch(tile, footprint, transform, patch_size, data_source, data_attrs) :
    """
    This function extracts a patch of `patch_size` pixels from the `tile`, centered around
    the GEDI `footprint`.

    Args:
    - tile: dictionary, with the tile data.
    - footprint: geopandas Series, GEDI footprint.
    - transform: affine.Affine, transform of the tile.
    - patch_size: tuple of ints, size of the patch to extract.

    Returns:
    - patch_data: dictionary, with 'bands' as key, and the corresponding 3d numpy array as value.
    """

    # Initialize the data
    patch_bands, patch_data = {}, {}

    # If the tile is None, fill the patch with NODATAVALS
    if tile is None:
        if data_source in ['ALOS', 'LC'] :
            patch_data['bands'] = np.full((len(data_attrs), patch_size[0], patch_size[1]), fill_value = NODATAVALS[data_source], dtype = data_attrs.values[0])
        elif data_source in ['CH', 'DEM'] :
            for attr, dtype in data_attrs.items() :
                patch_data[attr] = np.full(patch_size, fill_value = NODATAVALS[data_source], dtype = dtype)    
    
    else:

        # Get the row and column corresponding to the footprint center
        pt_x, pt_y = footprint.geometry.x, footprint.geometry.y
        x, y = rowcol(transform, pt_x, pt_y)
        
        # Get the size of the window to extract
        x_offset, y_offset = (patch_size[0] - 1) // 2, (patch_size[1] - 1) // 2

        # Crop the tile to the patch
        for attr, data in tile.items() :
            patch = data[x - x_offset : x + x_offset + 1, y - y_offset : y + y_offset + 1]
            assert patch.shape == (patch_size[0], patch_size[1]), f'{data_source} Patch shape is {patch.shape}, should be {patch_size} | attr {attr} | data shape {data.shape}'
            patch_bands[attr] = patch

        # Stack the bands
        if data_source in ['ALOS', 'LC'] :
            patch_data['bands'] = np.array([patch_bands[attr] for attr in data_attrs.keys()])
        
        elif data_source in ['CH', 'DEM'] :
            patch_data = patch_bands

    return patch_data


def radiometric_offset_values(path_s2, product, offset) :
    """
    This function extracts the BOA_QUANTIFICATION_VALUE and BOA_ADD_OFFSET_VALUES from the
    Sentinel-2 L2A product at hand, and returns them.

    Args:
    - path_s2: string, path to the Sentinel-2 data directory.
    - product: string, name of the Sentinel-2 L2A product.
    - offset: int, 1 if the product was acquired after January 25th, 2022; 0 otherwise.

    Returns:
    - None
    """

    # There is a mismatch between the names of the physical bands in the metadata file, and the
    # names of the bands in the IMG_DATA/ folder. This dictionary defines the mapping
    bands_mapping = {'B1': 'B01', 'B2': 'B02', 'B3': 'B03', 'B4': 'B04', 'B5': 'B05', 'B6': 'B06', 'B7': 'B07', \
                    'B8': 'B08', 'B8A': 'B8A', 'B9': 'B09', 'B10': 'B10', 'B11': 'B11', 'B12': 'B12'}

    # Parse the XML file
    tree = ET.parse(f'{join(path_s2, product)}.SAFE/MTD_MSIL2A.xml')
    root = tree.getroot()

    # Get the BOA_QUANTIFICATION_VALUE
    for elem in root.find('.//QUANTIFICATION_VALUES_LIST') :
        if elem.tag == 'BOA_QUANTIFICATION_VALUE' :
            boa_quantification_value = float(elem.text)
            assert boa_quantification_value == 10000, f'BOA_QUANTIFICATION_VALUE is {boa_quantification_value}, should be 10000'
        else: continue

    # Get the physical bands and their ids
    physical_bands = {elem.get('bandId'): elem.get('physicalBand') \
                      for elem in root.find('.//Spectral_Information_List')}

    if offset :
        
        # Check the BOA offset values (should be 1000)
        for elem in root.find('.//BOA_ADD_OFFSET_VALUES_LIST') :
            physical_band = physical_bands[elem.get('band_id')]
            actual_band = bands_mapping[physical_band]
            boa_add_offset_value = int(elem.text)
            assert boa_add_offset_value == 1000, f'BOA_ADD_OFFSET_VALUE is {boa_add_offset_value}, should be 1000 | band {actual_band}'


def process_S2_tile(product, path_s2) :
    """
    This function iterates over the bands of the Sentinel-2 L2A product at hand; reprojects them to
    EPSG 4326; upsamples them to 10m resolution (when needed) using cubic interpolation (nearest
    neighbor for the scene classification mask); and returns them.
    
    Args:
    - product: string, name of the Sentinel-2 L2A product.
    - path_s2: string, path to the Sentinel-2 data directory.

    Returns:
    - processed_bands: dictionary, with the band names as keys, and the corresponding 2d arrays as values.
    """

    # Get the path to the IMG_DATA/ folder of the Sentinel-2 product
    path_to_img_data = glob.glob(join(path_s2, product + '.SAFE', 'GRANULE', '*', 'IMG_DATA'))[0]

    # Get the date and tile name from the L2A product name
    _, _, date, _, _, tname, _ = product.split('_')
    year, month, day = int(date[:4]), int(date[4:6]), int(date[6:8])

    # Check the BOA quantification value (and BOA offsets if applicable)
    if dt.date(2022, 1, 25) <= dt.date(year, month, day) : boa_offset = 1 
    else: boa_offset = 0
    radiometric_offset_values(path_s2, product, boa_offset)

    # Iterate over the bands    
    processed_bands = {}
    for res, bands in S2_L2A_BANDS.items() :
        for band in bands :

            # Read the band data
            with rs.open(join(path_to_img_data, f'R{res}', f'{tname}_{date}_{band}_{res}.tif')) as src :
                band_data = src.read(1)
                transform = src.transform
                crs = src.crs
                bounds = src.bounds
            # Turn the band into a 2d array
            if len(band_data.shape) == 3 : band_data = band_data[0, :, :]

            # Use the 10m resolution B02 band as reference
            if res == '10m' :
                if band == 'B02' :
                    # Base the other bands' upsampling on this band's
                    upsampling_shape = band_data.shape
                    # Save the transform of this band
                    _transform = transform
            
            # Upsample the band to a 10m resolution if necessary
            else :
                # Order 1 indicates nearest interpolation, and order 3 indicates cubic interpolation                
                band_data = resize(band_data, upsampling_shape, order = 1 if band == 'SCL' else 3, preserve_range = True)
                band_data = band_data.astype(S2_attrs['bands'][band])

            # Store the transformed band
            assert band_data.shape == (upsampling_shape[0], upsampling_shape[1])
            processed_bands[band] = band_data

    return _transform, upsampling_shape, processed_bands, crs, bounds, boa_offset


def GEDI_to_S2_date(GEDI_date) :
    """
    This function converts the GEDI date of acquisition (number of days since the start of the
    mission) to a Sentinel-2 analysis compatible date, in YYYY-MM-DD format.

    Args:
    - GEDI_date: string, date of the GEDI footprint.

    Returns:
    - acq_date: string, date of the Sentinel-2 L2A product.
    """
    start_of_mission = dt.datetime.strptime('2019-04-17', '%Y-%m-%d')
    acq_date = start_of_mission + dt.timedelta(days = int(GEDI_date))
    return acq_date.strftime('%Y-%m-%d')


def match_s2_product(gedi_date, tile_name, path_s2) :
    """
    For a given Sentinel-2 tile and GEDI date of acquisition (YYYY-MM-DD), select the Sentinel-2
    L2A product that is closest in time to the GEDI footprint, but not after it.

    Args:
    - gedi_date: string, date of the GEDI footprint (format YYYY-MM-DD).
    - path_s2: string, path to the Sentinel-2 data directory.

    Returns:
    - s2_product: string, name of the Sentinel-2 L2A product.
    """

    # Get the year, month and day of the GEDI footprint
    year, month, day = GEDI_to_S2_date(gedi_date).split('-')

    # Check if there is a match for the current month, in which case we also need to
    # check the day of acquisition
    match = glob.glob(join(path_s2, f'*MSI*_{year}{month}*_T{tile_name}*.zip'))
    if match:
        match_day = basename(match[0]).split('_')[2].split('T')[0][-2:]
        if int(match_day) <= int(day):
            return basename(match[0]).strip('.zip')
    
    # Check the matches over the past six months
    year, month = int(year), int(month)
    for _ in range(1, 6) :
        month = int(month - 2) % 12 + 1
        year = year - 1 if month == 12 else year
        match = glob.glob(join(path_s2, f'*MSI*_{year}{month:02d}*_T{tile_name}*.zip'))
        if match: return basename(match[0]).strip('.zip')


def group_GEDI_by_S2(GEDI, tile_name, path_s2) :
    """
    Since many GEDI footprints were sampled on the same day, we can group them as to
    execute the `match_s2_product` fewer times. To this end, this function lists the
    unique GEDI acquisition dates, and for each of them, finds the corresponding S2
    L2A product. It then populates the GEDI dataframe with a column for the S2 product,
    and groups the footprints by S2 product.

    Args:
    - GEDI: geopandas dataframe, GEDI data.
    - tile_name: string, name of the Sentinel-2 tile.
    - path_s2: string, path to the Sentinel-2 data directory.

    Returns:
    - groups: pandas groupby object, GEDI data grouped by S2 product.
    """
    
    # Get all unique GEDI acquisition dates
    unique_dates = GEDI['date'].unique()
    
    # For each date, find the corresponding Sentinel-2 L2A product
    matches = {gedi_date: match_s2_product(gedi_date, tile_name, path_s2) \
                                for gedi_date in unique_dates}
    
    # Populate the GEDI dataframe with a column for the Sentinel-2 product
    GEDI['s2_product'] = GEDI['date'].map(matches)

    # Drop the footprints for which no match was found
    print(f'Dropping {GEDI.s2_product.isna().sum()}/{len(GEDI)} rows, for lack of S2 match.')
    GEDI = GEDI.dropna(subset = ['s2_product'])

    # Group the GEDI data by Sentinel-2 product
    return GEDI.groupby('s2_product')


def setup_output_files(output_path, output_fname, i, N) :
    """
    The patches extraction procedure takes place in parallel, executed by N processes. As each process creates new patches (one 
    after the other), they need to be saved somewhere. This function creates and initializes an output file per process, for them
    to write the patches to. 

    Args:
    - output_path: string, path to the output directory.
    - i: int, process split i/N.
    - N: int, total number of splits.

    Returns:
    - None
    """

    # Initialize output files for each process to write the patches to
    if output_fname == '' : fname = f'data_{i}-{N}.h5'
    else: fname = f'data_{output_fname}_{i}-{N}.h5'
    with h5py.File(join(output_path, fname), 'w') as file :
        print(f'Initializing output file for split {i}/{N}.')


def initialize_results(S1_flag, ALOS_flag, CH_flag, LC_flag, DEM_flag) :
    """
    This function initializes the results placeholder.

    Args:
    - None

    Returns:
    - (s2_data, s1_data, gedi_data, alos_data) : dictionaries, with the Sentinel-2/Sentinel-1/GEDI/ALOS attributes as keys, and empty
        lists as values.
    """

    # Populate the GEDI results placeholder
    gedi_data = {k: [] for k in GEDI_attrs.keys()}
    
    # Populate the S2 placeholder
    s2_data = {k: [] for k in list(S2_attrs['metadata'].keys()) + ['bands']}
    
    # Populate the S1 placeholder
    if S1_flag: s1_data = {k: [] for k in list(S1_attrs['metadata'].keys()) + ['bands']}
    else: s1_data = None

    # Populate the ALOS placeholder
    if ALOS_flag: alos_data = {'bands': []}
    else: alos_data = None

    # Populate the CH placeholder
    if CH_flag: ch_data = {k: [] for k in CH_attrs.keys()}
    else: ch_data = None

    # Populate the LC placeholder
    if LC_flag: lc_data = {'bands': []}
    else: lc_data = None

    # Populate the DEM placeholder
    if DEM_flag: dem_data = {k: [] for k in DEM_attrs.keys()}
    else: dem_data = None

    return s2_data, s1_data, gedi_data, alos_data, ch_data, lc_data, dem_data


def update_results(s2_data, s1_data, gedi_data, alos_data, ch_data, lc_data, dem_data, s2_footprint_data, s1_footprint_data, gedi_footprint_data, alos_footprint_data, ch_footprint_data, lc_footprint_data, dem_footprint_data) :
    """
    This function updates the results placeholder with the data from the current GEDI footprint.

    Args:
    - (s2_data, s1_data, gedi_data, alos_data) : dictionaries, placeholders for the Sentinel-2, Sentinel-1 and GEDI data.
    - (s2_footprint_data, s1_footprint_data, gedi_footprint_data, alos_footprint_data) : dictionaries, with the Sentinel-2, Sentinel-1,
        GEDI, and ALOS data to update the placeholders with.
    
    Returns:
    - (s2_data, s1_data, gedi_data, alos_data) : dictionaries, updated placeholders for the Sentinel-2, Sentinel-1 and GEDI data.
    """
    
    # Iterate over the placeholders and the new data, and update the placeholders
    for placeholder, new_data in zip([s2_data, s1_data, gedi_data, alos_data, ch_data, lc_data, dem_data], [s2_footprint_data, s1_footprint_data, gedi_footprint_data, alos_footprint_data, ch_footprint_data, lc_footprint_data, dem_footprint_data]) :
        if placeholder is None: continue
        for attr, data in new_data.items() :
            placeholder[attr].append(data)
    return s2_data, s1_data, gedi_data, alos_data, ch_data, lc_data, dem_data


def save_results(s2_data, s1_data, gedi_data, alos_data, ch_data, lc_data, dem_data, tile_name, chunk_size, file) :
    """
    This function: 1) identifies the PID of the current process; 2) opens the corresponding output file; and 3) appends the
    results to the output file, group by group, and attribute by attribute.

    Args:
    - (s2_data, s1_data, gedi_data) : dictionaries, placeholders for the Sentinel-2, Sentinel-1 and GEDI data.
    - tile_name: string, name of the Sentinel-2 tile.
    - chunk_size: int, number of patches to write to file at once.
    - file: h5py File object, opened hdf5 file.

    Returns:
    - None
    """

    # Iterate over the Sentinel-2 data
    if s2_data is not None:
        for attr, data in s2_data.items() :
            
            if attr in S2_attrs['metadata'].keys() :
                dset = file[tile_name]['Sentinel_metadata'][f'S2_{attr}']
                dset.resize(len(dset) + chunk_size, axis = 0)
                dset[slice(-chunk_size, None, None)] = np.array(data).astype(dset.dtype)
            
            elif attr == 'bands' :
                # Put the bands as the last axis
                data = np.moveaxis(np.array(data), 1, -1)
                
                # Save the 'normal' bands together
                dset = file[tile_name][f'S2_bands']
                dset.resize(len(dset) + chunk_size, axis = 0)
                dset[slice(-chunk_size, None, None)] = data[:, :, :, : -1]
                
                # Save the SCL band separately
                dset = file[tile_name][f'S2_SCL']
                dset.resize(len(dset) + chunk_size, axis = 0)
                dset[slice(-chunk_size, None, None)] = data[:, :, :, -1]

    # Iterate over the S1 data
    if s1_data is not None:
        for attr, data in s1_data.items() :
            if attr in S1_attrs['metadata'].keys() :
                dset = file[tile_name]['Sentinel_metadata'][f'S1_{attr}']
                dset.resize(len(dset) + chunk_size, axis = 0)
                dset[slice(-chunk_size, None, None)] = np.array(data).astype(dset.dtype)
            elif attr == 'bands' :
                dset = file[tile_name][f'S1_bands']
                dset.resize(len(dset) + chunk_size, axis = 0)
                # Put the bands as the last axis
                data = np.moveaxis(np.array(data), 1, -1).astype(dset.dtype)
                dset[slice(-chunk_size, None, None)] = data

    # Iterate over the ALOS data
    if alos_data is not None:
        dset = file[tile_name]['ALOS_bands']
        # Put the bands as the last axis
        dset.resize(len(dset) + chunk_size, axis = 0)
        data = np.moveaxis(alos_data['bands'], 1, -1).astype(dset.dtype)
        dset[slice(-chunk_size, None, None)] = data

    # Iterate over the GEDI data
    for attr, data in gedi_data.items() :
        dset = file[tile_name]['GEDI'][attr]
        dset.resize(len(dset) + chunk_size, axis = 0)
        dset[slice(-chunk_size, None, None)] = np.array(data).astype(dset.dtype)
    
    # Iterate over the CH data
    if ch_data is not None:
        for attr, data in ch_data.items() :
            dset = file[tile_name]['CH'][attr]
            dset.resize(len(dset) + chunk_size, axis = 0)
            dset[slice(-chunk_size, None, None)] = np.array(data).astype(dset.dtype)
    
    # Iterate over the LC data
    if lc_data is not None:
        dset = file[tile_name]['LC']
        # Put the bands as the last axis
        dset.resize(len(dset) + chunk_size, axis = 0)
        data = np.moveaxis(lc_data['bands'], 1, -1).astype(dset.dtype)
        dset[slice(-chunk_size, None, None)] = data
    
    # Iterate over the DEM data
    if dem_data is not None:
        dset = file[tile_name]['DEM']
        dset.resize(len(dset) + chunk_size, axis = 0)
        data = np.array(dem_data['dem']).astype(dset.dtype)
        dset[slice(-chunk_size, None, None)] = data


def init_h5_group(file, tile_name, patch_size, chunk_size, S1_flag, ALOS_flag, CH_flag, LC_flag, DEM_flag) :
    """
    For a given (opened) empty hdf5 `file`, create a group for the current `tile_name`, and initialize the datasets for the
    Sentinel-2, Sentinel-1 and GEDI data. In particular, this is the structure of the datasets:
		> (group) 32TNN
					> (dataset) S2_bands, N x 15 x 15 x 12 (uint16)
										> (attrs) order
                    > (dataset) S2_SCL, N x 15 x 15 x 1 (uint8)
					> (dataset) S1_bands, N x 15 x 15 x 4
										> (attrs) order
					> (group) Sentinel_metadata
								> (dataset) S2_vegetation_score, N x 1
								> (dataset) S2_date, N x 1
                                > (dataset) S2_pbn, N x 1
                                > (dataset) S2_ron, N x 1
                                > (dataset) S1_date, N x 1
                                > TODO the other S1 metadata
					> (group) GEDI
								> (dataset) agbd, N x 1
								> ...
								> (dataset) lon_decimal, N x 1
                    > (dataset) ALOS_bands, N x 15 x 15 x 2
                                        > (attrs) order
                    > (group) CH
                            > (dataset) ch, N x 15 x 15 x 1 (uint8)
                            > (dataset) std, N x 15 x 15 x 1 (float32)
                    > (dataset) LC, N x 15 x 15 x 2 (uint8)
                                        > (attrs) order
                    > (dataset) DEM, N x 15 x 15 x 1 (int16)
                                        
    Args:
    - file: h5py File object, opened hdf5 file.
    - tile_name: string, name of the Sentinel-2 tile.
    - patch_size: tuple of ints, size of the patches to extract.
    - chunk_size: int, number of patches to write to file at once.
    - S1_flag: bool, whether to extract Sentinel-1 data.
    - ALOS_flag: bool, whether to extract ALOS data.

    Returns:
    - None
    """

    # Initialize the group for the current tile
    h5_group = file.create_group(tile_name)

    # Create the S2_bands dataset
    num_s2_bands = len(S2_attrs['bands']) - 1
    dtype = S2_attrs['bands']['B01']
    dset = h5_group.create_dataset('S2_bands', shape = (0, patch_size[0], patch_size[1], num_s2_bands), \
                            maxshape = (None, patch_size[0], patch_size[1], num_s2_bands), \
                            dtype = dtype, compression = 'gzip', chunks = (chunk_size, patch_size[0], patch_size[1], num_s2_bands))
    dset.attrs['order'] = list(S2_attrs['bands'].keys())[:-1]

    # Create the S2_SCL dataset
    dtype = S2_attrs['bands']['SCL']
    dset = h5_group.create_dataset('S2_SCL', shape = (0, patch_size[0], patch_size[1]), \
                            maxshape = (None, patch_size[0], patch_size[1]), \
                            dtype = dtype, compression = 'gzip', chunks = (chunk_size, patch_size[0], patch_size[1]))

    # Create the ALOS_bands dataset
    if ALOS_flag:
        num_alos_bands = len(ALOS_attrs)
        dtype = ALOS_attrs['HH']
        dset = h5_group.create_dataset('ALOS_bands', shape = (0, patch_size[0], patch_size[1], num_alos_bands), \
                                maxshape = (None, patch_size[0], patch_size[1], num_alos_bands), \
                                dtype = dtype, compression = 'gzip', chunks = (chunk_size, patch_size[0], patch_size[1], num_alos_bands))
        dset.attrs['order'] = list(ALOS_attrs.keys())

    # Create the Sentinel_metadata group and corresponding datasets
    sentinel_metadata_group = h5_group.create_group('Sentinel_metadata')
    for prod, prod_attrs in zip(['S1', 'S2'], [S1_attrs, S2_attrs]) :

        # Skip Sentinel-1 if we don't want to extract it
        if prod == 'S1' and not S1_flag: continue

        for attr, val in prod_attrs['metadata'].items() :
            sentinel_metadata_group.create_dataset(f'{prod}_{attr}', shape = (0,), maxshape = (None,), dtype = val, \
                                                    compression = 'gzip', chunks = (chunk_size,))

    # Create the S1_bands dataset
    if S1_flag:
        num_s1_bands = len(S1_attrs['bands'])
        dtype = S1_attrs['bands']['HH']
        h5_group.create_dataset('S1_bands', shape = (0, patch_size[0], patch_size[1], num_s1_bands), \
                                maxshape = (None, patch_size[0], patch_size[1], num_s1_bands), \
                                dtype = dtype, compression = 'gzip', chunks = (chunk_size, patch_size[0], patch_size[1], num_s1_bands))

    # Create the GEDI group
    gedi_group = h5_group.create_group('GEDI')
    for attr, dtype in GEDI_attrs.items() :
        if dtype == str: dtype = h5py.string_dtype()
        gedi_group.create_dataset(attr, shape = (0,), maxshape = (None,), dtype = dtype, compression = 'gzip', chunks = (chunk_size,))

    # Create the CH group
    if CH_flag:
        ch_group = h5_group.create_group('CH')
        for attr, dtype in CH_attrs.items() :
            ch_group.create_dataset(attr, shape = (0, patch_size[0], patch_size[1]), maxshape = (None, patch_size[0], patch_size[1]), dtype = dtype, compression = 'gzip', chunks = (chunk_size, patch_size[0], patch_size[1]))

    # Create the LC dataset
    if LC_flag:
        dtype = LC_attrs['lc']
        num_bands = len(LC_attrs)
        h5_group.create_dataset('LC', shape = (0, patch_size[0], patch_size[1], num_bands), maxshape = (None, patch_size[0], patch_size[1], 2), dtype = dtype, compression = 'gzip', chunks = (chunk_size, patch_size[0], patch_size[1], num_bands))

    # Create the DEM dataset
    if DEM_flag :
        h5_group.create_dataset('DEM', shape = (0, patch_size[0], patch_size[1]), maxshape = (None, patch_size[0], patch_size[1]), dtype = DEM_attrs['dem'], compression = 'gzip', chunks = (chunk_size, patch_size[0], patch_size[1]))


def load_ALOS_data(tile_name, groups, path_alos) :
    """
    For a given Sentinel-2 `tile_name`, and the corresponding grouping of the matching Sentinel-2 products and GEDI footprints,
    this function loads the pre-processed ALOS PALSAR mosaics for the tile, and the years spanned by the products. We return a
    dictionary with the years spanned as keys, and the corresponding ALOS PALSAR data (in the format of a dictionary with keys
    'transform', 'HH', and 'HV') as values.

    Args:
    - tile_name: string, name of the Sentinel-2 tile.
    - groups: pandas groupby object, GEDI footprints for the tile, grouped by S2 product.
    - path_alos: string, path to the ALOS PALSAR data directory.

    Returns:
    - alos_tiles: dictionary, with the years spanned as keys, and the corresponding ALOS PALSAR data as values.
    """
    
    alos_tiles = {}
    
    years = np.unique([s2_prod.split('_')[2][:4] for s2_prod, _ in groups])
    for year in years :

        cropped_year = year[2:4]

        alos_tiles_year = {}
        if exists(join(path_alos, f'ALOS_{tile_name}_{cropped_year}.tif')) :
            
            with rs.open(join(path_alos, f'ALOS_{tile_name}_{cropped_year}.tif'), 'r') as dataset:
                alos_tiles_year['transform'] = dataset.transform
                band_1, band_2 = dataset.descriptions
                alos_tiles_year[band_1] = dataset.read(1)
                alos_tiles_year[band_2] = dataset.read(2)
            
        alos_tiles[year] = alos_tiles_year
    
    return alos_tiles


def load_CH_data(path_ch, tile_name, year) :
    """
    This function loads the CH data for the current tile and year.

    Args:
    - path_ch: string, path to the CH data directory.
    - tile_name: string, name of the Sentinel-2 tile.
    - year: str, year of the Sentinel-2 product.

    Returns:
    - CH: dictionary, with the CH attributes as keys, and the corresponding values as values.
    """

    CH = {}

    # TODO revert back to the original way of reading the data when the data transfer is done accordingly
    # for now it will always load the 2019 data

    # with rs.open(join(path_ch, tile_name, year, 'preds_inv_var_mean', f'{tile_name}_pred.tif')) as src:
    with rs.open(join(path_ch, f'{tile_name}_pred.tif')) as src:
        CH['ch'] = src.read(1)
        CH['transform'] = src.transform

    # with rs.open(join(path_ch, tile_name, year, 'preds_inv_var_mean', f'{tile_name}_std.tif')) as src:
    with rs.open(join(path_ch, f'{tile_name}_std.tif')) as src:
        CH['std'] = src.read(1)
    
    return CH


def ch_quality_score(patch, NO_DATA = 255) :
    """
    This function returns a quality score for the CH patch.
    The score is between 0 and 100, and is computed as the fraction of pixels that are not NO_DATA.
    If the central pixel is NO_DATA, the score is 0, independently of the other pixels.

    Args:
    - patch: 2d array of the CH agbd prediction.

    Returns:
    - quality_score: int between 0 and 100.
    """

    # Check if the center pixel is vegetated (i.e. not obstructed in any way)
    center_pixel = patch[patch.shape[0] // 2, patch.shape[1] // 2]
    if center_pixel != NO_DATA : return 0

    # Return the fraction of pixels that is vegetated
    fraction = (np.count_nonzero(patch == NO_DATA) / (patch.shape[0] * patch.shape[1]))
    
    return int(fraction * 100)


def load_LC_data(path_lc, tile_name, year) :
    """
    This function loads the LC data for the current tile and year.

    Args:
    - path_lc: string, path to the LC data directory.
    - tile_name: string, name of the Sentinel-2 tile.
    - year: str, year of the Sentinel-2 product.

    Returns:
    - LC: dictionary, with the LC attributes as keys, and the corresponding values as values.
    """

    LC = {}
    with rs.open(join(path_lc, f'LC_{tile_name}_{year}.tif'), 'r') as src :
        LC['lc'] = src.read(1)
        LC['prob'] = src.read(2)
        LC['transform'] = src.transform
    return LC


def load_DEM_data(path_dem, tile_name) :
    """
    This function loads the DEM data for the current tile.

    Args:
    - path_dem: string, path to the DEM data directory.
    - tile_name: string, name of the Sentinel-2 tile.

    Returns:
    - DEM: dictionary, with the DEM attributes as keys, and the corresponding values as values.
    """

    DEM = {}
    with rs.open(join(path_dem, f'DEM_{tile_name}.tif'), 'r') as src :
        DEM['dem'] = src.read(1)
        DEM['transform'] = src.transform
    return DEM


############################################################################################################################
# Main function

def extract_patches(tile_name, year, tile_geom, patch_size, chunk_size, path_gedi, path_s2, path_s1, path_alos, path_ch, path_lc, path_dem, output_path, output_fname, i, N, S1_flag, ALOS_flag, CH_flag, LC_flag, DEM_flag) :
    """
    This function extracts the GEDI footprint-centered (patch_size[0] x patch_size[1]) patches from the Sentinel-2 L2A products with `tile_name`, as well
    as the corresponding Sentinel-1 and ALOS PALSAR-2 data. The patches are iteratively saved to an hdf5 file.

    More specifically, the function:
    1) Loads the GEDI data using the geometry of the tile.
    2) Groups the footprints by their corresponding Sentinel-2 product (so that we only process each product once, and extract the data corresponding
        to the footprints).
    3) Loads the ALOS PALSAR mosaics for the tile, across the years spanned by the S2 products.
    4) Iterate over the S2 products, and for each product: reprojects to EPSG:4326 and upsamples to 10m resolution the S2 data; gets the corresponding S1
        product and processes it; crops to the product and upsamples to 10m resolution the ALOS tile; and extracts the patches for the footprints.
    5) Saves the results to the output file.

    Args:
    - tile_name: string, name of the Sentinel-2 tile.
    - tile_geom: shapely Polygon, geometry of the Sentinel-2 tile.
    - patch_size: tuple of ints, size of the patches to extract.
    - chunk_size: int, number of patches to write to file at once.
    - path_shp: string, path to the Sentinel-2 grid shapefile.
    - path_gedi: string, path to the GEDI data directory.
    - path_s2: string, path to the Sentinel-2 data directory.
    - path_s1: string, path to the Sentinel-1 data directory.
    - path_alos: string, path to the ALOS PALSAR data directory.
    - output_path: string, path to the output directory.
    - S1_flag: boolean, whether to extract Sentinel-1 data.
    - i: int, process split i/N.
    - N: int, total number of splits.
    - ALOS_flag: boolean, whether to extract ALOS data.

    Returns:
    - None
    """

    # Load the GEDI data using the geometry of the tile, and reproject it to the tile's CRS
    print(f'Loading GEDI data for tile {tile_name}.')
    GEDI, crs_1 = load_GEDI_data(path_gedi, tile_geom, tile_name)
    if GEDI.empty :
        print(f'No GEDI data for tile {tile_name}.')
        print(f'Done for tile {tile_name}!')
        return

    # Group the footprints by their corresponding Sentinel-2 product
    print(f'Grouping GEDI data by Sentinel-2 product.')
    groups = group_GEDI_by_S2(GEDI, tile_name, path_s2)
    if list(groups) == [] :
        print(f'No S2 match for tile {tile_name}.')
        print(f'Done for tile {tile_name}!')
        return

    # Load the ALOS PALSAR mosaics for the tile, and years spanned by the products
    if ALOS_flag: alos_raw = load_ALOS_data(tile_name, groups, path_alos)
    
    # Load the CH data for this year and tile
    if CH_flag: ch_raw = load_CH_data(path_ch, tile_name, year)

    # Load the LC data for this year and tile
    if LC_flag: lc_raw = load_LC_data(path_lc, tile_name, year)

    # Load the DEM data for this tile
    if DEM_flag: dem_raw = load_DEM_data(path_dem, tile_name)

    # Name the output file
    if output_fname == '' : fname = f'data_{i}-{N}.h5'
    else: fname = f'data_{output_fname}_{i}-{N}.h5'
    
    # Open the output file
    with h5py.File(join(output_path, fname), 'a') as file :

        # Initialize the h5py group for the current tile
        print(f'Initializing output file for tile {tile_name}.')
        init_h5_group(file, tile_name, patch_size, chunk_size, S1_flag, ALOS_flag, CH_flag, LC_flag, DEM_flag)

        # Iterate over the footprints with the same Sentinel-2 product
        print(f'Extracting patches for tile {tile_name}.')
        for s2_prod, footprints in groups :

            print(f'>> Extracting patches for product {s2_prod}.')

            # Unzip the S2 L2A product if it hasn't been done
            if not exists(join(path_s2, s2_prod + '.SAFE')) :
                try:
                    with ZipFile(join(path_s2, s2_prod + '.zip'), 'r') as zip_ref:
                        zip_ref.extractall('/')
                except Exception as e:
                    print(f'>> Could not unzip {s2_prod}.')
                    continue

            # Reproject and upsample the S2 bands            
            try: 
                transform, upsampling_shape, processed_bands, crs_2, bounds, boa_offset = process_S2_tile(s2_prod, path_s2)
            except Exception as e:
                print(f'>> Could not process product {s2_prod}.')
                print(e)
                continue
            
            assert crs_1 == crs_2 == footprints.crs, "CRS mismatch."

            # Get the corresponding S1 L1C product
            # TODO get an ASC one and a DESC one
            if S1_flag: s1_prod = get_s1_product(s2_prod, path_s1) # TODO

            # Process the ALOS tile corresponding to the S2 product
            if ALOS_flag: alos_tile = get_tile(alos_raw[year], transform, upsampling_shape, 'ALOS', ALOS_attrs)

            # Process the CH tile corresponding to the S2 product
            if CH_flag: ch_tile = get_tile(ch_raw, transform, upsampling_shape, 'CH', CH_attrs)

            # Process the LC tile corresponding to the S2 product
            if LC_flag: lc_tile = get_tile(lc_raw, transform, upsampling_shape, 'LC', LC_attrs)

            # Process the DEM tile corresponding to the S2 product
            if DEM_flag: dem_tile = get_tile(dem_raw, transform, upsampling_shape, 'DEM', DEM_attrs)

            # Initialize results placeholder
            s2_data, s1_data, gedi_data, alos_data, ch_data, lc_data, dem_data = initialize_results(S1_flag, ALOS_flag, CH_flag, LC_flag, DEM_flag)

            # Further crop the data to the product's bounds
            footprints = footprints[footprints.intersects(box(*bounds))]

            # Iterate over the footprints
            footprint_idx = 0
            debug_footprints = {'idx': [], 'geometry':[]}
            for footprint in footprints.itertuples() :

                if footprint_idx == 5: 
                    # save the footprints
                    gpd.GeoDataFrame(debug_footprints, crs = crs_2).to_file(f'/scratch2/gsialelli/patches/debug/footprints_{tile_name}.geojson', driver = 'GeoJSON', index = False)
                    exit()

                # Extract the Sentinel-2 data
                s2_footprint_data = get_sentinel2_patch(transform, processed_bands, footprint, patch_size, s2_prod, boa_offset)
                if s2_footprint_data is None: continue

                print(footprint_idx)

                debug_footprints['idx'].append(footprint_idx)
                debug_footprints['geometry'].append(footprint.geometry)


                # TODO debugging, remove
                s2_meta = {'driver': 'GTiff', 'height': patch_size[0], 'width': patch_size[1], 'transform': s2_footprint_data['patch_transform'], 'crs': crs_2, 'count' : 3, 'dtype' : np.uint16}
                with rs.open(f'/scratch2/gsialelli/patches/debug/s2_{tile_name}_{footprint_idx}.tif', 'w', **s2_meta) as dst:
                    dst.write(s2_footprint_data['bands'][0], 1)
                    dst.write(s2_footprint_data['bands'][1], 2)
                    dst.write(s2_footprint_data['bands'][2], 3)

                # Extract the Sentinel-1 data
                if S1_flag: s1_footprint_data = get_sentinel1_patch(s1_prod, footprint) # TODO
                else: s1_footprint_data = None

                # Extract the ALOS data
                if ALOS_flag: 
                    alos_footprint_data = get_patch(alos_tile, footprint, transform, patch_size, 'ALOS', ALOS_attrs)
                    # TODO debugging, remove
                    meta = {'driver': 'GTiff', 'height': patch_size[0], 'width': patch_size[1], 'transform': s2_footprint_data['patch_transform'], 'crs': crs_2, 'count' : 1, 'dtype' : np.uint16}
                    with rs.open(f'/scratch2/gsialelli/patches/debug/alos_{tile_name}_{footprint_idx}.tif', 'w', **meta) as dst:
                        dst.write(alos_footprint_data['bands'][0], 1)
                else: alos_footprint_data = None

                # Extract the CH data
                if CH_flag: 
                    ch_footprint_data = get_patch(ch_tile, footprint, transform, patch_size, 'CH', CH_attrs)
                    # TODO debugging, remove
                    meta = {'driver': 'GTiff', 'height': patch_size[0], 'width': patch_size[1], 'transform': s2_footprint_data['patch_transform'], 'crs': crs_2, 'count' : 1, 'dtype' : np.uint16}
                    with rs.open(f'/scratch2/gsialelli/patches/debug/ch_{tile_name}_{footprint_idx}.tif', 'w', **meta) as dst:
                        dst.write(ch_footprint_data['ch'], 1)
                else: ch_footprint_data = None

                # Extract the LC data
                if LC_flag: 
                    lc_footprint_data = get_patch(lc_tile, footprint, transform, patch_size, 'LC', LC_attrs)
                    # TODO debugging, remove
                    meta = {'driver': 'GTiff', 'height': patch_size[0], 'width': patch_size[1], 'transform': s2_footprint_data['patch_transform'], 'crs': crs_2, 'count' : 1, 'dtype' : np.uint8}
                    with rs.open(f'/scratch2/gsialelli/patches/debug/lc_{tile_name}_{footprint_idx}.tif', 'w', **meta) as dst:
                        dst.write(lc_footprint_data['bands'][0], 1)
                else: lc_footprint_data = None

                # Extract the DEM data
                if DEM_flag: 
                    dem_footprint_data = get_patch(dem_tile, footprint, transform, patch_size, 'DEM', DEM_attrs)
                    # TODO debugging, remove
                    meta = {'driver': 'GTiff', 'height': patch_size[0], 'width': patch_size[1], 'transform': s2_footprint_data['patch_transform'], 'crs': crs_2, 'count' : 1, 'dtype' : np.int16}
                    with rs.open(f'/scratch2/gsialelli/patches/debug/dem_{tile_name}_{footprint_idx}.tif', 'w', **meta) as dst:
                        dst.write(dem_footprint_data['dem'], 1)
                else: dem_footprint_data = None

                # Extract the GEDI data
                gedi_footprint_data = get_gedi_data(footprint)

                # Aggregate the results
                # TODO remove
                del s2_footprint_data['patch_transform']
                footprint_idx += 1
                s2_data, s1_data, gedi_data, alos_data, ch_data, lc_data, dem_data = update_results(s2_data, s1_data, gedi_data, alos_data, ch_data, lc_data, dem_data, s2_footprint_data, s1_footprint_data, gedi_footprint_data, alos_footprint_data, ch_footprint_data, lc_footprint_data, dem_footprint_data)

                # Write the results to file and reset the placeholders
                num_patches = len(gedi_data['agbd'])
                if (num_patches % chunk_size) == 0 :
                    save_results(s2_data, s1_data, gedi_data, alos_data, ch_data, lc_data, dem_data, tile_name, chunk_size, file)
                    s2_data, s1_data, gedi_data, alos_data, ch_data, lc_data, dem_data = initialize_results(S1_flag, ALOS_flag, CH_flag, LC_flag, DEM_flag)

            # Remove the unzipped S2 product
            rmtree(join(path_s2, s2_prod + '.SAFE'))
        
    print(f'Done for tile {tile_name}!')

############################################################################################################################
# Execute

import time

if __name__ == "__main__":

    # Parse the command line arguments
    tilenames, year, patch_size, chunk_size, path_shp, path_gedi, path_s2, path_s1, path_alos, path_ch, path_lc, path_dem, output_path, output_fname, i, N, S1_flag, ALOS_flag, CH_flag, LC_flag, DEM_flag = setup_parser()

    # Read the Sentinel-2 grid shapefile
    grid_df = gpd.read_file(path_shp, engine = 'pyogrio')

    # List all S2 tiles and their geometries
    tile_names, tile_geoms = list_s2_tiles(tilenames, grid_df, path_s2)

    # Split into N, and process the i-th split
    tile_names = np.array_split(tile_names, N)[i]
    tile_geoms = np.array_split(tile_geoms, N)[i]
    assert len(tile_names) == len(tile_geoms)

    setup_output_files(output_path, output_fname, i, N)

    start_time = time.time()

    for tile_name, tile_geom in zip(tile_names, tile_geoms) :
        extract_patches(tile_name, year, tile_geom, patch_size, chunk_size, path_gedi, path_s2, path_s1, path_alos, path_ch, path_lc, path_dem, output_path, output_fname, i, N, S1_flag, ALOS_flag, CH_flag, LC_flag, DEM_flag)
        # try: 
        #     extract_patches(tile_name, year, tile_geom, patch_size, chunk_size, path_gedi, path_s2, path_s1, path_alos, path_ch, path_lc, path_dem, output_path, output_fname, i, N, S1_flag, ALOS_flag, CH_flag, LC_flag, DEM_flag)
        # except Exception as e: 
        #     print(f"Couldn't extract patches for tile {tile_name}.", e)
        #     continue

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Elapsed time: {elapsed_time} seconds.')


# TODO make argument passing better, like with the * thingy

"""

python create_patches.py --tilenames /scratch2/gsialelli/BiomassDatasetCreation/patches/test.txt --chunk_size 2 --i 0 --N 5

"""