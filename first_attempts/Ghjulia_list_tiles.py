"""

This script lists all the ALOS PALSAR-2 tiles for given AOIs, or globally.

Execution:
    python list_tiles.py    --AOI California Cuba Paraguay UnitedRepublicofTanzania Ghana Austria Greece Nepal
                                ShaanxiProvince NewZealand FrenchGuiana 
                            --path_geojson {path_geojson} (optional)
                            --path_output /scratch2/gsialelli/BiomassDatasetCreation/ALOS/
"""

############################################################################################################################
# IMPORTS 

import math
import unittest
import argparse
import numpy as np
import geopandas as gpd
from os.path import join
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Rectangle

############################################################################################################################
# Helper functions

def setup_args_parser() :
    """ 
    Setup the arguments parser for the program.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_output', type = str, required = True,
                        help = 'Directory in which to write the `ALOS_<AOI>.txt` file.')
    parser.add_argument('--path_geojson', type = str,
                        help = 'Path to the .geojson file to consider for the geometries.',
                        default = join('/scratch2', 'gsialelli', 'BiomassDatasetCreation', 'Data', 'countrySelection', 'AOIs.geojson'))  
    parser.add_argument('--AOI', type = str, nargs = '*', default = 'global',
                        help = 'The AOI(s) for which to list the available granules')
    args = parser.parse_args()
    return args.path_output, args.path_geojson, args.AOI


def rounddown(x, integ = 1):
    """
    Round down to the nearest multiple of `integ`. We return positive values only.
    """
    return math.floor(abs(x) / integ) * integ


def roundup(x, integ = 1):
    """
    Round up to the nearest multiple of `integ`. We return positive values only.
    """
    return int(np.sign(x) * int(math.ceil(abs(x) / integ)) * integ)


class TestTile(unittest.TestCase) :
    """
    Test cases taken from https://www.eorc.jaxa.jp/ALOS/en/palsar_fnf/data/2019/map.htm
    Run with `unittest.main()`
    """
    def test_get_tiles_from_coordinates(self) :
        self.assertEqual(get_tiles_from_coordinates(0, 3, 170, 175), ['N00E170', 'N00E175'])
        self.assertEqual(get_tiles_from_coordinates(-4.617, 3.656, -72.145, -72.05), ['N04W073'])
        self.assertEqual(get_tiles_from_coordinates(-50, -45, -120, -110), ['S50W120', 'S50W115', 'S50W110', 'S45W120', 'S45W115', 'S45W110'])
    
    def test_round_tilename(self) :
        self.assertEqual(round_tilename('S03W072'), 'N00W075')
        self.assertEqual(round_tilename('S06W071'), 'S05W075')
        self.assertEqual(round_tilename('N54E001'), 'N55E000')
        self.assertEqual(round_tilename('N54W003'), 'N55W005')
        self.assertEqual(round_tilename('S01E024'), 'N00E020')
        self.assertEqual(round_tilename('N49E139'), 'N50E135')


def define_filename(AOI) :
    """
    For given AOI(s), this function defines the name of the file that will contain the list of available granules.
    
    Args:
    - AOI (str, or str list) : the AOI(s) for which to list the available granules

    Returns:
    - filename (str) : the name of the file containing the list of available granules
    """
    if AOI != 'global': AOI = '_'.join(AOI)
    return f"ALOS_{AOI}.txt"


############################################################################################################################
# Main functions

def list_all_ALOS_tiles() :
    """
    Write to file all of the ALOS PALSAR-2 tiles. We compute the tiles as follows: the ALOS grid is split into 5x5 degree
    tiles, each of which is further split into 1x1 degree tiles, the smallest unit. The naming convention for the unit tiles
    is <5x5 degree tile name>_<1x1 degree tile name>. We construct them iteratively. There are a few pitfalls and corner
    cases, which we elaborate on below.
    
    Note: we list the theoretically existing tiles, not the ones that are actually available for download. Which means that
    we list tiles over the ocean, or over the poles. It is up to the user of this list to filter out the tiles that are not
    available for download.

    Returns:
    - all_tiles (list of str) : the list of all the ALOS PALSAR-2 tiles
    """

    all_tiles = []

    # The longitude goes from W180 to W005 to E000 to E175
    longitudes = list(range(-180, 180, 5))
    
    # The latitude goes from N90 to N00 to S05 to S55
    latitudes = list(range(-55, 95, 5))
    
    for long in longitudes :
        lon_tiles = []
        
        # No corner cases for the longitude, e.g. tile W080 spans from W080 to W076
        # and tile E065 spans from E065 to E069
        if long < 0 : 
            lon_letter = 'W'
            for lon_1deg in list(range(long, long + 5, 1)) :
                lon_tiles.append(f'{lon_letter}{abs(lon_1deg):03n}')
        else:
            lon_letter = 'E'
            for lon_1deg in list(range(long, long + 5, 1)) :
                lon_tiles.append(f'{lon_letter}{lon_1deg:03n}')
                
        for lat in latitudes :
            lat_tiles = []
            
            # Corner-case 1 : N00 spans from N00 to S04
            if lat == 0 :
                lat_letter = 'N'
                lat_tiles += ['N00', 'S01', 'S02', 'S03', 'S04']
            # Corner-case 2 : N90 spans from N90 to N86
            elif lat > 0 : 
                lat_letter = 'N'
                for lat_1deg in range(lat - 4, lat + 1, 1) :
                    lat_tiles.append(f'{lat_letter}{lat_1deg:02n}')
            # Corner-case 3 : S55 spans from S55 to S59
            else:
                lat_letter = 'S'
                for lat_1deg in range(lat - 4, lat + 1, 1) :
                    lat_tiles.append(f'{lat_letter}{abs(lat_1deg):02n}')
            
            # Generate the 1x1 degree tile name
            suffixes = [f'{lat_tile}{lon_tile}' for lat_tile in lat_tiles for lon_tile in lon_tiles]
            assert len(suffixes) == 25

            # Generate the 5x5 degree tile name
            # prefix = f'{lat_letter}{abs(lat):02n}{lon_letter}{abs(long):03n}'
            prefixes = [round_tilename(suffix) for suffix in suffixes]
            
            # Generate the full tile name
            for prefix, suffix in zip(prefixes,suffixes) :
                all_tiles.append(f'{prefix}_{suffix}')
    
    return all_tiles


def get_tile_from_xy(lat, lon) :
    """
    For a given latitude and longitude, this function identifies the ALOS PALSAR-2 tile unit (1x1 degree) tile that
    contains the point. The naming convention for the tiles is <lat_letter><lat_number>_<lon_letter><lon_number>,
    where the letters are N or S for the latitude, and E or W for the longitude.
    
    We identify the tile by the following rules:
        .for latitude:
            if lat < 0 : N<roundup(abs(lat))>
            elif lat in [0;1] : N00
            else: S<rounddown(lat)>
        . for longitude:
            if lon < 0 : W<roundup(abs(lon))>
            else: E<rounddown(lon)>

    Args:
    - (lat, lon) (float) : the latitude and longitude of the point (in degrees

    Returns:
    - lat_letter (str) : the letter for the latitude
    - lat_number (int) : the number for the latitude
    - lon_letter (str) : the letter for the longitude
    - lon_number (int) : the number for the longitude
    """
    
    if lat > 0 :
        lat_letter = 'N'
        lat_number = roundup(lat)
    elif -1 < lat <= 0 : 
        lat_letter = 'N'
        lat_number = 0
    else:
        lat_letter = 'S'
        lat_number = rounddown(abs(lat))
    if lon < 0 :
        lon_letter = 'W'
        lon_number = roundup(abs(lon))
    else:
        lon_letter = 'E'
        lon_number = rounddown(lon)
    
    return lat_letter, lat_number, lon_letter, lon_number


def get_lat_lon_range_from_tile(lat_letter, lat_number, lon_letter, lon_number) :
    """
    For a given ALOS PALSAR-2 tile unit (1x1 degree) tile, this function returns the range of latitudes and longitudes
    that it spans. 

    Args:
    - lat_letter (str) : the letter for the latitude
    - lat_number (int) : the number for the latitude
    - lon_letter (str) : the letter for the longitude
    - lon_number (int) : the number for the longitude

    Returns:
    - latitudes (tuple of int) : the range of latitudes that the tile spans
    - longitudes (tuple of int) : the range of longitudes that the tile spans
    """

    # For the latitudes
    if lat_letter == 'S' : lat_number = -lat_number
    latitudes = (lat_number - 1, lat_number)

    # For the longitudes
    if lon_letter == 'W' : lon_number = -lon_number
    longitudes = (lon_number, lon_number + 1)

    return latitudes, longitudes


def get_tiles_from_coordinates(lat_min, lat_max, lon_min, lon_max, meridian_flag = False):
    """
    Given `lat_min, lat_max, lon_min, lon_max` this function returns all of the ALOS PALSAR-2 unit (1x1 degree)
    tiles that intersect the bounding box defined by these coordinates. The `meridian_flag` is set to True when
    the bounding box spans the 180th meridian. In this case, need to consider the tiles that span the meridian.

    Args:
    - (lat_min, lat_max, lon_min, lon_max) (float) : the bounding box coordinates
    - meridian_flag (bool) : whether the bounding box spans the 180th meridian

    Returns:
    - tiles (list of str) : the list of ALOS PALSAR-2 unit tiles that intersect the bounding box
    """

    # Identify the 1x1 degree tiles that contain the corners of the bounding box
    start_lat_letter, start_lat_number, start_lon_letter, start_lon_number = get_tile_from_xy(lat_min, lon_min)
    end_lat_letter, end_lat_number, end_lon_letter, end_lon_number = get_tile_from_xy(lat_max, lon_max)
    
    # Fill in the gap between the start and end tiles
    
    latitudes = []

    # If the latitudes are in the same hemisphere, simply get the tiles between the start and end tiles
    if start_lat_letter == end_lat_letter :

        # When in the negative range, we need to reverse the order of the tiles
        if start_lat_letter == 'S' : start_lat_number, end_lat_number = end_lat_number, start_lat_number
        for i in range(start_lat_number, end_lat_number + 1) :
            latitudes.append(f"{start_lat_letter}{i:02n}")
    
    # Otherwise, get the tiles from the start tile to the equator, and from the equator to the end tile
    else: 

        # Will be S, so from 1 to start_lat_number
        for i in range(1, start_lat_number + 1) :
            latitudes.append(f"{start_lat_letter}{i:02n}")
        # Will be N, so from 0 to end_lat_number
        for i in range(0, end_lat_number + 1) :
            latitudes.append(f"{end_lat_letter}{i:02n}")

    longitudes = []

    # If the longitudes are in the same hemisphere, simply get the tiles between the start and end tiles
    if start_lon_letter == end_lon_letter :

        # When in the negative range, we need to reverse the order of the tiles
        if start_lon_letter == 'W' : start_lon_number, end_lon_number = end_lon_number, start_lon_number
        for i in range(start_lon_number, end_lon_number + 1) :
            longitudes.append(f"{start_lon_letter}{i:03n}")
    
    # Otherwise, get the tiles from the start tile to the 0th meridian, and from the 0th meridian to the end tile
    else:
        # Corner case when the bounding box spans the 180th meridian
        if meridian_flag :
            # Go from the start tile to E179
            for i in range(start_lon_number, 180) :
                longitudes.append(f"{start_lon_letter}{i:03n}")
            # And from the end tile to W180
            for i in range(end_lon_number, 180 + 1) :
                longitudes.append(f"{end_lon_letter}{i:03n}")
        else:
            # Will be W, so from 1 to start_lon_number
            for i in range(1, start_lon_number + 1) :
                longitudes.append(f"{start_lon_letter}{i:03n}")
            # Will be E, so from 0 to end_lon_number
            for i in range(0, end_lon_number + 1) :
                longitudes.append(f"{end_lon_letter}{i:03n}")

    # Sanity check for the results
    results = [f"{lat}{lon}" for lat in latitudes for lon in longitudes]
    assert len(results) > 0, "No match found."

    return results


def round_tilename(name, integ = 5) :
    """
    Given a unit (1x1 degree) ALOS PALSAR-2 tile name, this function returns the name of the 5x5 degree tile
    that contains it. The naming convention for the unit tiles is <5x5 degree tile name>_<1x1 degree tile name>.
    We construct them iteratively. There are a few rules and corner cases, which we elaborate on below.

    Args:
    - name (str) : the name of the unit (1x1 degree) tile, in the format <lat_letter>xx<lon_letter>xxx

    Returns:
    - name (str) : the name of the 5x5 degree tile that contains the unit tile (<lat_letter>xx<lon_letter>xxx)
    """

    # Parse the unit tile name name
    lat_letter, lat_abs, lon_letter, lon_abs = name[0], int(name[1:3]), name[3], int(name[4:7])

    # Corner case: N00 spans from N00 to S04
    if lat_letter == 'S' and lat_abs <= 4 : lat_letter, lat_abs = 'N', 0
    # Round, e.g., N44 to N55 
    if lat_letter == 'N' : lat = roundup(lat_abs, integ = integ)
    # and S06 to S05
    else : lat = rounddown(lat_abs, integ = integ)
    # Round, e.g., W72 to W75
    if lon_letter == 'W' : lon = roundup(lon_abs, integ = integ)
    # and E03 to E00
    else: lon = rounddown(lon_abs, integ = integ)

    return "{}{:02n}{}{:03n}".format(lat_letter, lat, lon_letter, lon)


def get_true_bounds(geometry) :
    """
    For a given geometry, this function returns the "true" min/max lat/lon values. This is necessary when the geometry
    spans the 180th meridian, as the `bounds` method of the geometry object blindly returns the min/max lat/lon values,
    disregarding the mixture of positive and negative values. We need to separate the positive and negative longitudes,
    and define lon_min, lat_min, lon_max, lat_max = min(pos_lon), max(latitudes), max(neg_lon), min(latitudes). Note
    that the use of the `.geoms` attribute is necessary, cf. https://stackoverflow.com/a/76493457.

    Args:
    - geometry (geopandas.geoseries.GeoSeries) : the geometry for which to get the true bounds

    Returns:
    - lon_min, lat_min, lon_max, lat_max (float) : the true min/max lat/lon values    
    """

    points = []
    for polygon in geometry.values[0].geoms :
        points.extend(polygon.exterior.coords[:-1])
    longitudes, latitudes = [point[0] for point in points], [point[1] for point in points]
    pos_lon = [l for l in longitudes if l > 0]
    neg_lon = [l for l in longitudes if l <= 0]

    lat_min, lat_max = min(latitudes), max(latitudes)
    lon_min, lon_max = min(pos_lon), max(neg_lon)

    return lon_min, lat_min, lon_max, lat_max


def visual_inspection(tiles) :
    """
    This function allows for a visual inspection of the tiles on a map. Given a list of ALOS tiles (in their
    official format, i.e. <5x5 degree tile name>_<1x1 degree tile name>), this function plots the tiles on a map
    using the Basemap library.

    Args:
    - tiles (list of str) : the list of ALOS PALSAR-2 tiles to plot

    Returns:
    - None
    """

    # Setup the figure
    fig, ax = plt.subplots()
    m = Basemap(projection='merc', llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180, resolution='l')
    m.drawcoastlines()

    # Iterate over the tiles
    min_x1, max_x2, min_y1, max_y2 = np.inf, -np.inf, np.inf, -np.inf
    for tile in tiles :
        _, suffix = tile.split('_')
        lat_letter, lat_number, lon_letter, lon_number = suffix[0], int(suffix[1:3]), suffix[3], int(suffix[4:7])
        (min_lat, max_lat), (min_lon, max_lon) = get_lat_lon_range_from_tile(lat_letter, lat_number, lon_letter, lon_number)

        # Plot the tile on the map
        x1, y1 = m(min_lon, min_lat)
        x2, y2 = m(max_lon, max_lat)
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        rect = Rectangle((x1, y1), width, height, edgecolor = 'r', facecolor = 'none')
        ax.add_patch(rect)

        # Update the min/max lat/lon values
        min_x1, max_x2 = min(min_x1, x1), max(max_x2, x2)
        min_y1, max_y2 = min(min_y1, y1), max(max_y2, y2)

    # Set the aspect of the plot to be equal so that the square is not distorted
    ax.set_xlim(min_x1 - 1000000, max_x2 + 1000000)
    ax.set_ylim(min_y1 - 1000000, max_y2 + 1000000)
    ax.set_aspect('equal')
    plt.savefig('test.png')


############################################################################################################################
# Execute

if __name__ == '__main__' :

    path_output, path_geojson, AOI = setup_args_parser()
    all_tiles = []

    # Get the tiles for the whole ALOS PALSAR-2 coverage
    if AOI == 'global' :
        all_tiles = list_all_ALOS_tiles()

    else:
        for aoi in AOI:
            print(f'Extracting tiles for {aoi}...')
            countries = gpd.read_file(path_geojson)
            country = countries[countries['name'] == aoi]
            lon_min, lat_min, lon_max, lat_max = country.geometry.values[0].bounds

            # Corner case when an AOI spans the 180th meridian
            if abs(lon_min - lon_max) > 180 :
                meridian_flag = True
                lon_min, lat_min, lon_max, lat_max = get_true_bounds(country.geometry)
            else: meridian_flag = False

            tnames = get_tiles_from_coordinates(lat_min, lat_max, lon_min, lon_max, meridian_flag)
            prefixes = [round_tilename(tname) for tname in tnames]
            all_tiles.extend([prefix + '_' + tname for prefix, tname in zip(prefixes, tnames)])
    
    all_tiles = np.unique(all_tiles)

    # Write the list of tiles to a file
    fname = define_filename(AOI)
    with open(join(path_output, fname), 'w') as f:
        # We append an extra \n character, otherwise using `wc -l` will not count the last line
	    f.write('\n'.join(all_tiles) + '\n')
