""" 

Lists available GEDI L4A granules. 

Execution:

    python3 list_GEDI_granules.py   --path_output {path_output} 
                                    --AOI {AOI} (optional)
                                    --path_geojson {path_geojson} (optional)
                                    --d1 {d1} (optional)
                                    --d2 {d2} (optional)


Content: this program lists the available GEDI L4A granules, as follows:
    if --AOI is provided, the program will list the available granules for the given AOI
    if --AOI is not provided, the program will list the available granules globally (and AOI is set to 'global')
    if --d1 and --d2 are provided, the program will list the available granules for the given time period
    if --d1 and --d2 are not provided, the program will list the available granules for the whole GEDI time period
The listed files are written to `GEDI_L4A_<AOI>_<d1>_<d2>.txt` in the `path_output` folder.

Note: AOI should be the "name" of one of the features of the provided .geojson file.
    
"""

############################################################################################################################
# Imports

import argparse
import datetime as dt
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
import requests
import json
from shapely.ops import orient
from os.path import join
pd.options.mode.chained_assignment = None  # default='warn'

############################################################################################################################
# Helper functions

def setup_args_parser() :
    """ 
    Setup the arguments parser for the program.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_output', type = str,
                    help = 'Directory in which to write the `GEDI_L4A_<AOI>_<d1>_<d2>.txt` file.', default="/")
    parser.add_argument('--path_geojson', type = str,
                        help = 'Path to the .geojson file to consider for the geometries.', default="S2_tiles_Siberia_polybox/S2_tiles_Siberia_all.geojson")
    parser.add_argument('--AOI', type = str, nargs = '*', default = 'global',
                        help = 'The AOI(s) for which to list the available granules')
    parser.add_argument('--d1', type = dt.date.fromisoformat, required = False, default = None,
                        help = '(Optional) Date from which to start the search, format yyyy-mm-dd.')
    parser.add_argument('--d2', type = dt.date.fromisoformat, required = False, default = None,
                        help = '(Optional) Date until which to search, format yyyy-mm-dd.')
    args = parser.parse_args()

    # Set the dates if not provided, to cover the whole period of the GEDI L4A dataset
    d1, d2 = args.d1, args.d2
    if d1 == None : d1 = dt.datetime.strptime("2018-01-01", '%Y-%m-%d')
    if d2 == None : d2 = dt.date.today()

    return args.path_output, args.path_geojson, args.AOI, d1, d2


def get_granules(start_date, end_date, country) :
    """ 
    Lists the available GEDI L4A granules for the given time period and country.
    Adapted from https://github.com/ornldaac/gedi_tutorials/blob/main/1_gedi_l4a_search_download.ipynb 

    Args:
    - start_date (datetime.date) : date from which to start the search
    - end_date (datetime.date) : date until which to search
    - country (geopandas.GeoDataFrame or None) : the country for which to search the granules

    Returns:
    - list_of_granules (list) : list of the granules that are available for the given time period and country
    """

    correspondences = json.load(open('correspondences_id.json'))
    granule_list = list(set(int(value) for values in correspondences.values() for value in values))

    GLOBAL = True

    if country is not None:
        GLOBAL = False
        country.geometry = country.geometry.apply(orient, args=(1,))
        geojson = {"shapefile": ("country.geojson", country.geometry.to_json(), "application/geo+json")}

    # Setting up the parsing
    doi = '10.3334/ORNLDAAC/2186' # Icesat biomass DOI 
    cmrurl = 'https://cmr.earthdata.nasa.gov/search/' # CMR API base url
    doisearch = cmrurl + 'collections.json?doi=' + doi
    response = requests.get(doisearch)
    response.raise_for_status()
    concept_id = response.json()['feed']['entry'][0]['id']
    page_num = 1
    page_size = 2000 # CMR page size limit

    # CMR formatted start and end times
    dt_format = '%Y-%m-%dT%H:%M:%SZ'
    temporal_str = start_date.strftime(dt_format) + ',' + end_date.strftime(dt_format)
    
    # Collecting granules
    granule_arr = []
    
    while True:
        
        # defining parameters
        cmr_param = {
            "collection_concept_id": concept_id, 
            "page_size": page_size,
            "page_num": page_num,
            "temporal": temporal_str,
            "simplify-shapefile": 'true' # this is needed to bypass 5000 coordinates limit of CMR
        }
        
        granulesearch = cmrurl + 'granules.json'

        if GLOBAL: response = requests.post(granulesearch, data=cmr_param, )
        else: response = requests.post(granulesearch, data=cmr_param, files=geojson)

        # Path to your GeoJSON file
        geojson_path = 'S2_tiles_Siberia_polybox/S2_tiles_Siberia_all.geojson'

        # Load the GeoJSON file
        with open(geojson_path, 'r') as file:
            geojson_data = json.load(file)

        # Reverse the coordinates of each polygon
        for feature in geojson_data['features']:
            if feature['geometry']['type'] == 'Polygon':
                feature['geometry']['coordinates'] = [coords[::-1] for coords in feature['geometry']['coordinates']]
            elif feature['geometry']['type'] == 'MultiPolygon':
                feature['geometry']['coordinates'] = [[[coords[::-1] for coords in part] for part in polygon] for polygon in feature['geometry']['coordinates']]

        # Convert the modified GeoJSON object back to a string
        geojson_str = json.dumps(geojson_data)

        # Include the modified GeoJSON data in the request
        files = {'shapefile': ('geojson_file.geojson', geojson_str, 'application/geo+json')}

        response = requests.post(granulesearch, data=cmr_param, files=files)

        granules = response.json()['feed']['entry']
        
        if granules:
            for g in granules:
                granule_url = ''
                granule_poly = ''
                granule_id = ''
        
                # reading bounding geometries
                if 'boxes' in g:
                    lat1, lon1, lat2, lon2 = g['boxes'][0].split(' ')
                    bottom_left = (lon1, lat1)
                    bottom_right = (lon2, lat1)
                    top_left = (lon1, lat2)
                    top_right = (lon2, lat2)
                    granule_poly = Polygon([bottom_left, top_left, top_right, bottom_right, bottom_left])
                
                # Get URL to HDF5 files
                for links in g['links']:
                    if 'title' in links and links['title'].startswith('Download') \
                    and links['title'].endswith('.tif'):
                        granule_url = links['href']
                        granule_id = int(granule_url[-8:-4])
                        # TODO check if we want the link or just the filename
                if granule_id in granule_list:
                    granule_arr.append([granule_id, granule_url, granule_poly])
                
            page_num += 1
        else: 
            break

    # creating a pandas dataframe
    icesat_granules = pd.DataFrame(granule_arr, columns = ["granule_id", "granule_url", "granule_poly"])

    # Drop granules with empty geometry
    icesat_granules = icesat_granules[icesat_granules['granule_poly'] != '']

    # Return names of granules
    return icesat_granules['granule_url'].tolist()


############################################################################################################################
# Code execution

if __name__ == "__main__" :

    # Parse the arguments
    path_output, path_geojson, AOI, d1, d2 = setup_args_parser()
    all_granules = []

    # Get the granules for the whole GEDI coverage    
    if AOI == 'global':
        all_granules += get_granules(d1, d2, None)
    
    # Get the granules for the given AOIs
    else:
        for aoi in AOI:
            print(f'Extracting granules for {aoi}...') 
            countries = gpd.read_file(path_geojson)
            country = countries[countries['name'] == aoi]
            all_granules += get_granules(d1, d2, country)
    
    # Write the list of granules to a file
    fname = "granule_links_icesat.txt"
    with open(join(fname), 'w') as f:
        # we append an extra \n character, otherwise using `wc -l` will not count the last line
	    f.write('\n'.join(all_granules) + '\n')