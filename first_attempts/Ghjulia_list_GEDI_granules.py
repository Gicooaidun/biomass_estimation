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
from shapely.ops import orient
from os.path import join
from GEDI_settings import START_MISSION
pd.options.mode.chained_assignment = None  # default='warn'

############################################################################################################################
# Helper functions

def setup_args_parser() :
    """ 
    Setup the arguments parser for the program.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_output', type = str, required = True,
                        help = 'Directory in which to write the `GEDI_L4A_<AOI>_<d1>_<d2>.txt` file.')
    parser.add_argument('--path_geojson', type = str, required = True,
                        help = 'Path to the .geojson file to consider for the geometries.',
                        default = join('/scratch2', 'gsialelli', 'BiomassDatasetCreation', 'Data', 'countrySelection', 'AOIs.geojson'))     
    parser.add_argument('--AOI', type = str, nargs = '*', default = 'global',
                        help = 'The AOI(s) for which to list the available granules')
    parser.add_argument('--d1', type = dt.date.fromisoformat, required = False, default = None,
                        help = '(Optional) Date from which to start the search, format yyyy-mm-dd.')
    parser.add_argument('--d2', type = dt.date.fromisoformat, required = False, default = None,
                        help = '(Optional) Date until which to search, format yyyy-mm-dd.')
    args = parser.parse_args()

    # Set the dates if not provided, to cover the whole period of the GEDI L4A dataset
    d1, d2 = args.d1, args.d2
    if d1 == None : d1 = dt.datetime.strptime(START_MISSION, '%Y-%m-%d')
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

    GLOBAL = True

    if country is not None:
        GLOBAL = False
        country.geometry = country.geometry.apply(orient, args=(1,))
        geojson = {"shapefile": ("country.geojson", country.geometry.to_json(), "application/geo+json")}

    # Setting up the parsing
    doi = '10.3334/ORNLDAAC/2056' # GEDI L4A DOI 
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

        if GLOBAL: response.post(granulesearch, data=cmr_param)
        else: response = requests.post(granulesearch, data=cmr_param, files=geojson)
        
        granules = response.json()['feed']['entry']
        
        if granules:
            for g in granules:
                granule_url = ''
                granule_poly = ''
        
                # reading bounding geometries
                if 'polygons' in g:
                    polygons= g['polygons']
                    multipolygons = []
                    for poly in polygons:
                        i=iter(poly[0].split (" "))
                        ltln = list(map(" ".join,zip(i,i)))
                        multipolygons.append(Polygon([[float(p.split(" ")[1]), float(p.split(" ")[0])] for p in ltln]))
                    granule_poly = MultiPolygon(multipolygons)
                
                # Get URL to HDF5 files
                for links in g['links']:
                    if 'title' in links and links['title'].startswith('Download') \
                    and links['title'].endswith('.h5'):
                        granule_url = links['href']
                        # TODO check if we want the link or just the filename
                granule_arr.append([granule_url, granule_poly])
                
            page_num += 1
        else: 
            break

    # creating a pandas dataframe
    l4adf = pd.DataFrame(granule_arr, columns = ["granule_url", "granule_poly"])

    # Drop granules with empty geometry
    l4adf = l4adf[l4adf['granule_poly'] != '']

    # Return names of granules
    return l4adf['granule_url'].tolist()



def define_filename(AOI, d1, d2) :
    """
    Given AOI(s) and start and end dates, defines the name of the file containing the list of available granules.
    
    Args:
    - AOI (str, or str list) : the AOI(s) for which to list the available granules
    - d1 (datetime.date) : date from which to start the search
    - d2 (datetime.date) : date until which to search

    Returns:
    - filename (str) : the name of the file containing the list of available granules
    """
    
    dates = f"{d1.strftime('%Y-%m-%d')}_{d2.strftime('%Y-%m-%d')}"
    if AOI != 'global': AOI = '_'.join(AOI)
    
    return f"GEDI_L4A_{AOI}_{dates}.txt"


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
    fname = define_filename(AOI, d1, d2)
    with open(join(path_output, fname), 'w') as f:
        # we append an extra \n character, otherwise using `wc -l` will not count the last line
	    f.write('\n'.join(all_granules) + '\n')