import rasterio
import matplotlib.pyplot as plt
from pyproj import Transformer, CRS

transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326")

# dataset = rasterio.open('data/boreal_agb_202302061675660693_2656.tiff')
dataset = rasterio.open('reprojected data/ICESat2_52UEU.tif')

def all_info(dataset):
    print("======================================")
    #print("edges: Left: " + str(dataset.bounds[0]) + ", Right: " + str(dataset.bounds[2]) + ", Top: " + str(dataset.bounds[3]) + ", Bottom: " + str(dataset.bounds[1]))
    print(dataset.bounds)

    # Define the source CRS using the PROJ string
    src_crs = CRS(dataset.crs.wkt)
    # Define the target CRS as WGS84
    tgt_crs = CRS('EPSG:4326')
    # Create a transformer
    transformer = Transformer.from_crs(src_crs, tgt_crs)
    # Convert the bounding box coordinates
    left, bottom = transformer.transform(dataset.bounds[0], dataset.bounds[1])
    right, top = transformer.transform(dataset.bounds[2], dataset.bounds[3])

    print(left, bottom, right, top)
    print("number of bands: " + str(dataset.count))
    print("band names: " + str(dataset.descriptions))
    print("mapping band index to variable data type: " + str({i: dtype for i, dtype in zip(dataset.indexes, dataset.dtypes)}))
    print("height of dataset: " + str(dataset.height) + ", width of dataset: " + str(dataset.width))
    print("coordinate reference system crs: " + str(dataset.crs))
    print("transform: " + str(dataset.transform))
    print("driver: " + str(dataset.driver))
    print("nodata: " + str(dataset.nodata))
    print("closed: " + str(dataset.closed))
    print("compression: " + str(dataset.compression))
    print("block sizes: " + str(dataset.block_shapes))
    print("colorinterp: " + str(dataset.colorinterp))
    print("descriptions: " + str(dataset.descriptions))
    print("dtypes: " + str(dataset.dtypes))
    print("files: " + str(dataset.files))
    print("gcps: " + str(dataset.gcps))
    print("indexes: " + str(dataset.indexes))
    print("interleaving: " + str(dataset.interleaving))
    print("mask_flags: " + str(dataset.mask_flag_enums))
    print("meta: " + str(dataset.meta))
    print("======================================")
    
def plot_difference(band1, band2):
    # Calculate the difference between band1 and band2
    band_diff = band1 - band2
    # Plot the difference in color
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(band_diff, cmap='viridis')
    ax.set_title('Difference between Band 1 and Band 2')
    # Display the plot
    plt.show()

def plot_bands(band1, band2, plot_multiple=False):
    # Plot both bands in color
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(band1, cmap='viridis')
    ax1.set_title('aboveground_biomass_density (Mg ha-1)')
    ax2.imshow(band2, cmap='viridis')
    ax2.set_title('standard_deviation')
    if not plot_multiple:
        plt.show()

def plot_multiple(files):
    for dataset in files:
        print(all_info(dataset))
        band1 = dataset.read(1)
        band2 = dataset.read(2)
        plot_bands(band1, band2, plot_multiple=True)
    plt.show()

# all_info(dataset)
#plot all .tif files in the folder reprojected data
import os
import glob
files = glob.glob('reprojected data/*.tif')
for file in files:
    dataset = rasterio.open(file)
    plot_multiple([dataset])

plot_multiple([dataset])
    
# Read the bands
# band1 = dataset.read(1)
# band2 = dataset.read(2)

# plot_bands(band1, band2)
# plot_difference(band1, band2)
