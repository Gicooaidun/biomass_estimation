"""  


    Content: script that will be called by `run-inference.py` to perform inference for a given model on a given tile.
    Execution:
                    python3 inference.py --path
                                         --model_name
                                         --arch
                                         --tile_name
                                         --year
                                         --patch_size <patch_height patch_width>
                                         --overlap_size <overlap_height overlap_width>
                                         --saving_dir
                                         --dw
"""


#######################################################################################################################
# Imports


import time
from os.path import join
import os, pickle, argparse
import torch
import numpy as np
import rasterio as rs
from pyproj import Transformer
import clemence
from skimage.transform import rescale


#######################################################################################################################
# Helper functions


def setup_parser():
    """
        Main function. Returns an `ArgumentParser()` object containing the command-line arguments.
    """


    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type = str, required = True, help = "Either 'local' or the path to the data.")
    parser.add_argument('--model_name', nargs = '+', type = str, help = 'Model name(s), whitespace-separated.')
    parser.add_argument("--arch", required = True, type = str, help = 'Network architecture.')
    parser.add_argument('--saving_dir', type = str, help = 'Directory in which to save the plots.')
    parser.add_argument("--tile_name", required = True, type = str, help = 'Tile on which to run the prediction.')
    parser.add_argument("--year", required = True, type = int, help = 'Year for which to run the prediction.')
    parser.add_argument("--dw", action = 'store_true', help = 'Downsample the preds to 50m resolution.')
    parser.add_argument("--patch_size", nargs = 2, type = int, default = [200,200], help = 'Size (height,width) of the patches.')
    parser.add_argument("--overlap_size", nargs = 2, type = int, default = [100,100], help = 'Size (height,width) of the patches.')
    return parser


def encode_tile(tile_reader, transformer) :
    """
        Content: from a DatasetReader for a tile, get the latitude and longitude per pixel in EPSG:4326, and encode
        those geographical coordinates to cyclic geographical coordinates.


        input:
        - `tile_reader` (rasterio DatasetReader) : tile to encode;
        - `transformer` (pyproj.Transformer) : transformer from one CRS to another one.


        output (3 x np.ndarray) : encoded latitude, longitude_1, and longitude_2
    """


    width, height = tile_reader.width, tile_reader.height
    top, bottom, right = tile_reader.xy(0,0), tile_reader.xy(height, 0), tile_reader.xy(0, width)
    top, bottom, right = [transformer.transform(x,y) for (x,y) in [top, bottom, right]]
   
    # For longitude, we only need to calculate the first row, which we do by interpolating
    dist = np.abs(top[0] - right[0])
    incr = dist / (width - 1)
    row = np.append(np.arange(start = top[0], stop = right[0], step = incr), right[0])[:10980]
    row_1, row_2 = np.cos(np.pi * row / 180), np.sin(np.pi * row / 180)


    # For latitude, we only need to calculate the first column, which we do by interpolating
    dist = np.abs(top[1] - bottom[1])
    incr = dist / (height - 1)
    column = np.append(np.arange(start = top[1], stop = bottom[1], step = incr), bottom[1])[:10980]
    column = np.sin(np.pi * column / 180)


    # Now we duplicate the relevant row and column to have the desired shape
    lat, lon_1, lon_2 = np.zeros((height, width)), np.zeros((height, width)), np.zeros((height, width))
    for i in range(width): lat[:, i] = column
    for i in range(height) :
        lon_1[i, :] = row_1
        lon_2[i, :] = row_2
   
    return lat, lon_1, lon_2


def load_ch_tile(PATH_TO_TILE, tile_name):


    # load canopy height tile
    ch = rs.open(join(PATH_TO_TILE, f'{tile_name}_pred.tif'))
    nodataval = ch.nodata
   
    transformer = Transformer.from_crs(ch.crs, 'EPSG:4326')
    lat, lon_1, lon_2 = encode_tile(ch, transformer)
   
    ch = ch.read(1).astype(np.float32)


    # load standard deviation tile
    ch_std = rs.open(join(PATH_TO_TILE, f'{tile_name}_std.tif')).read(1).astype(np.float32)


    # get mask
    mask = (ch == nodataval)


    return ch[:10980, :10980], ch_std[:10980, :10980], lat[:10980, :10980], lon_1[:10980, :10980], lon_2[:10980, :10980], mask[:10980, :10980]




def load_input(paths, tile_name, norm_values, cfg):
   
    """
        Reads the input tile specified in tile_name, as well as the corresponding encoded geographical coordinates,
        and normalize the input. Sets lat, lon_1, lon_2, img.
    """
   
    start_time = time.time()
    print('Loading input...')
    ch, ch_std, lat, lon_1, lon_2, mask = load_ch_tile(paths['tiles'], tile_name)
    ch = (ch - norm_values['mean_ch']) / norm_values['sigma_ch']
    ch_std = (ch_std - norm_values['mean_ch_std']) / norm_values['sigma_ch_std']
    ch, ch_std, lat, lon_1, lon_2 = [np.expand_dims(x, axis = 0) for x in [ch, ch_std, lat, lon_1, lon_2]]
    if cfg['latlon']: img = np.concatenate([ch, ch_std, lat, lon_1, lon_2], axis = 0)
    else: img = np.concatenate([ch, ch_std, lat], axis = 0)
    img = np.swapaxes(np.swapaxes(img, 0, 1), 1, 2)
    img = torch.from_numpy(img).to(torch.float32)
    print('done!')
    end_time = time.time()
    print(f'Loading input took {end_time - start_time} seconds.')


    return img, mask




def predict_patch(model, patch):


    """
        Predict patch for AGBD.


        output: (np.ndarray) where the first dimension is the predicted mean, and the second is the variance
    """


    patch = torch.unsqueeze(torch.permute(patch, [2,0,1]), 0).to(0)
    preds = model.model(patch).cpu().detach().numpy()
    return preds[0, 0 : 2, :, :]




def predict_tile(img, size, model_names, models, patch_size, overlap_size):


    """
        Split 100km x 100km tile, ie. of shape (10 000, 10 000, num_features), into patches of size `patch_size`, with overlap by `overlap_size`.


        - `patch_size` - (int, int) : Size (width, height) of the patches to extract.
        - `overlap_size` - (int, int) : Size (width, height) of the desired overlap between two patches.
   
        Best practices:
        . choose patch_size such that patch_size / 5 is an integer
        . choose overlap_size such that overlap_size / 2 is an integer


    """


    # Define variables for the splitting of the Sentinel-2 tile into patches ######################################
   
    # Width and height of the input Sentinel-2 tile
    img_height, img_width, _ = img.shape
    # Width and height of the desired patches
    patch_height, patch_width = patch_size
    # Width and height of the desired overlap between two patches
    overlap_height, overlap_width = overlap_size
    # Step in the width/height dimension: width/height of the patch minus width/height of the overlap
    step_height, step_width = patch_height - overlap_height, patch_width - overlap_width
    # Find the number of times the patch will fit entirely in the image
    n_height, n_width = (img_height - overlap_height) / (patch_height - overlap_height), (img_width - overlap_width) / (patch_width - overlap_width)
    overload_height, overload_width = True, True
    if (n_height % 1) == 0 : overload_height = False
    else: n_height = np.ceil(n_height)
    if (n_width % 1) == 0 : overload_width = False
    else: n_width = np.ceil(n_width)


    # Define variables for the predictions mosaicing ##############################################################
   
    # Downsampling factor, to predict at a 50m resolution per pixel if enabled
    dw_factor = 15 // size
    # Width and height of the prediction patch: the (downsampled) width/height of the patch
    pred_patch_width, pred_patch_height  = int(np.ceil(patch_width  / dw_factor)), int(np.ceil(patch_height / dw_factor))
    # Width and height of the prediction patch overlap: the (downsampled) width/height of the overlap
    pred_overlap_width, pred_overlap_height  = int(np.ceil(overlap_width  / dw_factor)), int(np.ceil(overlap_height / dw_factor))
    # Width and height of the mosaiced predictions: the (downsampled) width/height of the Sentinel-2 tile
    pred_width, pred_height = img_width // dw_factor, img_height // dw_factor
    # Step in the width/height dimension: width/height of the prediction patch minus width/height of the prediction overlap
    pred_step_width, pred_step_height = pred_patch_width - pred_overlap_width, pred_patch_height - pred_overlap_height
   
   
    # Place-holder for the put-together predictions
    predictions = {}
    for model_name in model_names :
        predictions[model_name] = np.zeros((2, pred_height, pred_width))


    # Actual prediction ###########################################################################################


    print('Actual tile prediction...')
    start_time = time.time()
    # Iterate over the patches and predict the AGBD
    for i, i_p in zip(range(0, img_height - patch_height + 1, step_height), range(0, pred_height - pred_patch_height + 1, pred_step_height)) :
        off_h = 0 if i_p == 0 else overlap_height // (2 * dw_factor) # to limit border-effect
        for j, j_p in zip(range(0, img_width - patch_width + 1, step_width), range(0, pred_width - pred_patch_width + 1, pred_step_width)) :
            off_w = 0 if j_p == 0 else overlap_width // (2 * dw_factor) # to limit border-effect
            patch = img[i : i + patch_height, j : j + patch_width, :]
            for model_name, model in zip(model_names, models) :
                predictions[model_name][:, i_p + off_h : i_p + pred_patch_height, j_p + off_w : j_p + pred_patch_width] = predict_patch(model, patch)[:, off_h : , off_w :]
        # Last column, if patches don't equally fit in the image
        if overload_width :
            patch = img[i : i + patch_height, - patch_width : , :]
            for model_name, model in zip(model_names, models) :
                predictions[model_name][:, i_p + off_h : i_p + pred_patch_height, - pred_patch_width + off_w : ] = predict_patch(model, patch)[:, off_h : , off_w :]
    # Last row, if patches don't equally fit in the image
    if overload_height :
        for j, j_p in zip(range(0, img_width - patch_width + 1, step_width), range(0, pred_width - pred_patch_width + 1, pred_step_width)) :
            patch = img[ - patch_height : , j : j + patch_width, :]
            for model_name, model in zip(model_names, models) :
                predictions[model_name][:, - pred_patch_height + off_h : , j_p + off_w : j_p + pred_patch_width] = predict_patch(model, patch)[:, off_h : , off_w :]
   
    # Divide by the number of times a value was in an overlap to get the mean
    print('done!')
    end_time = time.time()
    print(f'Actual tile prediction took {end_time - start_time} seconds.')
    return predictions




#######################################################################################################################
# Inference class definition


class Inference:


    """
        An `Inference` object loads a PyTorch model and performs AGBD inference at the Sentinel-2 tile level.
    """


    def __init__(self, arch, model_name, paths, tile_name, cfg):


        """
            Initialization method.


            input:
            - `arch` (str) : the architecture of the model to load
            - `model_name` (str) : the name (<JOB_ID>-<model_idx> format) of the model to load
            - `paths` (dict) : dictionary with keys `norm`, `tiles`, and `ckpt` and with values
                the paths to the corresponding file/folder
            - `tile_name` (str) : the name of the Sentinel-2 tile to perform inference on
            - `cfg` (wandb.config) : dict with training configuration of the model to load
        """


        self.arch = arch
        self.model_name = model_name
        self.paths = paths
        self.tile_name = tile_name
        self.cfg = cfg        
        self.load_model()
   
    def load_model(self):  # sourcery skip: raise-specific-error


        """
            Loads the model, setting self.model
        """


        self.model = clemence.Net('fcn_6_gaussian', num_outputs = 2, in_features = 3, downsample = "average" if self.cfg['downsample'] else None)
        ckpt = torch.load(join(self.paths['ckpt'], self.model_name))
        self.model.load_state_dict(ckpt['model_state_dict'], strict=True)
        self.model.to(0)
        self.model.eval()




#######################################################################################################################
# Code execution


def run_inference():
   
    # Get the command line arguments and set the global variables
    args, _ = setup_parser().parse_known_args()


    # Define paths
    local_dataset_paths = {'norm' : '',
                           'tiles': join('..', 'Nico', 'global-canopy-height-model', f'{args.tile_name}', str(args.year), 'preds_inv_var_mean'),
                           'ckpt' : 'pretrained_models'}
    if args.path == 'local' :
        paths = local_dataset_paths
    else:
        paths = {k: args.path for k in local_dataset_paths}
    paths['saving_dir'] = join(args.saving_dir, str(args.year))




    # Here you can modify the configuation, e.g. playing with `downsample`
    cfg = {'latlon': False, 'bands': [], 'aux_vars': [], 'downsample': args.dw, 'augment': False, 'reweighting': 'no', 'ch': True}


    # Load the input
    with open(os.path.join(paths['norm'], 'normalization_values.pkl'), mode = 'rb') as f: norm_values = pickle.load(f)
    img, mask = load_input(paths, args.tile_name, norm_values, cfg)
    size = 3 if cfg['downsample'] else 15
    pred_mask = rescale(mask, size / 15)


    # Load the models
    inference_objects = [Inference(arch = args.arch, model_name = model_name, paths = paths, tile_name = args.tile_name, cfg = cfg) for model_name in args.model_name]
    models = [inference_object.model for inference_object in inference_objects]


    # Get the predictions
    ensemble_predictions = predict_tile(img, size, args.model_name, models, args.patch_size, args.overlap_size)


    # Ensemble
    preds_variables, preds_variances = [], []
    for model_name in args.model_name:


        print(f'Ensembling predictions for {model_name}...')
       
        # Get the predictions for this model
        predictions = ensemble_predictions[model_name]


        # We ignore the predictions that correspond to where the input was masked
        predictions[:, pred_mask] = np.nan


        # De-normalize the data
        preds = predictions[0, :, :] * norm_values['sigma_agbd'] + norm_values['mean_agbd']
        preds_variables.append(preds)


        # Get the variance from the log-variance (we add a small constant for numerical stability)
        variances = np.exp(predictions[1, :, :]) # + 1e-6
        # De-normalize the variance by multiplying by the target variance
        variances = variances * norm_values['sigma_agbd'] ** 2


        preds_variances.append(variances)


    # Aggregate the predictions
    preds_variables = np.array(preds_variables, dtype = np.float32) # (n_models, n, m)
    preds_variances = np.array(preds_variances, dtype = np.float32) # (n_models, n, m)


    # Extract uncertainties
    avg_preds_variables = np.nanmean(preds_variables, axis = 0) # (n, m)
    aleatoric_unc = np.nanmean(preds_variances, axis = 0) # (n, m)
    epistemic_unc = np.nanvar(preds_variables, axis = 0) # (n, m)
    uncertainty = np.sqrt(aleatoric_unc + epistemic_unc) # (n, m)


    # Cast negative AGB values to 0, and all values to uint16
    avg_preds_variables[avg_preds_variables < 0] = 0
    avg_preds_variables[avg_preds_variables > 65535] = 65535
    avg_preds_variables[np.isinf(avg_preds_variables)] = 65535
    avg_preds_variables[np.isnan(avg_preds_variables)] = 65535
    avg_preds_variables = avg_preds_variables.astype(np.uint16)


    # Get the metadata from the original tile
    print(f'Saving predictions to {os.path.join(paths["saving_dir"], f"{args.tile_name}_<agb/std>.tif")}')
    with rs.open(os.path.join(paths['tiles'], f'{args.tile_name}_pred.tif'), 'r') as f: meta = f.meta
   
    # Save the AGB predictions to a GeoTIFF, with dtype uint16
    meta.update(driver = 'GTiff', dtype = np.uint16, count = 1, compress = 'lzw', nodata = 65535)
    if not os.path.exists(paths['saving_dir']): os.makedirs(paths['saving_dir'])
    with rs.open(os.path.join(paths['saving_dir'], f'{args.tile_name}_agb.tif'), 'w', **meta) as f:
        f.write(avg_preds_variables, 1)


    # Save the uncertainties estimation to another GeoTIFF, with dtype float32
    meta.update(dtype = np.float32, nodata = np.nan)
    with rs.open(os.path.join(paths['saving_dir'], f'{args.tile_name}_std.tif'), 'w', **meta) as f:
        f.write(uncertainty, 1)




if __name__ == '__main__':
    run_inference()
    print('Inference done!')
