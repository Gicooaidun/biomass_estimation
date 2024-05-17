import icepyx as ipx

import numpy as np
import xarray as xr
import pandas as pd

import h5py
import os,json
from pprint import pprint

region = ipx.Query('ATL08',[130, 46, 131, 47],['2019-02-22','2019-02-28'], \
                           start_time='00:00:00', end_time='23:59:59')
region.show_custom_options(dictview=True)
print("======================================")
print(region.order_vars.avail())