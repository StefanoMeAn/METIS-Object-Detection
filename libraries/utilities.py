import os
import numpy as np
import astropy.wcs
import math
import matplotlib.pyplot as plt
from scipy.stats import scoreatpercentile
import urllib.request
from astropy.io import fits
import pandas as pd




    ### FITS RELATED LIBRARIES ###

def extract_timestamp(fits_data):
    """
    Extract timestamp from a .fits object.

    Parameters:
        fits_data (astropy object): fits loaded.
    Output:
        timestamp (float): time where image was taken in UTC.
    """
    # Extrat time.
    obt_beg = fits_data[0].header['OBT_BEG'] 
    obt_end = fits_data[0].header['OBT_END']
    obt_avg = (obt_beg + obt_end) / 2
    frac, whole = math.modf(obt_avg)
    frac *= 65536.
    return str(int(whole))+':'+str(int(frac))

def fits_loader(path, keys):

    # Load .fits file
    fits_file = fits.open(path)
    # Extract timestamp 
    timestamp = extract_timestamp(fits_file)
    # Extract required headers for storing in .csv
    fits_header = [fits_file[0].header[headers] for headers in keys]
    # Extract image.
    image = fits_file[0].data
    # Close file
    fits_file.close()

    return timestamp, fits_header, image

def plot_fits(fits, waveband = None):
    """Plot a .fits file with its corresponding tags"""
    fig ,axis = plt.subplots(1, 1, figsize= (5,5))
    low, high = scoreatpercentile(fits[0].data, per=(10, 99), limit=(-np.inf ,np.inf))
    axis.imshow(fits[0].data, cmap = "gray", vmin = low, vmax = high, origin = "lower")
    axis.set_xlabel("x detector"), axis.set_ylabel("y detector")
    if fits[0].header["LEVEL"] == "L2":
        axis.set_title(f'{fits[0].header["LEVEL"]} {fits[0].header["WAVEBAND"]} \n {fits[0].header["DATE-OBS"]}')

        return fig, axis
    else:
        axis.set_title(f'{fits[0].header["LEVEL"]} {waveband} \n {fits[0].header["FILE_RAW"][:20]}')
        return fig, axis
    
def download_fits(url, filename, folder, timeout = 5):
    """Download a fits file and save it in a specific path.

    Parameters:
        url (str): url to retrieve file.
        filename (str): name of the file to be saved.
        folder (str): directory where file will be stored.    
    """

    with urllib.request.urlopen(url, timeout=timeout) as response, open(os.path.join(folder, filename), 'wb') as out_file:
            out_file.write(response.read())

def extract_headers(df, headers_new):

    """
    Save headers in each fits into a dataframe and later save it into pkl.

    Parameters: 
        df (str): name of the original pandas df.
        headers_new (astropy headers): headers from fits file.
    """

    # Load pkl file.
    dataframe = pd.read_pickle(df)
    # Concat dfs.
    dataframe = pd.concat([dataframe, pd.DataFrame([headers_new])], ignore_index= True)
    # Save dataframes.
    dataframe.to_pickle(df)


### EXTRAS ###
def sorter(list):
    """
    Sort a list with elements of the type ABCNN, where A, B, C are letters and N is a number.

    Args:
        list (list): List with LTPs or STPs.
    Returns:
        Sorted list.
    """
    return sorted(list, key=lambda x: int(x[3:]))

def get_iterable(x):
    """
    Check if a variable is a list. If it is not, covert it into a list.
    """
    if isinstance(x, list):
        return x
    else:
        return (x,)
