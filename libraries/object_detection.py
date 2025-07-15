

import pandas as pd
import numpy as np
import math
from astropy.stats import sigma_clipped_stats
from photutils.detection import find_peaks
from libraries.utilities import get_iterable
import itertools
import libraries.starfunctions as sf
import os
import libraries.utilities as ut
import csv
import sqlite3

def save_peaks_to_sql(peaks_df, db_path):
    import pickle
    peaks_df = peaks_df.copy()
    peaks_df["REGION"] = peaks_df["REGION"].apply(lambda x: pickle.dumps(x))
    with sqlite3.connect(db_path) as conn:
        peaks_df.to_sql("psfs", conn, if_exists="append", index=False)


def init_sqlite_db(db_path):
    with sqlite3.connect(db_path) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS psfs (
                LTP TEXT,
                STP TEXT,
                IDX INTEGER,
                PEAK_VAL REAL,
                X_COORD INTEGER,
                Y_COORD INTEGER,
                PRE_LABEL TEXT,
                INFO TEXT,
                REGION BLOB,
                FILENAME TEXT
            )
        ''')



def image_slicer(image, size, overlapping_size = 3):
    """
    Slice a 2D-numpy array into several symmetrical regions that may share borders
    with its neighbors.


    Args:
        image (array): .fits image.
        size (int): size of the proposal region.
        overlapping (int): indicate length of share region (overlapping).

    Return:
        proposal_regions (list): list with proposed regions.
        proposal_coordinates (list): list with the coordinates from the original image.

    """

    # Extract size of image.
    size_x, size_y = image.shape
    # Create grid for storing positions

   
    image_coordinates = np.zeros((size_x,size_y), dtype=object)
    size_x_vals = np.arange(size_x)
    size_y_vals = np.arange(size_y)

    for idx, valx in enumerate(size_x_vals):
        for idy, valy in enumerate(size_y_vals):
            image_coordinates[idx, idy] = [valx, valy]

    # Add padding for obtaining uniform regions.
    n_regionsx = math.ceil(size_x/size)
    n_regionsy = math.ceil(size_y/size)

    padding_x = (n_regionsx*size - size_x)/2
    padding_y = (n_regionsy*size - size_y)/2

    # Add "False" padding to the borders of the image. If padding is even, then it is symmetrical.
    image = np.pad(image, [(math.ceil(padding_x), math.floor(padding_x) + overlapping_size),
                           (math.ceil(padding_y), math.floor(padding_y) + overlapping_size)],
                           constant_values=0)
    image_coordinates = np.pad(image_coordinates, [(math.ceil(padding_x), math.floor(padding_x)+ overlapping_size),
                           (math.ceil(padding_y), math.floor(padding_y) + overlapping_size)],
                           constant_values=0)
    
    size_x, size_y = image.shape

    # Create list for storing proposals and its corresponding coordinates
    proposal_regions = []
    proposal_coordinates = []

    # Crop symmetrical regions of size sizeXsize and store them into a list.
    for i in range(int(size_x/size)):
        for j in range(int(size_y/size)):
            proposal_regions.append(image[i*size:(i+1)*size + overlapping_size, j*size:(j+1)*size+overlapping_size])
            proposal_coordinates.append(image_coordinates[i*size:(i+1)*size + overlapping_size, j*size:(j+1)*size+overlapping_size])


    return proposal_regions, proposal_coordinates

def create_coordinates(size_x, size_y):
    """
    Create coordinate matrix made of pairs that represents the position of the value in the grid.

    Parameters:
        size_x (int): size in x dimension.
        size_y (int): size in y dimension.
    
    Returns:
        grid (array): 2D grid with ij coordinates as elements.

    """

    # Create array with indexes of a given image.
    X = np.arange(size_x)
    Y = np.arange(size_y)

    # Create meshgrid
    Xmsh, Ymsh = np.meshgrid(X, Y)
    
    # Merge values
    pairs = np.stack((Xmsh, Ymsh), axis=-1)
    return pairs


def statistics_region(region, sigma = 3.0):
    """
    Create a mask for 0 values in a region and compute statistics.

    Parameters:
    
        region (2d array): region to be masked and analyzed.
        sigma (float): sigma value for statistics.
    
    Return:
        mask (2d array): mask for peak detection.
        std (float): standard deviation of region.
        median (float): median of the data.
    """
    
    # Create mask for 0-valued items in the region.
    mask = region == 0
    # Compute statistics.
    mean, median, std = sigma_clipped_stats(region[~mask], sigma=3.0)

    return mask, std, median


import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

def peak_detector_main(ltp, stp, id, image, coordinates, coeff, full_image, lim, size, filename):
    """
    Detect points-of-interest in a 2d-array.

    Parameters:
        image (2d-array): image where points-of-interest may be found.
        coordinates (2d-array): real coordinates of the proposal region.
        coeff (int): coefficient for threshold in find_peaks.
    
    Return:
        points_of_interest (dataframe): dataframe with the detected points.
    """
    # Generate dataframe
    # Extract size
    pd_df = pd.DataFrame( columns= ["LTP", "STP", "IDX", "PEAK_VAL", "X_COORD", "Y_COORD",
                                     "PRE_LABEL", "INFO", "REGION", "FILENAME"])
    size= image[0].shape[0]

    # Open all proposed regions and read one by one.
    for n_reg, reg in enumerate(image):
    
        # Create mask and compute statistics
        masked, std, median = statistics_region(reg)
        # Start peak detection.
        points = find_peaks(reg-median, threshold=coeff*std, box_size=size/2, 
                           npeaks=1, mask=masked)
    
        # Check for obtained values
        if points:
            # Iterate all over the rows in the detected points dataframe                
                real_pos = coordinates[n_reg][int(points["y_peak"]), int(points["x_peak"])]

                if len(get_iterable(real_pos)) == 2:
                    pd_df.loc[len(pd_df)] = [ltp, stp, id, points["peak_value"][0]+median, real_pos[1],
                                              real_pos[0], "object", "info", cropped_region(full_image, real_pos[1], real_pos[0], lim),
                                              filename]
                
    return pd_df     

def cropped_region(image, x_pos, y_pos, lim):
    """
    Crop a small region from a given image. If coordinates are near boundaries, generate zero padding.

    Parameters:
        image (2d-array): .fits image with data.
        x_pos (int): x-coordinate with an identified object.
        y_pos (int): y-coordinate with an identified object.
        lim (int): radius of cropped region.

    Output:
        cropped_image (2d-array): cropped region from .fits image.
    """

    # Extract size of the image.
    H, W = image.shape
    # Calculate final size of the cropped region.
    crop_size = 2 * lim + 1

    # Generate cropping boundaries.
    # Add +1 factor in the cropped region so xy coordinates are in the center of the image.
    y_min = max(y_pos - lim, 0)
    y_max = min(y_pos + lim + 1, H)
    x_min = max(x_pos - lim, 0)
    x_max = min(x_pos + lim + 1, W)

    # Generate cropped image.
    # As coordinates are in x-y cartesian plane, for converting them into indexes, flip them.
    cropped = image[y_min:y_max, x_min:x_max]

    # Check if image is in-boundaries.
    if cropped.shape[0] == crop_size and cropped.shape[1] == crop_size:
        return cropped
    
    # If not, apply zero padding by generating a new image.
    padded_crop = np.zeros((crop_size, crop_size), dtype=image.dtype)

    # Determine placement in the zero-padded array
    pad_y_start = max(lim - y_pos, 0)
    pad_y_end = pad_y_start + (y_max - y_min)
    pad_x_start = max(lim - x_pos, 0)
    pad_x_end = pad_x_start + (x_max - x_min)

    # Sum up both regions.
    padded_crop[pad_y_start:pad_y_end, pad_x_start:pad_x_end] = cropped

    return padded_crop


def crop_star_position(ltp, stp, id, df, star_df, image, lim):
    """
    Crop an area where a star is supposed to be located.

    Parameters:
        df (pandas): pandas dataframe with detected objects.
        star_df (pandas): pandas dataframe with star position.
        image (2d-array): .fits image.
        lim (int): size of the cropped image.
    
    Output:
        updated_df (pandas): updated dataframe with added star images.
    """
    # Iterate over stars.
    for ids in range(len(star_df)):
        # Create region with star.
        x = int(star_df["xsensor"].iloc[ids])
        y = int(star_df["ysensor"].iloc[ids])
        cropped = cropped_region(image, x , y, lim )
        df.loc[len(df)] = [ltp, stp, id, image[y,x], x, y, "star", "non detected", cropped ]
    
    return df


### OBJECT PRE-CLASSIFICATION ###

def star_comparison(dataframe, star_catalogue):
    """
    Check if a found object is a previously detected star by computing the euclidean distance between
    both points.
    
    Parameters:
        dataframe (table): astropy table with detected peaks.
        star_catalogue (table): astropy table with detected stars.
    
    """
    # Extract coordinates from the detected object.
    values = len(dataframe)
    for idx in range(values):

        x1 = dataframe["X_COORD"].iloc[idx]
        y1 = dataframe["Y_COORD"].iloc[idx]

        # Run for all detected stars in star_catalogue.

        for ids in range(len(star_catalogue)):
            # Extract star position.
            x2, y2 = star_catalogue["xsensor"].iloc[ids], star_catalogue["ysensor"].iloc[ids]
            # Compute euclidean distance.
            dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if dist < 10:
                dataframe["PRE_LABEL"].iloc[idx] = "star"
                dataframe["INFO"].iloc[idx] = str(star_catalogue["MAIN_ID"].iloc[ids])
        

    return dataframe, star_catalogue


def remove_similar_objects(pandas_df, threshold = 5):
    """
    Given a pandas dataframe with detected objects, remove the duplicated ones.

    Parameters:
        pandas_df (dataframe): dataframe with object position and peak value.
        threshold (int): pixel distance between two objects for being considered same object.
    Return:
        cleaned_df (dataframe): cleaned dataframe with no duplicates.
    """

    # Create an array from 0 to size of dataframe.
    elements = np.arange(len(pandas_df))
    pairs = []

    # Create non repetitive pairs of indexes.
    for x, y in itertools.combinations(elements, 2):
        pairs.append([x,y])

    # Create list for duplicated elements
    remove_indexes = []

    # Iterate all over the pairs generated
    for pair in pairs:
        px, py = pair[0], pair[1]
        # Compute euclidean distance of detected objects
        y = (pandas_df["Y_COORD"].iloc[px] - pandas_df["Y_COORD"].iloc[py])**2
        x = (pandas_df["X_COORD"].iloc[px] - pandas_df["X_COORD"].iloc[py])**2
        dist = np.sqrt(y+x)
        # Store the index object whose peak was lower.
        if dist < threshold:
            index = [px if pandas_df["PEAK_VAL"].iloc[px] < pandas_df["PEAK_VAL"].iloc[py] else py]
            # Store index.
            remove_indexes.append(pandas_df.iloc[index[0]].name)

    # Remove repeated objec with lowest peak value.
    pandas_df = pandas_df.drop(remove_indexes)

    return pandas_df    

def points_inside_fov(x, y, center, scale, factor = 0.1):
    """
    Remove objects whose coordinates are outside metis detection region.

    Parameters:
        x (float): x coordinate of the detected object.
        y (float): y coordinate of the detected object.
        center (list): x, y coordinate of the metis center POV.
        scale (float): scale factor for matching FOV limits with the image.
        factor (float): multiplication factor for tunning up FOV limit.

    Return:
        (bool): True if detected object falls in new region.
    """

    # Metis FOV.
    radius1 = (sf.METIS_fov_min + factor)/scale
    radius2 = (sf.METIS_fov - factor)/scale

    # Check if object is outside first boundary.
    outside_first = (x - center[0])**2 + (y - center[1])**2 >= radius1**2
    # Check if object is inside second boundary.
    inside_second = (x - center[0])**2 + (y - center[1])**2 <= radius2**2

    return outside_first and inside_second

def remove_objects_fov(df, center, scale):
    """
    Remove objects that are outside METIS FOV.

    Parameters:
        df (pandas): dataframe with stars
        center (list): x, y coordinate of FOV center
        scale (float): scale.

    Return:
        df (pandas): filtered dataframe.
    """
    vals = []

    for i in range(len(df)):  
        x = df["X_COORD"].iloc[i]
        y = df["Y_COORD"].iloc[i]
        accept = points_inside_fov(x, y, center, scale)

        if not accept:
            vals.append(df.index[i])  

    df = df.drop(vals)  
    return df

def checkpoint_status(csv_file, csv_folder):
    """
    Check if a previously .csv exists. If so, extract the LTP, STP and FN from the
    last checked file. 

    Parameters:
        csv_file (str): name of the .csv file to check.
    
    Output:
        checkpoint (list): LTP, STP, FN from last checked FITS.
    """
    
    # Retrieve files names in folder.
    files_list = os.listdir(csv_folder)
    checkpoint = []
    # Check files.
    if csv_file in files_list:
        file = pd.read_pickle(csv_file)
        # Store info into list.
        checkpoint.append(file["LTP"].iloc[-1])
        checkpoint.append(file["STP"].iloc[-1])
        checkpoint.append(file["IDX"].iloc[-1])
    else:

        checkpoint = False
    
    return checkpoint

def folder_reader(csvs_fail, pkl_fits, pkl_obj, metis_folder, pkls_folder,
                  KERNEL_PATH, KERNEL_NAME, magnitude,uv, catalog,
                  size, overlapping, std, lim, filter = "vl-image"
                  ):
    """"
    Extract point-of-interest in a given set of L0 images. 

    """

    # Normal structure of a LO FITS file.
    headers_fits = ['APID', 'BITPIX', 'BLANK', 'CHECKSUM', 'COMMENT', 'COMPRESS',
       'COMP_RAT', 'CREATOR', 'DATAMAX', 'DATAMIN', 'DATASUM', 'EXTEND',
       'FILENAME', 'FILE_RAW', 'HISTORY', 'INSTRUME', 'LEVEL', 'LONGSTRN',
       'NAXIS', 'NAXIS1', 'NAXIS2', 'OBT_BEG', 'OBT_END', 'ORIGIN', 'SIMPLE',
       'VERSION', 'VERS_SW']

    ### DATA LOADING ###
    # Define full paths for savind data.
    obj_path = os.path.join(pkls_folder, pkl_obj)
    fits_path = os.path.join(pkls_folder, pkl_fits)
    csvf_path = os.path.join(pkls_folder, csvs_fail)


    # Check if there is already a previous pickle file.
    checkpoints = checkpoint_status(pkl_fits, pkls_folder)

    # SQL APPROACH.
    sqlite_path = os.path.join(pkls_folder, "psf_metadata.db")
    init_sqlite_db(sqlite_path)


    # If it is first run, create the pickle files with their corresponding headers.
    if not checkpoints:
        # Create file for fits.
        fits_file_pkl = pd.DataFrame(columns = headers_fits + ["LTP", "STP", "IDX",
                                                               "TIMESTAMP", "STARS", "OBJECTS"])

        # Create file for object detection.
        objs_file_pkl = pd.DataFrame(columns = ["LTP", "STP", "IDX",
                                                 "PEAK_VAL", "X_COORD", "Y_COORD",
                                                   "PRE_LABEL", "INFO", "REGION",
                                                   "FILENAME"])

        # Save DFs into pickle for preserving array structure.
        fits_file_pkl.to_pickle(fits_path)
        objs_file_pkl.to_pickle(obj_path)
        id_ck = -1
    
    # If pkl_fits already exist, then extract the LTP, STP, ID from the last analyzed file.
    else:
        ltp_ck, stp_ck, id_ck = checkpoints

    # Load pickles files
    fits_file_pkl = pd.read_pickle(fits_path)
    objs_file_pkl = pd.read_pickle(obj_path)

    ### FOLDER ITERATION ###
    
    # Extract folder with LTPs.
    LTPs_folder = ut.sorter(get_iterable(os.listdir(metis_folder)))

    # Load checkpoint for LTP.
    if checkpoints:
        LTPs_folder = LTPs_folder[LTPs_folder.index(ltp_ck):]

    # Iterate in all the LTP folders.
    for ltp_folder in LTPs_folder:
        print(f"Starting analysis with: {ltp_folder}") 

        # Extract folder with STPs.
        STPs_folder = ut.sorter(get_iterable(os.listdir(os.path.join(metis_folder, ltp_folder))))
        
        
        # Load checkpoint in STP.
        if checkpoints:
            STPs_folder = STPs_folder[STPs_folder.index(stp_ck):]

        # Iterate in all the STP folders.
        for stp_folder in STPs_folder:
            print(f"\t {stp_folder}") 

            # Extract files in each folder.
            fits_folder = sorted(get_iterable(os.listdir(os.path.join(metis_folder, ltp_folder, stp_folder))))
            # Apply filter.
            fits_folder = [file for file in fits_folder if filter in file]

            
            if checkpoints:
                # Load checkpoint in FITSs.
                fits_folder = fits_folder[id_ck + 1 :]
        
            ### DEACTIVATE CHECKPOINT ###
                checkpoints = False
            
            ### FITS ANALYSIS AND OBJECT DETECTION ###

            if not fits_folder:
                continue

            for id, fits_file in enumerate(fits_folder):
                print(f"\t\t File {id}")

                try: 
                    ## STAR DETECTION ##

                    # Extract timestamp, headers and the image of every .fits file
                    timestamp, headers, image = ut.fits_loader(os.path.join(metis_folder, ltp_folder, stp_folder, fits_file))
                    # Retrieve star position.
                    stars_detected, et, scale, center = sf.star_detector_offline(KERNEL_NAME, KERNEL_PATH, timestamp, uv, catalog, magnitude)
                    # Compute UTC_TIME.
                    timestamp = sf.et2utc(et)


                    ## OBJECT DETECTION ##

                    # Split image into several proposal regions.
                    regions, coordinates = image_slicer(image, size, overlapping)
                    # Search possible peaks given a threshold.
                    peaks = peak_detector_main(ltp_folder, stp_folder, id, regions, coordinates, std, image, lim, size, headers["FILENAME"])

                    ## OBJECT PRE-LABELING ##

                    # Check if stars are found.
                    if len(stars_detected)>0:
                        peaks, stars_remained = star_comparison(peaks, stars_detected)

                    # Remove similar objects.
                    peaks = remove_similar_objects(peaks, 5)
                    # Remove peaks outside metis foc
                    peaks = remove_objects_fov(peaks, center, scale)


                    ## DATA SAVING ##

                    # NORMAL APPROACH: Merge and save data in objects file. 
                    #objs_file_pkl = pd.concat([objs_file_pkl, peaks])
                    #objs_file_pkl.to_pickle(obj_path)

                    # SQL APPROACH: Use databases to store metadata and store
                    # each PSFs as separated file.
                    save_peaks_to_sql(peaks, sqlite_path)




                    # Append data in fits file.
                    headers = pd.DataFrame([headers])
                    headers_extra = pd.DataFrame([[ltp_folder, stp_folder, id,
                                                                        timestamp, len(peaks[peaks["PRE_LABEL"] =="star"]) ,
                                                                        len(peaks[peaks["PRE_LABEL"] =="object"])]], columns= ["LTP", "STP", "IDX",
                                                               "TIMESTAMP", "STARS", "OBJECTS"])
        
                    fits_file_pkl.loc[len(fits_file_pkl)] = pd.concat([headers, headers_extra], axis = 1).iloc[0]
                    fits_file_pkl.to_pickle(fits_path)

                except Exception as e:
                    with open(csvf_path, "a") as f:
                        writer = csv.writer(f)
                        writer.writerow([ltp_folder, stp_folder, fits_file, str(e)])


