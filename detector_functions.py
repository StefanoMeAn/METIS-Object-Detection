
import os
import pandas as pd
import math
import collections
import spicepy
from starcatalogs import StarCatalog
from scipy.stats import iqr
from scipy.optimize import curve_fit
from scipy.spatial import distance
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import find_peaks
import itertools
from star_functions import*


import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

### STAR DETECTION ###

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

def star_detector_offline(KERNEL_NAME, KERNEL_PATH, time, UV, cat, max_mag):
    """
    Simulate the Field-of-view of the metis project and extract the possible visible stars.
    For fast-performance, query the stars in a previously downloaded catalog.
    Simulate the possible position of the on-orbit telescope and simulate its FOV.

    Parameters:
        KERNEL_NAME (str): name of the kernel to be used.
        KERNEL_PATH (str): path to the kernels.
        time (float): timestamp at which the .fits was generated.
        UV (bool): waveband in which .fits was generated.
        cat (starcatalogs): catalog with stars up to magnitude 7.
        mag_stars (float): max magnitude for the star detection.
    
    Output:
        catalog_stars (pandas-df): table with the obtained stars.
    """

    # Convert timestamp into ephemeris time.
    et = scs2et(time, KERNEL_PATH)
    # Extract angles.
    ra, dec, roll = boresight(et, KERNEL_PATH,  UV=False)
    # Get status vector vs SSB.
    posvel, dist = spkezr(KERNEL_PATH,"SSB", et)
    # Build wcs from apparent Metis pointing.
    wcs = wcs_from_boresight(ra, dec, roll, UV)
    # Adjust boresight with nominal wcs.
    ra_adj, dec_adj = wcs.wcs_pix2world([wcs.wcs.crpix], 0).flatten()
    # Find stars through catalog search.
    catalog_stars = cat.query(ra_adj, dec_adj, METIS_fov, METIS_fov_min)
    # Add stellar aberration corrected coordinates.
    ra_abcorr, dec_abcorr = stelab(KERNEL_PATH, catalog_stars["ra"], 
                                         catalog_stars["dec"], et)
    catalog_stars["ra_abcorr"] = ra_abcorr
    catalog_stars["dec_abcorr"] = dec_abcorr
    # Add sensor coordinates to catalog star list.
    x, y = wcs.wcs_world2pix(ra_abcorr, dec_abcorr, 0)
    catalog_stars["xsensor"] = x
    catalog_stars["ysensor"] = y
    # Remove stars outside the sensor
    catalog_stars = catalog_stars[
    (catalog_stars.xsensor > 0) & (catalog_stars.xsensor < wcs.pixel_shape[0]) &
    (catalog_stars.ysensor > 0) & (catalog_stars.ysensor < wcs.pixel_shape[1])
        ]
    # Filter stars according to the star magnitude.
    catalog_stars = catalog_stars[catalog_stars["Vmag"]<=max_mag]
    return catalog_stars.reset_index().drop(["index"], axis = 1), et

def sorter(list):
    """
    Sort a list with elements of the type ABCNN, where A, B, C are letters and N is a number.

    Args:
        list (list): List with LTPs or STPs.
    Returns:
        Sorted list.
    """
    return sorted(list, key=lambda x: int(x[3:]))

### PEAK DETECTOR ###

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

def peak_detector_main(columns, ltp, stp, id, image, coordinates, coeff, full_image, lim):
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
    pd_df = pd.DataFrame( columns= columns)
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
                                              real_pos[0], "object","info", cropped_region(full_image, real_pos[1], real_pos[0], lim)]
                
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


from scipy.stats import iqr
def iqr_test(array, x, y):
    """
    Detect outliers in a numpy 2d array.

    Parameters:
        array (array): 2d array from a cropped region.
        pixel (array): xy coordinate with pixel position.
    
    """
    # Flatten 2D image.
    pixel = array[x, y]
    flatten_array = array.flatten()
    iqr_val = iqr(flatten_array)

    # Compute Q1, Q3.
    Q1 = np.percentile(flatten_array, 25)
    Q3 = np.percentile(flatten_array, 75)
    lower_bound = Q1 - 1.5 * iqr_val
    upper_bound = Q3 + 1.5 * iqr_val

    if (pixel < lower_bound) | (pixel > upper_bound):
        return "bright pixel"
    else:
        return None


def detect_outliers(dataframe):
    """
    Given a pandas dataframe, check each proposal region and distinguish which ones are broken pixels.

    Parameters:
        dataframe (dataframe): pandas dataframe with detected objects.
    
    Return:
        classified_dataframe:
    """

    # Extract size.
    size_df = len(dataframe)
    size_reg = dataframe["REGION"].iloc[0].shape[0]
    
    for idx in range(size_df):
        if dataframe["PRE_LABEL"].iloc[idx]!= "star":
            # Extract region number.
            region = dataframe["REGION"].iloc[idx]

            # Compute coordinate with respect of the proposed region.
            x = int(size_reg/2)
            y = int(size_reg/2)

            # Check if detected point is an outlier
            val = iqr_test(region, x, y)
            
            if val:
                dataframe["PRE_LABEL"].iloc[idx] = val
    
    return dataframe



def gaussian_2d(coords, A, x0, y0, sigma_x, sigma_y, C):
    """2D Gaussian with offset"""
    x, y = coords
    return A * np.exp(-(((x - x0)**2) / (2 * sigma_x**2) + ((y - y0)**2) / (2 * sigma_y**2))) + C

def gaussian_fitter(image, x, y):
    """
    Fit a 2D-Gaussian function in a image and return the obtained parameters.

    Params:
        image (2d-array): image with possible gaussian.

    Returns:
        A (float): amplitude.
        ux (float): mean in x.
        uy (float): mean in y.
        stdx (float): std in x.
        stdy (float): std in y.
    """

    # Convert 2D into 1D data.
    x_flatted = x.ravel()
    y_flatted = y.ravel()
    image_flatted = image.ravel()

    # Compute intial guess.
    A_0 = np.max(image)
    x0_0 = image.shape[1] // 2  
    y0_0 = image.shape[0] // 2  
    stdx_0 = image.shape[1] / 10  
    stdy_0 = image.shape[0] / 10
    C_0 = np.min(image) 

    # Start fitting
    popt, pcov = curve_fit(gaussian_2d, (x_flatted, y_flatted),
                            image_flatted, p0=(A_0, x0_0, y0_0, stdx_0, stdy_0, C_0))
    
    # Extract parameters
    A, ux, uy, stdx, stdy, C = popt
    
    return A, ux, uy, stdx, stdy, C




##### FUNCTIONS FOR DETECTION #####

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



def folder_reader(pkl_fits, pkl_obj, metis_folder, pkls_folder,
                  KERNEL_PATH, KERNEL_NAME, magnitude,uv, catalog,
                  headers_fits, headers_fits_extra, headers_objs,
                  size, overlapping, std, lim
                  ):
    """"
    Extract point-of-interest in a given set of L0 images. 

    """

    
    ### DATA LOADING ###
    # Define full paths for savind data.
    obj_path = os.path.join(pkls_folder, pkl_obj)
    fits_path = os.path.join(pkls_folder, pkl_fits)


    # Check if there is already a previous pickle file.
    checkpoints = checkpoint_status(pkl_fits, pkls_folder)

    # If it is first run, create the pickle files with their corresponding headers.
    if not checkpoints:
        # Create file for fits.
        fits_file_pkl = pd.DataFrame(columns = headers_fits + headers_fits_extra)

        # Create file for object detection.
        objs_file_pkl = pd.DataFrame(columns = headers_objs)

        # Save DFs into pickle for preserving array structure.
        fits_file_pkl.to_pickle(fits_path)
        objs_file_pkl.to_pickle(obj_path)
    
    # If pkl_fits already exist, then extract the LTP, STP, ID from the last analyzed file.
    else:
        ltp_ck, stp_ck, id_ck = checkpoints

    # Load pickles files
    fits_file_pkl = pd.read_pickle(fits_path)
    objs_file_pkl = pd.read_pickle(obj_path)

    ### FOLDER ITERATION ###
    
    # Extract folder with LTPs.
    LTPs_folder = sorter(get_iterable(os.listdir(metis_folder)))

    # Load checkpoint for LTP.
    if checkpoints:
        LTPs_folder = LTPs_folder[LTPs_folder.index(ltp_ck):]

    # Iterate in all the LTP folders.
    for ltp_folder in LTPs_folder:
        print(f"Starting analysis with: {ltp_folder}") 

        # Extract folder with STPs.
        STPs_folder = sorter(get_iterable(os.listdir(os.path.join(metis_folder, ltp_folder))))
        
        
        # Load checkpoint in STP.
        if checkpoints:
            STPs_folder = STPs_folder[STPs_folder.index(stp_ck):]

        # Iterate in all the STP folders.
        for stp_folder in STPs_folder:
            print(f"\t {stp_folder}") 

            # Extract files in each folder.
            fits_folder = sorted(get_iterable(os.listdir(os.path.join(metis_folder, ltp_folder, stp_folder))))
            
            if checkpoints:
                # Load checkpoint in FITSs.
                fits_file_chk = fits_folder[id_ck]
                fits_folder = fits_folder[fits_folder.index(fits_file_chk) +1 :]
            
            ### DEACTIVATE CHECKPOINT ###
                checkpoints = False

            ### FITS ANALYSIS AND OBJECT DETECTION ###

            for id, fits_file in enumerate(fits_folder):
                print(f"\t\t File {id}")

                ## STAR DETECTION ##

                # Extract timestamp, headers and the image of every .fits file
                timestamp, headers, image = fits_loader(os.path.join(metis_folder, ltp_folder, stp_folder, fits_file), headers_fits)
                # Retrieve star position.
                stars_detected, et = star_detector_offline(KERNEL_NAME, KERNEL_PATH, timestamp, uv, catalog, magnitude)
                # Compute UTC_TIME.
                timestamp = et2utc(et)


                ## OBJECT DETECTION ##

                # Split image into several proposal regions.
                regions, coordinates = image_slicer(image, size, overlapping)
                # Search possible peaks given a threshold.
                peaks = peak_detector_main(headers_objs, ltp_folder, stp_folder, id, regions, coordinates, 9, image, lim)

                ## OBJECT PRE-LABELING ##

                # Check if stars are found.
                if len(stars_detected)>0:
                    peaks, stars_remained = star_comparison(peaks, stars_detected)

                # Remove similar objects.
                peaks = remove_similar_objects(peaks, 5)


                ## DATA SAVING ##

                # Merge and save data in objects file.
                objs_file_pkl = pd.concat([objs_file_pkl, peaks])
                objs_file_pkl.to_pickle(obj_path)

                # Append data in fits file.
                fits_file_pkl.loc[len(fits_file_pkl)] = headers + [ltp_folder, stp_folder, id,
                                                                    timestamp, len(peaks[peaks["PRE_LABEL"] =="star"]) ,
                                                                    len(peaks[peaks["PRE_LABEL"] =="object"])]
                fits_file_pkl.to_pickle(fits_path)
