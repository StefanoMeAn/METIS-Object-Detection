from detector_functions import*


"""------PARAMETERS FOR DATA LOADING------""" 
# Path to the metis_folder.
DATA_DIR = "/home/stefano98/University of Padua/thesis/Packages/METIS-Object-Detection/metis_vis"

# Path to write .csvs
CSV_DIR = "/home/stefano98/University of Padua/thesis/Packages/METIS-Object-Detection"


"""------PARAMETERS FOR DATA SAVING------"""
# Name for fits saving file.
CVS_FILE_FITS = "fits_files.pkl"

# Name for object saving file.
OBS_FILE = "objects_files.pkl"


"""------STAR DETECTION PARAMETERS-------"""
# Kernel information.
KERNEL_PATH = "/home/stefano98/University of Padua/thesis/notebooks/kernels/solar-orbiter/kernels/mk"
KERNEL_NAME = "solo_ANC_soc-flown-mk.tm"

# Define waveband in which pictures were taken.
UV = False

# Maximum stellar magnitude.
MAX_MAG = 6

# Catalog with stars.
CAT = StarCatalog('Simbad') # Put this in the CSV_DIR.

"""------OBJECT DETECTION PARAMETERS------"""
# Size for region proposal.
BOX_SIZE = 20
# Size for overlapping between regions.
OVERLAPPING = 3
# Threshold for peak finding.
THRESHOLD = 8
# Size for cropped region (total_size = LIM*2 +1)
LIM = 10 


"""------HEADERS FOR PICKLE FILES------"""
# Headers to extract from the .fits file (L0).
HEADERS_TO_EXTRACT = ["FILENAME", "OBT_BEG", "OBT_END", "DATAMIN", "DATAMAX"]

# Additional headers to be created in the FITS_FILE.csv.
HEADERS_TO_ADD = ["LTP", "STP", "IDX","TIMESTAMP", "STARS", "OBJECTS"]

# Headers to create in the object detection .csv.
HEADERS_OBJ = ["LTP", "STP", "IDX", "PEAK_VAL", "X_COORD", "Y_COORD", "PRE_LABEL", "INFO", "REGION"]


#### CODE RUNNING ####

# Load kernel.
spice = load_kernel(KERNEL_NAME, KERNEL_PATH)

# Run object detection.

folder_reader(CVS_FILE_FITS, OBS_FILE, DATA_DIR, CSV_DIR,
            KERNEL_PATH, KERNEL_NAME, MAX_MAG, UV, CAT, 
            HEADERS_TO_EXTRACT, HEADERS_TO_ADD, HEADERS_OBJ,
              BOX_SIZE, OVERLAPPING, THRESHOLD, LIM)
