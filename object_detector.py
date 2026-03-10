import libraries.object_detection as od
from libraries.starcatalogs import StarCatalog
import libraries.starfunctions as sf

"""------PARAMETERS FOR DATA LOADING------"""
# Path to the metis folder.
DATA_DIR = "/home/stefano98/University of Padua/thesis/Packages/METIS-Object-Detection/metis_vis"

# Path where the SQLite database will be written.
OUTPUT_DIR = "/home/stefano98/University of Padua/thesis/Packages/METIS-Object-Detection"

# SQLite database filename.
DB_NAME = "psf_metadata.db"


"""------STAR DETECTION PARAMETERS-------"""
# Kernel information.
KERNEL_PATH = "/home/stefano98/University of Padua/thesis/Packages/Solar-orbiter/kernels/mk"
KERNEL_NAME = "solo_ANC_soc-flown-mk.tm"

# Define waveband in which pictures were taken.
UV = False

# Maximum stellar magnitude.
MAX_MAG = 8

# Catalog with stars.
CAT = StarCatalog("Simbad")


"""------OBJECT DETECTION PARAMETERS------"""
# Size for region proposal.
BOX_SIZE = 30

# Size for overlapping between regions.
OVERLAPPING = 3

# Threshold for peak finding.
THRESHOLD = 5

# Size for cropped region (total_size = LIM*2 + 1)
LIM = 30


#### CODE RUNNING ####

# Load kernel.
spice = sf.load_kernel(KERNEL_NAME, KERNEL_PATH)

# Run object detection.
od.folder_reader(
    DATA_DIR,
    OUTPUT_DIR,
    KERNEL_PATH,
    KERNEL_NAME,
    MAX_MAG,
    UV,
    CAT,
    BOX_SIZE,
    OVERLAPPING,
    THRESHOLD,
    LIM,
    db_name=DB_NAME
)