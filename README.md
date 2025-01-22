# METIS-Object-Detection
This project performs object detection on FITS images from the METIS dataset. The code first identifies stars in the images using a previouly downloaded star catalogs, then applies peak detection across the entire image to identify points-of-interest. Detected objects and headers are saved into Pickle files.

---

## Features
- **Star Detection**: Uses predefined star catalogs to detect stars within the images using algorith develop by Paolo Chioetto.
- **Peak Detection**: Applies region proposals and peak-finding algorithms to detect points-of-interest.
- **Data Saving**: Saves results, including metadata, in pickle files.

---


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/StefanoMeAn/METIS-Object-Detection.git
   cd METIS-Object-Detection
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run it
   ```bash
   python object_detector.py

### Directory Setup
Ensure your directories are correctly set up:
- **`DATA_DIR`**: Path to the folder containing FITS images.
- **`CSV_DIR`**: Path to save CSV and pickle results.
- **`KERNEL_PATH`**: Path to SPICE kernel files.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

