import spiceypy
import contextlib
import os
import numpy as np
from astroquery.simbad import Simbad
import astropy.wcs
import math
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.stats import scoreatpercentile


METIS_fov = 3.4 # METIS Field of View in deg, for astrocatalog searches
METIS_fov_min = 1.6 
SUN_RADIUS = 6.957e5   # Nominal Solar radius (km)
SOLO_naif_id = -144

# Functions for running star detection
solar_orbiter_naif_id = -144
@contextlib.contextmanager
def chdir(path):
    """Function needed to change directory as needed"""
    CWD = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(CWD)

def scs2et(obt_string, path):
    with chdir(path):
        return spiceypy.scs2e(solar_orbiter_naif_id, obt_string)

def et2utc(et):
    return spiceypy.et2utc(et, 'C', 4)

def load_kernel(kernel, path):
    """Load kernel"""
    with chdir(path):
        spiceypy.furnsh(kernel)

def boresight(et, path, UV=False):
     
    channel = 'SOLO_METIS_EUV_ILS' if UV else 'SOLO_METIS_VIS_ILS'
    with chdir(path):
        rot = spiceypy.pxform(channel, 'J2000', et) # rotation matrix from Metis local frame to J2000
        res = np.dot(rot, [-1, 0, 0])               # transform Metis boresight to J2000
        radec = spiceypy.recrad(res)                # returns [distance, RA, DEC] of vector
        ra, dec = np.rad2deg(radec[1:3])            # ra, dec in degrees
        _, _, roll = np.rad2deg(spiceypy.m2eul(rot, 3, 2, 1)) # convert rot matrix to Euler angles. The one corresponding to x axis is Metis roll
        
    return ra, dec, roll

def wcs_from_boresight(ra, dec, roll, UV=False):
    '''
    Returns WCS coordinates transformation object from RA, Dec and Roll of Metis.
    The WCS converts from celestial and sensor coordinates (and vice versa).
    VL sensor coordinates have inverted x axis and are rotated 90 deg clockwise
    wrt. celestial.
    https://fits.gsfc.nasa.gov/fits_wcs.html

    '''
    
    if UV:
        scx = -20.401/3600      # platescale in deg
        scy = 20.401/3600       
        bx, by = 512+2.6, 512-4.2  # boresight position in px
        det_angle = 0
        flip_xy = True          # UV detector appears to be flipped

    else:    
        scx = -10.137/3600  # platescale in deg (negative to invert axis)
        scy = 10.137/3600       
        bx = 966.075        # boresight position in px
        by = 2048-1049.130 
        det_angle = -90     # deg, detector base rotation wrt. sky
        flip_xy = False
    
    roll_offset = 0.077
    
    # build a WCS between sensor and sky coords
    # trasformation order pix->sky: matrix (rotation) -> translation -> scale
    w = astropy.wcs.WCS(naxis=2)
    w.pixel_shape = (1024, 1024) if UV else (2048, 2048)
    w.wcs.crpix = [bx, by]     # for boresight translation (center in pixels)         
    w.wcs.cdelt = [scx, scy]   # for scaling
    w.wcs.crval = [ra, dec]    # boresight in celestial coords
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]  # projection type from plane to spherical (TAN is gnomonic)
    w.wcs.cunit = ["deg", "deg"]            # unit for crdelt and crval
    t = -np.deg2rad(roll+roll_offset+det_angle)  # negative wrt the one returned by the rot matrix. Why?
    c, s = np.cos(t), np.sin(t)
    w.wcs.pc = np.array([[c, -s], [s, c]]) # rotation matrix accounting for roll angle
    if flip_xy:
        w.wcs.pc = w.wcs.pc @ np.array([[0,-1],[-1,0]])
    
    return w      

def simbad_search(ra, dec, max_mag=6):
    
    # ref https://astroquery.readthedocs.io/en/latest/simbad/simbad.html    
    cs = Simbad()  # custom query fields
    cs.add_votable_fields('ra(d;A;ICRS;J2000;2000)', 'dec(d;D;ICRS;J2000;2000)') # ICRS decimal coords
    cs.add_votable_fields('flux(V)') # Visible flux
    cs.add_votable_fields('id(HD)')
    cs.remove_votable_fields('coordinates')
    # cs.get_votable_fields()
    # Simbad.get_field_description('id')
    
    # query SIMBAD database
    res = cs.query_criteria(f'region(circle, ICRS, {ra} {dec}, {METIS_fov}d) & Vmag < {max_mag}')
    if res is None:
        return None
    # rename columns for plotting routines
    res.rename_columns(['RA_d_A_ICRS_J2000_2000', 'DEC_d_D_ICRS_J2000_2000', 'FLUX_V'], ['ra', 'dec', 'mag'])
    
    # remove stars in the inner Metis FOV
    res = res[np.sqrt((res['ra'] - ra)**2 + (res['dec'] - dec)**2) > METIS_fov_min]    

    return res


def stars_detector(KERNEL_NAME, KERNEL_PATH, time, UV):
    """Retrieve a dataframe with stars on a given time coordinate according to Metis"""
    # Load kernel and extract stars
    spice = load_kernel(KERNEL_NAME, KERNEL_PATH)
    # Convert into ephemeris time
    et = scs2et(time, KERNEL_PATH)
    # Extract angles
    ra, dec, roll = boresight(et, KERNEL_PATH,  UV=False)
    wcs = wcs_from_boresight(ra, dec, roll, False)
    # Search stars
    catalog_stars = simbad_search(ra, dec, max_mag=6)
    if catalog_stars is None:
        with chdir(KERNEL_PATH):
            spiceypy.unload(KERNEL_NAME)
            return None

    # Transform star coordinates into pixel coordinates
    x, y = wcs.wcs_world2pix(catalog_stars['ra'], catalog_stars['dec'], 0)
    # Generate new columns in catalog star dataframe
    catalog_stars['xsensor'] = x
    catalog_stars['ysensor'] = y
    # Remove stars that are not in range
    catalog_stars.remove_rows((x < 0) | (y < 0) | 
                          (x > wcs.pixel_shape[0]) | (y > wcs.pixel_shape[0]))
    if len(catalog_stars) == 0:
        with chdir(KERNEL_PATH):
            spiceypy.unload(KERNEL_NAME)
            return None
    # Unload kernel
    with chdir(KERNEL_PATH):
        spiceypy.unload(KERNEL_NAME)

    return catalog_stars, -wcs.wcs.cdelt[0], wcs.wcs.crpix

def get_iterable(x):
    """
    Check if a variable is a list. If it is not, covert it into a list.
    """
    if isinstance(x, list):
        return x
    else:
        return (x,)

#### SPICELIB LIBRARIES ####

def spkezr(path, targ, et, ref='J2000', abcorr='LT'):
        with chdir(path):
            return spiceypy.spkezr(targ=targ, et=et, ref=ref, abcorr=abcorr, obs=str(SOLO_naif_id))    
        
def stelab(path, ra, dec, et):
    """
    Corrects a list of celestial direction vectors in J2000 for stellar 
    aberrations wrt. Metis velocity, returning apparent positions of 
    objects from nominal positions. Used for determining apparent star 
    positions in Metis FoV.

    Parameters
    ----------
    ra, dec : Numpy arrays, [degrees]
        celestial coordinates of object, in degrees.
    et : float, ephemeris time
        observer epoch.

    Returns
    -------
    ra, dec : Numpy arrays, [degrees]
    celestial coordinates of apparent object position.

    """
    with chdir(path):
        # Get the geometric state of the observer (Metis) relative to 
        # the solar system barycenter.
        s_metis = spiceypy.spkssb(SOLO_naif_id, et, "J2000")
        
        ra_corr = []
        dec_corr = []
        for r, d in zip(np.deg2rad(ra), np.deg2rad(dec)):
            # Convert range, ra, dec to cartesian coordinates
            # Range is set to 1 as it should not matter for stelab
            pos = spiceypy.radrec(1, r, d)
        
            # apply stellar aberration correction to the boresight
            # s_metis[3:6] is Metis velocity
            pos_abcorr = spiceypy.stelab(pos, s_metis[3:6])
        
            # Convert back to ra, dec
            r_, r_corr, d_corr = spiceypy.recrad(pos_abcorr)
            
            ra_corr.append(r_corr)
            dec_corr.append(d_corr)
        
        return np.rad2deg(ra_corr), np.rad2deg(dec_corr)
    

    ### FITS RELATED LIBRARIES ###

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
    # Close file
    fits_file.close()

    return timestamp, fits_header


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