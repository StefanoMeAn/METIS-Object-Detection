"""

Functions developed for star detection in METIS Solar Orbiter
    Retrieved from: https://github.com/chioettop/metis_star_utils
    By Paolo Chioetto.

"""

# BASIC LIBRARIES
import os
import spiceypy
import numpy as np
import astropy.wcs
import contextlib
from astroquery.simbad import Simbad


# IMPORTANT PARAMETERS
METIS_fov = 3.4         # METIS Field of View in deg.
METIS_fov_min = 1.6     
SUN_RADIUS = 6.957e5    # Nominal Solar radius (km).
SOLO_naif_id = -144     # ID for METIS solar orbiter.



@contextlib.contextmanager
def chdir(path):
    """
    Change directory as needed.
    
    Parameters:
        path (str): path to SPICE KERNELS.
    """

    CWD = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(CWD)


def load_kernel(kernel, path):
    """
    Loads SPICE KERNEL

    Parameters:
        kernel (str): SPICE Kernel.
        path (str): path to SPICE KERNELS.
    """

    with chdir(path):
        spiceypy.furnsh(kernel)
        
def unload_kernel(kernel, path):
    """
    Unloads SPICE KERNEL

    Parameters:
        kernel (str): SPICE Kernel.
        path (str): path to SPICE KERNELS.
    """

    with chdir(path):
        spiceypy.unload(kernel)

def scs2et(obt_string, path):
    """
    Converts Spacecraft clock string into ephemeris time.

    Parameters:
        obt_string (str): SCS timestamp from METIS.
        path (str): path to SPICE KERNELS.
    """

    with chdir(path):
        return spiceypy.scs2e(SOLO_naif_id, obt_string)
    

def et2utc(et):
    """
    Converts Ephemeris time into UTC.

    Parameters:
        et (float): SPICE continuos time.
    Output:
        utc (timestamp): UTC time.
    """

    utc = spiceypy.et2utc(et, 'C', 4)
    return utc



def boresight(et, path, UV=False):
    """
    Retrieves the RA, DEC, and ROLL angles from a given et in the METIS solar orbiter.
    
    Parameters:
        et (float): SPICE continuos time.
        path (str): path to SPICE KERNELS.
        UV (bool): specified bandwidth. If false, visible-light bandwidth is chosen.
    Output:
        ra (float): right ascension angle.
        dec (float): declination angle.
        roll (float): rotation angle.
    """
     
    channel = 'SOLO_METIS_EUV_ILS' if UV else 'SOLO_METIS_VIS_ILS'

    with chdir(path):
        # Rotation matrix from Metis local frame to J2000.
        rot = spiceypy.pxform(channel, 'J2000', et) 

        # Transform Metis boresight to J2000
        res = np.dot(rot, [-1, 0, 0])  

        # Returns [distance, RA, DEC] of vector             
        radec = spiceypy.recrad(res)  

        # ra, dec in degrees  
        ra, dec = np.rad2deg(radec[1:3])

        # Convert rot matrix to Euler angles. The one corresponding to x axis is Metis roll.       
        _, _, roll = np.rad2deg(spiceypy.m2eul(rot, 3, 2, 1)) 
        
    return ra, dec, roll

def wcs_from_boresight(ra, dec, roll, UV=False, L0 = True):
    """
    Returns WCS coordinates transformation object from RA, Dec and Roll of Metis.
    The WCS converts from celestial and sensor coordinates (and vice versa).
    VL sensor coordinates have inverted x axis and are rotated 90 deg clockwise
    wrt. celestial. More here -> https://fits.gsfc.nasa.gov/fits_wcs.html

    Parameters:
        ra (float): right ascension angle.
        dec (float): declination angle.
        roll (float): rotation angle.
        UV (bool): specified bandwidth. If false, visible-light bandwidth is chosen.
    Output:
        w (wcs object): object for performing coordinate transformations.
        
    """

    size_vl = 2048
    size_uv = 1024

    if not L0:
        size_vl = 2048
        size_uv = 1024


    if UV:
        scx = -20.401/3600      # platescale in deg
        scy = 20.401/3600       
        bx, by = size_uv+2.6, size_uv-4.2  # boresight position in px
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
    
    ##  Build a WCS between sensor and sky coords.

    # Trasformation order pix->sky: matrix (rotation) -> translation -> scale.
    w = astropy.wcs.WCS(naxis=2)

    # Transformatios for L0 FITS.
    w.pixel_shape = (1024, 1024) if UV else (2048, 2048)

    # Boresight translation (center in pixels).
    w.wcs.crpix = [bx, by]

    # Scaling.              
    w.wcs.cdelt = [scx, scy] 

    # Boresight in celestial coordinates.
    w.wcs.crval = [ra, dec]    
    # Projection type from plane to spherical (TAN is gnomonic).
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]  
    # Units for crdelt and crval.
    w.wcs.cunit = ["deg", "deg"]            
    # Negative wrt the one returned by the rot matrix. Why?
    t = -np.deg2rad(roll+roll_offset+det_angle)  
    c, s = np.cos(t), np.sin(t)
    # Rotation matrix accounting for roll angle.
    w.wcs.pc = np.array([[c, -s], [s, c]]) 
    if flip_xy:
        w.wcs.pc = w.wcs.pc @ np.array([[0,-1],[-1,0]])
    
    return w  

def simbad_search(ra, dec, max_mag=6):
    """
    Performs queering in SIMBAD CATALOGUE and retrieve stars position in a given position.
    More info -> https://astroquery.readthedocs.io/en/latest/simbad/simbad.html   

    Parameters: 
        ra (float): right ascension angle.
        dec (float): declination angle.
        max_magnitude (float): threshold for maximum apparent magnitude.
    Output:
        res (astropy table): table with detected stars in a region.
    """
    
    # Load catalogue.
    cs = Simbad() 

    # ICRS decimal coords
    cs.add_votable_fields('ra(d;A;ICRS;J2000;2000)', 'dec(d;D;ICRS;J2000;2000)') 
    cs.add_votable_fields('flux(V)') # Visible flux
    cs.add_votable_fields('id(HD)')
    cs.remove_votable_fields('coordinates')
    
    # Query SIMBAD database
    res = cs.query_criteria(f'region(circle, ICRS, {ra} {dec}, {METIS_fov}d) & Vmag < {max_mag}')

    # If no stars are detected, exit.
    if res is None:
        return None
    
    # Rename columns for plotting routines.
    res.rename_columns(['RA_d_A_ICRS_J2000_2000', 'DEC_d_D_ICRS_J2000_2000', 'FLUX_V'], ['ra', 'dec', 'mag'])
    
    # Remove stars in the inner Metis FOV.
    res = res[np.sqrt((res['ra'] - ra)**2 + (res['dec'] - dec)**2) > METIS_fov_min]    

    return res


def stars_detector(KERNEL_NAME, KERNEL_PATH, time, UV=False):
    """
    Retrieve a dataframe with stars on a given time coordinate according to Metis.

    Parameters:
        KERNEL_NAME (str): SPICE kernel.
        KERNEL_PATH (str): path to SPICE kernels.
        time (str): time in SCS.
        UV (bool): specified bandwidth. If false, visible-light bandwidth is chosen.

    Returns:
        catalog_stars (table): astropy table with detected stars.
        -wcs.wcs.cdelt[0] (float): pixel scale.
        wcs.wcs.crpix (tuple): center of image.
    """

    try:
        # Load kernel.
        spice = load_kernel(KERNEL_NAME, KERNEL_PATH)

        # Convert time to ephemeris time.
        et = scs2et(time, KERNEL_PATH)

        # Calculate boresight orientation.
        ra, dec, roll = boresight(et, KERNEL_PATH, UV)
        wcs = wcs_from_boresight(ra, dec, roll, UV)

        # Query stars from Simbad catalog.
        catalog_stars = simbad_search(ra, dec, max_mag=6)
        if catalog_stars is None or len(catalog_stars) == 0:
            return None

        # Transform star coordinates to sensor coordinates.
        x, y = wcs.wcs_world2pix(catalog_stars["ra"], catalog_stars["dec"], 0)

        # Add sensor coordinates to the DataFrame.
        catalog_stars["xsensor"] = x
        catalog_stars["ysensor"] = y

        # Filter stars outside the sensor bounds.
        in_bounds = (x >= 0) & (y >= 0) & (x <= wcs.pixel_shape[1]) & (y <= wcs.pixel_shape[0])
        catalog_stars = catalog_stars[in_bounds]
        if len(catalog_stars) == 0:
            return None

        return catalog_stars, -wcs.wcs.cdelt[0], wcs.wcs.crpix

    except Exception as e:
        print(f"Error: {e}")
        return None

    finally:

        # Unload kernel.
        with chdir(KERNEL_PATH):
            spiceypy.unload(KERNEL_NAME)

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
        scale (float): scale for fov radius.
        center (float): center of the sun.
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

    return catalog_stars.reset_index().drop(["index"], axis = 1), et, -wcs.wcs.cdelt[0], wcs.wcs.crpix
