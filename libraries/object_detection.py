import pandas as pd
import numpy as np
import math
from astropy.stats import sigma_clipped_stats
from photutils.detection import find_peaks, DAOStarFinder
from libraries.utilities import get_iterable
import itertools
import libraries.starfunctions as sf
import os
import libraries.utilities as ut
import sqlite3
import warnings

warnings.filterwarnings("ignore")


def init_sqlite_db(db_path):
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS psfs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                LTP TEXT,
                STP TEXT,
                IDX INTEGER,
                PEAK_VAL REAL,
                X_COORD INTEGER,
                Y_COORD INTEGER,
                PRE_LABEL TEXT,
                INFO TEXT,
                FILENAME TEXT,
                REGION_BLOB BLOB,
                REGION_SHAPE_Y INTEGER,
                REGION_SHAPE_X INTEGER,
                REGION_DTYPE TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS processed_fits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                LTP TEXT,
                STP TEXT,
                IDX INTEGER,
                TIMESTAMP TEXT,
                FILENAME TEXT,
                STARS INTEGER,
                OBJECTS INTEGER,
                APID TEXT,
                BITPIX TEXT,
                BLANK TEXT,
                CHECKSUM TEXT,
                COMMENT TEXT,
                COMPRESS TEXT,
                COMP_RAT TEXT,
                CREATOR TEXT,
                DATAMAX TEXT,
                DATAMIN TEXT,
                DATASUM TEXT,
                EXTEND TEXT,
                FILE_RAW TEXT,
                HISTORY TEXT,
                INSTRUME TEXT,
                LEVEL TEXT,
                LONGSTRN TEXT,
                NAXIS TEXT,
                NAXIS1 TEXT,
                NAXIS2 TEXT,
                OBT_BEG TEXT,
                OBT_END TEXT,
                ORIGIN TEXT,
                SIMPLE TEXT,
                VERSION TEXT,
                VERS_SW TEXT,
                UNIQUE(LTP, STP, FILENAME)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS failed_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                LTP TEXT,
                STP TEXT,
                FILENAME TEXT,
                ERROR TEXT
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_processed_fits_key
            ON processed_fits (LTP, STP, FILENAME)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_psfs_key
            ON psfs (LTP, STP, FILENAME)
        """)


def region_to_sql_fields(region):
    """
    Convert a numpy region into SQLite-storable fields.
    """
    region = np.asarray(region, dtype=np.float32)
    return (
        sqlite3.Binary(region.tobytes()),
        int(region.shape[0]),
        int(region.shape[1]),
        str(region.dtype)
    )


def load_region_from_row(region_blob, shape_y, shape_x, dtype):
    """
    Reconstruct a numpy array from SQLite row fields.
    """
    if region_blob is None:
        return None

    arr = np.frombuffer(region_blob, dtype=np.dtype(dtype))
    return arr.reshape((shape_y, shape_x))


def load_region_by_id(conn, psf_id):
    """
    Load a region directly from the psfs table using the row id.
    """
    row = conn.execute("""
        SELECT REGION_BLOB, REGION_SHAPE_Y, REGION_SHAPE_X, REGION_DTYPE
        FROM psfs
        WHERE id = ?
    """, (psf_id,)).fetchone()

    if row is None:
        raise ValueError(f"No psfs row found with id={psf_id}")

    return load_region_from_row(*row)


def save_peaks_to_sql(rows, conn):
    if not rows:
        return

    values = []
    for row in rows:
        region_blob, region_shape_y, region_shape_x, region_dtype = region_to_sql_fields(row["REGION"])

        values.append((
            row["LTP"],
            row["STP"],
            row["IDX"],
            row["PEAK_VAL"],
            row["X_COORD"],
            row["Y_COORD"],
            row["PRE_LABEL"],
            row["INFO"],
            row["FILENAME"],
            region_blob,
            region_shape_y,
            region_shape_x,
            region_dtype
        ))

    conn.executemany("""
        INSERT INTO psfs (
            LTP, STP, IDX, PEAK_VAL, X_COORD, Y_COORD,
            PRE_LABEL, INFO, FILENAME,
            REGION_BLOB, REGION_SHAPE_Y, REGION_SHAPE_X, REGION_DTYPE
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, values)


def save_processed_fits(conn, headers, ltp, stp, idx, timestamp, n_stars, n_objects):
    row = {
        "LTP": ltp,
        "STP": stp,
        "IDX": idx,
        "TIMESTAMP": str(timestamp),
        "FILENAME": str(headers.get("FILENAME", "")),
        "STARS": int(n_stars),
        "OBJECTS": int(n_objects),
        "APID": str(headers.get("APID", "")),
        "BITPIX": str(headers.get("BITPIX", "")),
        "BLANK": str(headers.get("BLANK", "")),
        "CHECKSUM": str(headers.get("CHECKSUM", "")),
        "COMMENT": str(headers.get("COMMENT", "")),
        "COMPRESS": str(headers.get("COMPRESS", "")),
        "COMP_RAT": str(headers.get("COMP_RAT", "")),
        "CREATOR": str(headers.get("CREATOR", "")),
        "DATAMAX": str(headers.get("DATAMAX", "")),
        "DATAMIN": str(headers.get("DATAMIN", "")),
        "DATASUM": str(headers.get("DATASUM", "")),
        "EXTEND": str(headers.get("EXTEND", "")),
        "FILE_RAW": str(headers.get("FILE_RAW", "")),
        "HISTORY": str(headers.get("HISTORY", "")),
        "INSTRUME": str(headers.get("INSTRUME", "")),
        "LEVEL": str(headers.get("LEVEL", "")),
        "LONGSTRN": str(headers.get("LONGSTRN", "")),
        "NAXIS": str(headers.get("NAXIS", "")),
        "NAXIS1": str(headers.get("NAXIS1", "")),
        "NAXIS2": str(headers.get("NAXIS2", "")),
        "OBT_BEG": str(headers.get("OBT_BEG", "")),
        "OBT_END": str(headers.get("OBT_END", "")),
        "ORIGIN": str(headers.get("ORIGIN", "")),
        "SIMPLE": str(headers.get("SIMPLE", "")),
        "VERSION": str(headers.get("VERSION", "")),
        "VERS_SW": str(headers.get("VERS_SW", "")),
    }

    pd.DataFrame([row]).to_sql("processed_fits", conn, if_exists="append", index=False)


def already_processed(conn, ltp, stp, filename):
    cur = conn.execute("""
        SELECT 1
        FROM processed_fits
        WHERE LTP = ? AND STP = ? AND FILENAME = ?
        LIMIT 1
    """, (ltp, stp, filename))
    return cur.fetchone() is not None


def save_failed_file(conn, ltp, stp, filename, error_msg):
    conn.execute("""
        INSERT INTO failed_files (LTP, STP, FILENAME, ERROR)
        VALUES (?, ?, ?, ?)
    """, (ltp, stp, filename, str(error_msg)))


def image_slicer(image, size, overlapping_size=3):
    """
    Slice a 2D-numpy array into several symmetrical regions that may share borders
    with its neighbors.
    """

    size_x, size_y = image.shape

    image_coordinates = np.zeros((size_x, size_y), dtype=object)
    size_x_vals = np.arange(size_x)
    size_y_vals = np.arange(size_y)

    for idx, valx in enumerate(size_x_vals):
        for idy, valy in enumerate(size_y_vals):
            image_coordinates[idx, idy] = [valx, valy]

    n_regionsx = math.ceil(size_x / size)
    n_regionsy = math.ceil(size_y / size)

    padding_x = (n_regionsx * size - size_x) / 2
    padding_y = (n_regionsy * size - size_y) / 2

    image = np.pad(
        image,
        [
            (math.ceil(padding_x), math.floor(padding_x) + overlapping_size),
            (math.ceil(padding_y), math.floor(padding_y) + overlapping_size),
        ],
        constant_values=0,
    )

    image_coordinates = np.pad(
        image_coordinates,
        [
            (math.ceil(padding_x), math.floor(padding_x) + overlapping_size),
            (math.ceil(padding_y), math.floor(padding_y) + overlapping_size),
        ],
        constant_values=0,
    )

    size_x, size_y = image.shape

    proposal_regions = []
    proposal_coordinates = []

    for i in range(int(size_x / size)):
        for j in range(int(size_y / size)):
            proposal_regions.append(
                image[i * size:(i + 1) * size + overlapping_size,
                      j * size:(j + 1) * size + overlapping_size]
            )
            proposal_coordinates.append(
                image_coordinates[i * size:(i + 1) * size + overlapping_size,
                                  j * size:(j + 1) * size + overlapping_size]
            )

    return proposal_regions, proposal_coordinates


def create_coordinates(size_x, size_y):
    X = np.arange(size_x)
    Y = np.arange(size_y)
    Xmsh, Ymsh = np.meshgrid(X, Y)
    pairs = np.stack((Xmsh, Ymsh), axis=-1)
    return pairs


def statistics_region(region, sigma=3.0):
    mask = region == 0
    valid = region[~mask]

    if valid.size == 0:
        return mask, 0.0, 0.0

    mean, median, std = sigma_clipped_stats(valid, sigma=sigma)
    return mask, float(std), float(median)


def cropped_region(image, x_pos, y_pos, lim):
    H, W = image.shape
    crop_size = 2 * lim + 1

    y_min = max(y_pos - lim, 0)
    y_max = min(y_pos + lim + 1, H)
    x_min = max(x_pos - lim, 0)
    x_max = min(x_pos + lim + 1, W)

    cropped = image[y_min:y_max, x_min:x_max]

    if cropped.shape[0] == crop_size and cropped.shape[1] == crop_size:
        return cropped

    padded_crop = np.zeros((crop_size, crop_size), dtype=image.dtype)

    pad_y_start = max(lim - y_pos, 0)
    pad_y_end = pad_y_start + (y_max - y_min)
    pad_x_start = max(lim - x_pos, 0)
    pad_x_end = pad_x_start + (x_max - x_min)

    padded_crop[pad_y_start:pad_y_end, pad_x_start:pad_x_end] = cropped
    return padded_crop


def peak_detector_main(ltp, stp, id, image, coordinates, coeff, full_image, lim, size, filename):
    rows = []
    region_size = image[0].shape[0]

    for n_reg, reg in enumerate(image):
        masked, std, median = statistics_region(reg)

        if std <= 0:
            continue

        points = find_peaks(
            reg - median,
            threshold=coeff * std,
            box_size=max(1, region_size // 2),
            npeaks=1,
            mask=masked
        )

        if points is not None and len(points) > 0:
            real_pos = coordinates[n_reg][int(points["y_peak"][0]), int(points["x_peak"][0])]

            if len(get_iterable(real_pos)) == 2:
                rows.append({
                    "LTP": ltp,
                    "STP": stp,
                    "IDX": id,
                    "PEAK_VAL": float(points["peak_value"][0] + median),
                    "X_COORD": int(real_pos[1]),
                    "Y_COORD": int(real_pos[0]),
                    "PRE_LABEL": "object",
                    "INFO": "info",
                    "REGION": cropped_region(full_image, real_pos[1], real_pos[0], lim),
                    "FILENAME": filename
                })

    return rows


def peak_detector_main_DAOS(ltp, stp, id, image, coordinates, coeff, full_image, lim, size, filename):
    rows = []
    fwhm = 2.0

    for n_reg, reg in enumerate(image):
        masked, std, median = statistics_region(reg)

        if std <= 0:
            continue

        data = reg - median

        finder = DAOStarFinder(
            threshold=coeff * std,
            fwhm=fwhm,
            exclude_border=False
        )

        sources = finder(data, mask=masked)

        if sources is None or len(sources) == 0:
            continue

        brightest_idx = np.argmax(sources["peak"])
        src = sources[brightest_idx]

        x = int(round(src["xcentroid"]))
        y = int(round(src["ycentroid"]))

        if x < 0 or y < 0 or y >= reg.shape[0] or x >= reg.shape[1]:
            continue

        real_pos = coordinates[n_reg][y, x]

        if len(get_iterable(real_pos)) == 2:
            rows.append({
                "LTP": ltp,
                "STP": stp,
                "IDX": id,
                "PEAK_VAL": float(src["peak"] + median),
                "X_COORD": int(real_pos[1]),
                "Y_COORD": int(real_pos[0]),
                "PRE_LABEL": "object",
                "INFO": "info",
                "REGION": cropped_region(full_image, real_pos[1], real_pos[0], lim),
                "FILENAME": filename
            })

    return rows


def crop_star_position(ltp, stp, id, df, star_df, image, lim, filename=""):
    for ids in range(len(star_df)):
        x = int(star_df["xsensor"].iloc[ids])
        y = int(star_df["ysensor"].iloc[ids])
        cropped = cropped_region(image, x, y, lim)
        df.loc[len(df)] = [ltp, stp, id, image[y, x], x, y, "star", "non detected", cropped, filename]

    return df


def star_comparison(dataframe, star_catalogue):
    values = len(dataframe)

    for idx in range(values):
        x1 = dataframe["X_COORD"].iloc[idx]
        y1 = dataframe["Y_COORD"].iloc[idx]

        for ids in range(len(star_catalogue)):
            x2 = star_catalogue["xsensor"].iloc[ids]
            y2 = star_catalogue["ysensor"].iloc[ids]
            dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            if dist < 10:
                dataframe.loc[dataframe.index[idx], "PRE_LABEL"] = "star"
                dataframe.loc[dataframe.index[idx], "INFO"] = str(star_catalogue["MAIN_ID"].iloc[ids])

    return dataframe, star_catalogue


def remove_similar_objects(pandas_df, threshold=5):
    if len(pandas_df) <= 1:
        return pandas_df

    elements = np.arange(len(pandas_df))
    remove_indexes = []

    for px, py in itertools.combinations(elements, 2):
        y = (pandas_df["Y_COORD"].iloc[px] - pandas_df["Y_COORD"].iloc[py]) ** 2
        x = (pandas_df["X_COORD"].iloc[px] - pandas_df["X_COORD"].iloc[py]) ** 2
        dist = np.sqrt(y + x)

        if dist < threshold:
            index = px if pandas_df["PEAK_VAL"].iloc[px] < pandas_df["PEAK_VAL"].iloc[py] else py
            remove_indexes.append(pandas_df.iloc[index].name)

    pandas_df = pandas_df.drop(remove_indexes)
    pandas_df = pandas_df.reset_index(drop=True)
    return pandas_df


def points_inside_fov(x, y, center, scale, factor=0.1):
    radius1 = (sf.METIS_fov_min + factor) / scale
    radius2 = (sf.METIS_fov - factor) / scale

    outside_first = (x - center[0]) ** 2 + (y - center[1]) ** 2 >= radius1 ** 2
    inside_second = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius2 ** 2

    return outside_first and inside_second


def remove_objects_fov(df, center, scale):
    vals = []

    for i in range(len(df)):
        x = df["X_COORD"].iloc[i]
        y = df["Y_COORD"].iloc[i]
        accept = points_inside_fov(x, y, center, scale)

        if not accept:
            vals.append(df.index[i])

    df = df.drop(vals)
    df = df.reset_index(drop=True)
    return df


def folder_reader(metis_folder, output_folder,
                  KERNEL_PATH, KERNEL_NAME, magnitude, uv, catalog,
                  size, overlapping, std, lim, filter="vl-image",
                  db_name="psf_metadata_sqlite.db"):
    """
    Extract point-of-interest in a given set of L0 images and save
    everything directly to SQLite.
    """

    DAOS = False

    sqlite_path = os.path.join(output_folder, db_name)

    init_sqlite_db(sqlite_path)

    conn = sqlite3.connect(sqlite_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")

    try:
        LTPs_folder = ut.sorter(get_iterable(os.listdir(metis_folder)))
        processed_counter = 0

        for ltp_folder in LTPs_folder:
            print(f"Starting analysis with: {ltp_folder}")

            stp_dir = os.path.join(metis_folder, ltp_folder)
            STPs_folder = ut.sorter(get_iterable(os.listdir(stp_dir)))

            for stp_folder in STPs_folder:
                print(f"\t{stp_folder}")

                fits_dir = os.path.join(metis_folder, ltp_folder, stp_folder)
                fits_folder = sorted(get_iterable(os.listdir(fits_dir)))
                fits_folder = [file for file in fits_folder if filter in file]

                if not fits_folder:
                    continue

                for idx, fits_file in enumerate(fits_folder):
                    print(f"\t\tFile {idx}: {fits_file}")

                    if already_processed(conn, ltp_folder, stp_folder, fits_file):
                        print(f"\t\tSkipping already processed file: {fits_file}")
                        continue

                    try:
                        full_path = os.path.join(fits_dir, fits_file)

                        timestamp, headers, image = ut.fits_loader(full_path)

                        stars_detected, et, scale, center = sf.star_detector_offline(
                            KERNEL_NAME, KERNEL_PATH, timestamp, uv, catalog, magnitude
                        )

                        timestamp = sf.et2utc(et)

                        regions, coordinates = image_slicer(image, size, overlapping)

                        if DAOS:
                            peaks_rows = peak_detector_main_DAOS(
                                ltp_folder, stp_folder, idx, regions, coordinates,
                                std, image, lim, size, headers.get("FILENAME", fits_file)
                            )
                        else:
                            peaks_rows = peak_detector_main(
                                ltp_folder, stp_folder, idx, regions, coordinates,
                                std, image, lim, size, headers.get("FILENAME", fits_file)
                            )

                        peaks = pd.DataFrame(peaks_rows)

                        if len(peaks) > 0 and len(stars_detected) > 0:
                            peaks, _ = star_comparison(peaks, stars_detected)

                        if len(peaks) > 0:
                            peaks = remove_similar_objects(peaks, 5)
                            peaks = remove_objects_fov(peaks, center, scale)

                        if len(peaks) > 0:
                            peaks_rows = peaks.to_dict(orient="records")
                            save_peaks_to_sql(peaks_rows, conn)

                        n_stars = len(peaks[peaks["PRE_LABEL"] == "star"]) if len(peaks) > 0 else 0
                        n_objects = len(peaks[peaks["PRE_LABEL"] == "object"]) if len(peaks) > 0 else 0

                        save_processed_fits(
                            conn=conn,
                            headers=headers,
                            ltp=ltp_folder,
                            stp=stp_folder,
                            idx=idx,
                            timestamp=timestamp,
                            n_stars=n_stars,
                            n_objects=n_objects
                        )

                        processed_counter += 1

                        if processed_counter % 20 == 0:
                            conn.commit()
                            print(f"\t\tCommitted after {processed_counter} processed FITS files.")

                    except Exception as e:
                        save_failed_file(conn, ltp_folder, stp_folder, fits_file, e)
                        conn.commit()
                        print(f"\t\tError processing {fits_file}: {e}")

        conn.commit()

    finally:
        conn.close()