# from logging import getLogger, Formatter, StreamHandler, INFO
# from pathlib import Path
# import argparse
# import math
# import glob
# import warnings

# import click
# import scipy
# import tqdm
# import tables as tb
# import pandas as pd
# import numpy as np
# import skimage.transform
# from skimage.draw import polygon
# import rasterio
# import shapely.wkt

from logging import getLogger, Formatter, StreamHandler, INFO, FileHandler
from pathlib import Path
import subprocess
import argparse
import math
import glob
import sys
import json
import re
import warnings

import scipy
import tqdm
import click
import tables as tb
import pandas as pd
import numpy as np
from keras.models import Model
# from keras.engine.topology import merge as merge_l
from keras.layers import merge as merge_l
from keras.layers import (
    Input, Convolution2D, MaxPooling2D, UpSampling2D,
    Reshape, core, Dropout,
    Activation, BatchNormalization)
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, History
from keras import backend as K
import skimage.transform
import skimage.morphology
import rasterio.features
import shapely.wkt
import shapely.ops
import shapely.geometry
import theano

MY_DIRECTORY = '/home/dhanraj/Documents/Semester_3_2/Machine_Learning/LAB/practice/BuildingDetectors_Round2/SpaceNet_Buildings_Competition_Round2_Sample'


ORIGINAL_SIZE = 650
INPUT_SIZE = 256

BASE_TRAIN_DIR = MY_DIRECTORY  
WORKING_DIR = MY_DIRECTORY + "/working"
IMAGE_DIR = MY_DIRECTORY + "/working/images/dhanraj"


# Parameters
MIN_POLYGON_AREA = 30

#---------------------------------------------------------------


MODEL_NAME = 'dhanraj'

MODEL_DIR = MY_DIRECTORY + "/working/models/{}".format(MODEL_NAME)

FMT_VALTRAIN_IMAGELIST_PATH = IMAGE_DIR + "/{prefix:s}_valtrain_ImageId.csv"

FMT_VALMODEL_PATH = MODEL_DIR + "/{}_val_weights.h5"

FMT_VALMODEL_LAST_PATH = MODEL_DIR + "/{}_val_weights_last.h5"

FMT_VALMODEL_HIST = MODEL_DIR + "/{}_val_hist.csv"

FMT_VALMODEL_EVALHIST = MODEL_DIR + "/{}_val_evalhist.csv"

FMT_VALMODEL_EVALTHHIST = MODEL_DIR + "/{}_val_evalhist_th.csv"

FMT_MULMEAN = IMAGE_DIR + "/{}_mulmean.h5"

FMT_TESTPOLY_PATH = MODEL_DIR + "/{}_poly.csv"

FMT_TESTPRED_PATH = MODEL_DIR + "/{}_pred.h5"

FMT_VALTRAIN_IMAGELIST_PATH = IMAGE_DIR + "/{prefix:s}_valtrain_ImageId.csv"
FMT_VALTEST_IMAGELIST_PATH = IMAGE_DIR + "/{prefix:s}_valtest_ImageId.csv"

FMT_VALTEST_MASK_STORE = IMAGE_DIR + "/valtest_{}_mask.h5"
FMT_VALTRAIN_MUL_STORE = IMAGE_DIR + "/valtrain_{}_mul.h5"
FMT_VALTRAIN_MASK_STORE = IMAGE_DIR + "/valtrain_{}_mask.h5"
FMT_VALTEST_MUL_STORE = IMAGE_DIR + "/valtest_{}_mul.h5"

FMT_VALTESTPRED_PATH = MODEL_DIR + "/{}_eval_pred.h5"
FMT_VALTESTPOLY_PATH = MODEL_DIR + "/{}_eval_poly.csv"
FMT_VALTESTTRUTH_PATH = MODEL_DIR + "/{}_eval_poly_truth.csv"

#-----------------------------------------------------------------
FMT_IMAGE_RGBPAN_PATH = str(
    Path("{datapath:s}/") /
    Path("RGB-PanSharpen/")
        )
FMT_IMAGE_MUL_PATH = str(
    Path("{datapath:s}/") /
    Path("MUL/")
        )
FMT_IMAGE_PAN_PATH = str(
    Path("{datapath:s}/") /
    Path("PAN/")
        )
FMT_IMAGE_MULPAN_PATH = str(
    Path("{datapath:s}/") /
    Path("MUL-PanSharpen/")
        )

# Input files
FMT_TRAIN_SUMMARY_PATH = str(
    Path(BASE_TRAIN_DIR) /
    Path("{prefix:s}_Train/") /
    Path("summaryData/{prefix:s}_Train_Building_Solutions.csv"))
FMT_TRAIN_RGB_IMAGE_PATH = str(
    Path("{datapath:s}/") /
    Path("RGB-PanSharpen/RGB-PanSharpen_{image_id:s}.tif"))
FMT_TEST_RGB_IMAGE_PATH = str(
    Path("{datapath:s}/") /
    Path("RGB-PanSharpen/RGB-PanSharpen_{image_id:s}.tif"))
FMT_TRAIN_MSPEC_IMAGE_PATH = str(
    Path("{datapath:s}/") /
    Path("MUL-PanSharpen/MUL-PanSharpen_{image_id:s}.tif"))
FMT_TEST_MSPEC_IMAGE_PATH = str(
    Path("{datapath:s}/") /
    Path("MUL-PanSharpen/MUL-PanSharpen_{image_id:s}.tif"))

# Preprocessing result
FMT_RGB_BANDCUT_TH_PATH = IMAGE_DIR + "/rgb_bandcut{}.csv"
FMT_MUL_BANDCUT_TH_PATH = IMAGE_DIR + "/mul_bandcut{}.csv"

# Image list, Image container and mask container
FMT_VALTRAIN_IMAGELIST_PATH = IMAGE_DIR + "/{prefix:s}_valtrain_ImageId.csv"
FMT_VALTRAIN_MASK_STORE = IMAGE_DIR + "/valtrain_{}_mask.h5"
FMT_VALTRAIN_IM_STORE = IMAGE_DIR + "/valtrain_{}_im.h5"
FMT_VALTRAIN_MUL_STORE = IMAGE_DIR + "/valtrain_{}_mul.h5"

FMT_VALTEST_IMAGELIST_PATH = IMAGE_DIR + "/{prefix:s}_valtest_ImageId.csv"
FMT_VALTEST_MASK_STORE = IMAGE_DIR + "/valtest_{}_mask.h5"
FMT_VALTEST_IM_STORE = IMAGE_DIR + "/valtest_{}_im.h5"
FMT_VALTEST_MUL_STORE = IMAGE_DIR + "/valtest_{}_mul.h5"

FMT_IMMEAN = IMAGE_DIR + "/{}_immean.h5"
FMT_MULMEAN = IMAGE_DIR + "/{}_mulmean.h5"

FMT_TEST_IMAGELIST_PATH = IMAGE_DIR + "/{prefix:s}_test_ImageId.csv"
FMT_TEST_IM_STORE = IMAGE_DIR + "/test_{}_im.h5"
FMT_TEST_MUL_STORE = IMAGE_DIR + "/test_{}_mul.h5"

# Logger
warnings.simplefilter("ignore", UserWarning)
handler = StreamHandler()
handler.setLevel(INFO)
handler.setFormatter(Formatter('%(asctime)s %(levelname)s %(message)s'))

logger = getLogger('spacenet2')
logger.setLevel(INFO)


if __name__ == '__main__':
    logger.addHandler(handler)


# Fix seed for reproducibility
np.random.seed(1145141919)


def directory_name_to_area_id(datapath):
    """
    Directory name to AOI number

    Usage:

        >>> directory_name_to_area_id("/data/test/AOI_2_Vegas")
        2
    """
    dir_name = Path(datapath).name
    if dir_name.startswith('AOI_2_Vegas'):
        return 2
    elif dir_name.startswith('AOI_3_Paris'):
        return 3
    elif dir_name.startswith('AOI_4_Shanghai'):
        return 4
    elif dir_name.startswith('AOI_5_Khartoum'):
        return 5
    else:
        raise RuntimeError("Unsupported city id is given.")


def calc_rgb_multiband_cut_threshold(area_id, datapath):
    rows = []
    band_cut_th = __calc_rgb_multiband_cut_threshold(area_id, datapath)
    prefix = area_id_to_prefix(area_id)
    row = dict(prefix=area_id_to_prefix(area_id))
    row['area_id'] = area_id
    for chan_i in band_cut_th.keys():
        row['chan{}_max'.format(chan_i)] = band_cut_th[chan_i]['max']
        row['chan{}_min'.format(chan_i)] = band_cut_th[chan_i]['min']
    rows.append(row)
    pd.DataFrame(rows).to_csv(
        FMT_RGB_BANDCUT_TH_PATH.format(prefix), index=False)


def __calc_rgb_multiband_cut_threshold(area_id, datapath):
    prefix = area_id_to_prefix(area_id)
    band_values = {k: [] for k in range(3)}
    band_cut_th = {k: dict(max=0, min=0) for k in range(3)}

    image_id_list = pd.read_csv(FMT_VALTRAIN_IMAGELIST_PATH.format(
        prefix=prefix)).ImageId.tolist()
    for image_id in tqdm.tqdm(image_id_list[:500]):
        image_fn = get_train_image_path_from_imageid(image_id, datapath)
        logger.info("rasterio file path {}".format(image_fn))
        with rasterio.open(image_fn, 'r') as f:
            values = f.read().astype(np.float32)
            for i_chan in range(3):
                values_ = values[i_chan].ravel().tolist()
                values_ = np.array(
                    [v for v in values_ if v != 0]
                )  # Remove sensored mask
                band_values[i_chan].append(values_)

    image_id_list = pd.read_csv(FMT_VALTEST_IMAGELIST_PATH.format(
        prefix=prefix)).ImageId.tolist()
    for image_id in tqdm.tqdm(image_id_list[:500]):
        image_fn = get_train_image_path_from_imageid(image_id, datapath)
        with rasterio.open(image_fn, 'r') as f:
            values = f.read().astype(np.float32)
            for i_chan in range(3):
                values_ = values[i_chan].ravel().tolist()
                values_ = np.array(
                    [v for v in values_ if v != 0]
                )  # Remove sensored mask
                band_values[i_chan].append(values_)

    logger.info("Calc percentile point ...")
    for i_chan in range(3):
        band_values[i_chan] = np.concatenate(
            band_values[i_chan]).ravel()
        band_cut_th[i_chan]['max'] = scipy.percentile(
            band_values[i_chan], 98)
        band_cut_th[i_chan]['min'] = scipy.percentile(
            band_values[i_chan], 2)
    return band_cut_th


def calc_mul_multiband_cut_threshold(area_id, datapath):
    rows = []
    band_cut_th = __calc_mul_multiband_cut_threshold(area_id, datapath)
    prefix = area_id_to_prefix(area_id)
    row = dict(prefix=area_id_to_prefix(area_id))
    row['area_id'] = area_id
    for chan_i in band_cut_th.keys():
        row['chan{}_max'.format(chan_i)] = band_cut_th[chan_i]['max']
        row['chan{}_min'.format(chan_i)] = band_cut_th[chan_i]['min']
    rows.append(row)
    pd.DataFrame(rows).to_csv(
        FMT_MUL_BANDCUT_TH_PATH.format(prefix),
        index=False)


def __calc_mul_multiband_cut_threshold(area_id, datapath):
    prefix = area_id_to_prefix(area_id)
    band_values = {k: [] for k in range(8)}
    band_cut_th = {k: dict(max=0, min=0) for k in range(8)}

    image_id_list = pd.read_csv(FMT_VALTRAIN_IMAGELIST_PATH.format(
        prefix=prefix)).ImageId.tolist()
    for image_id in tqdm.tqdm(image_id_list[:500]):
        image_fn = get_train_image_path_from_imageid(
            image_id, datapath, mul=True)
        with rasterio.open(image_fn, 'r') as f:
            values = f.read().astype(np.float32)
            for i_chan in range(8):
                values_ = values[i_chan].ravel().tolist()
                values_ = np.array(
                    [v for v in values_ if v != 0]
                )  # Remove sensored mask
                band_values[i_chan].append(values_)

    image_id_list = pd.read_csv(FMT_VALTEST_IMAGELIST_PATH.format(
        prefix=prefix)).ImageId.tolist()
    for image_id in tqdm.tqdm(image_id_list[:500]):
        image_fn = get_train_image_path_from_imageid(
            image_id, datapath, mul=True)
        with rasterio.open(image_fn, 'r') as f:
            values = f.read().astype(np.float32)
            for i_chan in range(8):
                values_ = values[i_chan].ravel().tolist()
                values_ = np.array(
                    [v for v in values_ if v != 0]
                )  # Remove sensored mask
                band_values[i_chan].append(values_)

    logger.info("Calc percentile point ...")
    for i_chan in range(8):
        band_values[i_chan] = np.concatenate(
            band_values[i_chan]).ravel()
        band_cut_th[i_chan]['max'] = scipy.percentile(
            band_values[i_chan], 98)
        band_cut_th[i_chan]['min'] = scipy.percentile(
            band_values[i_chan], 2)
    return band_cut_th


def image_mask_resized_from_summary(df, image_id):
    im_mask = np.zeros((650, 650))

    if len(df[df.ImageId == image_id]) == 0:
        raise RuntimeError("ImageId not found on summaryData: {}".format(
            image_id))

    for idx, row in df[df.ImageId == image_id].iterrows():
        shape_obj = shapely.wkt.loads(row.PolygonWKT_Pix)
        if shape_obj.exterior is not None:
            coords = list(shape_obj.exterior.coords)
            x = np.array([round(float(pp[0])) for pp in coords])
            y = np.array([round(float(pp[1])) for pp in coords])
            yy, xx = skimage.draw.polygon(y, x, (650, 650))
            im_mask[yy, xx] = 1

            interiors = shape_obj.interiors
            for interior in interiors:
                coords = list(interior.coords)
                x = np.array([round(float(pp[0])) for pp in coords])
                y = np.array([round(float(pp[1])) for pp in coords])
                yy, xx = skimage.draw.polygon(y, x, (650, 650))
                im_mask[yy, xx] = 0
    im_mask = skimage.transform.resize(im_mask, (INPUT_SIZE, INPUT_SIZE))
    im_mask = (im_mask > 0.5).astype(np.uint8)
    return im_mask


def generate_test_mul_image_prep(area_id):
    prefix = area_id_to_prefix(area_id)

    df_test = pd.read_csv(
        FMT_TEST_IMAGELIST_PATH.format(prefix=prefix),
        index_col='ImageId')
    band_mul_th = __load_band_cut_th(
        FMT_MUL_BANDCUT_TH_PATH.format(prefix), bandsz=8)[area_id]

    fn = FMT_TEST_MUL_STORE.format(prefix)
    logger.info("Prepare image container: {}".format(fn))
    with tb.open_file(fn, 'w') as f:
        for image_id in tqdm.tqdm(df_test.index, total=len(df_test)):
            im = get_resized_raster_8chan_image_test(
                image_id, band_rgb_th, band_mul_th)
            atom = tb.Atom.from_dtype(im.dtype)
            filters = tb.Filters(complib='blosc', complevel=9)
            ds = f.create_carray(f.root, image_id, atom, im.shape,
                                 filters=filters)
            ds[:] = im


def prep_image_mask(area_id, is_valtrain=True):
    prefix = area_id_to_prefix(area_id)
    logger.info("prep_image_mask for {}".format(prefix))
    if is_valtrain:
        fn_list = FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix)
        fn_mask = FMT_VALTRAIN_MASK_STORE.format(prefix)
    else:
        fn_list = FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
        fn_mask = FMT_VALTEST_MASK_STORE.format(prefix)

    df = pd.read_csv(fn_list, index_col='ImageId')
    df_summary = _load_train_summary_data(area_id)
    logger.info("Prepare image container: {}".format(fn_mask))
    with tb.open_file(fn_mask, 'w') as f:
        for image_id in tqdm.tqdm(df.index, total=len(df)):
            im_mask = image_mask_resized_from_summary(df_summary, image_id)
            atom = tb.Atom.from_dtype(im_mask.dtype)
            filters = tb.Filters(complib='blosc', complevel=9)
            print (" dhanraj image_id = ", image_id)
            ds = f.create_carray(f.root, image_id, atom, im_mask.shape,
                                 filters=filters)
            ds[:] = im_mask


def prep_rgb_image_store_train(area_id, datapath, is_valtrain=True):
    prefix = area_id_to_prefix(area_id)
    bandstats = __load_rgb_bandstats(area_id)

    logger.info("prep_rgb_image_store_train for {}".format(prefix))
    if is_valtrain:
        fn_list = FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix)
        fn_store = FMT_VALTRAIN_IM_STORE.format(prefix)
    else:
        fn_list = FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
        fn_store = FMT_VALTEST_IM_STORE.format(prefix)

    df_list = pd.read_csv(fn_list, index_col='ImageId')

    logger.info("Image store file: {}".format(fn_store))
    with tb.open_file(fn_store, 'w') as f:
        for image_id in tqdm.tqdm(df_list.index, total=len(df_list)):
            im = get_resized_3chan_image_train(image_id, datapath, bandstats)
            atom = tb.Atom.from_dtype(im.dtype)
            filters = tb.Filters(complib='blosc', complevel=9)
            ds = f.create_carray(f.root, image_id, atom, im.shape,
                                 filters=filters)
            ds[:] = im


def prep_rgb_image_store_test(area_id, datapath):
    prefix = area_id_to_prefix(area_id)
    bandstats = __load_rgb_bandstats(area_id)

    logger.info("prep_rgb_image_store_test for {}".format(prefix))
    fn_list = FMT_TEST_IMAGELIST_PATH.format(prefix=prefix)
    fn_store = FMT_TEST_IM_STORE.format(prefix)
    df_list = pd.read_csv(fn_list, index_col='ImageId')

    logger.info("Image store file: {}".format(fn_store))
    with tb.open_file(fn_store, 'w') as f:
        for image_id in tqdm.tqdm(df_list.index, total=len(df_list)):
            im = get_resized_3chan_image_test(image_id, datapath, bandstats)
            atom = tb.Atom.from_dtype(im.dtype)
            filters = tb.Filters(complib='blosc', complevel=9)
            ds = f.create_carray(f.root, image_id, atom, im.shape,
                                 filters=filters)
            ds[:] = im


def get_resized_3chan_image_train(image_id, datapath, bandstats):
    fn = get_train_image_path_from_imageid(image_id, datapath)
    with rasterio.open(fn, 'r') as f:
        values = f.read().astype(np.float32)
        for chan_i in range(3):
            min_val = bandstats[chan_i]['min']
            max_val = bandstats[chan_i]['max']
            values[chan_i] = np.clip(values[chan_i], min_val, max_val)
            values[chan_i] = (values[chan_i] - min_val) / (max_val - min_val)

    values = np.swapaxes(values, 0, 2)
    values = np.swapaxes(values, 0, 1)
    values = skimage.transform.resize(values, (INPUT_SIZE, INPUT_SIZE))
    return values


def get_resized_3chan_image_test(image_id, datapath, bandstats):
    fn = get_test_image_path_from_imageid(image_id, datapath)
    with rasterio.open(fn, 'r') as f:
        values = f.read().astype(np.float32)
        for chan_i in range(3):
            min_val = bandstats[chan_i]['min']
            max_val = bandstats[chan_i]['max']
            values[chan_i] = np.clip(values[chan_i], min_val, max_val)
            values[chan_i] = (values[chan_i] - min_val) / (max_val - min_val)

    values = np.swapaxes(values, 0, 2)
    values = np.swapaxes(values, 0, 1)
    values = skimage.transform.resize(values, (INPUT_SIZE, INPUT_SIZE))
    return values


def prep_mul_image_store_train(area_id, datapath, is_valtrain=True):
    prefix = area_id_to_prefix(area_id)
    bandstats_rgb = __load_rgb_bandstats(area_id)
    bandstats_mul = __load_mul_bandstats(area_id)

    logger.info("prep_mul_image_store_train for ".format(prefix))
    if is_valtrain:
        fn_list = FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix)
        fn_store = FMT_VALTRAIN_MUL_STORE.format(prefix)
    else:
        fn_list = FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
        fn_store = FMT_VALTEST_MUL_STORE.format(prefix)

    df_list = pd.read_csv(fn_list, index_col='ImageId')

    logger.info("Image store file: {}".format(fn_store))
    with tb.open_file(fn_store, 'w') as f:
        for image_id in tqdm.tqdm(df_list.index, total=len(df_list)):
            im = get_resized_8chan_image_train(
                image_id, datapath, bandstats_rgb, bandstats_mul)
            atom = tb.Atom.from_dtype(im.dtype)
            filters = tb.Filters(complib='blosc', complevel=9)
            ds = f.create_carray(f.root, image_id, atom, im.shape,
                                 filters=filters)
            ds[:] = im


def prep_mul_image_store_test(area_id, datapath):
    prefix = area_id_to_prefix(area_id)
    bandstats_rgb = __load_rgb_bandstats(area_id)
    bandstats_mul = __load_mul_bandstats(area_id)

    logger.info("prep_mul_image_store_test for ".format(prefix))
    fn_list = FMT_TEST_IMAGELIST_PATH.format(prefix=prefix)
    fn_store = FMT_TEST_MUL_STORE.format(prefix)

    df_list = pd.read_csv(fn_list, index_col='ImageId')

    logger.info("Image store file: {}".format(fn_store))
    with tb.open_file(fn_store, 'w') as f:
        for image_id in tqdm.tqdm(df_list.index, total=len(df_list)):
            im = get_resized_8chan_image_test(
                image_id, datapath, bandstats_rgb, bandstats_mul)
            atom = tb.Atom.from_dtype(im.dtype)
            filters = tb.Filters(complib='blosc', complevel=9)
            ds = f.create_carray(f.root, image_id, atom, im.shape,
                                 filters=filters)
            ds[:] = im


def get_resized_8chan_image_train(image_id, datapath, bs_rgb, bs_mul):
    im = []

    fn = get_train_image_path_from_imageid(image_id, datapath)
    with rasterio.open(fn, 'r') as f:
        values = f.read().astype(np.float32)
        for chan_i in range(3):
            min_val = bs_rgb[chan_i]['min']
            max_val = bs_rgb[chan_i]['max']
            values[chan_i] = np.clip(values[chan_i], min_val, max_val)
            values[chan_i] = (values[chan_i] - min_val) / (max_val - min_val)
            im.append(skimage.transform.resize(
                values[chan_i],
                (INPUT_SIZE, INPUT_SIZE)))

    fn = get_train_image_path_from_imageid(image_id, datapath, mul=True)
    with rasterio.open(fn, 'r') as f:
        values = f.read().astype(np.float32)
        usechannels = [1, 2, 5, 6, 7]
        for chan_i in usechannels:
            min_val = bs_mul[chan_i]['min']
            max_val = bs_mul[chan_i]['max']
            values[chan_i] = np.clip(values[chan_i], min_val, max_val)
            values[chan_i] = (values[chan_i] - min_val) / (max_val - min_val)
            im.append(skimage.transform.resize(
                values[chan_i],
                (INPUT_SIZE, INPUT_SIZE)))

    im = np.array(im)  # (ch, w, h)
    im = np.swapaxes(im, 0, 2)  # -> (h, w, ch)
    im = np.swapaxes(im, 0, 1)  # -> (w, h, ch)
    return im


def get_resized_8chan_image_test(image_id, datapath, bs_rgb, bs_mul):
    im = []

    fn = get_test_image_path_from_imageid(image_id, datapath)
    with rasterio.open(fn, 'r') as f:
        values = f.read().astype(np.float32)
        for chan_i in range(3):
            min_val = bs_rgb[chan_i]['min']
            max_val = bs_rgb[chan_i]['max']
            values[chan_i] = np.clip(values[chan_i], min_val, max_val)
            values[chan_i] = (values[chan_i] - min_val) / (max_val - min_val)
            im.append(skimage.transform.resize(
                values[chan_i],
                (INPUT_SIZE, INPUT_SIZE)))

    fn = get_test_image_path_from_imageid(image_id, datapath, mul=True)
    with rasterio.open(fn, 'r') as f:
        values = f.read().astype(np.float32)
        usechannels = [1, 2, 5, 6, 7]
        for chan_i in usechannels:
            min_val = bs_mul[chan_i]['min']
            max_val = bs_mul[chan_i]['max']
            values[chan_i] = np.clip(values[chan_i], min_val, max_val)
            values[chan_i] = (values[chan_i] - min_val) / (max_val - min_val)
            im.append(skimage.transform.resize(
                values[chan_i],
                (INPUT_SIZE, INPUT_SIZE)))

    im = np.array(im)  # (ch, w, h)
    im = np.swapaxes(im, 0, 2)  # -> (h, w, ch)
    im = np.swapaxes(im, 0, 1)  # -> (w, h, ch)
    return im


def _load_train_summary_data(area_id):
    prefix = area_id_to_prefix(area_id)
    fn = FMT_TRAIN_SUMMARY_PATH.format(prefix=prefix)
    df = pd.read_csv(fn)
    # df.loc[:, 'ImageId'] = df.ImageId.str[4:]
    return df


def __load_rgb_bandstats(area_id):
    """
    Usage:

        >>> __load_rgb_bandstats(3)
        {
          0: {
              'max': 462.0,
              'min': 126.0,
          },
          1: {
              'max': 481.0,
              'min': 223.0,
          },
          2: {
              'max': 369.0,
              'min': 224.0,
          },
        }
    """
    prefix = area_id_to_prefix(area_id)
    fn_stats = FMT_RGB_BANDCUT_TH_PATH.format(prefix)
    df_stats = pd.read_csv(fn_stats, index_col='area_id')
    r = df_stats.loc[area_id]

    stats_dict = {}
    for chan_i in range(3):
        stats_dict[chan_i] = dict(
            min=r['chan{}_min'.format(chan_i)],
            max=r['chan{}_max'.format(chan_i)])
    return stats_dict


def __load_mul_bandstats(area_id):
    prefix = area_id_to_prefix(area_id)
    fn_stats = FMT_MUL_BANDCUT_TH_PATH.format(prefix)
    df_stats = pd.read_csv(fn_stats, index_col='area_id')
    r = df_stats.loc[area_id]

    stats_dict = {}
    for chan_i in range(8):
        stats_dict[chan_i] = dict(
            min=r['chan{}_min'.format(chan_i)],
            max=r['chan{}_max'.format(chan_i)])
    return stats_dict


def __load_band_cut_th(band_fn):
    df = pd.read_csv(band_fn, index_col='area_id')
    all_band_cut_th = {area_id: {} for area_id in range(2, 6)}
    for area_id, row in df.iterrows():
        for chan_i in range(3):
            all_band_cut_th[area_id][chan_i] = dict(
                min=row['chan{}_min'.format(chan_i)],
                max=row['chan{}_max'.format(chan_i)],
            )
    return all_band_cut_th


def prep_valtrain_valtest_imagelist(area_id):
    prefix = area_id_to_prefix(area_id)
    df = _load_train_summary_data(area_id)
    df_agg = df.groupby('ImageId').agg('first')

    image_id_list = df_agg.index.tolist()
    np.random.shuffle(image_id_list)
    sz_valtrain = int(len(image_id_list) * 0.7)
    sz_valtest = len(image_id_list) - sz_valtrain

    base_dir = Path(FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix)).parent
    if not base_dir.exists():
        base_dir.mkdir(parents=True)

    pd.DataFrame({'ImageId': image_id_list[:sz_valtrain]}).to_csv(
        FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix),
        index=False)
    pd.DataFrame({'ImageId': image_id_list[sz_valtrain:]}).to_csv(
        FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix),
        index=False)


def prep_test_imagelist(area_id, datapath):
    prefix = area_id_to_prefix(area_id)

    image_id_list = glob.glob(str(
        Path(datapath) /
        Path("./PAN/PAN_{prefix:s}_*.tif")).format(prefix=prefix))
    image_id_list = [path.split("PAN_")[-1][:-4] for path in image_id_list]
    pd.DataFrame({'ImageId': image_id_list}).to_csv(
        FMT_TEST_IMAGELIST_PATH.format(prefix=prefix),
        index=False)


def get_train_image_path_from_imageid(image_id, datapath, mul=False):
    prefix = image_id_to_prefix(image_id)
    if mul:
        return FMT_TRAIN_MSPEC_IMAGE_PATH.format(
            datapath=datapath, prefix=prefix, image_id=image_id)
    else:
        return FMT_TRAIN_RGB_IMAGE_PATH.format(
            datapath=datapath, prefix=prefix, image_id=image_id)


def get_test_image_path_from_imageid(image_id, datapath, mul=False):
    prefix = image_id_to_prefix(image_id)
    if mul:
        return FMT_TEST_MSPEC_IMAGE_PATH.format(
            datapath=datapath, image_id=image_id)
    else:
        return FMT_TEST_RGB_IMAGE_PATH.format(
            datapath=datapath, image_id=image_id)


def image_id_to_prefix(image_id):
    """
    `AOI_3_Paris_img585` -> `AOI_3_Paris`
    """
    prefix = image_id.split('img')[0][:-1]
    return prefix


def prefix_to_area_id(prefix):
    area_dict = {
        'AOI_2_Vegas': 2,
        'AOI_3_Paris': 3,
        'AOI_4_Shanghai': 4,
        'AOI_5_Khartoum': 5,
    }
    return area_dict[area_id]


def area_id_to_prefix(area_id):
    area_dict = {
        2: 'AOI_2_Vegas',
        3: 'AOI_3_Paris',
        4: 'AOI_4_Shanghai',
        5: 'AOI_5_Khartoum',
    }
    return area_dict[area_id]


def prep_immean(area_id, datapath):
    prefix = area_id_to_prefix(area_id)
    X_train = []

    # Load valtrain
    fn_im = FMT_VALTRAIN_IM_STORE.format(prefix)
    image_list = pd.read_csv(FMT_VALTRAIN_IMAGELIST_PATH.format(
        prefix=prefix)).ImageId.tolist()
    with tb.open_file(fn_im, 'r') as f:
        for idx, image_id in enumerate(image_list):
            im = np.array(f.get_node('/' + image_id))
            im = np.swapaxes(im, 0, 2)
            im = np.swapaxes(im, 1, 2)
            X_train.append(im)

    # Load valtest
    fn_im = FMT_VALTEST_IM_STORE.format(prefix)
    image_list = pd.read_csv(FMT_VALTEST_IMAGELIST_PATH.format(
        prefix=prefix)).ImageId.tolist()
    with tb.open_file(fn_im, 'r') as f:
        for idx, image_id in enumerate(image_list):
            im = np.array(f.get_node('/' + image_id))
            im = np.swapaxes(im, 0, 2)
            im = np.swapaxes(im, 1, 2)
            X_train.append(im)

    X_mean = np.array(X_train).mean(axis=0)

    fn = FMT_IMMEAN.format(prefix)
    logger.info("Prepare mean image: {}".format(fn))
    with tb.open_file(fn, 'w') as f:
        atom = tb.Atom.from_dtype(X_mean.dtype)
        filters = tb.Filters(complib='blosc', complevel=9)
        ds = f.create_carray(f.root, 'immean', atom, X_mean.shape,
                             filters=filters)
        ds[:] = X_mean


def prep_mulmean(area_id):
    prefix = area_id_to_prefix(area_id)
    X_train = []

    # Load valtrain
    fn_im = FMT_VALTRAIN_MUL_STORE.format(prefix)
    image_list = pd.read_csv(FMT_VALTRAIN_IMAGELIST_PATH.format(
        prefix=prefix)).ImageId.tolist()
    with tb.open_file(fn_im, 'r') as f:
        for idx, image_id in enumerate(image_list):
            im = np.array(f.get_node('/' + image_id))
            im = np.swapaxes(im, 0, 2)
            im = np.swapaxes(im, 1, 2)
            X_train.append(im)

    # Load valtest
    fn_im = FMT_VALTEST_MUL_STORE.format(prefix)
    image_list = pd.read_csv(FMT_VALTEST_IMAGELIST_PATH.format(
        prefix=prefix)).ImageId.tolist()
    with tb.open_file(fn_im, 'r') as f:
        for idx, image_id in enumerate(image_list):
            im = np.array(f.get_node('/' + image_id))
            im = np.swapaxes(im, 0, 2)
            im = np.swapaxes(im, 1, 2)
            X_train.append(im)

    X_mean = np.array(X_train).mean(axis=0)

    fn = FMT_MULMEAN.format(prefix)
    logger.info("Prepare mean image: {}".format(fn))
    with tb.open_file(fn, 'w') as f:
        atom = tb.Atom.from_dtype(X_mean.dtype)
        filters = tb.Filters(complib='blosc', complevel=9)
        ds = f.create_carray(f.root, 'mulmean', atom, X_mean.shape,
                             filters=filters)
        ds[:] = X_mean


# My method
# def get_unet():
#     conv_params = dict(activation='relu', border_mode='same')
#     merge_params = dict(mode='concat', concat_axis=0)
#     inputs = Input((8,256,256))
#     conv1 = Convolution2D(32, 3, 3, **conv_params)(inputs)
#     conv1 = Convolution2D(32, 3, 3, **conv_params)(conv1)
#     print("conv1.shape = ",conv1.shape)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#     print("pool1.shape = ",pool1.shape)

#     conv2 = Convolution2D(64, 3, 3, **conv_params)(pool1)
#     conv2 = Convolution2D(64, 3, 3, **conv_params)(conv2)
#     print("conv2.shape = ",conv2.shape)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#     print("pool2.shape = ",pool2.shape)

#     conv3 = Convolution2D(128, 3, 3, **conv_params)(pool2)
#     conv3 = Convolution2D(128, 3, 3, **conv_params)(conv3)
#     print("conv3.shape = ",conv3.shape)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#     print("pool3.shape = ",pool3.shape)

#     conv4 = Convolution2D(256, 3, 3, **conv_params)(pool3)
#     conv4 = Convolution2D(256, 3, 3, **conv_params)(conv4)
#     print("conv4.shape = ",conv4.shape)
#     pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
#     print("pool4.shape = ",pool4.shape)

#     conv5 = Convolution2D(256, 3, 3, **conv_params)(pool4)
#     conv5 = Convolution2D(256, 3, 3, **conv_params)(conv5)
#     print("conv5.shape = ",conv5.shape)

#     up6 = merge_l([UpSampling2D(size=(2, 2))(conv5), conv4], **merge_params)
#     print("up6.shape = ",up6.shape)
#     conv6 = Convolution2D(128, 3, 3, **conv_params)(up6)
#     conv6 = Convolution2D(128, 3, 3, **conv_params)(conv6)
#     print("conv6.shape = ",conv6.shape)

#     up7 = merge_l([UpSampling2D(size=(2, 2))(conv6), conv3], **merge_params)
#     conv7 = Convolution2D(64, 3, 3, **conv_params)(up7)
#     conv7 = Convolution2D(64, 3, 3, **conv_params)(conv7)
#     print("conv7.shape = ",conv7.shape)

#     up8 = merge_l([UpSampling2D(size=(2, 2))(conv7), conv2], **merge_params)
#     conv8 = Convolution2D(32, 3, 3, **conv_params)(up8)
#     conv8 = Convolution2D(32, 3, 3, **conv_params)(conv8)
#     print("conv8.shape = ",conv8.shape)

#     up9 = merge_l([UpSampling2D(size=(2, 2))(conv8), conv1], **merge_params)
#     conv9 = Convolution2D(32, 3, 3, **conv_params)(up9)
#     conv9 = Convolution2D(32, 3, 3, **conv_params)(conv9)

#     conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)
#     adam = Adam()

#     model = Model(input=inputs, output=conv10)
#     model.compile(optimizer=adam,
#                   loss='binary_crossentropy',
#                   metrics=['accuracy', jaccard_coef, jaccard_coef_int])
#     return model


# their's method
def get_unet():
    conv_params = dict(activation='relu', border_mode='same')
    merge_params = dict(mode='concat', concat_axis=1)
    inputs = Input((8, 256, 256))
    conv1 = Convolution2D(32, 3, 3, **conv_params)(inputs)
    conv1 = Convolution2D(32, 3, 3, **conv_params)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, **conv_params)(pool1)
    conv2 = Convolution2D(64, 3, 3, **conv_params)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, **conv_params)(pool2)
    conv3 = Convolution2D(128, 3, 3, **conv_params)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, **conv_params)(pool3)
    conv4 = Convolution2D(256, 3, 3, **conv_params)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, **conv_params)(pool4)
    conv5 = Convolution2D(512, 3, 3, **conv_params)(conv5)

    up6 = merge_l([UpSampling2D(size=(2, 2))(conv5), conv4], **merge_params)
    conv6 = Convolution2D(256, 3, 3, **conv_params)(up6)
    conv6 = Convolution2D(256, 3, 3, **conv_params)(conv6)

    up7 = merge_l([UpSampling2D(size=(2, 2))(conv6), conv3], **merge_params)
    conv7 = Convolution2D(128, 3, 3, **conv_params)(up7)
    conv7 = Convolution2D(128, 3, 3, **conv_params)(conv7)

    up8 = merge_l([UpSampling2D(size=(2, 2))(conv7), conv2], **merge_params)
    conv8 = Convolution2D(64, 3, 3, **conv_params)(up8)
    conv8 = Convolution2D(64, 3, 3, **conv_params)(conv8)

    up9 = merge_l([UpSampling2D(size=(2, 2))(conv8), conv1], **merge_params)
    conv9 = Convolution2D(32, 3, 3, **conv_params)(up9)
    conv9 = Convolution2D(32, 3, 3, **conv_params)(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)
    adam = Adam()

    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=adam,
                  loss='binary_crossentropy',
                  metrics=['accuracy', jaccard_coef, jaccard_coef_int])
    return model


def jaccard_coef(y_true, y_pred):
    smooth = 1e-12
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
   # print("y_true = ",y_true," y_pred = ",y_pred)
    smooth = 1e-12
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)


def _get_valtrain_mul_data(area_id):
    prefix = area_id_to_prefix(area_id)
    fn_train = FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix)
    df_train = pd.read_csv(fn_train)

    X_val = []
    fn_im = FMT_VALTRAIN_MUL_STORE.format(prefix)
    with tb.open_file(fn_im, 'r') as f:
        for idx, image_id in enumerate(df_train.ImageId.tolist()):
            im = np.array(f.get_node('/' + image_id))
            im = np.swapaxes(im, 0, 2)
            im = np.swapaxes(im, 1, 2)
            X_val.append(im)
    X_val = np.array(X_val)

    y_val = []
    fn_mask = FMT_VALTRAIN_MASK_STORE.format(prefix)
    with tb.open_file(fn_mask, 'r') as f:
        for idx, image_id in enumerate(df_train.ImageId.tolist()):
            mask = np.array(f.get_node('/' + image_id))
            mask = (mask > 0.5).astype(np.uint8)
            y_val.append(mask)
    y_val = np.array(y_val)
    y_val = y_val.reshape((-1, 1, INPUT_SIZE, INPUT_SIZE))

    return X_val, y_val

def _get_valtest_mul_data(area_id):
    prefix = area_id_to_prefix(area_id)
    fn_test = FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
    df_test = pd.read_csv(fn_test)

    X_val = []
    fn_im = FMT_VALTEST_MUL_STORE.format(prefix)
    with tb.open_file(fn_im, 'r') as f:
        for idx, image_id in enumerate(df_test.ImageId.tolist()):
            im = np.array(f.get_node('/' + image_id))
            im = np.swapaxes(im, 0, 2)
            im = np.swapaxes(im, 1, 2)
            X_val.append(im)
    X_val = np.array(X_val)

    y_val = []
    fn_mask = FMT_VALTEST_MASK_STORE.format(prefix)
    with tb.open_file(fn_mask, 'r') as f:
        for idx, image_id in enumerate(df_test.ImageId.tolist()):
            mask = np.array(f.get_node('/' + image_id))
            # mask = np.array(f.get_node(image_id))
            mask = (mask > 0.5).astype(np.uint8)
            y_val.append(mask)
    y_val = np.array(y_val)
    y_val = y_val.reshape((-1, 1, INPUT_SIZE, INPUT_SIZE))

    return X_val, y_val


def get_mul_mean_image(area_id):
    prefix = area_id_to_prefix(area_id)

    with tb.open_file(FMT_MULMEAN.format(prefix), 'r') as f:
        im_mean = np.array(f.get_node('/mulmean'))
    return im_mean


def _internal_validate_fscore_wo_pred_file(area_id,
                                           epoch=3,
                                           min_th=MIN_POLYGON_AREA,
                                           enable_tqdm=False):
    prefix = area_id_to_prefix(area_id)

    # Prediction phase
    logger.info("Prediction phase")
    y_pred = _internal_validate_predict(
        area_id,
        epoch=epoch,
        save_pred=True,
        enable_tqdm=enable_tqdm)

    # print("y_pred = ",y_pred)

    # Postprocessing phase
    logger.info("Postprocessing phase")
    fn_test = FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
    df_test = pd.read_csv(fn_test, index_col='ImageId')

    fn = FMT_VALTESTPRED_PATH.format(prefix)
    fn_out = FMT_VALTESTPOLY_PATH.format(prefix)
    with open(fn_out, 'w') as f:
        f.write("ImageId,BuildingId,PolygonWKT_Pix,Confidence\n")
        test_list = df_test.index.tolist()
        iterator = enumerate(test_list)

        for idx, image_id in tqdm.tqdm(iterator, total=len(test_list)):
            print("y_pred[",idx,"][0] = ",y_pred[idx][0])
            df_poly = mask_to_poly(y_pred[idx][0], min_polygon_area_th=min_th)
            print("length of df_poly = ",len(df_poly))
            if len(df_poly) > 0:
                for i, row in df_poly.iterrows():
                    line = "{},{},\"{}\",{:.6f}\n".format(
                        image_id,
                        row.bid,
                        row.wkt,
                        row.area_ratio)
                    line = _remove_interiors(line)
                    f.write(line)
            else:
                f.write("{},{},{},0\n".format(
                    image_id,
                    -1,
                    "POLYGON EMPTY"))

    # ------------------------
    # Validation solution file
    logger.info("Validation solution file")
    fn_true = FMT_TRAIN_SUMMARY_PATH.format(prefix=prefix)
    df_true = pd.read_csv(fn_true)

    # # Remove prefix "PAN_"
    # df_true.loc[:, 'ImageId'] = df_true.ImageId.str[4:]

    fn_test = FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
    df_test = pd.read_csv(fn_test)
    df_test_image_ids = df_test.ImageId.unique()

    fn_out = FMT_VALTESTTRUTH_PATH.format(prefix)
    with open(fn_out, 'w') as f:
        f.write("ImageId,BuildingId,PolygonWKT_Pix,Confidence\n")
        df_true = df_true[df_true.ImageId.isin(df_test_image_ids)]
        for idx, r in df_true.iterrows():
            f.write("{},{},\"{}\",{:.6f}\n".format(
                r.ImageId,
                r.BuildingId,
                r.PolygonWKT_Pix,
                1.0))


def _internal_validate_predict(area_id,
                               epoch=3,
                               save_pred=True,
                               enable_tqdm=False):
    prefix = area_id_to_prefix(area_id)
    X_mean = get_mul_mean_image(area_id)

    # Load model weights
    # Predict and Save prediction result
    fn_model = FMT_VALMODEL_PATH.format(prefix + '_{epoch:02d}')
    fn_model = fn_model.format(epoch=epoch)
    model = get_unet()
    model.load_weights(fn_model)

    fn_test = FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
    df_test = pd.read_csv(fn_test, index_col='ImageId')

    y_pred = model.predict_generator(
        generate_valtest_batch(
            area_id,
            batch_size=64,
            immean=X_mean,
            enable_tqdm=enable_tqdm,
        ),
        val_samples=len(df_test),
    )
    del model

    # Save prediction result
    if save_pred:
        fn = FMT_VALTESTPRED_PATH.format(prefix)
        with tb.open_file(fn, 'w') as f:
            atom = tb.Atom.from_dtype(y_pred.dtype)
            filters = tb.Filters(complib='blosc', complevel=9)
            ds = f.create_carray(f.root,
                                 'pred',
                                 atom,
                                 y_pred.shape,
                                 filters=filters)
            ds[:] = y_pred
    return y_pred


def mask_to_poly(mask, min_polygon_area_th=MIN_POLYGON_AREA):
    """
    Convert from 256x256 mask to polygons on 650x650 image
    """
    mask = (skimage.transform.resize(mask, (650, 650)) > 0.3).astype(np.uint8)
    shapes = rasterio.features.shapes(mask.astype(np.int16), mask > 0)
    # poly_list = []
    df = []
    mp = shapely.ops.cascaded_union(
        shapely.geometry.MultiPolygon([
            shapely.geometry.shape(shape)
            for shape, value in shapes
        ]))

    # print("mp = ", mp)

    if isinstance(mp, shapely.geometry.Polygon):
        df = pd.DataFrame({
            'area_size': [mp.area],
            'poly': [mp],
        })
    else:
        df = pd.DataFrame({
            'area_size': [p.area for p in mp],
            'poly': [p for p in mp],
        })

    df = df[df.area_size > min_polygon_area_th].sort_values(
        by='area_size', ascending=False)
    df.loc[:, 'wkt'] = df.poly.apply(lambda x: shapely.wkt.dumps(
        x, rounding_precision=0))
    df.loc[:, 'bid'] = list(range(1, len(df) + 1))
    df.loc[:, 'area_ratio'] = df.area_size / df.area_size.max()
    # print("mask_to_poly, df = ", df)
    return df

def _remove_interiors(line):
    if "), (" in line:
        line_prefix = line.split('), (')[0]
        line_terminate = line.split('))",')[-1]
        line = (
            line_prefix +
            '))",' +
            line_terminate
        )
    return line


def generate_valtest_batch(area_id,
                           batch_size=8,
                           immean=None,
                           enable_tqdm=False):
    prefix = area_id_to_prefix(area_id)
    df_train = pd.read_csv(FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix))
    fn_im = FMT_VALTEST_MUL_STORE.format(prefix)
    fn_mask = FMT_VALTEST_MASK_STORE.format(prefix)

    image_id_list = df_train.ImageId.tolist()

    if enable_tqdm:
        pbar = tqdm.tqdm(total=len(image_id_list))

    while 1:
        total_sz = len(image_id_list)
        n_batch = int(math.floor(total_sz / batch_size) + 1)
        with tb.open_file(fn_im, 'r') as f_im,\
                tb.open_file(fn_mask, 'r') as f_mask:
            for i_batch in range(n_batch):
                target_image_ids = image_id_list[
                    i_batch*batch_size:(i_batch+1)*batch_size
                ]
                if len(target_image_ids) == 0:
                    continue

                X_train = []
                y_train = []
                for image_id in target_image_ids:
                    im = np.array(f_im.get_node('/' + image_id))
                    im = np.swapaxes(im, 0, 2)
                    im = np.swapaxes(im, 1, 2)

                    X_train.append(im)
                    mask = np.array(f_mask.get_node('/' + image_id))
                    mask = (mask > 0).astype(np.uint8)
                    y_train.append(mask)
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                y_train = y_train.reshape((-1, 1, INPUT_SIZE, INPUT_SIZE))

                if immean is not None:
                    X_train = X_train - immean

                if enable_tqdm:
                    pbar.update(y_train.shape[0])

                yield (X_train, y_train)

        if enable_tqdm:
            pbar.close()



def _calc_fscore_per_aoi(area_id):
    prefix = area_id_to_prefix(area_id)
    truth_file = FMT_VALTESTTRUTH_PATH.format(prefix)
    poly_file = FMT_VALTESTPOLY_PATH.format(prefix)

    # cmd = [
    #     'java',
    #     '-jar',
    #     '/home/dhanraj/Documents/Semester_3_2/Machine_Learning/LAB/practice/BuildingDetectors_Round2/1-XD_XD/visualizer-2.0/visualizer.jar',
    #     '-truth',
    #     truth_file,
    #     '-solution',
    #     poly_file,
    #     '-no-gui',
    #     '-band-triplets',
    #     '/home/dhanraj/Documents/Semester_3_2/Machine_Learning/LAB/practice/BuildingDetectors_Round2/1-XD_XD/visualizer-2.0/data/band-triplets.txt',
    #     '-image-dir',
    #     'pass',
    # ]

    cmd = [
        'java',
        '-jar',
        '/home/dhanraj/Documents/Semester_3_2/Machine_Learning/LAB/practice/BuildingDetectors_Round2/1-XD_XD/visualizer-2.0/visualizer.jar',
        '-truth',
        truth_file,
        '-solution',
        poly_file,
        '-band-triplets',
        '/home/dhanraj/Documents/Semester_3_2/Machine_Learning/LAB/practice/BuildingDetectors_Round2/1-XD_XD/visualizer-2.0/data/band-triplets.txt',
        '-image-dir',
        '/home/dhanraj/Documents/Semester_3_2/Machine_Learning/LAB/practice/BuildingDetectors_Round2/SpaceNet_Buildings_Competition_Round2_Sample/dhanraj',
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout_data, stderr_data = proc.communicate()
    lines = [line for line in stdout_data.decode('utf8').split('\n')[-10:]]

    # print(stdout_data)

    """
Overall F-score : 0.85029

AOI_2_Vegas:
  TP       : 27827
  FP       : 4999
  FN       : 4800
  Precision: 0.847712
  Recall   : 0.852883
  F-score  : 0.85029
    """

    if stdout_data.decode('utf8').strip().endswith("Overall F-score : 0"):
        overall_fscore = 0
        tp = 0
        fp = 0
        fn = 0
        precision = 0
        recall = 0
        fscore = 0

    elif len(lines) > 0 and lines[0].startswith("Overall F-score : "):
        assert lines[0].startswith("Overall F-score : ")
        assert lines[2].startswith("AOI_")
        assert lines[3].strip().startswith("TP")
        assert lines[4].strip().startswith("FP")
        assert lines[5].strip().startswith("FN")
        assert lines[6].strip().startswith("Precision")
        assert lines[7].strip().startswith("Recall")
        assert lines[8].strip().startswith("F-score")

        overall_fscore = float(re.findall("([\d\.]+)", lines[0])[0])
        tp = int(re.findall("(\d+)", lines[3])[0])
        fp = int(re.findall("(\d+)", lines[4])[0])
        fn = int(re.findall("(\d+)", lines[5])[0])
        precision = float(re.findall("([\d\.]+)", lines[6])[0])
        recall = float(re.findall("([\d\.]+)", lines[7])[0])
        fscore = float(re.findall("([\d\.]+)", lines[8])[0])
    else:
        logger.warn("Unexpected data >>> " + stdout_data.decode('utf8'))
        raise RuntimeError("Unsupported format")

    return {
        'overall_fscore': overall_fscore,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'fscore': fscore,
    }



def _internal_validate_fscore(area_id,
                              epoch=3,
                              predict=True,
                              min_th=MIN_POLYGON_AREA,
                              enable_tqdm=False):
    prefix = area_id_to_prefix(area_id)

    # Prediction phase
    logger.info("Prediction phase")
    if predict:
        _internal_validate_predict(
            area_id,
            epoch=epoch,
            enable_tqdm=enable_tqdm)

    # Postprocessing phase
    logger.info("Postprocessing phase")
    fn_test = FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
    df_test = pd.read_csv(fn_test, index_col='ImageId')

    fn = FMT_VALTESTPRED_PATH.format(prefix)
    fn_out = FMT_VALTESTPOLY_PATH.format(prefix)
    with open(fn_out, 'w') as f,\
            tb.open_file(fn, 'r') as fr:

        y_pred = np.array(fr.get_node('/pred'))

        f.write("ImageId,BuildingId,PolygonWKT_Pix,Confidence\n")
        test_list = df_test.index.tolist()
        iterator = enumerate(test_list)

        for idx, image_id in tqdm.tqdm(iterator, total=len(test_list)):
            df_poly = mask_to_poly(y_pred[idx][0], min_polygon_area_th=min_th)
            if len(df_poly) > 0:
                for i, row in df_poly.iterrows():
                    line = "{},{},\"{}\",{:.6f}\n".format(
                        image_id,
                        row.bid,
                        row.wkt,
                        row.area_ratio)
                    line = _remove_interiors(line)
                    f.write(line)
            else:
                f.write("{},{},{},0\n".format(
                    image_id,
                    -1,
                    "POLYGON EMPTY"))

    # ------------------------
    # Validation solution file
    logger.info("Validation solution file")
    # if not Path(FMT_VALTESTTRUTH_PATH.format(prefix)).exists():
    if True:
        fn_true = FMT_TRAIN_SUMMARY_PATH.format(prefix=prefix)
        df_true = pd.read_csv(fn_true)
        # # Remove prefix "PAN_"
        # df_true.loc[:, 'ImageId'] = df_true.ImageId.str[4:]

        fn_test = FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
        df_test = pd.read_csv(fn_test)
        df_test_image_ids = df_test.ImageId.unique()

        fn_out = FMT_VALTESTTRUTH_PATH.format(prefix)
        with open(fn_out, 'w') as f:
            f.write("ImageId,BuildingId,PolygonWKT_Pix,Confidence\n")
            df_true = df_true[df_true.ImageId.isin(df_test_image_ids)]
            for idx, r in df_true.iterrows():
                f.write("{},{},\"{}\",{:.6f}\n".format(
                    r.ImageId,
                    r.BuildingId,
                    r.PolygonWKT_Pix,
                    1.0))


def _internal_test(area_id, enable_tqdm=False):
    prefix = area_id_to_prefix(area_id)
    y_pred = _internal_test_predict_best_param(area_id, save_pred=False)

    param = _get_model_parameter(area_id)
    min_th = param['min_poly_area']

    # Postprocessing phase
    logger.info("Postprocessing phase")
    fn_test = FMT_TEST_IMAGELIST_PATH.format(prefix=prefix)
    df_test = pd.read_csv(fn_test, index_col='ImageId')

    fn_out = FMT_TESTPOLY_PATH.format(prefix)
    with open(fn_out, 'w') as f:
        f.write("ImageId,BuildingId,PolygonWKT_Pix,Confidence\n")

        test_image_list = df_test.index.tolist()
        for idx, image_id in tqdm.tqdm(enumerate(test_image_list),
                                       total=len(test_image_list)):
            df_poly = mask_to_poly(y_pred[idx][0], min_polygon_area_th=min_th)
            if len(df_poly) > 0:
                for i, row in df_poly.iterrows():
                    line = "{},{},\"{}\",{:.6f}\n".format(
                        image_id,
                        row.bid,
                        row.wkt,
                        row.area_ratio)
                    line = _remove_interiors(line)
                    f.write(line)
            else:
                f.write("{},{},{},0\n".format(
                    image_id,
                    -1,
                    "POLYGON EMPTY"))



def _internal_test_predict_best_param(area_id,
                                      save_pred=True):
    prefix = area_id_to_prefix(area_id)
    param = _get_model_parameter(area_id)
    epoch = param['fn_epoch']
    min_th = param['min_poly_area']

    # Prediction phase
    logger.info("Prediction phase: {}".format(prefix))

    X_mean = get_mul_mean_image(area_id)

    # Load model weights
    # Predict and Save prediction result
    fn = FMT_TESTPRED_PATH.format(prefix)
    fn_model = FMT_VALMODEL_PATH.format(prefix + '_{epoch:02d}')
    fn_model = fn_model.format(epoch=epoch)
    model = get_unet()
    model.load_weights(fn_model)

    fn_test = FMT_TEST_IMAGELIST_PATH.format(prefix=prefix)
    df_test = pd.read_csv(fn_test, index_col='ImageId')

    y_pred = model.predict_generator(
        generate_test_batch(
            area_id,
            batch_size=64,
            immean=X_mean,
            enable_tqdm=True,
        ),
        val_samples=len(df_test),
    )
    del model

    # Save prediction result
    if save_pred:
        with tb.open_file(fn, 'w') as f:
            atom = tb.Atom.from_dtype(y_pred.dtype)
            filters = tb.Filters(complib='blosc', complevel=9)
            ds = f.create_carray(f.root, 'pred', atom, y_pred.shape,
                                 filters=filters)
            ds[:] = y_pred

    return y_pred


def _get_model_parameter(area_id):
    prefix = area_id_to_prefix(area_id)
    fn_hist = FMT_VALMODEL_EVALTHHIST.format(prefix)
    best_row = pd.read_csv(fn_hist).sort_values(
        by='fscore',
        ascending=False,
    ).iloc[0]

    param = dict(
        fn_epoch=int(best_row['zero_base_epoch']),
        min_poly_area=int(best_row['min_area_th']),
    )
    return param


def generate_test_batch(area_id,
                        batch_size=64,
                        immean=None,
                        enable_tqdm=False):
    prefix = area_id_to_prefix(area_id)
    df_test = pd.read_csv(FMT_TEST_IMAGELIST_PATH.format(prefix=prefix))
    fn_im = FMT_TEST_MUL_STORE.format(prefix)

    image_id_list = df_test.ImageId.tolist()

    if enable_tqdm:
        pbar = tqdm.tqdm(total=len(image_id_list))

    while 1:
        total_sz = len(image_id_list)
        n_batch = int(math.floor(total_sz / batch_size) + 1)
        with tb.open_file(fn_im, 'r') as f_im:
            for i_batch in range(n_batch):
                target_image_ids = image_id_list[
                    i_batch*batch_size:(i_batch+1)*batch_size
                ]
                if len(target_image_ids) == 0:
                    continue

                X_test = []
                y_test = []
                for image_id in target_image_ids:
                    im = np.array(f_im.get_node('/' + image_id))
                    im = np.swapaxes(im, 0, 2)
                    im = np.swapaxes(im, 1, 2)

                    X_test.append(im)
                    mask = np.zeros((INPUT_SIZE, INPUT_SIZE)).astype(np.uint8)
                    y_test.append(mask)
                X_test = np.array(X_test)
                y_test = np.array(y_test)
                y_test = y_test.reshape((-1, 1, INPUT_SIZE, INPUT_SIZE))

                if immean is not None:
                    X_test = X_test - immean

                if enable_tqdm:
                    pbar.update(y_test.shape[0])

                yield (X_test, y_test)

        if enable_tqdm:
            pbar.close()


# >>> -------------------------------------------------------------


@click.group()
def cli():
    pass


@cli.command()
@click.argument('datapath', type=str)
def preproc_train(datapath):
    """ train.sh """
    area_id = directory_name_to_area_id(datapath)
    prefix = area_id_to_prefix(area_id)
    logger.info("Preproc for training on {}".format(prefix))

    # Imagelist
    if Path(FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix)).exists():
        logger.info("Generate IMAGELIST csv ... skip")
    else:
        logger.info("Generate IMAGELIST csv")
        prep_valtrain_valtest_imagelist(area_id)
    if Path(FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)).exists():
        logger.info("Generate IMAGELIST csv ... skip")
    else:
        logger.info("Generate IMAGELIST csv")
        prep_valtrain_valtest_imagelist(area_id)

    # Band stats (RGB)
    if Path(FMT_RGB_BANDCUT_TH_PATH.format(prefix)).exists():
        logger.info("Generate band stats csv (RGB) ... skip")
    else:
        logger.info("Generate band stats csv (RGB)")
        calc_rgb_multiband_cut_threshold(area_id, datapath)

    # Band stats (MUL)
    if Path(FMT_MUL_BANDCUT_TH_PATH.format(prefix)).exists():
        logger.info("Generate band stats csv (MUL) ... skip")
    else:
        logger.info("Generate band stats csv (MUL)")
        calc_mul_multiband_cut_threshold(area_id, datapath)

    # Mask (Target output)
    if Path(FMT_VALTRAIN_MASK_STORE.format(prefix)).exists():
        logger.info("Generate MASK (valtrain) ... skip")
    else:
        logger.info("Generate MASK (valtrain)")
        prep_image_mask(area_id, is_valtrain=True)
    if Path(FMT_VALTEST_MASK_STORE.format(prefix)).exists():
        logger.info("Generate MASK (valtest) ... skip")
    else:
        logger.info("Generate MASK (valtest)")
        prep_image_mask(area_id, is_valtrain=False)

    # Image HDF5 store (RGB)
    if Path(FMT_VALTRAIN_IM_STORE.format(prefix)).exists():
        logger.info("Generate RGB_STORE (valtrain) ... skip")
    else:
        logger.info("Generate RGB_STORE (valtrain)")
        prep_rgb_image_store_train(area_id, datapath, is_valtrain=True)
    if Path(FMT_VALTEST_IM_STORE.format(prefix)).exists():
        logger.info("Generate RGB_STORE (valtest) ... skip")
    else:
        logger.info("Generate RGB_STORE (valtest)")
        prep_rgb_image_store_train(area_id, datapath, is_valtrain=False)

    # Image HDF5 store (MUL)
    if Path(FMT_VALTRAIN_MUL_STORE.format(prefix)).exists():
        logger.info("Generate MUL_STORE (valtrain) ... skip")
    else:
        logger.info("Generate MUL_STORE (valtrain)")
        prep_mul_image_store_train(area_id, datapath, is_valtrain=True)
    if Path(FMT_VALTEST_MUL_STORE.format(prefix)).exists():
        logger.info("Generate MUL_STORE (valtest) ... skip")
    else:
        logger.info("Generate MUL_STORE (valtest)")
        prep_mul_image_store_train(area_id, datapath, is_valtrain=False)

    # Image Mean (RGB)
    if Path(FMT_IMMEAN.format(prefix)).exists():
        logger.info("Generate RGBMEAN ... skip")
    else:
        logger.info("Generate RGBMEAN")
        prep_immean(area_id, datapath)

    # Image Mean (MUL)
    if Path(FMT_MULMEAN.format(prefix)).exists():
        logger.info("Generate MULMEAN ... skip")
    else:
        logger.info("Generate MULMEAN")
        prep_mulmean(area_id)

    # DONE!
    logger.info("Preproc for training on {} ... done".format(prefix))


@cli.command()
@click.argument('datapath', type=str)
def preproc_test(datapath):
    """ test.sh """
    area_id = directory_name_to_area_id(datapath)
    prefix = area_id_to_prefix(area_id)
    logger.info("preproc_test for {}".format(prefix))

    # Imagelist
    if Path(FMT_TEST_IMAGELIST_PATH.format(prefix=prefix)).exists():
        logger.info("Generate IMAGELIST for inference ... skip")
    else:
        logger.info("Generate IMAGELIST for inference")
        prep_test_imagelist(area_id, datapath)

    # Image HDF5 store (RGB)
    if Path(FMT_TEST_IM_STORE.format(prefix)).exists():
        logger.info("Generate RGB_STORE (test) ... skip")
    else:
        logger.info("Generate RGB_STORE (test)")
        prep_rgb_image_store_test(area_id, datapath)

    # Image HDF5 store (MUL)
    if Path(FMT_TEST_MUL_STORE.format(prefix)).exists():
        logger.info("Generate MUL_STORE (test) ... skip")
    else:
        logger.info("Generate MUL_STORE (test)")
        prep_mul_image_store_test(area_id, datapath)

    logger.info("preproc_test for {} ... done".format(prefix))




@cli.command()
@click.argument('datapath', type=str)
def validate(datapath):
    area_id = directory_name_to_area_id(datapath)
    prefix = area_id_to_prefix(area_id)
    logger.info(">> validate sub-command: {}".format(prefix))

    X_mean = get_mul_mean_image(area_id)
    X_val, y_val = _get_valtest_mul_data(area_id)
    X_val = X_val - X_mean

    if not Path(MODEL_DIR).exists():
        Path(MODEL_DIR).mkdir(parents=True)

    logger.info("load valtrain")
    X_trn, y_trn = _get_valtrain_mul_data(area_id)
    X_trn = X_trn - X_mean

    print("X_trn = ",X_trn," y_trn = ",y_trn)

    model = get_unet()
   # logger.info(FMT_VALMODEL_PATH.format(prefix + "_{epoch:02d}"))
    model_checkpoint = ModelCheckpoint(
        FMT_VALMODEL_PATH.format(prefix + "_{epoch:02d}"),
        monitor='val_jaccard_coef_int',
        save_best_only=False)
    model_earlystop = EarlyStopping(
        monitor='val_jaccard_coef_int',
        patience=10,
        verbose=1,
        mode='max')
    model_history = History()

    df_train = pd.read_csv(FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix))
    logger.info("Fit")
    model.fit(
        X_trn, y_trn,
        nb_epoch=12,
        shuffle=True,
        verbose=1,
        validation_data=(X_val, y_val),
        callbacks=[model_checkpoint, model_earlystop, model_history])
    model.save_weights(FMT_VALMODEL_LAST_PATH.format(prefix))

    # Save evaluation history
    pd.DataFrame(model_history.history).to_csv(
        FMT_VALMODEL_HIST.format(prefix), index=False)
    logger.info(">> validate sub-command: {} ... Done".format(prefix))



@cli.command()
@click.argument('datapath', type=str)
def evalfscore(datapath):
    area_id = directory_name_to_area_id(datapath)
    prefix = area_id_to_prefix(area_id)
    logger.info("Evaluate fscore on validation set: {}".format(prefix))

    # for each epoch
    # if not Path(FMT_VALMODEL_EVALHIST.format(prefix)).exists():
    if True:
        df_hist = pd.read_csv(FMT_VALMODEL_HIST.format(prefix))
        df_hist.loc[:, 'epoch'] = list(range(1, len(df_hist)+1))
        print("df_hist = ",df_hist)

        rows = []
        for zero_base_epoch in range(1, len(df_hist)+1):
            logger.info(">>> Epoch: {}".format(zero_base_epoch))
            _internal_validate_fscore_wo_pred_file(
                area_id,
                epoch=zero_base_epoch,
                min_th=MIN_POLYGON_AREA,
                enable_tqdm=True)
            evaluate_record = _calc_fscore_per_aoi(area_id)
            evaluate_record['zero_base_epoch'] = zero_base_epoch
            evaluate_record['min_area_th'] = MIN_POLYGON_AREA
            evaluate_record['area_id'] = area_id
            logger.info("\n" + json.dumps(evaluate_record, indent=4))
            rows.append(evaluate_record)

        pd.DataFrame(rows).to_csv(
            FMT_VALMODEL_EVALHIST.format(prefix),
            index=False)

    # find best min-poly-threshold
    df_evalhist = pd.read_csv(FMT_VALMODEL_EVALHIST.format(prefix))
    best_row = df_evalhist.sort_values(by='fscore', ascending=False).iloc[0]
    best_epoch = int(best_row.zero_base_epoch)
    best_fscore = best_row.fscore

    # optimize min area th
    rows = []
    for th in [30, 60, 90, 120, 150, 180, 210, 240]:
        logger.info(">>> TH: {}".format(th))
        predict_flag = False
        if th == 30:
            predict_flag = True

        _internal_validate_fscore(
            area_id,
            epoch=best_epoch,
            enable_tqdm=True,
            min_th=th,
            predict=predict_flag)
        evaluate_record = _calc_fscore_per_aoi(area_id)
        evaluate_record['zero_base_epoch'] = best_epoch
        evaluate_record['min_area_th'] = th
        evaluate_record['area_id'] = area_id
        logger.info("\n" + json.dumps(evaluate_record, indent=4))
        rows.append(evaluate_record)

    pd.DataFrame(rows).to_csv(
        FMT_VALMODEL_EVALTHHIST.format(prefix),
        index=False)

    logger.info("Evaluate fscore on validation set: {} .. done".format(prefix))


@cli.command()
@click.argument('datapath', type=str)
def testproc(datapath):
    area_id = directory_name_to_area_id(datapath)
    prefix = area_id_to_prefix(area_id)

    logger.info(">>>> Test proc for {}".format(prefix))
    _internal_test(area_id)
    logger.info(">>>> Test proc for {} ... done".format(prefix))


if __name__ == '__main__':
    cli()
