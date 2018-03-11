from collections import defaultdict
import glob
import logging
import numpy as np
import os
from PIL import Image
import sys


logging.basicConfig(format='%(asctime)s %(name)s:%(lineno)d %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_image(datastream):
    """Load image from a file handle.

    Returns None if the image is not RGB, as apparently some aren't.
    """
    im = Image.open(datastream)
    if im.mode != 'RGB':
        logger.warn('Image using {}, not RGB'.format(im.mode))
        return None
    data = np.array(im.getdata(), dtype=np.float32).reshape(im.size + (3,))
    return data


def load_from_dir(dirpath, ext=None):
    imgs = []
    fileglob = os.path.join(dirpath, '*.' + ext if ext else '*')
    paths = glob.iglob(fileglob)
    for path in paths:
        with open(path, 'rb') as imgfile:
            data = load_image(path)
            if data is not None:
                imgs.append((path, load_image(path)))
            else:
                logger.warn('Bad image: {}'.path)
    return imgs


def load_tinyimagenet_train(traindir):
    output = {}
    objdirs = os.listdir(traindir)  # dir name is also value
    for objdir in objdirs:
        IMGDIR = 'images'
        imgs = load_from_dir(os.path.join(traindir, objdir, IMGDIR))
        imgarr = np.zeros((len(imgs),) + imgs[0][1].shape, dtype=np.float32)
        for idx, (_, img) in enumerate(imgs):
            imgarr[idx, :, :, :] = img
        output[objdir] = imgarr
    return output


def load_tinyimagenet_val(valdir):
    imgdict = defaultdict(list)
    VAL_LABELS = 'val_annotations.txt'
    labelfile = os.path.join(valdir, VAL_LABELS)
    with open(labelfile, 'r') as fin:
        for line in fin:
            filename, lbl = line.split('\t')[:2]
            logger.debug('Reading file: {}'.format(filename))
            with open(os.path.join(valdir, 'images', filename), 'rb') as fin:
                img = load_image(fin)
            if img is None:
                logger.warn('Bad image: {}'.format(filename))
            else:
                imgdict[lbl].append(img)
    output = {}
    for lbl, imglist in imgdict.items():
        combined = np.zeros((len(imglist),) + imglist[0].shape)
        logger.debug(str(imglist[0].shape))
        for idx, img in enumerate(imglist):
            combined[idx, :, :, :] = img
        output[lbl] = combined
    return output


def build_parallel_arrays(valuedict):
    imgarray = np.vstack(valuedict.values())
    labels = []
    for lbl, data in valuedict.items():
        labels.extend([lbl] * data.shape[0])
    return imgarray, labels


def shuffle(*arrays):
    state = np.random.get_state()
    if len(arrays) == 0:
        return
    np.random.shuffle(arrays[0])
    for arr in arrays[1:]:
        np.random.set_state(state)
        np.random.shuffle(arr)
