from collections import defaultdict
import glob
import logging
import numpy as np
import os
from PIL import Image
import torch


logging.basicConfig(format='%(asctime)s %(name)s:%(lineno)d %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_image(inputfile):
    """Load image from a file handle.

    Returns None if the image is not RGB, as apparently some aren't.
    """
    im = Image.open(inputfile)
    # im = im.convert(mode='L')
    if im.mode != 'RGB':
        # logger.debug('Image using {}, not RGB'.format(im.mode))
        im = im.convert(mode='RGB')
        # return None
    data = torch.FloatTensor(np.array(im, dtype=np.float32)).permute(2, 0, 1) / 255
    return data


def load_paths(paths):
    """Load images from the specified paths.

    Returns a torch tensor, arranged in the same order as the paths that were
    passed in.
    """
    images = [load_image(p) for p in paths]
    return torch.cat([im.unsqueeze(0) for im in images if im is not None])


def tinyimagenet_train_paths(traindir):
    output = {}
    objdirs = os.listdir(traindir)  # dir name is also value
    for objdir in objdirs:
        IMGDIR = 'images'
        imgpath = os.path.join(traindir, objdir, IMGDIR)
        fileglob = os.path.join(imgpath, '*')
        paths = glob.glob(fileglob)
        output[objdir] = paths
    return output


def build_parallel_paths(valuedict):
    pathlist = []
    for ls in valuedict.values():
        pathlist += ls
    labels = []
    for lbl, ls in valuedict.items():
        labels.extend([lbl] * len(ls))
    return pathlist, torch.LongTensor(labels)


def load_from_dir(dirpath, ext=None):
    imgs = []
    fileglob = os.path.join(dirpath, '*.' + ext if ext else '*')
    paths = glob.iglob(fileglob)
    for path in paths:
        with open(path, 'rb') as imgfile:
            data = load_image(imgfile)
            if data is not None:
                imgs.append((path, load_image(path)))
            else:
                logger.warn('Bad image: {}'.format(path))
    return imgs


def load_tinyimagenet_train(traindir):
    output = {}
    objdirs = os.listdir(traindir)  # dir name is also value
    for objdir in objdirs:
        IMGDIR = 'images'
        imgs = load_from_dir(os.path.join(traindir, objdir, IMGDIR))
        imgarr = torch.zeros((len(imgs),) + imgs[0][1].size())
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
        combined = torch.zeros((len(imglist),) + imglist[0].size())
        logger.debug(str(imglist[0].shape))
        for idx, img in enumerate(imglist):
            combined[idx, :, :, :] = img
        output[lbl] = combined
    return output


def build_parallel_arrays(valuedict):
    imgarray = torch.cat(valuedict.values())
    labels = []
    for lbl, data in valuedict.items():
        labels.extend([lbl] * data.shape[0])
    return imgarray, torch.LongTensor(labels)


def encode_numeric(valuedict):
    enc = {}
    encoded = 0
    dec = []
    for k in valuedict:
        enc[k] = encoded
        dec.append(k)
        encoded += 1
    transformed = {enc[k]: v for k, v in valuedict.items()}
    return transformed, enc, dec


def shuffle(*arrays):
    state = np.random.get_state()
    if len(arrays) == 0:
        return
    np.random.shuffle(arrays[0])
    for arr in arrays[1:]:
        np.random.set_state(state)
        np.random.shuffle(arr)
