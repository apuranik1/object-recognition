import argh
import logging
import torch

import cnn_training
import image_loading as il


logging.basicConfig(format='%(asctime)s %(name)s:%(lineno)d %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def train(traindir, valdir, save_as):
    traindict = il.load_tinyimagenet_train(traindir)
    for v in traindict.values():
        im_dim = v.shape[1:]
        break
    # valdict = None
    # fuck validation for now
    network, enc, dec = cnn_training.train_obj_detector(traindict, None, im_dim)
    torch.save_model(network, save_as)  # TODO: switch this to saving only weights


if __name__ == '__main__':
    argh.dispatch_command(train)
