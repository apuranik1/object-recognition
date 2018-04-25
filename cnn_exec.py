import argh
import logging
import torch

import cnn_training
import image_loading as il


logging.basicConfig(format='%(asctime)s %(name)s:%(lineno)d %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def train(traindir, valdir, save_as):
    traindict = il.tinyimagenet_train_paths(traindir)
    im_dim = (3, 64, 64)
    # valdict = None
    # fuck validation for now
    network, enc, dec = cnn_training.train_obj_detector(traindict, None,
                                                        im_dim, batch_size=512,
                                                        epochs=100, lr=0.01)
    with open(save_as, 'wb') as fout:
        torch.save(network, fout)  # TODO: switch this to saving only weights


if __name__ == '__main__':
    argh.dispatch_command(train)
