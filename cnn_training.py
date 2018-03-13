import logging
import torch
from torch import nn
from torch.autograd import Variable

from cnn_builder import ObjectRecognitionCNN, PoolingTypes, LayerSpec
import image_loading as il


logging.basicConfig(format='%(asctime)s %(name)s:%(lineno)d %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


CUDA = torch.cuda.is_available()


CNN_SPEC = [
    LayerSpec(16, 3, 1, PoolingTypes.MAX, 2, 2),
    LayerSpec(32, 3, 1, PoolingTypes.MAX, 2, 2),
    LayerSpec(64, 5, 2, PoolingTypes.NONE, 0, 0)
    ]


def train_classification_batch(network, data, target, loss, optimizer):
    optimizer.zero_grad()
    input_var = Variable(torch.FloatTensor(data))
    if CUDA:
        input_var = input_var.cuda()
    pred = network(input_var)
    output = loss(pred, target)
    output.backward()
    optimizer.step()
    return output


def train_obj_detector(train_dict, val_dict, picture_dim, batch_size=128, lr=0.01, momentum=0.9, epochs=10):
    data, encoder, decoder = il.encode_numeric(train_dict)
    trainX, trainY = il.build_parallel_arrays(data)
    logging.debug('Training data: {}'.format(trainX.shape))
    il.shuffle(trainX, trainY)
    num_points = trainY.shape[0]
    network = ObjectRecognitionCNN(picture_dim[1:], picture_dim[0], CNN_SPEC, len(encoder))
    if CUDA:
        network = network.cuda()
    optimizer = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9)
    lossfunc = nn.CrossEntropyLoss()
    logger.info('Starting optimization')
    for epoch in range(epochs):
        # sketchy ceiling division
        total_loss = 0
        for batch in range(-(-num_points // batch_size)):
            start = batch * batch_size
            end = start + batch_size
            xBatch = trainX[start:end]
            logger.info('Input dimension: {}'.format(xBatch.shape))
            loss = train_classification_batch(network, trainX[start:end],
                                              trainY[start:end], lossfunc,
                                              optimizer)
            total_loss += loss.sum()
        # print status update
        logger.info('{} epochs completed. Training loss: {}'.format(epoch + 1, total_loss))
    return network, encoder, decoder
