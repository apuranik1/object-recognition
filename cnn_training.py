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
    LayerSpec(64, 7, 1, PoolingTypes.NONE, 2, 2),
    LayerSpec(128, 7, 2, PoolingTypes.MAX, 2, 2),
    LayerSpec(256, 3, 1, PoolingTypes.NONE, 0, 0),
    ]


def train_classification_batch(network, data, target, loss, optimizer):
    optimizer.zero_grad()
    input_var = Variable(data)
    target_var = Variable(target)
    if CUDA:
        input_var = input_var.cuda()
        target_var = target_var.cuda()
    pred = network(input_var)
    # print(torch.cat((pred.data, target_var.view(-1, 1).data.float()), dim=1))
    # input()
    output = loss(pred, target_var)
    output.backward()
    optimizer.step()
    return output


def train_obj_detector(train_dict, val_dict, picture_dim, batch_size=128, lr=0.01, momentum=0.9, epochs=10):
    data, encoder, decoder = il.encode_numeric(train_dict)
    trainX, trainY = il.build_parallel_paths(data)
    # logger.debug('Training data: {}'.format(trainX.shape))
    il.shuffle(trainX, trainY)
    num_points = trainY.shape[0]
    logger.info('Number of classes: {}'.format(len(decoder)))
    network = ObjectRecognitionCNN(picture_dim[1:], picture_dim[0], CNN_SPEC, len(decoder))
    if CUDA:
        network = network.cuda()
    optimizer = torch.optim.Adam(network.parameters(), lr=lr) #, momentum=momentum)
    lossfunc = nn.CrossEntropyLoss()
    # logger.info('Starting optimization')
    for epoch in range(epochs):
        # sketchy ceiling division
        epoch_loss = 0
        batch_count = -(-num_points // batch_size)
        for batch in range(batch_count):
            # logger.info('Batch {}'.format(batch))
            start = batch * batch_size
            end = start + batch_size
            xBatch = il.load_paths(trainX[start:end])
            # logger.info('Input dimension: {}'.format(xBatch.shape))
            loss = train_classification_batch(network, xBatch,
                                              trainY[start:end], lossfunc,
                                              optimizer)
            del xBatch
            batch_loss = loss.sum()
            logger.info("Batch {} - loss: {}".format(batch, batch_loss))
            epoch_loss += batch_loss
        avg_loss = epoch_loss.data / batch_count
        # print status update
        logger.info('{} epochs completed. Training loss: {}'.format(epoch + 1, avg_loss))
    return network, encoder, decoder
