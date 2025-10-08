import os
import sys
from .stu_upload.layers_1 import FullyConnectedLayer, ReLULayer, SoftmaxLossLayer
from .stu_upload.mnist_mlp_cpu import MNIST_MLP, build_mnist_mlp
import numpy as np
import struct
import time

def evaluate(mlp):
    pred_results = np.zeros([mlp.test_data.shape[0]])
    full_batches = mlp.test_data.shape[0] // mlp.batch_size
    for idx in range(full_batches):
        batch_images = mlp.test_data[idx*mlp.batch_size:(idx+1)*mlp.batch_size, :-1]
        start = time.time()
        prob = mlp.forward(batch_images)
        end = time.time()
        print("inferencing time: %f" % (end-start))
        pred_labels = np.argmax(prob, axis=1)
        pred_results[idx*mlp.batch_size:(idx+1)*mlp.batch_size] = pred_labels
    remainder = mlp.test_data.shape[0] % mlp.batch_size
    if remainder > 0:
        last_batch_start = mlp.test_data.shape[0] - remainder
        batch_images = mlp.test_data[last_batch_start:, :-1]
        prob = mlp.forward(batch_images)
        pred_labels = np.argmax(prob, axis=1)
        pred_results[last_batch_start:] = pred_labels
    accuracy = np.mean(pred_results == mlp.test_data[:,-1])
    print('Accuracy in test set: %f' % accuracy)

def run_test():
    mlp = build_mnist_mlp('weight.npy')
    evaluate(mlp)

if __name__ == '__main__':
    run_test()
