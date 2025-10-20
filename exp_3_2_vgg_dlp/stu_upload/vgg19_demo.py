# -*- coding: UTF-8 -*- 
import pycnnl
import time
import numpy as np
import os
import scipy.io

class VGG19(object):
    def __init__(self):
        # set up net
        
        self.net = pycnnl.CnnlNet()
        self.input_quant_params = []
        self.filter_quant_params = []

   
    def build_model(self, param_path='../../imagenet-vgg-verydeep-19.mat'):
        self.param_path = param_path

       
        # TODO: 使用net的createXXXLayer接口搭建VGG19网络
        # creating layers
        self.net.setInputShape(1, 3, 224, 224)
        # conv1_1
       
        input_shape1=pycnnl.IntVector(4)
        input_shape1[0]=1
        input_shape1[1]=3
        input_shape1[2]=224
        input_shape1[3]=224
        self.net.createConvLayer('conv1_1', input_shape1,64, 3, 1, 1, 1)
             
        # relu1_1
        self.net.createReLuLayer('relu1_1')
        # conv1_2
        
        input_shape12=pycnnl.IntVector(4)
        input_shape12[0]=1
        input_shape12[1]=64
        input_shape12[2]=224
        input_shape12[3]=224

        self.net.createConvLayer('conv1_2',input_shape12, 64, 3, 1, 1, 1)
        
        # relu1_2
        self.net.createReLuLayer('relu1_2')
        
        # pool1
        input_shape3 = pycnnl.IntVector(4)
        input_shape3[0] = 1
        input_shape3[1] = 64
        input_shape3[2] = 224
        input_shape3[3] = 224
        # 2x2核, 步长2
        self.net.createPoolingLayer('pool1', input_shape3, 2, 2)

        # conv2_1
        input_shape4 = pycnnl.IntVector(4)
        input_shape4[0] = 1
        input_shape4[1] = 64
        input_shape4[2] = 112
        input_shape4[3] = 112
        # (input_shape, out_channel, kernel_size, stride, pad, bias)
        self.net.createConvLayer('conv2_1', input_shape4, 128, 3, 1, 1, 1)
        # relu2_1
        self.net.createReLuLayer('relu2_1')
        
        # conv2_2
        input_shape5 = pycnnl.IntVector(4)
        input_shape5[0] = 1
        input_shape5[1] = 128
        input_shape5[2] = 112
        input_shape5[3] = 112
        self.net.createConvLayer('conv2_2', input_shape5, 128, 3, 1, 1, 1)
        # relu2_2
        self.net.createReLuLayer('relu2_2')
        
        # pool2
        input_shape6 = pycnnl.IntVector(4)
        input_shape6[0] = 1
        input_shape6[1] = 128
        input_shape6[2] = 112
        input_shape6[3] = 112
        self.net.createPoolingLayer('pool2', input_shape6, 2, 2)

        # conv3_1
        input_shape7 = pycnnl.IntVector(4)
        input_shape7[0] = 1
        input_shape7[1] = 128
        input_shape7[2] = 56
        input_shape7[3] = 56
        self.net.createConvLayer('conv3_1', input_shape7, 256, 3, 1, 1, 1)
        self.net.createReLuLayer('relu3_1')

        # conv3_2
        input_shape8 = pycnnl.IntVector(4)
        input_shape8[0] = 1
        input_shape8[1] = 256
        input_shape8[2] = 56
        input_shape8[3] = 56
        self.net.createConvLayer('conv3_2', input_shape8, 256, 3, 1, 1, 1)
        self.net.createReLuLayer('relu3_2')

        # conv3_3
        input_shape9 = pycnnl.IntVector(4)
        input_shape9[0] = 1
        input_shape9[1] = 256
        input_shape9[2] = 56
        input_shape9[3] = 56
        self.net.createConvLayer('conv3_3', input_shape9, 256, 3, 1, 1, 1)
        self.net.createReLuLayer('relu3_3')

        # conv3_4
        input_shape10 = pycnnl.IntVector(4)
        input_shape10[0] = 1
        input_shape10[1] = 256
        input_shape10[2] = 56
        input_shape10[3] = 56
        self.net.createConvLayer('conv3_4', input_shape10, 256, 3, 1, 1, 1)
        self.net.createReLuLayer('relu3_4')

        # pool3
        input_shape11 = pycnnl.IntVector(4)
        input_shape11[0] = 1
        input_shape11[1] = 256
        input_shape11[2] = 56
        input_shape11[3] = 56
        self.net.createPoolingLayer('pool3', input_shape11, 2, 2)

        # conv4_1
        input_shape12 = pycnnl.IntVector(4)
        input_shape12[0] = 1
        input_shape12[1] = 256
        input_shape12[2] = 28
        input_shape12[3] = 28
        self.net.createConvLayer('conv4_1', input_shape12, 512, 3, 1, 1, 1)
        self.net.createReLuLayer('relu4_1')

        # conv4_2
        input_shape13 = pycnnl.IntVector(4)
        input_shape13[0] = 1
        input_shape13[1] = 512
        input_shape13[2] = 28
        input_shape13[3] = 28
        self.net.createConvLayer('conv4_2', input_shape13, 512, 3, 1, 1, 1)
        self.net.createReLuLayer('relu4_2')

        # conv4_3
        input_shape14 = pycnnl.IntVector(4)
        input_shape14[0] = 1
        input_shape14[1] = 512
        input_shape14[2] = 28
        input_shape14[3] = 28
        self.net.createConvLayer('conv4_3', input_shape14, 512, 3, 1, 1, 1)
        self.net.createReLuLayer('relu4_3')

        # conv4_4
        input_shape15 = pycnnl.IntVector(4)
        input_shape15[0] = 1
        input_shape15[1] = 512
        input_shape15[2] = 28
        input_shape15[3] = 28
        self.net.createConvLayer('conv4_4', input_shape15, 512, 3, 1, 1, 1)
        self.net.createReLuLayer('relu4_4')

        # pool4
        input_shape16 = pycnnl.IntVector(4)
        input_shape16[0] = 1
        input_shape16[1] = 512
        input_shape16[2] = 28
        input_shape16[3] = 28
        self.net.createPoolingLayer('pool4', input_shape16, 2, 2)

        # conv5_1
        input_shape17 = pycnnl.IntVector(4)
        input_shape17[0] = 1
        input_shape17[1] = 512
        input_shape17[2] = 14
        input_shape17[3] = 14
        self.net.createConvLayer('conv5_1', input_shape17, 512, 3, 1, 1, 1)
        self.net.createReLuLayer('relu5_1')

        # conv5_2
        input_shape18 = pycnnl.IntVector(4)
        input_shape18[0] = 1
        input_shape18[1] = 512
        input_shape18[2] = 14
        input_shape18[3] = 14
        self.net.createConvLayer('conv5_2', input_shape18, 512, 3, 1, 1, 1)
        self.net.createReLuLayer('relu5_2')

        # conv5_3
        input_shape19 = pycnnl.IntVector(4)
        input_shape19[0] = 1
        input_shape19[1] = 512
        input_shape19[2] = 14
        input_shape19[3] = 14
        self.net.createConvLayer('conv5_3', input_shape19, 512, 3, 1, 1, 1)
        self.net.createReLuLayer('relu5_3')

        # conv5_4
        input_shape20 = pycnnl.IntVector(4)
        input_shape20[0] = 1
        input_shape20[1] = 512
        input_shape20[2] = 14
        input_shape20[3] = 14
        self.net.createConvLayer('conv5_4', input_shape20, 512, 3, 1, 1, 1)
        self.net.createReLuLayer('relu5_4')

        # pool5
        input_shape21 = pycnnl.IntVector(4)
        input_shape21[0] = 1
        input_shape21[1] = 512
        input_shape21[2] = 14
        input_shape21[3] = 14
        self.net.createPoolingLayer('pool5', input_shape21, 2, 2)

        # fc6
        input_shape_fc6 = pycnnl.IntVector(4)
        input_shape_fc6[0] = 1
        input_shape_fc6[1] = 1
        input_shape_fc6[2] = 1
        input_shape_fc6[3] = 25088 # 512 * 7 * 7
        weight_shape_fc6 = pycnnl.IntVector(4)
        weight_shape_fc6[0] = 1
        weight_shape_fc6[1] = 1
        weight_shape_fc6[2] = 25088
        weight_shape_fc6[3] = 4096
        output_shape_fc6 = pycnnl.IntVector(4)
        output_shape_fc6[0] = 1
        output_shape_fc6[1] = 1
        output_shape_fc6[2] = 1
        output_shape_fc6[3] = 4096
        self.net.createMlpLayer('fc6', input_shape_fc6, weight_shape_fc6, output_shape_fc6)
        self.net.createReLuLayer('relu6')

        # fc7
        input_shape_fc7 = pycnnl.IntVector(4)
        input_shape_fc7[0] = 1
        input_shape_fc7[1] = 1
        input_shape_fc7[2] = 1
        input_shape_fc7[3] = 4096
        weight_shape_fc7 = pycnnl.IntVector(4)
        weight_shape_fc7[0] = 1
        weight_shape_fc7[1] = 1
        weight_shape_fc7[2] = 4096
        weight_shape_fc7[3] = 4096
        output_shape_fc7 = pycnnl.IntVector(4)
        output_shape_fc7[0] = 1
        output_shape_fc7[1] = 1
        output_shape_fc7[2] = 1
        output_shape_fc7[3] = 4096
        self.net.createMlpLayer('fc7', input_shape_fc7, weight_shape_fc7, output_shape_fc7)
        self.net.createReLuLayer('relu7')

        # fc8
        
        input_shapem3=pycnnl.IntVector(4)
        input_shapem3[0]=1
        input_shapem3[1]=1
        input_shapem3[2]=1
        input_shapem3[3]=4096
        weight_shapem3=pycnnl.IntVector(4)
        weight_shapem3[0]=1
        weight_shapem3[1]=1
        weight_shapem3[2]=4096
        weight_shapem3[3]=1000
        output_shapem3=pycnnl.IntVector(4)
        output_shapem3[0]=1
        output_shapem3[1]=1
        output_shapem3[2]=1
        output_shapem3[3]=1000

        self.net.createMlpLayer('fc8', input_shapem3,weight_shapem3,output_shapem3)
        
        # softmax
        
        input_shapes=pycnnl.IntVector(3)
        input_shapes[0]=1
        input_shapes[1]=1
        input_shapes[2]=1000
    

        self.net.createSoftmaxLayer('softmax',input_shapes ,1)
    
    def load_model(self):
        # loading params ...
        print('Loading parameters from file ' + self.param_path)
        params = scipy.io.loadmat(self.param_path)
        self.image_mean = params['normalization'][0][0][0]
        self.image_mean = np.mean(self.image_mean, axis=(0, 1))

        for idx in range(self.net.size()):
            layer_name = self.net.getLayerName(idx)
            if 'conv' in layer_name:
                print('loading params for layer %s ...' % layer_name)
                weight, bias = params['layers'][0][idx][0][0][0][0]
                # matconvnet: [height, width, in_channel, out_channel]
                # pycnnl: [out_channel, height, width, in_channel]
                weight = np.transpose(weight, [3, 0, 1, 2]).flatten().astype(np.float)
                bias = bias.reshape(-1).astype(np.float)
                self.net.loadParams(idx, weight, bias)

            if 'fc' in layer_name:
                print('loading params for layer %s ...' % layer_name)
                weight, bias = params['layers'][0][idx][0][0][0][0]
                
                # 将4D权重 (e.g., 7x7x512x4096) 转换为 2D (25088x4096)
                shape = weight.shape
                weight = weight.reshape([shape[0] * shape[1] * shape[2], shape[3]])
                
                # 直接展平，不进行转置
                weight = weight.flatten().astype(np.float)
                bias = bias.reshape(-1).astype(np.float)
                self.net.loadParams(idx, weight, bias)

    def load_image(self, image_dir):
        # 读取图像数据
        self.image = image_dir
        # VGG19在ImageNet上训练的均值 (BGR 顺序)
        image_mean = np.array([123.68, 116.779, 103.939])
        print("Loading and preprocessing image from %s" % image_dir)
        
        # 1. 以 RGB 顺序加载图像
        input_image = scipy.misc.imread(image_dir)
        input_image = scipy.misc.imresize(input_image, [224, 224, 3])
        input_image = np.array(input_image).astype(np.float32)
        
        # 2. 将 RGB 转换为 BGR
        input_image = input_image[:, :, ::-1]
        
        # 3. 减去 BGR 均值
        input_image -= image_mean
        
        # 4. 转换为 NHWC 格式 (1, 224, 224, 3)
        input_image = np.reshape(input_image, [1] + list(input_image.shape))
        
        # !!! 错误的一行已被删除： np.transpose(input_image, [0, 3, 1, 2]) !!!
        
        # 5. 从 NHWC 格式展平并加载到 DLP
        input_data = input_image.flatten().astype(np.float)
        self.net.setInputData(input_data)

    def forward(self):
        return self.net.forward()
    
    def get_top5(self, label):
        start = time.time()
        self.forward()
        end = time.time()

        result = self.net.getOutputData()

        # loading labels
        labels = []
        with open('../synset_words.txt', 'r') as f:
            labels = f.readlines()

        # print results
        top1 = False
        top5 = False
        print('------ Top 5 of ' + self.image + ' ------')
        prob = sorted(list(result), reverse=True)[:6]
        if result.index(prob[0]) == label:
            top1 = True
        for i in range(5):
            top = prob[i]
            idx = result.index(top)
            if idx == label:
                top5 = True
            print('%f - '%top + labels[idx].strip())

        print('inference time: %f'%(end - start))
        return top1,top5
    
    def evaluate(self, file_list):
        top1_num = 0
        top5_num = 0
        total_num = 0

        start = time.time()
        with open(file_list, 'r') as f:
            file_list = f.readlines()
            total_num = len(file_list)
            for line in file_list:
                image = line.split()[0].strip()
                label = int(line.split()[1].strip())
                vgg.load_image(image)
                top1,top5 = vgg.get_top5(label)
                if top1 :
                    top1_num += 1
                if top5 :
                    top5_num += 1
        end = time.time()

        print('Global accuracy : ')
        print('accuracy1: %f (%d/%d) '%(float(top1_num)/float(total_num), top1_num, total_num))
        print('accuracy5: %f (%d/%d) '%(float(top5_num)/float(total_num), top5_num, total_num))
        print('Total execution time: %f'%(end - start))


if __name__ == '__main__':
    vgg = VGG19()
    vgg.build_model()
    vgg.load_model()
    vgg.evaluate('../file_list')
