import sys
import numpy as np
import struct
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from layers_1 import FullyConnectedLayer, ReLULayer, SoftmaxLossLayer

# MNIST 数据集的文件路径
MNIST_DIR = "../mnist_data"
TRAIN_DATA = "train-images-idx3-ubyte"
TRAIN_LABEL = "train-labels-idx1-ubyte"
TEST_DATA = "t10k-images-idx3-ubyte"
TEST_LABEL = "t10k-labels-idx1-ubyte"


def show_matrix(mat, name):
    # 用于调试，打印矩阵的形状、均值和标准差
    # print(name + str(mat.shape) + ' mean %f, std %f' % (mat.mean(), mat.std()))
    pass


class MNIST_MLP(object):
    '''
    多层感知机网络
    '''
    def __init__(self, batch_size=100, input_size=784, hidden1=32, hidden2=16, out_classes=10, lr=0.01, max_epoch=1, print_iter=100):
        '''
        网络初始化
        :param batch_size: 批处理大小
        :param input_size: 输入层维度
        :param hidden1: 隐藏层1的维度
        :param hidden2: 隐藏层2的维度
        :param out_classes: 输出层维度
        :param lr: 学习率
        :param max_epoch: 最大训练轮数
        :param print_iter: 每隔多少次迭代打印一次损失
        '''
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.out_classes = out_classes
        self.lr = lr
        self.max_epoch = max_epoch
        self.print_iter = print_iter

    def load_mnist(self, file_dir, is_images='True'):
        '''
        加载 MNIST 数据集
        :param file_dir: 文件路径
        :param is_images: 是否为图像文件
        :return: 加载的数据，格式为 numpy 数组
        '''
        # 以二进制只读方式打开文件
        bin_file = open(file_dir, 'rb')
        bin_data = bin_file.read()
        bin_file.close()

        # 根据文件类型解析文件头
        if is_images:
            # 图像文件的前 16 个字节是文件头，包含魔数、图像数量、行数和列数
            fmt_header = '>iiii'
            magic, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, 0)
        else:
            # 标签文件的前 8 个字节是文件头，包含魔数和标签数量
            fmt_header = '>ii'
            magic, num_images = struct.unpack_from(fmt_header, bin_data, 0)
            num_rows, num_cols = 1, 1
        
        # 解析数据部分
        data_size = num_images * num_rows * num_cols
        mat_data = struct.unpack_from('>' + str(data_size) + 'B', bin_data, struct.calcsize(fmt_header))
        
        # 将数据重塑为 numpy 数组
        mat_data = np.reshape(mat_data, [num_images, num_rows * num_cols])
        print('Load images from %s, number: %d, data shape: %s' % (file_dir, num_images, str(mat_data.shape)))
        return mat_data

    def load_data(self):
        '''
        加载并预处理 MNIST 训练和测试数据
        '''
        # TODO: 调用函数 load_mnist 读取和预处理 MNIST 中训练数据和测试数据的图像和标记
        print('Loading MNIST data from files...')
        # 加载训练图像和标签
        train_images = self.load_mnist(os.path.join(MNIST_DIR, TRAIN_DATA), True)
        train_labels = self.load_mnist(os.path.join(MNIST_DIR, TRAIN_LABEL), False)
        # 加载测试图像和标签
        test_images = self.load_mnist(os.path.join(MNIST_DIR, TEST_DATA), True)
        test_labels = self.load_mnist(os.path.join(MNIST_DIR, TEST_LABEL), False)

        # 将图像和标签合并，方便后续处理
        self.train_data = np.append(train_images, train_labels, axis=1)
        self.test_data = np.append(test_images, test_labels, axis=1)

    def shuffle_data(self):
        '''
        随机打乱训练数据
        '''
        print('Randomly shuffle MNIST data...')
        np.random.shuffle(self.train_data)

    def build_model(self):
        '''
        建立三层神经网络结构
        '''
        # TODO：建立三层神经网络结构
        print('Building multi-layer perception model...')
        # 第一层：全连接层 + ReLU激活
        self.fc1 = FullyConnectedLayer(self.input_size, self.hidden1)
        self.relu1 = ReLULayer()
        # 第二层：全连接层 + ReLU激活
        self.fc2 = FullyConnectedLayer(self.hidden1, self.hidden2)
        self.relu2 = ReLULayer()
        # 第三层：全连接层 + Softmax损失层
        self.fc3 = FullyConnectedLayer(self.hidden2, self.out_classes)
        self.softmax = SoftmaxLossLayer()
        # 将需要更新参数的层存入列表
        self.update_layer_list = [self.fc1, self.fc2, self.fc3]

    def init_model(self):
        '''
        初始化模型参数
        '''
        print('Initializing parameters of each layer in MLP...')
        for layer in self.update_layer_list:
            layer.init_param()

    def load_model(self, param_dir):
        '''
        从文件加载模型参数
        :param param_dir: 参数文件路径
        '''
        print('Loading parameters from file ' + param_dir)
        params = np.load(param_dir, allow_pickle=True).item()
        # 加载各层的权重和偏置
        self.fc1.load_param(params['w1'], params['b1'])
        self.fc2.load_param(params['w2'], params['b2'])
        self.fc3.load_param(params['w3'], params['b3'])

    def save_model(self, param_dir):
        '''
        保存模型参数到文件
        :param param_dir: 参数文件路径
        '''
        print('Saving parameters to file ' + param_dir)
        params = {}
        # 保存各层的权重和偏置
        params['w1'], params['b1'] = self.fc1.save_param()
        params['w2'], params['b2'] = self.fc2.save_param()
        params['w3'], params['b3'] = self.fc3.save_param()
        np.save(param_dir, params)

    def forward(self, input):
        '''
        神经网络的前向传播
        :param input: 输入数据
        :return: 输出的概率分布
        '''
        # TODO：神经网络的前向传播
        # 第一层
        h1 = self.fc1.forward(input)
        h1 = self.relu1.forward(h1)
        # 第二层
        h2 = self.fc2.forward(h1)
        h2 = self.relu2.forward(h2)
        # 第三层
        h3 = self.fc3.forward(h2)
        # Softmax输出
        prob = self.softmax.forward(h3)
        return prob

    def backward(self):
        '''
        神经网络的反向传播
        '''
        # TODO：神经网络的反向传播
        # 从 Softmax 层开始反向传播
        dloss = self.softmax.backward()
        # 第三层
        dh3 = self.fc3.backward(dloss)
        # 第二层
        dh2 = self.relu2.backward(dh3)
        dh2 = self.fc2.backward(dh2)
        # 第一层
        dh1 = self.relu1.backward(dh2)
        dh1 = self.fc1.backward(dh1)

    def update(self, lr):
        '''
        使用学习率更新网络参数
        :param lr: 学习率
        '''
        for layer in self.update_layer_list:
            layer.update_param(lr)

    def train(self):
        '''
        训练神经网络
        '''
        # 计算每个 epoch 的批次数
        max_batch = self.train_data.shape[0] // self.batch_size
        print('Start training...')
        for idx_epoch in range(self.max_epoch):
            # 每个 epoch 开始时打乱数据
            self.shuffle_data()
            for idx_batch in range(max_batch):
                # 获取一个批次的数据
                batch_images = self.train_data[idx_batch * self.batch_size:(idx_batch + 1) * self.batch_size, :-1]
                batch_labels = self.train_data[idx_batch * self.batch_size:(idx_batch + 1) * self.batch_size, -1]
                
                # 前向传播
                prob = self.forward(batch_images)
                # 计算损失
                loss = self.softmax.get_loss(batch_labels)
                # 反向传播
                self.backward()
                # 更新参数
                self.update(self.lr)
                
                # 打印训练信息
                if idx_batch % self.print_iter == 0:
                    print('Epoch %d, iter %d, loss: %.6f' % (idx_epoch, idx_batch, loss))

    def evaluate(self):
        '''
        在测试集上评估模型性能
        '''
        pred_results = np.zeros([self.test_data.shape[0]])
        for idx in range(int(self.test_data.shape[0] / self.batch_size)):
            batch_images = self.test_data[idx * self.batch_size:(idx + 1) * self.batch_size, :-1]
            prob = self.forward(batch_images)
            pred_labels = np.argmax(prob, axis=1)
            pred_results[idx * self.batch_size:(idx + 1) * self.batch_size] = pred_labels
        
        # 处理剩余不足一个批次的数据
        if self.test_data.shape[0] % self.batch_size > 0:
            last_batch_start = int(self.test_data.shape[0] / self.batch_size) * self.batch_size
            batch_images = self.test_data[last_batch_start:, :-1]
            prob = self.forward(batch_images)
            pred_labels = np.argmax(prob, axis=1)
            pred_results[last_batch_start:] = pred_labels
            
        accuracy = np.mean(pred_results == self.test_data[:, -1])
        print('Accuracy in test set: %f' % accuracy)


def build_mnist_mlp(param_dir='weight.npy'):
    '''
    构建、训练和评估 MNIST MLP 模型
    :param param_dir: 模型参数保存路径
    :return: 训练好的模型
    '''
    # 定义网络超参数
    h1, h2, e = 256, 128, 10  # 增加隐藏层节点数和训练轮数以提高精度
    mlp = MNIST_MLP(batch_size=10000, hidden1=h1, hidden2=h2, max_epoch=e)
    mlp.load_data()
    mlp.build_model()
    mlp.init_model()
    # mlp.train()
    
    # 保存和加载模型以验证
    model_path = 'mlp-%d-%d-%depoch.npy' % (h1, h2, e)
    # mlp.save_model(model_path)
    mlp.load_model(param_dir)
    
    return mlp


if __name__ == '__main__':
    mlp = build_mnist_mlp()
    mlp.evaluate()
