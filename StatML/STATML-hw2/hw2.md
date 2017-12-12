# 第二次作业
- 秦绪博
- 2017000621

---
### 数据集描述
当我们学习编程的时候，编写的第一个程序一般是实现打印"Hello World"。而机器学习（或深度学习）的入门教程，一般都是 MNIST 数据库上的手写识别问题。如下图所示：

![MNIST](http://paddlepaddle.org/docs/develop/book/02.recognize_digits/image/mnist_example_image.png)

图1. MNIST图片示例

这里我们使用TensorFlow内置的MNIST数据集，可以使用以下代码进行导入：
```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```
查看MNIST数据集的基本情况：
```python
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)
```
可以得到以下输出结果：
```
(55000, 784) (55000, 10)
(10000, 784) (10000, 10)
(5000, 784) (5000, 10)
```
可以看到，训练集有55000个样本，测试集有10000个样本，同时验证集有5000个样本，每个样本都有自己的标签label。
接下来将在训练集上训练模型，在验证集上检验效果并决定什么时候完成训练，最后在测试集上进行评测。

首先使用Softmax模型进行初步训练，此处把28X28像素的图片展开成一个一维向量，一共有28X28=784个维度。这里先简化问题，后面将使用卷积神经网络对空间信息进行利用。

### Softmax回归(Softmax Regression)

最简单的Softmax回归模型是先将输入层经过一个全连接层得到的特征，然后直接通过softmax 函数进行多分类。

输入层的数据$X$传到输出层，在激活操作之前，会乘以相应的权重 $W$ ，并加上偏置变量 $b$ ，具体如下：

$$ y_i = \text{softmax}(\sum_j W_{i,j}x_j + b_i) $$

其中 $ \text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}} $

对于有 $N$ 个类别的多分类问题，指定 $N$ 个输出节点，$N$ 维结果向量经过softmax将归一化为 $N$ 个[0,1]范围内的实数值，分别表示该样本属于这 $N$ 个类别的概率。此处的 $y_i$ 即对应该图片为数字 $i$ 的预测概率。

在分类问题中，我们一般采用交叉熵代价损失函数（cross entropy），公式如下：

$$  \text{crossentropy}(label, y) = -\sum_i label_ilog(y_i) $$

Softmax相当于一个没有隐含层的，最简单的神经网络，此处将用做性能评测的基线。

这部分的完整代码如下所示：
```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

import tensorflow as tf
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b) #定义模型

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])) #设置损失函数

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) # 设置随机梯度下降和优化目标

tf.global_variables_initializer().run()

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100) #每次抽取100个样本——如果每次训练都使用全部样本，则计算量太大，并且也不容易跳出局部最优
    train_step.run({x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))#对模型进行验证

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))#将correct_prediction输出的bool值转换为float32再求平均

print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))

```
最后可以得到Softmax模型的训练结果：
```
0.9203
```