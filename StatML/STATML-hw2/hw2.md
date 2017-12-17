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
这个准确率仍然不够高，在实际应用中缺乏价值。因此，接下来将引入更加复杂的模型来提高训练精度。

### 多层感知机(Multilayer Perceptron, MLP)
Softmax回归模型采用了最简单的两层神经网络，即只有输入层和输出层，因此其拟合能力有限。为了达到更好的识别效果，我们考虑在输入层和输出层中间加上若干个隐藏层。

- 经过第一个隐藏层，可以得到 $ H_1 = \phi(W_1X + b_1) $，其中$\phi$代表激活函数，常见的有sigmoid、tanh或ReLU等函数。
- 经过第二个隐藏层，可以得到 $ H_2 = \phi(W_2H_1 + b_2) $。
- 最后，再经过输出层，得到的$Y=\text{softmax}(W_3H_2 + b_3)$，即为最后的分类结果向量。


下图为多层感知器的网络结构图，图中权重用蓝线表示、偏置用红线表示、+1代表偏置参数的系数为1。

![](http://paddlepaddle.org/docs/develop/book/02.recognize_digits/image/mlp.png)

配图来自百度PaddlePaddle官方教程，下同


```python
in_units = 784
h1_units = 300
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))
```
in_units 表示输入节点数，h1_units 表示隐含层输出节点数，设置为300。W1，b1是隐含层的权重和偏置，此处将偏置全都设置为0，并将权重初始化为截断的正态分布，标准差为0.1,。

模型使用ReLU作为激活函数，为此需要使用正态分布给参数加噪声，以打破完全对称。

对于最后输出层的Softmax，直接将W2和b2全赋值为0即可。

多层感知机相当于在Softmax的基础上加上隐含层，接下来我们将定义模型结构：

```python
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)
```
hidden1 是一个隐含层，使用ReLU作为激活函数，计算公式$y=relu(W_1x+b_1)$。接下来使用Dropout功能，随机将一部分节点置为0，这里的keep_prob参数即为保留数据而不置为0的比例，在训练时应当小鱼1，防止过拟合；在预测时应当等于1，即使用全部特征来进行预测。
最后是输出层，同样使用softmax，与之前一部分内容一致。

损失函数仍然使用交叉熵，而优化函数选择自适应优化其Adagrad，学习速率设置为0.3
```python
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)
```

全部代码如下所示：
```python
# Create the model
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()

in_units = 784
h1_units = 300
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32, [None, in_units])
keep_prob = tf.placeholder(tf.float32)

hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

# Train
tf.global_variables_initializer().run()
for i in range(3000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
```
最后可以得到训练精度：
```
0.981
```

### 卷积神经网络
在多层感知器模型中，将图像展开成一维向量输入到网络中，忽略了图像的位置和结构信息，而卷积神经网络能够更好的利用图像的结构信息。LeNet-5 是一个较简单的卷积神经网络。

图 2 显示了其结构：输入的二维图像，先经过两次卷积层到池化层，再经过全连接层，最后使用 softmax 分类作为输出层。下面我们主要介绍卷积层和池化层。

![CNN](http://paddlepaddle.org/docs/develop/book/02.recognize_digits/image/mlp.png)

图 2. LeNet-5 卷积神经网络结构


#### 卷积层

卷积层是卷积神经网络的核心基石。在图像识别里我们提到的卷积是二维卷积，即离散二维滤波器（也称作卷积核）与二维图像做卷积操作，简单的讲是二维滤波器滑动到二维图像上所有位置，并在每个位置上与该像素点及其领域像素点做内积。卷积操作被广泛应用与图像处理领域，不同卷积核可以提取不同的特征，例如边沿、线性、角等特征。在深层卷积神经网络中，通过卷积操作可以提取出图像低级到复杂的特征。


![](http://paddlepaddle.org/docs/develop/book/02.recognize_digits/image/conv_layer.png)

卷积层图片

---
上图给出一个卷积计算过程的示例图，输入图像大小为 $H=5,W=5,D=3$，即 $5\times5\times5$ 大小的 $3$ 通道（RGB，也称作深度）彩色图像。

这个示例图中包含两（用 $K$ 表示）组卷积核，即图中滤波器 $W_0$ 和 $W_1$。在卷积计算中，通常对不同的输入通道采用不同的卷积核，如图示例中每组卷积核包含 $(D=3)$ 个 $3\times3\times3$（用 $F\times F\times F$ 表示）大小的卷积核。

另外，这个示例中卷积核在图像的水平方向（$W$ 方向）和垂直方向（$H$ 方向）的滑动步长为 $2$（用 $S$ 表示）；对输入图像周围各填充 $1$（用 $P$ 表示）个 $0$，即图中输入层原始数据为蓝色部分，灰色部分是进行了大小为 $1$ 的扩展，用 $0$ 来进行扩展。

---
经过卷积操作得到输出为 $3\times3\times2$（用 $H_0\times W_0\times K$ 表示）大小的特征图，即 $3\times3$ 大小的 $2$ 通道特征图，其中 $H_0$ 计算公式为：$H_0=(H-F+2\times P)/S$，$W_0$同理。 而输出特征图中的每个像素，是每组滤波器与输入图像每个特征图的内积再求和，再加上偏置 $b_0$，偏置通常对于每个输出特征图是共享的。输出特征图 $o[:,:,0]$ 中的最后一个 $-2$ 计算如上图右下角公式所示。


在卷积操作中卷积核是可学习的参数，经过上面示例介绍，每层卷积的参数大小为 $D\times F \times K$。在多层感知器模型中，神经元通常是全部连接，参数较多。而卷积层的参数较少，这也是由卷积层的主要特性即局部连接和共享权重所决定。

---
#### 局部连接

每个神经元仅与输入神经元的一块区域连接，这块局部区域称作感受野（receptive field）。在图像卷积操作中，即神经元在空间维度（spatial dimension，即上图示例 $H$ 和 $W$ 所在的平面）是局部连接，但在深度上是全部连接。对于二维图像本身而言，也是局部像素关联较强。这种局部连接保证了学习后的过滤器能够对于局部的输入特征有最强的响应。局部连接的思想，也是受启发于生物学里面的视觉系统结构，视觉皮层的神经元就是局部接受信息的。

---
#### 权重共享

计算同一个深度切片的神经元时采用的滤波器是共享的。例如上图中计算 $o[:,:,0]$ 的每个每个神经元的滤波器均相同，都为 $W_0$，这样可以很大程度上减少参数。

共享权重在一定程度上讲是有意义的，例如图片的底层边缘特征与特征在图中的具体位置无关。但是在一些场景中是无意的，比如输入的图片是人脸，眼睛和头发位于不同的位置，希望在不同的位置学到不同的特征（参考斯坦福大学公开课）。请注意权重只是对于同一深度切片的神经元是共享的，在卷积层，通常采用多组卷积核提取不同特征，即对应不同深度切片的特征，不同深度切片的神经元权重是不共享。另外，偏重对同一深度切片的所有神经元都是共享的。

---
通过介绍卷积计算过程及其特性，可以看出卷积是线性操作，并具有平移不变性（shift-invariant），平移不变性即在图像每个位置执行相同的操作。卷积层的局部连接和权重共享使得需要学习的参数大大减小，这样也有利于训练较大卷积神经网络。

---
#### 池化层

池化是非线性下采样的一种形式，主要作用是通过减少网络的参数来减小计算量，并且能够在一定程度上控制过拟合。通常在卷积层的后面会加上一个池化层。池化包括最大池化、平均池化等。其中最大池化是用不重叠的矩形框将输入层分成不同的区域，对于每个矩形框的数取最大值作为输出层，如图所示。

![level](http://paddlepaddle.org/docs/develop/book/02.recognize_digits/image/max_pooling.png)

---
更详细的关于卷积神经网络的具体知识可以参考[斯坦福大学公开课](http://cs231n.github.io/convolutional-networks/)和[图像分类](https://github.com/PaddlePaddle/book/blob/develop/image_classification/README.md)教程。


接下来我们继续使用TensorFlow 来实现卷积神经网络，并在MNIST数据集上验证训练精度：
```python
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()
```
定义标准差和偏差函数，此处设置0.1的偏差值，打破完全对称，避免出现死亡结点：
```python
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
```
定义二维卷积函数和最大池化函数：
```python
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')  
```
此处x是输入，W是卷积参数，strides表示移动的步长，此处全部设定为1，即不会错过任何一个点。

接下来定义变量：
```python
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1,28,28,1])
```
跟MLP等不同，在卷积神经网络中，我们保留原始的空间结构信息，所以此处将其恢复成28X28的2D结构。

定义第一个卷积层和池化层：
```python
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
```
[5, 5, 1, 32]代表卷积核的尺寸为5X5，1个颜色通道，32个不同的卷积核，然后使用ReLU函数进行非线性处理，使用最大池化函数对输出结果进行池化操作。

第二个卷积层与之类似：
```python
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
```

经过两次最大池化之后，