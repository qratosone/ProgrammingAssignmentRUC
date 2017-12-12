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

配图来自百度PaddlePaddle官方教程


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
