# Deep_learning_tutorial
## 深度学习教程
### 前言
主要是练习李沐老师教程中的源码，我在每一章中容易有疑惑的地方进行了更细致的补充，另外补充了练习题的答案。所用的深度学习框架为Pytorch。
### TODO
- [x] 参加Kaggle竞赛
- [ ] 完成教程中的概念源码学习，以及练习课后题
- [X] 把每章中的每节进行编号
### 章节
#### 01.preliminarier
* [01.数据操作。讲解pytorch中的张量(tensor)的一些使用方法；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/01.preliminaries/01.ndarray.ipynb)
* [02.数据预处理。讲解使用pandas对数据进行一系列的预处理操作；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/01.preliminaries/02.pandas.ipynb)
* [03.线性代数。讲解标量、向量、矩阵、张量的基本概念和性质；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/01.preliminaries/03.linear-algebra.ipynb)
* [04.微积分。讲解导数和微分、偏导数、梯度、链式法则等基本概念性质，以及怎么使用matplotlib画图；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/01.preliminaries/04.calculus.ipynb)
* [05.自动微分(autograd)。Pytorch可以自动求导。Pytorch函数表达式是隐式构造的，求解导数时是反向累积的；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/01.preliminaries/05.autograd.ipynb)
#### 02.linear-networks
* [01.线性回归。讲解线性模型，它是一个单层神经网络；机器学习模型中最关键要素是训练数据、损失函数、优化算法，还有模型本身；使用最小二乘法和极大似然估计这两种方法去求解 w 和 b 是等价的；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/02.linear-networks/01.linear-regression.ipynb)
* [02.线性回归的源码实现。讲解线性回归的具体源码实现，包括初始化模型参数、定义模型、定义损失函数、定义优化算法、训练；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/02.linear-networks/02.linear-regression-scratch.ipynb)
* [03.线性回归的框架实现。讲解线性回归的框架实现，在pytorch中，data模块提供了数据处理工具，nn模块定义了大量的神经网络层和常见损失函数，optim模块有实现优化算法；使用深度学习框架可以高效地搭建神经网络；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/02.linear-networks/03.linear-regression-concise.ipynb)
* [04.softmax回归。讲解使用softmax回归去解决分类问题，它使用了softmax运算中输出类别的概率分布。详细内容包括网络架构、softmax运算、损失函数、softmax及其导数和交叉熵损失；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/02.linear-networks/04.softmax-regression.ipynb)
* [05.图像分类数据集。本节介绍了一个服装分类数据集Fashion-MNIST，它由10个类别的图像组成；同时实现了一个函数用于获取和读取数据集，函数返回训练集和验证集的数据迭代器；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/02.linear-networks/05.image-classification-dataset.ipynb)
* [06.softmax回归的源码实现。本节实现了一个softmax回归服装多分类的模型，首先先读取数据，再定义模型和损失函数，然后使用优化算法训练模型；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/02.linear-networks/06.softmax-regression-scratch.ipynb)
* [07.softmax回归的框架实现。讲解softmax的框架实现，框架实现要比从0开始编写模型的健壮性更强，框架实现可以避免计算过程中出现数值为0、inf或nan（不是数字）的情况；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/02.linear-networks/07.softmax-regression-concise.ipynb)
#### 03.multilayer-perceptrons
* [01.多层感知机序言。最简单的深度网络称为多层感知机。总领接下来会讲到的MLP基本概念，包括过拟合、欠拟合和模型选择，以及为了解决这些问题会使用的权重衰减和暂退法等正则化技术；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/03.multilayer-perceptrons/01.index.ipynb)
* [02.多层感知机。讲解多层感知机的概念，多层感知机在输出层和输入层之间增加一个或多个全连接隐藏层，并通过激活函数转换隐藏层的输出；常用的激活函数包括ReLU、sigmoid和tanh函数，在神经网络中引入这些非线性激活函数，使得网络能够学习和模拟更加复杂的函数映射；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/03.multilayer-perceptrons/02.mlp.ipynb)
* [03.多层感知机的源码实现。本节实现了一个具有单隐藏层的多层感知机，其中有256个隐藏单元，num_inputs=784，num_outputs=10，激活函数用的ReLU，用这个模型去分类Fashion-MNIST数据集；练习题讲解了一些调整超参数的技巧；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/03.multilayer-perceptrons/03.mlp_scratch.ipynb)
* [04.多层感知机的框架实现。本节使用高级API更简洁地实现多层感知机；对于相同的分类问题，多层感知机的实现与softmax回归的实现相同，只是多层感知机的实现里增加了带有激活函数的隐藏层；练习题做了一些实验对比选取哪些激活函数、哪些初始化权重的方法，效果好；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/03.multilayer-perceptrons/04.mlp_concise.ipynb)
* [05.模型选择、欠拟合和过拟合。本节介绍了过拟合、欠拟合的概念以及模型选择的一些技巧；模型选择中，明确说明测试集和验证集是不同的；当训练数据稀缺时，可以选择K折交叉验证；选择合适的模型复杂度是很重要的，合适的模型复杂度可以最小化泛化损失；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/03.multilayer-perceptrons/05.underfit-overfit.ipynb)
* [06.权重衰减。权重衰减是一种正则化技术，在损失函数中加上一个平方L2范数（称为惩罚项），去惩罚权重向量的大小，防止过拟合；本节进行了weight decay的源码实现和框架实现，框架实现的优点是运行得更快，也更容易实现，权重衰减在优化器中提供；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/03.multilayer-perceptrons/06.weight-decay.ipynb)
* [07.暂退法(Dropout)。暂退法也是一种正则化技术，暂退法在训练过程中引入了噪声，迫使网络学习更加鲁棒的特征表示，同时通过调整未丢弃节点的输出，确保了网络的期望输出不受影响。这种方法有效地提高了模型的泛化能力，减少了过拟合的风险。同时本节讲解了暂退法的源码实现和框架实现；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/03.multilayer-perceptrons/07.dropout.ipynb)
* [08.前向传播、反向传播和计算图。本节通过一些基本的数学和计算图，深入探讨了反向传播的细节；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/03.multilayer-perceptrons/08.backprop.ipynb)
* [09.数值稳定性和模型初始化。初始化方案的选择在神经网络学习中起着举足轻重的作用，它对保持数值稳定性至关重要；ReLU激活函数可以缓解梯度消失的问题，加速收敛；随机初始化是保证在进行优化前打破对称性的关键，从而实现网络的表达能力；Xavier初始化表明，每一层输出的方差不受输入数量的影响，任何梯度的方差不受输出数量的影响；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/03.multilayer-perceptrons/09.numerical-stability-and-init.ipynb)
* [10.环境和分布偏移。训练集和测试集不来自同一个分布，就会有分布偏移。经验风险是训练数据的平均损失，用于近似真实风险。在实践中，我们要进行经验风险最小化；在测试时可以检测并纠正协变量偏移和标签偏移；在某些情况下，环境可能会记住自动操作并以令人惊讶的方式做出响应；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/03.multilayer-perceptrons/10.environment.ipynb)
* [11.实战Kaggle比赛：预测房价。本节讲解了实际应用和做比赛的一些技巧，如何做Kaggle比赛，首先可能需要对数据做预处理，做数据归一化，用均值替换缺失值；将类别特征可以转为使用独热向量来表示；使用K折交叉验证来选择模型并调整超参数；对数对于相对误差很有用；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/03.multilayer-perceptrons/11.kaggle-house-price.ipynb)
#### 04.deep-learning-computation
* [01.深度学习计算序言。总领接下来要讲的深度学习计算的关键组件：模型构建、参数访问与初始化、设计自定义层和块、将模型读写到磁盘，以及利用GPU实现显著的加速；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/04.deep-learning-computation/01.index.ipynb)
* [02.层和块。讲解模型构建的内容，一个块可以由许多层组成，一个块也可以由许多块组成；块可以包含代码；块负责大量的内部处理，包括参数初始化和反向传播；层和块的顺序连接由Sequential块处理；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/04.deep-learning-computation/02.model-construction.ipynb)
* [03.参数管理。介绍了几种访问、初始化和绑定模型参数的方法；可以自定义初始化方法；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/04.deep-learning-computation/03.parameters.ipynb)
* [04.延后初始化。延后初始化使框架能够自动推断参数形状，使修改模型架构变得容易，避免了一些常见错误；可以通过模型传递数据，使框架最终初始化参数；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/04.deep-learning-computation/04.deferred-init.ipynb)
* [05.自定义层。深度学习中可以构建自定义层，比如可以构建不带参数的层，也可以构建带参数的层；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/04.deep-learning-computation/05.custom-layer.ipynb)
* [06.读写文件。本节介绍如何保存和加载训练的模型，可以使用save和load函数实现模型保存和加载，其中保存的是模型的参数而不是整个模型，为了恢复模型，需要用代码生成架构，然后从磁盘加载参数；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/04.deep-learning-computation/06.read-write.ipynb)
* [07.使用GPU。本节介绍做深度学习运算时可以指定计算设备，存储在不同设备上的数据做运算会导致异常，必须复制到同一设备才可以做运算；神经网络模型也可以指定设备；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/04.deep-learning-computation/07.use-gpu.ipynb)
#### 05.convolutional-neural-networks
* [01.卷积神经网络序言。总领接下来要详细介绍的卷积神经网络的内容：包括卷积本身、填充(padding)和步幅(stride)的基本细节、用于在相邻区域汇聚信息的汇聚层(pooling)、在每一层中多通道(channel)的使用，以及有关现代卷积网络架构的仔细讨论。](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/05.convolutional-neural-networks/01.index.ipynb)
* [02.从全连接层到卷积。对于高维感知数据，多层感知机这种缺少结构的网络可能会变得不实用；卷积层通常比全连接层需要更少的参数，而且依旧获得高效用的模型；图像的平移不变性在处理局部图像时，可以不在乎它的位置；局部性计算相应隐藏层只需要一小部分局部图像像素；多个输入输出通道使模型在每个空间位置可以获取图像的多方面特征；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/05.convolutional-neural-networks/02.why-conv.ipynb)
* [03.图像卷积。二维卷积层的核心计算是二维互相关运算，最简单的形式是对二维输入数据和卷积核执行互相关操作，然后添加一个偏置；可以从数据中学习卷积核的参数；学习卷积核时，无论用严格卷积运算或互相关运算，卷积层的输出不会受太大影响；当需要检测输入特征中更广区域时，我们可以构建一个更深的卷积网络；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/05.convolutional-neural-networks/03.conv-layer.ipynb)
* [04.填充和步幅。填充可以增加输出的高度和宽度，常用来使输出与输入具有相同的高和宽；步幅可以减少输出的高和宽；填充为了做更深的卷积，步幅为了快速减小大小减少计算量；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/05.convolutional-neural-networks/04.padding-and-strides.ipynb)
* [05.多输入多输出通道。多输入多输出通道可以用来扩展卷积层的模型；当以每像素为基础应用时，1 * 1卷积层相当于全连接层；1 * 1卷积层通常用于调整网络层的通道数量和控制模型复杂性；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/05.convolutional-neural-networks/05.channels.ipynb)
* [06.汇聚层。池运算是确定性的，他不同于卷积层中的输入与卷积核之间的互相关运算，汇聚层不包含参数；有最大汇聚层和平均汇聚层；汇聚层的主要优点之一是减轻卷积层对位置的过度敏感；可以指定汇聚层的填充和步幅；使用最大汇聚层以及大于1的步幅，可减少空间维度（如高度和宽度）；汇聚层的输出通道数与输入通道数相同；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/05.convolutional-neural-networks/06.pooling.ipynb)