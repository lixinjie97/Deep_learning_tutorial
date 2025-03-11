## 深度学习教程

<div align="center">

  ![counter](https://counter.seku.su/cmoe?name=Deep_learning_tutorial&theme=moebooru)

</div>

### 前言
主要是练习李沐老师教程中的源码，我在每一章中容易有疑惑的地方进行了更细致的补充，另外补充了练习题的答案。所用的深度学习框架为Pytorch。
### Requirement
d2l==0.17.5

numpy==1.22
### TODO
- [x] 参加Kaggle竞赛
- [ ] 完成教程中的概念源码学习，以及练习课后题
- [X] 把每章中的每节进行编号
### 章节
<details>
  <summary><b><u>01.preliminarier</u></b></summary>

  * [01.数据操作。讲解pytorch中的张量(tensor)的一些使用方法；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/01.preliminaries/01.ndarray.ipynb)
  * [02.数据预处理。讲解使用pandas对数据进行一系列的预处理操作；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/01.preliminaries/02.pandas.ipynb)
  * [03.线性代数。讲解标量、向量、矩阵、张量的基本概念和性质；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/01.preliminaries/03.linear-algebra.ipynb)
  * [04.微积分。讲解导数和微分、偏导数、梯度、链式法则等基本概念性质，以及怎么使用matplotlib画图；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/01.preliminaries/04.calculus.ipynb)
  * [05.自动微分(autograd)。Pytorch可以自动求导。Pytorch函数表达式是隐式构造的，求解导数时是反向累积的；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/01.preliminaries/05.autograd.ipynb)
</details>

<details>
  <summary><b><u>02.linear-networks</u></b></summary>

  * [01.线性回归。讲解线性模型，它是一个单层神经网络；机器学习模型中最关键要素是训练数据、损失函数、优化算法，还有模型本身；使用最小二乘法和极大似然估计这两种方法去求解 w 和 b 是等价的；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/02.linear-networks/01.linear-regression.ipynb)
  * [02.线性回归的源码实现。讲解线性回归的具体源码实现，包括初始化模型参数、定义模型、定义损失函数、定义优化算法、训练；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/02.linear-networks/02.linear-regression-scratch.ipynb)
  * [03.线性回归的框架实现。讲解线性回归的框架实现，在pytorch中，data模块提供了数据处理工具，nn模块定义了大量的神经网络层和常见损失函数，optim模块有实现优化算法；使用深度学习框架可以高效地搭建神经网络；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/02.linear-networks/03.linear-regression-concise.ipynb)
  * [04.softmax回归。讲解使用softmax回归去解决分类问题，它使用了softmax运算中输出类别的概率分布。详细内容包括网络架构、softmax运算、损失函数、softmax及其导数和交叉熵损失；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/02.linear-networks/04.softmax-regression.ipynb)
  * [05.图像分类数据集。本节介绍了一个服装分类数据集Fashion-MNIST，它由10个类别的图像组成；同时实现了一个函数用于获取和读取数据集，函数返回训练集和验证集的数据迭代器；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/02.linear-networks/05.image-classification-dataset.ipynb)
  * [06.softmax回归的源码实现。本节实现了一个softmax回归服装多分类的模型，首先先读取数据，再定义模型和损失函数，然后使用优化算法训练模型；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/02.linear-networks/06.softmax-regression-scratch.ipynb)
  * [07.softmax回归的框架实现。讲解softmax的框架实现，框架实现要比从0开始编写模型的健壮性更强，框架实现可以避免计算过程中出现数值为0、inf或nan（不是数字）的情况；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/02.linear-networks/07.softmax-regression-concise.ipynb)
</details>

<details>
  <summary><b><u>03.multilayer-perceptrons</u></b></summary>

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
</details>

<details>
  <summary><b><u>04.deep-learning-computation</u></b></summary>

  * [01.深度学习计算序言。总领接下来要讲的深度学习计算的关键组件：模型构建、参数访问与初始化、设计自定义层和块、将模型读写到磁盘，以及利用GPU实现显著的加速；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/04.deep-learning-computation/01.index.ipynb)
  * [02.层和块。讲解模型构建的内容，一个块可以由许多层组成，一个块也可以由许多块组成；块可以包含代码；块负责大量的内部处理，包括参数初始化和反向传播；层和块的顺序连接由Sequential块处理；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/04.deep-learning-computation/02.model-construction.ipynb)
  * [03.参数管理。介绍了几种访问、初始化和绑定模型参数的方法；可以自定义初始化方法；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/04.deep-learning-computation/03.parameters.ipynb)
  * [04.延后初始化。延后初始化使框架能够自动推断参数形状，使修改模型架构变得容易，避免了一些常见错误；可以通过模型传递数据，使框架最终初始化参数；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/04.deep-learning-computation/04.deferred-init.ipynb)
  * [05.自定义层。深度学习中可以构建自定义层，比如可以构建不带参数的层，也可以构建带参数的层；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/04.deep-learning-computation/05.custom-layer.ipynb)
  * [06.读写文件。本节介绍如何保存和加载训练的模型，可以使用save和load函数实现模型保存和加载，其中保存的是模型的参数而不是整个模型，为了恢复模型，需要用代码生成架构，然后从磁盘加载参数；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/04.deep-learning-computation/06.read-write.ipynb)
  * [07.使用GPU。本节介绍做深度学习运算时可以指定计算设备，存储在不同设备上的数据做运算会导致异常，必须复制到同一设备才可以做运算；神经网络模型也可以指定设备；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/04.deep-learning-computation/07.use-gpu.ipynb)
</details>

<details>
  <summary><b><u>05.convolutional-neural-networks</u></b></summary>

  * [01.卷积神经网络序言。总领接下来要详细介绍的卷积神经网络的内容：包括卷积本身、填充(padding)和步幅(stride)的基本细节、用于在相邻区域汇聚信息的汇聚层(pooling)、在每一层中多通道(channel)的使用，以及有关现代卷积网络架构的仔细讨论。](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/05.convolutional-neural-networks/01.index.ipynb)
  * [02.从全连接层到卷积。对于高维感知数据，多层感知机这种缺少结构的网络可能会变得不实用；卷积层通常比全连接层需要更少的参数，而且依旧获得高效用的模型；图像的平移不变性在处理局部图像时，可以不在乎它的位置；局部性计算相应隐藏层只需要一小部分局部图像像素；多个输入输出通道使模型在每个空间位置可以获取图像的多方面特征；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/05.convolutional-neural-networks/02.why-conv.ipynb)
  * [03.图像卷积。二维卷积层的核心计算是二维互相关运算，最简单的形式是对二维输入数据和卷积核执行互相关操作，然后添加一个偏置；可以从数据中学习卷积核的参数；学习卷积核时，无论用严格卷积运算或互相关运算，卷积层的输出不会受太大影响；当需要检测输入特征中更广区域时，我们可以构建一个更深的卷积网络；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/05.convolutional-neural-networks/03.conv-layer.ipynb)
  * [04.填充和步幅。填充可以增加输出的高度和宽度，常用来使输出与输入具有相同的高和宽；步幅可以减少输出的高和宽；填充为了做更深的卷积，步幅为了快速减小大小减少计算量；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/05.convolutional-neural-networks/04.padding-and-strides.ipynb)
  * [05.多输入多输出通道。多输入多输出通道可以用来扩展卷积层的模型；当以每像素为基础应用时，1 * 1卷积层相当于全连接层；1 * 1卷积层通常用于调整网络层的通道数量和控制模型复杂性；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/05.convolutional-neural-networks/05.channels.ipynb)
  * [06.汇聚层。池运算是确定性的，他不同于卷积层中的输入与卷积核之间的互相关运算，汇聚层不包含参数；有最大汇聚层和平均汇聚层；汇聚层的主要优点之一是减轻卷积层对位置的过度敏感；可以指定汇聚层的填充和步幅；使用最大汇聚层以及大于1的步幅，可减少空间维度（如高度和宽度）；汇聚层的输出通道数与输入通道数相同；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/05.convolutional-neural-networks/06.pooling.ipynb)
  * [07.卷积神经网络（LeNet）。本节复现了经典的LeNet网络，和在fashion_mnist数据集上跑了一次实验，得到的测试准确率为0.803；CNN是一类使用卷积层的网络；在CNN中，组合使用卷积层、非线性激活函数和汇聚层；为了构造高性能的CNN，通常对卷积层进行排列，逐渐降低其表示的空间分辨率，同时增加通道数；在传统的CNN中，卷积块编码得到的表征在输出之前需由一个或多个全连接层进行处理；LeNet是最早发布的卷积神经网络之一；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/05.convolutional-neural-networks/07.lenet.ipynb)
</details>

<details>
  <summary><b><u>06.convolutional-modern</u></b></summary>

  * [01.现代卷积神经网络。总领介绍现代的卷积神经网络架构，有AlexNet、VGG、NiN、GoogLenet、ResNet、DenseNet；不同的网络架构和超参数选择，神经网络的性能会发生很大的变化；神经网络是将人类直觉和相关数学见解结合后，经过大量研究试错后的结晶；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/06.convolutional-modern/01.index.ipynb)
  * [02.深度卷积神经网络（AlexNet）。AlexNet的架构与LeNet相似，但使用了更多的卷积层和更多的参数来拟合大规模的ImageNet数据集；AlexNet在深度学习中是一个里程碑性质的模型，它是从浅层网络到深层网络的关键一步；大规模数据集和算力的支持使得AlexNet这种深层网络的问世成为可能；Dropout、ReLU和预处理都可以提升计算机视觉任务的性能；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/06.convolutional-modern/02.alexnet.ipynb)
  * [03.使用块的网络（VGG）。VGG-11使用可复用的卷积块构造网络，不同的VGG模型可通过每个块中卷积层数量和输出通道数量的差异来定义；块的使用导致网络定义的非常简洁，使用块可以有效地设计复杂的网络；在VGG论文中，Simonyan和Ziserman尝试了各种架构，特别是他们发现深层且窄的卷积(即3*3)比较浅层且宽的卷积更有效；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/06.convolutional-modern/03.vgg.ipynb)
  * [04.网络中的网络（NiN）。NiN使用由一个卷积层和多个1*1卷积层组成的块，该块可以在卷积神经网络中使用，以允许更多的每像素非线性；NiN在最后去除了容易造成过拟合的全连接层，将它们替换为全局平均汇聚层（即在所有位置上进行求和），该汇聚层通道数量为所需的输出数量；移除全连接层可减少过拟合，同时显著减少NiN的参数；NiN的设计影响了许多后续卷积神经网络的设计；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/06.convolutional-modern/04.nin.ipynb)
  * [05.含并行连结的网络（GoogLeNet）。GoogLeNet将多个设计精细的Inception块与其它层（卷积层、全连接层）串联起来，其中Inception块的通道数分配之比是在ImageNet数据集上通过大量的实验得来的；Inception块相当于有4条路径的子网络，它通过不同窗口形状的卷积层和最大汇聚层来并行抽取信息，并使用1*1卷积层减少每像素级别上的通道维数从而降低模型复杂度；GoogLeNet和它的后继者们一度是ImageNet上最有效的模型之一：它以较低的计算复杂度提供了类似的测试精度；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/06.convolutional-modern/05.googlenet.ipynb)
  * [06.批量规范化（batch normalization）。在模型训练过程中，批量规范化利用小批量的均值和标准差，不断调整神经网络的中间输出，使整个神经网络各层的中间输出值更加稳定；批量规范化在全连接层和卷积层的使用略有不同；批量规范化和暂退法一样，在训练模式和预测模式下计算不同；批量规范化有许多有益的副作用，主要是正则化，另一方面，“减少内部协变量偏移”的原始动机似乎不是一个有效地解释；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/06.convolutional-modern/06.batch-norm.ipynb)
  * [07.残差网络（ResNet）。学习嵌套函数（nested function）是训练神经网络的理想情况，在深层神经网络中，学习另一层作为恒等映射（identity function）较容易；残差映射可以更容易地学习同一函数，例如将权重层中的参数近似为零；利用残差块（residual blocks）可以训练出一个有效的深层神经网络：输入可以通过层间的残余连接更快地向前传播；残差网络（ResNet）对随后的深层神经网络设计产生了深远影响；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/06.convolutional-modern/07.resnet.ipynb)
  * [08.稠密连接网络（DenseNet）。在跨层连接上，不同于ResNet中将输入与输出相加，稠密连接网络（DenseNet）在通道维上连接输入与输出；DenseNet的主要构建模块是稠密块和过渡层；在构建DenseNet时，我们需要通过添加过渡层来控制网络的维数，从而再次减少通道的数量；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/06.convolutional-modern/08.densenet.ipynb)
</details>

<details>
  <summary><b><u>07.computational-performance</u></b></summary>

  * [01.计算性能。本章主要讨论影响计算性能的主要因素：命令式编程、符号编程、异步计算、自动并行和多GPU计算，学习本章可以进一步提高之前实现模型的计算性能；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/07.computational-performance/01.index.ipynb)
  * [02.编译器和解释器。Python是一种解释型语言；命令式编程使得新模型的设计变得容易，因为可以依据控制流编写代码，并拥有相对成熟的Python软件生态；符号式编程要求我们先定义并且编译程序，然后在执行程序，其好处是提高了计算性能；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/07.computational-performance/02.hybridize.ipynb)
  * [03.深度学框架可以将Python前端的控制与后端的执行耦合，使得命令可以快速地异步插入后端、并行执行；异步产生了一个相当灵活的前端，但请注意：过度填充任务队列可能会导致内存消耗过多。建议对每个小批量进行同步，以保持前端和后端的大致同步；芯片供应商提供了复杂的性能分析工具，已获得对深度学习效率更精确的洞察；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/07.computational-performance/03.async-computation.ipynb)
  * [04.自动并行。现代系统拥有多种设备，如多个GPU和多个CPU，还可以并行地、异步地使用它们；现代系统还拥有各种通信资源，如PCI Express、存储（固态硬盘或网络存储）和网络带宽，为了达到最高效率可以并行使用它们；后端可以通过自动化地并行计算和通信来提高性能；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/07.computational-performance/04.auto-parallelism.ipynb)
  * [05.硬件。设备有运行开销，因此，数据传输要争取量大次少而不是量少次多，这适用于RAM、固态驱动器、网络和GPU；在训练过程中数据类型过小导致的数值溢出可能是个问题（在推断过程中则影响不大）；训练硬件和推断硬件在性能和价格方面有不同的优点；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/07.computational-performance/05.hardware.ipynb)
  * [06.多GPU训练。有多种方法可以在多个GPU上拆分深度网络的训练，拆分可以在层之间、跨层或跨数据上实现，前两者需要对数据传输过程中进行严格编排，而最后一种则是最简单的策略；在数据并行中，数据需要跨多个GPU拆分，其中每个GPU执行自己的前向传播和反向传播，随后所有的梯度被聚合为一，之后聚合结果向所有GPU广播；小批量数据量更大时，学习率也需要稍微提高一些；数据并行训练本身是不复杂的，它通过增加有效的小批量数据量的大小提高了训练效率；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/07.computational-performance/06.multiple-gpus.ipynb)
  * [07.多GPU的简洁实现。把数据分批地放到不同的GPU上训练，每个GPU都算梯度，然后梯度汇总到一个GPU上再梯度求和，更新梯度，再广播到所有GPU；利用nn.DataParallel，神经网络可以在单GPU上自动评估；每台设备上的网络需要先初始化，然后再尝试访问该设备上的参数，否则会遇到错误；优化算法在多个GPU上自动聚合；更大的batch_size需要更大的epoch模型才能收敛；如果批量大小增加，学习率也可以适当增加，较大的批量提供了更稳定的梯度估计，允许模型在每次迭代中采取更大的步长；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/07.computational-performance/07.multiple-gpus-concise.ipynb)
  * [08.参数服务器。同步需要高度适应特定的网络基础设施和服务器内的连接，这种适应会严重影响同步所需的时间；环同步对于p3和DGX-2服务器是最佳的，而对于其他服务器则未必；当添加多个参数服务器以增加带宽时，分层同步策略可以工作的很好；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/07.computational-performance/08.parameterserver.ipynb)
</details>

<details>
  <summary><b><u>08.computer-vision</u></b></summary>

  * [01.计算机视觉。本章主要讨论深度学习的应用领域之一：计算机视觉；开头，介绍两种可以改进模型泛化的方法，即图像增广和微调；然后介绍目标检测、语义分割、样式迁移的知识；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/08.computer-vision/01.index.ipynb)
  * [02.图像增广。图像增广基于现有的训练数据生成随机图像，来提高模型泛化能力；为了在预测过程中得到确切的结果，我们通常只对训练样本进行图像增广，而在预测过程中不使用带随机操作的图像增广；深度学习框架提供了许多不同的图像增广的方法，这些方法可以被同时应用；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/08.computer-vision/02.image-augmentation.ipynb)
  * [03.微调。从头训练一个模型需要更多的精力和和时间，所以就有了微调，微调是算法工程师工作过程中用到最多的技术手段，它不仅保证了结果质量还提高了效率；迁移学习将从源数据集中学到的知识迁移到目标数据集，微调是迁移学习的常见技巧；除输出层外，目标模型从源模型中复制所有模型设计及其参数，并根据目标数据集对这些参数进行微调。但是目标模型的输出层需要从头开始训练；通常，微调参数使用较小的学习率，而从头开始训练输出层可以使用更大的学习率；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/08.computer-vision/03.fine-tuning.ipynb)
  * [04.目标检测和边界框。目标检测不仅可以识别图像中所有感兴趣的物体，还可以识别它们的位置，该位置通常由矩形边界框表示；我们可以在两种常用的边界框表示（中间，宽度，高度）和（左上，右下）坐标之间进行转换；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/08.computer-vision/04.bounding-box.ipynb)
  * [05.锚框。以图像的每个像素为中心生成不同形状的锚框，考虑计算复杂度，取每个像素n+m-1个锚框；交并比（IoU）也被称为杰卡德系数，用于衡量两个边界框的相似性，它是相交面积与相并面积的比率；在训练集中，我们需要给每个锚框两种类型的标签，一个是锚框中目标检测的类别，另一个是锚框真实相对于边界框的偏移量；预测期间可以使用非极大值抑制（NMS）来移除类似的预测边界框，从而简化输出；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/08.computer-vision/05.anchor.ipynb)
  * [06.多尺度目标检测。为了减少图像上锚框数量，我们可以在输入图像中均匀采样一小部分像素，并以它们为中心生成锚框；在多个尺度下，我们可以生成不同尺寸的锚框来检测不同尺寸的目标；通过定义特征图的形状，我们可以决定任何图像上均匀采样的锚框的中心；我们使用输入图像在某个感受野区域内的信息，来预测输入图像上与该区域位置相近的锚框类别和偏移量；我们可以通过深入学习，在多个层次上的图像分层表示进行多尺度目标检测；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/08.computer-vision/06.multiscale-object-detection.ipynb)
  * [07.目标检测数据集。收集香蕉检测数据集可用于演示目标检测模型；目标检测的数据加载与图像分类的数据加载类似，但是，在目标检测中，标签还包含真实边界框的信息，图像分类则没有；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/08.computer-vision/07.object-detection-dataset.ipynb)
  * [08.单发多框检测（SSD）。单发多框检测是一种多尺度目标检测模型，ssd中接近顶部的多尺度特征图较小，但具有较大的感受野，它们适合检测较少但较大的物体；ssd基于基础网络块和各个多尺度特征块，单发多框检测生成不同数量和不同大小的锚框，并通过预测这些锚框的类别和偏移量检测不同大小的目标；在训练单发多框检测模型时，损失函数是根据锚框的类别和偏移量的预测及标注值计算得出的；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/08.computer-vision/08.ssd.ipynb)
  * [09.区域卷积神经网络（R-CNN）系列。R-CNN对图像选取若干提议区域，使用卷积神经网络对每个提议区域执行前向传播以抽取其特征，然后再用这些特征来预测提议区域的类别和边界框；Fast R-CNN对R-CNN的一个主要改进：只对整个图像做卷积神经网络的前向传播。它还引入了兴趣区域汇聚层，从而为具有不同形状的兴趣区域抽取相同形状的特征；Faster R-CNN将Fast R-CNN中使用的选择性搜索替换为参与训练的区域提议网络，这样后者可以在减少提议区域数量的情况下仍保证目标检测的精度；Mask R-CNN在Faster R-CNN的基础上引入了一个全卷积网络，从而借助目标的像素级位置进一步提升目标检测的精度；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/08.computer-vision/09.rcnn.ipynb)
  * [10.语义分割和数据集。语义分割通过将图像划分为属于不同语义类别的区域，来识别并理解图像中像素级别的内容；语义分割的一个重要数据集叫做Pascal VOC2012；由于语义分割的输入图像和标签在像素上一一对应，输入图像会被随机裁剪为固定尺寸而不是缩放；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/08.computer-vision/10.semantic-segmentation-and-dataset.ipynb)
  * [11.转置卷积。转置卷积的作用是将输入张量的特征图进行上采样（即扩大尺寸），通常用于生成模型或语义分割等任务中；与通过卷积核减少输入元素的常规卷积相反，转置卷积通过卷积核广播输入元素，从而产生形状大于输入的输出；如果我们将f(X)输入卷积层f来获得输出Y=f(X)并创造一个与f有相同的超参数、但输出通道数是X中通道数的转置卷积层g，那么g(Y)的形状将与X相同；我们可以使用矩阵乘法来实现卷积；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/08.computer-vision/11.transposed-conv.ipynb)
  * [12.全卷积网络。全卷积网络先使用卷积神经网络抽取图像特征，然后通过1*1卷积层将通道数变换为类别个数，最后通过转置卷积层将特征图的高和宽变换为输入图像尺寸；在全卷积网络中，我们可以将转置卷积层初始化为双线性插值的上采样；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/08.computer-vision/12.fcn.ipynb)
  * [13.风格迁移。风格迁移的常用的损失函数由3部分组成：content_loss内容损失使合成图像与内容图像在内容特征上接近；style_loss风格损失令合成图像与风格图像在风格特征上接近；tv_loss全变分损失则有助于减少合成图像中的噪点；我们可以通过预训练的卷积神经网络来抽取图像的特征，并通过最小化损失函数来不断更新合成图像来作为模型参数；我们使用格拉姆矩阵表达风格层输出的风格；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/08.computer-vision/13.neural-style.ipynb)
  * [14.实战 Kaggle 比赛：图像分类(CIFAR-10)。将包含原始图像文件的数据集组织为所需格式后，我们可以读取它们；我们可以在图像分类竞赛中使用卷积神经网络和图像增广；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/08.computer-vision/14.kaggle-cifar10.ipynb)
  * [15.实战 Kaggle 比赛：狗的品种(ImageNet Dogs)。ImageNet数据集中的图像比CIFAR-10图像尺寸大，我们可能会修改不同数据集上任务的图像增广操作；要对ImageNet数据集的子集进行分类，我们可以利用完整ImageNet数据集上的预训练模型来提取特征并仅训练小型自定义输出网络，这将减少时间和节省内存空间；]()
</details>

### Acknowledgment
[d2l](https://zh.d2l.ai/)
