# Deep_learning_tutorial
## 深度学习教程
### 前言
主要是练习李沐老师教程中的源码，我在每一章中容易有疑惑的地方进行了更细致的补充，另外补充了练习题的答案。所用的深度学习框架为Pytorch。
### 章节
#### 01.preliminarier
* [数据操作。讲解pytorch中的张量(tensor)的一些使用方法；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/01.preliminaries/ndarray.ipynb)
* [数据预处理。讲解使用pandas对数据进行一系列的预处理操作；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/01.preliminaries/pandas.ipynb)
* [线性代数。讲解标量、向量、矩阵、张量的基本概念和性质；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/01.preliminaries/linear-algebra.ipynb)
* [微积分。讲解导数和微分、偏导数、梯度、链式法则等基本概念性质，以及怎么使用matplotlib画图；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/01.preliminaries/calculus.ipynb)
* [自动微分(autograd)。Pytorch可以自动求导。Pytorch函数表达式是隐式构造的，求解导数时是反向累积的；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/01.preliminaries/autograd.ipynb)
#### 02.linear-networks
* [线性回归。讲解深度学习中最简单的模型：线性模型，它又是一个单层神经网络；机器学习模型中最重要的关键要素是训练数据、损失函数、优化算法，还有模型本身；对数据进行矢量化表达可以使表达更简洁，并且运行也更快，比直接跑for循环做运算要快；根据已知的数据$\mathbf{x}$和$\mathbf{y}$求$\mathbf{w}$和$b$]，使用最小二乘法和极大似然估计这两种方法去求解是等价的；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/02.linear-networks/linear-regression.ipynb)