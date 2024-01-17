## 两分类-软分类-概率生成模型-高斯判别分析 GDA

生成模型中，我们对联合概率分布进行建模，然后采用 MAP 来获得参数的最佳值。两分类的情况，我们采用的假设：

1.  $y\sim Bernoulli(\phi)$
2.  $x|y=1\sim\mathcal{N}(\mu_1,\Sigma)$
3.  $x|y=0\sim\mathcal{N}(\mu_0,\Sigma)$

那么独立全同的数据集最大后验概率可以表示为：
$$
\begin{align}
\mathop{argmax}_{\phi,\mu_0,\mu_1,\Sigma}\log p(X|Y)p(Y)=\mathop{argmax}_{\phi,\mu_0,\mu_1,\Sigma}\sum\limits_{i=1}^N (\log p(x_i|y_i)+\log p(y_i))\nonumber\\
=\mathop{argmax}_{\phi,\mu_0,\mu_1,\Sigma}\sum\limits_{i=1}^N((1-y_i)\log\mathcal{N}(\mu_0,\Sigma)+y_i\log \mathcal{N}(\mu_1,\Sigma)+y_i\log\phi+(1-y_i)\log(1-\phi))
\end{align}
$$

* 首先对 $\phi$ 进行求解，将式子对 $\phi$ 求偏导：
  $$
  \begin{align}\sum\limits_{i=1}^N\frac{y_i}{\phi}+\frac{y_i-1}{1-\phi}=0\nonumber\\
  \Longrightarrow\phi=\frac{\sum\limits_{i=1}^Ny_i}{N}=\frac{N_1}{N}
  \end{align}
  $$

* 然后求解 $\mu_1$：
  $$
  \begin{align}\hat{\mu_1}&=\mathop{argmax}_{\mu_1}\sum\limits_{i=1}^Ny_i\log\mathcal{N}(\mu_1,\Sigma)\nonumber\\
  &=\mathop{argmin}_{\mu_1}\sum\limits_{i=1}^Ny_i(x_i-\mu_1)^T\Sigma^{-1}(x_i-\mu_1)
  \end{align}
  $$
  由于：
  $$
  \sum\limits_{i=1}^Ny_i(x_i-\mu_1)^T\Sigma^{-1}(x_i-\mu_1)=\sum\limits_{i=1}^Ny_ix_i^T\Sigma^{-1}x_i-2y_i\mu_1^T\Sigma^{-1}x_i+y_i\mu_1^T\Sigma^{-1}\mu_1
  $$


  求微分左边乘以 $\Sigma$ 可以得到：
  $$
  \begin{align}\sum\limits_{i=1}^N-2y_i\Sigma^{-1}x_i+2y_i\Sigma^{-1}\mu_1=0\nonumber\\
  \Longrightarrow\mu_1=\frac{\sum\limits_{i=1}^Ny_ix_i}{\sum\limits_{i=1}^Ny_i}=\frac{\sum\limits_{i=1}^Ny_ix_i}{N_1}
  \end{align}
  $$

* 求解 $\mu_0$，由于正反例是对称的，所以：
  $$
  \mu_0=\frac{\sum\limits_{i=1}^N(1-y_i)x_i}{N_0}
  $$

* 最为困难的是求解 $\Sigma$，我们的模型假设对正反例采用相同的协方差矩阵，当然从上面的求解中我们可以看到，即使采用不同的矩阵也不会影响之前的三个参数。首先我们有：
  $$
  \begin{align}
  \sum\limits_{i=1}^N\log\mathcal{N}(\mu,\Sigma)&=\sum\limits_{i=1}^N\log(\frac{1}{(2\pi)^{p/2}|\Sigma|^{1/2}})+(-\frac{1}{2}(x_i-\mu)^T\Sigma^{-1}(x_i-\mu))\nonumber\\
  &=Const-\frac{1}{2}N\log|\Sigma|-\frac{1}{2}Trace((x_i-\mu)^T\Sigma^{-1}(x_i-\mu))\nonumber\\
  &=Const-\frac{1}{2}N\log|\Sigma|-\frac{1}{2}Trace((x_i-\mu)(x_i-\mu)^T\Sigma^{-1})\nonumber\\
  &=Const-\frac{1}{2}N\log|\Sigma|-\frac{1}{2}NTrace(S\Sigma^{-1})
  \end{align}
  $$
  在这个表达式中，我们在标量上加入迹从而可以交换矩阵的顺序，对于包含绝对值和迹的表达式的导数，我们有：
  $$
  \begin{align}
  \frac{\partial}{\partial A}(|A|)&=|A|A^{-1}\\
  \frac{\partial}{\partial A}Trace(AB)&=B^T
  \end{align}
  $$
  因此：
  $$
  \begin{align}[\sum\limits_{i=1}^N((1-y_i)\log\mathcal{N}(\mu_0,\Sigma)+y_i\log \mathcal{N}(\mu_1,\Sigma)]'
  \nonumber\\=Const-\frac{1}{2}N\log|\Sigma|-\frac{1}{2}N_1Trace(S_1\Sigma^{-1})-\frac{1}{2}N_2Trace(S_2\Sigma^{-1})
  \end{align}
  $$
  其中，$S_1,S_2$ 分别为两个类数据内部的协方差矩阵，于是：
  $$
  \begin{align}N\Sigma^{-1}-N_1S_1^T\Sigma^{-2}-N_2S_2^T\Sigma^{-2}=0\nonumber
  \\\Longrightarrow\Sigma=\frac{N_1S_1+N_2S_2}{N}
  \end{align}
  $$
这里应用了类协方差矩阵的对称性。

于是我们就利用最大后验的方法求得了我们模型假设里面的所有参数，根据模型，可以得到联合分布，也就可以得到用于推断的条件分布了。

![](https://cdn.jsdelivr.net/gh/Rosefinch-Midsummer/MyImagesHost03/img/202401141124893.png)

![](https://cdn.jsdelivr.net/gh/Rosefinch-Midsummer/MyImagesHost03/img/202401141124951.png)


![](https://cdn.jsdelivr.net/gh/Rosefinch-Midsummer/MyImagesHost03/img/202401141123602.png)








