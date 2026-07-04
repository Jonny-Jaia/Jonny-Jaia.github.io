# 机器学习基础

## 当前定位

这章用于沉淀传统机器学习基础：优化、线性模型、正则化、损失函数、贝叶斯方法和集成学习。它服务于大模型算法面试中的“底层学习理论”和“训练直觉”。

> **面试抓手**：机器学习基础的核心不是模型名，而是“目标函数、假设空间、优化方法、泛化约束”四件事。

```flow
Data assumption -> Objective function -> Optimizer -> Regularization -> Generalization
```

## ReadyBlog 来源

- ReadyBlog: `机器学习/机器学习基础.md`
- ReadyBlog: `机器学习/机器学习课程笔记.md`
- ReadyBlog: `机器学习/集成学习.md`
- ReadyBlog: `机器学习/优化器.md`
- ReadyBlog: `机器学习/GradientDescent.py`
- ReadyBlog: `笔试记录/反复做的一些题目/逻辑回归梯度下降实现.py`

## 知识画像

| 主题 | 必须掌握 | 面试表达重点 |
|---|---|---|
| 梯度下降 | BGD、SGD、Mini-batch、学习率 | 方差、收敛速度、批大小和泛化 |
| 线性/逻辑回归 | sigmoid、交叉熵、决策边界 | 为什么逻辑回归用交叉熵而不是 MSE |
| 正则化 | L1、L2、Dropout、Early stopping | 控制模型复杂度和泛化误差 |
| 损失函数 | CE、MSE、hinge、ranking loss | 任务假设决定 loss 选择 |
| 集成学习 | Bagging、Boosting、GBDT、XGBoost、LightGBM | 降方差 vs 降偏差，树模型如何迭代 |
| 优化器 | Momentum、Adam、AdamW、学习率衰减 | 一阶动量、二阶动量、权重衰减解耦 |

## 核心公式

逻辑回归预测：

$$
p(y=1\mid x)=\sigma(w^\top x+b)
$$

二分类交叉熵：

$$
\mathcal{L}
=
-\left[y\log p+(1-y)\log(1-p)\right]
$$

L2 正则化：

$$
\mathcal{L}_{reg}=\mathcal{L}+\lambda\|w\|_2^2
$$

## 核心结论

- **优化器解决的是“怎么走”**：SGD 噪声大但可能泛化好，Adam 收敛快但要注意权重衰减和泛化。
- **正则化解决的是“别学太死”**：L1 倾向稀疏，L2 倾向平滑，Early stopping 用验证集控制过拟合。
- **集成学习解决的是“单模型不稳或偏差大”**：Bagging 降方差，Boosting 逐步修正残差/错误样本。
- **面试时要能从 loss 反推任务假设**：分类、回归、排序、偏好优化的目标函数不一样。

## 面试 QA

**Q：为什么逻辑回归常用交叉熵？**

A：逻辑回归建模的是伯努利分布，最大似然推导会得到交叉熵。交叉熵在分类错误且置信度高时惩罚更大，梯度性质也比 sigmoid + MSE 更适合分类优化。

**Q：L1 和 L2 正则有什么区别？**

A：L1 更容易产生稀疏权重，适合特征选择；L2 会让权重整体变小、更平滑，常用于抑制过拟合。几何上，L1 的约束边界更容易和坐标轴相交。

**Q：Bagging 和 Boosting 的区别是什么？**

A：Bagging 并行训练多个基模型，通过平均/投票降低方差；Boosting 串行训练，每一轮关注前一轮错误或残差，主要降低偏差。

## 后续补全计划

- 将逻辑回归梯度下降实现迁入“原理代码”。
- 从 ReadyBlog 优化器笔记中补 Adam / AdamW 对比小节。
- 为 GBDT / XGBoost / LightGBM 单独做面试卡片。
