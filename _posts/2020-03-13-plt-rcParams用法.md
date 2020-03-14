---
layout:     post
title:      plt.rcParams用法
subtitle:   小技巧
date:       2020-03-13
author:     Jonny
header-img: img/post-bg-2015.jpg
catalog: true
tags:
    - python
    - matplotlib

---

> 搬运相关资料



# Matplotlib中plt.rcParams用法

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
%matplotlib inline    

# 生成数据
x = np.linspace(0, 4*np.pi)
y = np.sin(x)

plt.rcParams['figure.figsize'] = (5.0, 4.0)     # 显示图像的最大范围

plt.rcParams['image.interpolation'] = 'nearest' # 差值方式，设置 interpolation style

plt.rcParams['image.cmap'] = 'gray'             # 灰度空间

#设置rc参数显示中文标题

#设置字体为SimHei显示中文
plt.rcParams['font.sans-serif'] = 'SimHei'

#设置正常显示字符
plt.rcParams['axes.unicode_minus'] = False
plt.title('sin曲线')

#设置线条样式
plt.rcParams['lines.linestyle'] = '-.'
#设置线条宽度
plt.rcParams['lines.linewidth'] = 3
#绘制sin曲线
plt.plot(x, y, label='$sin(x)$')
 
# plt.savefig('sin.png')
# plt.show()
```

# pandas cumsum函数用法

> 累加作用

```python
import numpy as np  
arr  = np.array([[[1,2,3],[3,4,5]],[[6,7,8],[1,2,3]]])   
print(arr)
print(arr.cumsum(0))  
print(arr.cumsum(1))  
print(arr.cumsum(2))
```

结果：

```
#arr：
[[[1 2 3]
  [3 4 5]]
 [[6 7 8]
  [1 2 3]]]
#arr.cumsum(0)
[[[ 1  2  3]
  [ 3  4  5]]
 [[ 7  9 11]
  [ 4  6  8]]]
#arr.cumsum(1)
[[[ 1  2  3]
  [ 4  6  8]]
 [[ 6  7  8]
  [ 7  9 11]]]
#arr.cumsum(2)
[[[ 1  3  6]
  [ 3  7 12]]
 [[ 6 13 21]
  [ 1  3  6]]]
```

**注释：**

arr是一个2*2*3三维矩阵，索引值为0，1，2

cumsum(0)：实现0轴上的累加：以最外面的数组元素为单位，以[[1,2,3],[4,5,6]]为开始实现后面元素的对应累加

cumsum(1)：实现1轴上的累加：以中间数组元素为单位，以[1,2,3]为开始，实现后面元素的对应累加

cumsum(2)：实现2轴上的累加：以最里面的元素为累加单位，即1为开始，实现后面的元素累加 



# numpy查漏知识

> 预防忘记

## **1. arange函数**

```python
a = np.arange(0, 100, 10)
b = np.arange(0, 3, 0.3)
a.reshape(5,-1)
#结果：
[ 0 10 20 30 40 50 60 70 80 90]
[0.  0.3 0.6 0.9 1.2 1.5 1.8 2.1 2.4 2.7]
[[ 0, 10],
 [20, 30],
 [40, 50],
 [60, 70],
 [80, 90]]
```

## **2. linspace函数**

>在生成浮点数列表时，最好不要使用`np.arange`，而是使用`np.linspace`，linspace函数通过指定开始值、终值和元素个数来创建一维数组，可以通过endpoint关键字指定是否包括终值，缺省设置是包括终值。

```python
np.linspace(start,stop,num)
np.linspace(0, 1, 10) # 步长为1/9
np.linspace(0, 1, 10, endpoint=False) # 步长为1/10
```

## **3. logspace函数**