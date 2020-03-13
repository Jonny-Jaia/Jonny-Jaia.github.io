---
layout:     post
title:      plt.rcParams用法  
subtitle:   matplotlib可视化
date:       2020-03-13
author:     Jonny
header-img: img/post-bg-ios9-web.jpg
catalog:true
tags:
    - python
    - matplotlib
    - 可视化
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

