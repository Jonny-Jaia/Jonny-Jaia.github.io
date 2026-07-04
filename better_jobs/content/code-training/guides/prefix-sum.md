# 前缀和

## 适用信号

- 子数组和
- 区间和查询
- 需要把区间和转为两个前缀差

## 固定框架

```python
count = 0
prefix = 0
seen = {0: 1}

for value in nums:
    prefix += value
    count += seen.get(prefix - target, 0)
    seen[prefix] = seen.get(prefix, 0) + 1
```

## 常见坑

- 忘记初始化 `{0: 1}`
- 子数组必须连续
- 负数存在时不能用滑动窗口替代前缀和
