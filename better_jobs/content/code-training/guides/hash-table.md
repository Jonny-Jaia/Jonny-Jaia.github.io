# 哈希表

## 适用信号

- 快速查找是否出现过
- 计数、分组、映射关系
- 需要把 `O(n^2)` 查找降到 `O(n)`

## 固定框架

```python
seen = {}

for index, value in enumerate(nums):
    key = build_key(value)
    if need(key) in seen:
        return build_answer(seen[need(key)], index)
    seen[key] = index
```

## 常见坑

- 先查再存，还是先存再查，要看是否允许使用当前元素
- key 设计错误，比如异位词分组时没有排序或计数
- 覆盖旧下标导致边界错误
