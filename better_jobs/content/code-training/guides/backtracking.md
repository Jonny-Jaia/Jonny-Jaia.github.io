# 回溯

## 适用信号

- 全排列、组合、子集
- 需要枚举所有方案
- 有选择、撤销选择、剪枝

## 固定框架

```python
path = []
answer = []

def backtrack(start):
    if is_complete(path):
        answer.append(path[:])
        return

    for i in range(start, len(candidates)):
        if should_skip(i):
            continue
        path.append(candidates[i])
        backtrack(next_start(i))
        path.pop()
```

## 常见坑

- 忘记 `path[:]` 复制
- 去重条件写错
- 组合和排列的 `start` 语义混淆
