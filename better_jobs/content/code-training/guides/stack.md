# 栈与单调栈

## 适用信号

- 最近一个更大 / 更小元素
- 括号匹配
- 需要维护一个单调结构

## 单调栈框架

```python
stack = []
answer = default_answer()

for i, value in enumerate(nums):
    while stack and should_pop(nums[stack[-1]], value):
        j = stack.pop()
        answer[j] = build_answer(j, i)
    stack.append(i)
```

## 常见坑

- 栈里存值还是下标要想清楚
- 单调递增和单调递减方向写反
- 没处理栈中最后没有匹配的元素
