# 滑动窗口

## 适用信号

- 连续子数组或连续子串
- 最长、最短、满足条件的区间
- 题目可以通过移动左右边界维护状态

## 固定框架

```python
left = 0
state = ...
answer = ...

for right, value in enumerate(nums):
    add(value)

    while window_is_invalid():
        remove(nums[left])
        left += 1

    answer = update(answer, left, right, state)
```

## 面试讲法

窗口内维护一个可快速判断合法性的状态。右边界负责扩张，左边界负责恢复合法性。只要每个元素最多进出窗口一次，总复杂度就是 `O(n)`。

## 常见坑

- 左边界回退
- 更新答案的位置不对
- `while` 和 `if` 的选择不对
- 忘记在左边界移动时同步删除状态
