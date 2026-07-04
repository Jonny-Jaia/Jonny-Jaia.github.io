# 双指针

## 适用信号

- 数组、链表、字符串两端移动
- 有序数组中的二数和、三数和
- 原地删除、合并、反转

## 固定框架

```python
left, right = 0, len(nums) - 1

while left < right:
    current = combine(nums[left], nums[right])
    if current == target:
        handle_answer()
        left += 1
        right -= 1
    elif current < target:
        left += 1
    else:
        right -= 1
```

## 常见坑

- 去重逻辑不完整
- 指针移动条件写反
- 忘记有序性是双指针成立的前提之一
