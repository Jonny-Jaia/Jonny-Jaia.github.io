# 二分查找

## 适用信号

- 有序数组或局部有序结构
- 要求 `O(log n)`
- 查找边界、最小可行值、最大可行值

## 固定框架：闭区间

```python
left, right = 0, len(nums) - 1

while left <= right:
    mid = (left + right) // 2
    if nums[mid] == target:
        return mid
    if should_go_left(mid):
        right = mid - 1
    else:
        left = mid + 1

return -1
```

## 固定框架：答案二分

```python
left, right = min_value, max_value

while left < right:
    mid = (left + right) // 2
    if feasible(mid):
        right = mid
    else:
        left = mid + 1

return left
```

## 常见坑

- `left <= right` 和 `left < right` 混用
- `mid` 更新后边界不收缩导致死循环
- 旋转数组中没有先判断哪一侧有序
