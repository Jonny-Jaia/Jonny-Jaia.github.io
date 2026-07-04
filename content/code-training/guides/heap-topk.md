# 堆与 TopK

## 适用信号

- 第 K 大 / 第 K 小
- 动态维护最值
- 合并多个有序结构

## 固定框架

```python
import heapq

heap = []
for value in nums:
    heapq.heappush(heap, value)
    if len(heap) > k:
        heapq.heappop(heap)

return heap[0]
```

## 常见坑

- Python 默认是小根堆
- 第 K 大通常维护大小为 K 的小根堆
- 复杂度是 `O(n log k)`，不是 `O(n log n)`
