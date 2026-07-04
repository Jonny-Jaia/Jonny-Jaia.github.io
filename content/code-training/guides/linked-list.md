# 链表

## 适用信号

- 反转、合并、删除节点
- 快慢指针
- 需要 dummy node 简化头节点处理

## 反转框架

```python
prev = None
cur = head

while cur:
    nxt = cur.next
    cur.next = prev
    prev = cur
    cur = nxt

return prev
```

## 常见坑

- 断链前没有保存 `next`
- 删除头节点时没有使用 dummy
- 快慢指针终止条件不清楚
