# 图搜索

## 适用信号

- 岛屿数量、连通块
- 最短路径
- 拓扑排序
- 网格搜索

## DFS 网格框架

```python
def dfs(r, c):
    if out_of_bound(r, c) or visited_or_blocked(r, c):
        return
    mark_visited(r, c)
    for dr, dc in directions:
        dfs(r + dr, c + dc)
```

## BFS 最短路框架

```python
from collections import deque

queue = deque([(start, 0)])
visited = {start}

while queue:
    node, dist = queue.popleft()
    if is_target(node):
        return dist
    for nxt in neighbors(node):
        if nxt not in visited:
            visited.add(nxt)
            queue.append((nxt, dist + 1))
```

## 常见坑

- 访问标记太晚导致重复入队
- 网格边界判断遗漏
- DFS 递归深度过大时需要改迭代或 BFS
