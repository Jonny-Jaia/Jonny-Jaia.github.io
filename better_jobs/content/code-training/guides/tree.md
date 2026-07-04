# 二叉树

## 适用信号

- DFS 递归
- BFS 层序遍历
- 二叉搜索树性质
- 最近公共祖先

## DFS 框架

```python
def dfs(node):
    if not node:
        return base_value

    left = dfs(node.left)
    right = dfs(node.right)
    return combine(node, left, right)
```

## BFS 框架

```python
from collections import deque

queue = deque([root])
while queue:
    for _ in range(len(queue)):
        node = queue.popleft()
        handle(node)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
```

## 常见坑

- 递归返回值没有定义清楚
- BST 只比较父子节点，没有传上下界
- 层序遍历没有固定当前层长度
