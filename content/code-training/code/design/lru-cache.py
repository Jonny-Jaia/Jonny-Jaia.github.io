class Node:
    """双向链表节点：同时保存 key 和 value，方便淘汰尾节点时从哈希表删除。"""

    def __init__(self, key: int = 0, value: int = 0):
        self.key = key
        self.value = value
        self.prev: Node | None = None
        self.next: Node | None = None


class LRUCache:
    """
    LRU 缓存手撕模板：哈希表 + 双向链表。

    不变量：
    - head 后面的节点永远是最近使用的节点。
    - tail 前面的节点永远是最久未使用的节点。
    - cache[key] 直接定位到链表节点，因此 get/put 都是 O(1)。

    复杂度：get O(1)，put O(1)，额外空间 O(capacity)。
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: dict[int, Node] = {}

        # 使用哨兵头尾节点，可以避免插入/删除时反复判断空链表或边界节点。
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1

        node = self.cache[key]
        # 只要被访问，就变成“最近使用”，移动到头部。
        self._move_to_front(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            self._move_to_front(node)
            return

        # 新节点一定是最近使用的节点，插到 head 后面。
        node = Node(key, value)
        self.cache[key] = node
        self._add_after_head(node)

        # 超出容量时，淘汰 tail 前面的最久未使用节点。
        if len(self.cache) > self.capacity:
            removed = self._remove_tail()
            del self.cache[removed.key]

    def _move_to_front(self, node: Node) -> None:
        self._remove(node)
        self._add_after_head(node)

    def _add_after_head(self, node: Node) -> None:
        first = self.head.next
        node.prev = self.head
        node.next = first
        first.prev = node
        self.head.next = node

    def _remove(self, node: Node) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev

    def _remove_tail(self) -> Node:
        node = self.tail.prev
        self._remove(node)
        return node
