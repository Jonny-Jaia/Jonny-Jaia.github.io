def num_islands(grid: list[list[str]]) -> int:
    """
    岛屿数量：网格 DFS flood fill 模板。

    思路：遇到一个未访问的陆地 '1'，岛屿数量 +1，然后把与它连通的所有陆地都淹掉。
    访问标记：直接把 '1' 改成 '0'，避免额外 visited 数组。

    时间复杂度：O(mn)，每个格子最多访问一次。
    空间复杂度：O(mn)，最坏情况下递归栈可能覆盖整张网格。
    """
    if not grid or not grid[0]:
        return 0

    rows, cols = len(grid), len(grid[0])
    count = 0

    def dfs(row: int, col: int) -> None:
        # 越界或遇到水域/已访问陆地，直接返回。
        if row < 0 or row >= rows or col < 0 or col >= cols:
            return
        if grid[row][col] != "1":
            return

        # 标记当前陆地已访问，再向四个方向扩散。
        grid[row][col] = "0"
        dfs(row + 1, col)
        dfs(row - 1, col)
        dfs(row, col + 1)
        dfs(row, col - 1)

    for row in range(rows):
        for col in range(cols):
            if grid[row][col] == "1":
                count += 1
                dfs(row, col)

    return count
