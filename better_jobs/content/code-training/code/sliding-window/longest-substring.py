def length_of_longest_substring(s: str) -> int:
    """
    无重复字符最长子串：滑动窗口模板。

    不变量：窗口 [left, right] 内始终没有重复字符。
    last_seen 记录字符上一次出现的位置；如果重复字符落在当前窗口内，就把 left 跳到重复字符后一位。

    时间复杂度：O(n)，每个字符最多被 right 扫一次、left 跳过一次。
    空间复杂度：O(k)，k 是字符集大小。
    """
    last_seen: dict[str, int] = {}
    left = 0
    best = 0

    for right, char in enumerate(s):
        # 只有上次出现位置在当前窗口内，才需要移动 left。
        if char in last_seen and last_seen[char] >= left:
            left = last_seen[char] + 1

        last_seen[char] = right
        best = max(best, right - left + 1)

    return best
