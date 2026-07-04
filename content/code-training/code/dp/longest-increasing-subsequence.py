def length_of_lis(nums: list[int]) -> int:
    """
    最长递增子序列长度：贪心 + 二分，也叫 patience sorting 模板。

    tails[i] 表示“长度为 i + 1 的递增子序列中，最小可能结尾值”。
    tails 不是原数组里的真实子序列，但它的长度一定等于当前 LIS 长度。

    为什么可以替换：同样长度下结尾越小，后续越容易接上更大的数。
    时间复杂度：O(n log n)，空间复杂度：O(n)。
    """
    tails: list[int] = []

    for num in nums:
        left, right = 0, len(tails)

        # 找到第一个 >= num 的位置：num 可以成为该长度下更小的结尾。
        while left < right:
            mid = (left + right) // 2
            if tails[mid] < num:
                left = mid + 1
            else:
                right = mid

        if left == len(tails):
            # num 比所有结尾都大，可以扩展出更长的递增子序列。
            tails.append(num)
        else:
            # 用更小的结尾替换，保持未来扩展空间最大。
            tails[left] = num

    return len(tails)
