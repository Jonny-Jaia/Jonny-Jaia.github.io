def search_rotated_array(nums: list[int], target: int) -> int:
    """
    旋转有序数组搜索：在没有重复元素的前提下，用二分保持 O(log n)。

    核心判断：每一轮至少有一半区间是严格有序的。
    - 如果 nums[left] <= nums[mid]，说明左半边有序。
    - 否则右半边有序。
    只要 target 落在有序半边，就收缩到那一半；否则去另一半。

    时间复杂度：O(log n)
    空间复杂度：O(1)
    """
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid

        # 左半区间 [left, mid] 有序：可以直接判断 target 是否落在这段单调区间里。
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # 否则右半区间 [mid, right] 有序，同理判断 target 是否落在右半边。
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1

    return -1
