"""
LeetCode 33 - Search in Rotated Sorted Array

Task:
Given a rotated sorted array with distinct values and a target, return the
target index. Return -1 if the target is not present.

Edit only search first. Then run:

    python content/code-training/practice/lc_0033_search_rotated_array.py
"""


def search(nums: list[int], target: int) -> int:
    """
    Implement your solution here.

    Interview target:
    - Time: O(log n)
    - Space: O(1)
    - Pattern: binary search on a partially ordered array
    """
    # TODO: write your solution
    # 采用二分查找
    if len(nums) == 0:
        return -1
    if len(nums)==1:
        return 0 if target==nums[0] else -1
    ind = 0
    start = 0
    while 0<=ind<=len(nums)-1:
        if target==nums[ind]:
            return ind
        elif target>nums[ind]:
            if ind+1<=len(nums)-1 and nums[ind+1]>nums[ind]:
                ind += 1
            else:
                return -1
        else:
            if ind==0 and start<1:
                ind = len(nums)-1
                start += 1
            elif ind-1>=0 and nums[ind-1]<nums[ind]:
                ind -= 1
            else:
                return -1
    return -1
        


def run_tests() -> None:
    cases = [
        ([4, 5, 6, 7, 0, 1, 2], 0, 4),
        ([4, 5, 6, 7, 0, 1, 2], 3, -1),
        ([1], 0, -1),
        ([1], 1, 0),
        ([3, 1], 1, 1),
        ([3, 1], 3, 0),
        ([5, 1, 3], 5, 0),
        ([5, 1, 3], 3, 2),
        ([1, 3, 5], 3, 1),
        ([1, 3, 5], 0, -1),
        ([6, 7, 1, 2, 3, 4, 5], 6, 0),
        ([6, 7, 1, 2, 3, 4, 5], 5, 6),
    ]

    failed = 0
    for nums, target, expected in cases:
        actual = search(nums, target)
        if actual != expected:
            failed += 1
            print(f"FAIL nums={nums!r}, target={target}: expected {expected}, got {actual}")
        else:
            print(f"PASS nums={nums!r}, target={target}: {actual}")

    if failed:
        print(f"\n{failed}/{len(cases)} cases failed. Keep going.")
    else:
        print("\nAll cases passed.")


if __name__ == "__main__":
    run_tests()
