"""
LeetCode 3 - Longest Substring Without Repeating Characters

Task:
Given a string s, return the length of the longest substring without repeating
characters.

Edit only length_of_longest_substring first. Then run:

    python content/code-training/practice/lc_0003_longest_substring.py
"""


def length_of_longest_substring(s: str) -> int:
    """
    Implement your solution here.

    Interview target:
    - Time: O(n)
    - Space: O(min(n, charset_size))
    - Pattern: sliding window + hash map / hash set
    """
    # TODO: write your solution
    if len(s) == 0:
        return 0
    record = set()
    ind = 0
    ret = -1
    for i in range(len(s)):
        while ind < len(s) and s[ind] not in record:
            record.add(s[ind])
            ind += 1
        ret = max(ret, ind-i)
        record.remove(s[i])
    return ret
            
            
        


def run_tests() -> None:
    cases = [
        ("", 0),
        ("a", 1),
        ("abcabcbb", 3),
        ("bbbbb", 1),
        ("pwwkew", 3),
        ("abba", 2),
        ("dvdf", 3),
        ("anviaj", 5),
    ]

    failed = 0
    for s, expected in cases:
        actual = length_of_longest_substring(s)
        if actual != expected:
            failed += 1
            print(f"FAIL s={s!r}: expected {expected}, got {actual}")
        else:
            print(f"PASS s={s!r}: {actual}")

    if failed:
        print(f"\n{failed}/{len(cases)} cases failed. Keep going.")
    else:
        print("\nAll cases passed.")


if __name__ == "__main__":
    run_tests()
