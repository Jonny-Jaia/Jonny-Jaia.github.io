window.BJ_CONTENT = window.BJ_CONTENT || {};

window.BJ_CONTENT.codeTraining = {
  id: "code-training",
  title: "代码训练",
  eyebrow: "Coding Interview Training",
  description:
    "独立于知识库的手撕代码训练区。题单维护状态、薄弱点、标签和复盘；具体代码与算法模板分别放在独立文档中。",
  assessment: {
    updatedAt: "2026-06-28",
    level: "起步阶段：滑动窗口已通过验证",
    summary:
      "已完成 LC3，能正确使用 set + 双指针维护无重复窗口，并通过固定边界与随机交叉验证。下一阶段重点诊断二分边界、图搜索 visited 时机和 DP 状态定义。",
    strengths: ["滑动窗口不变量理解正确", "能通过额外随机验证", "代码复杂度达到 O(n)"],
    weaknesses: ["需要在面试表达中更自然地处理初始化", "二分边界能力待诊断", "DP 与图搜索尚未开始系统训练"],
    nextFocus: ["二分查找", "图 / DFS", "动态规划"],
    completedProblems: ["lc-3"]
  },
  algorithmGuides: [
    { id: "guide-sliding-window", title: "滑动窗口", docRef: "sliding-window", tags: ["数组", "字符串"] },
    { id: "guide-binary-search", title: "二分查找", docRef: "binary-search", tags: ["数组", "边界"] },
    { id: "guide-two-pointers", title: "双指针", docRef: "two-pointers", tags: ["数组", "字符串"] },
    { id: "guide-hash-table", title: "哈希表", docRef: "hash-table", tags: ["查找", "计数"] },
    { id: "guide-stack", title: "栈与单调栈", docRef: "stack", tags: ["栈", "单调结构"] },
    { id: "guide-linked-list", title: "链表", docRef: "linked-list", tags: ["指针", "链表"] },
    { id: "guide-tree", title: "二叉树", docRef: "tree", tags: ["DFS", "BFS"] },
    { id: "guide-graph-search", title: "图搜索", docRef: "graph-search", tags: ["DFS", "BFS"] },
    { id: "guide-backtracking", title: "回溯", docRef: "backtracking", tags: ["枚举", "剪枝"] },
    { id: "guide-dp", title: "动态规划", docRef: "dynamic-programming", tags: ["状态", "转移"] },
    { id: "guide-heap", title: "堆与 TopK", docRef: "heap-topk", tags: ["堆", "TopK"] },
    { id: "guide-prefix-sum", title: "前缀和", docRef: "prefix-sum", tags: ["区间", "哈希表"] }
  ],
  problemSets: [
    {
      id: "lc-3",
      group: "滑动窗口",
      title: "3. Longest Substring Without Repeating Characters",
      difficulty: "Medium",
      source: "https://leetcode.com/problems/longest-substring-without-repeating-characters/",
      tags: ["滑动窗口", "哈希表"],
      rewrittenPrompt: "给定字符串，求不含重复字符的最长连续子串长度。",
      status: "已完成",
      weakness: "整体掌握；注意面试中可把 ret 初始化为 0，减少空串特判。",
      codeRef: "sliding-window/longest-substring",
      guideRef: "sliding-window",
      reviewRef: "lc-0003-longest-substring",
      reviewPlan: "已完成。后续复盘重点：讲清窗口不变量和 O(n) 原因。"
    },
    {
      id: "lc-33",
      group: "二分查找",
      title: "33. Search in Rotated Sorted Array",
      difficulty: "Medium",
      source: "https://leetcode.com/problems/search-in-rotated-sorted-array/",
      tags: ["二分", "数组"],
      rewrittenPrompt: "在旋转后的升序数组中查找目标值，要求对数复杂度。",
      status: "待练习",
      weakness: "待诊断",
      codeRef: "binary-search/search-rotated-array",
      guideRef: "binary-search",
      reviewRef: "",
      reviewPlan: "训练如何判断哪一侧有序，并避免边界死循环。"
    },
    {
      id: "lc-146",
      group: "设计题",
      title: "146. LRU Cache",
      difficulty: "Medium",
      source: "https://leetcode.com/problems/lru-cache/",
      tags: ["哈希表", "双向链表", "设计"],
      rewrittenPrompt: "实现固定容量缓存，访问和写入都需要维护最近使用顺序。",
      status: "待练习",
      weakness: "待诊断",
      codeRef: "design/lru-cache",
      guideRef: "linked-list",
      reviewRef: "",
      reviewPlan: "重点练哈希表加双向链表的节点移动。"
    },
    {
      id: "lc-200",
      group: "图 / DFS",
      title: "200. Number of Islands",
      difficulty: "Medium",
      source: "https://leetcode.com/problems/number-of-islands/",
      tags: ["DFS", "BFS", "网格"],
      rewrittenPrompt: "统计二维网格中相连陆地块的数量。",
      status: "待练习",
      weakness: "待诊断",
      codeRef: "graph/number-of-islands",
      guideRef: "graph-search",
      reviewRef: "",
      reviewPlan: "练习 visited 标记、边界检查和 flood fill。"
    },
    {
      id: "lc-300",
      group: "动态规划",
      title: "300. Longest Increasing Subsequence",
      difficulty: "Medium",
      source: "https://leetcode.com/problems/longest-increasing-subsequence/",
      tags: ["DP", "二分"],
      rewrittenPrompt: "求数组中最长严格递增子序列长度。",
      status: "待练习",
      weakness: "待诊断",
      codeRef: "dp/longest-increasing-subsequence",
      guideRef: "dynamic-programming",
      reviewRef: "",
      reviewPlan: "先掌握 O(n^2) DP，再进阶到 patience sorting。"
    }
  ]
};
