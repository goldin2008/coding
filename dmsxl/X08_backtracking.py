"""
回溯法也可以叫做回溯搜索法，它是一种搜索的方式。
在二叉树系列中，我们已经不止一次，提到了回溯，例如二叉树：以为使用了递归，其实还隐藏着回溯 。
回溯是递归的副产品，只要有递归就会有回溯。
所以以下讲解中，回溯函数也就是递归函数，指的都是一个函数。

因为回溯的本质是穷举, 穷举所有可能, 然后选出我们想要的答案, 如果想让回溯法高效一些, 可以加一些剪枝的操作, 但也改不了回溯法就是穷举的本质。
回溯法解决的问题都可以抽象为树形结构, 所有回溯法的问题都可以抽象为树形结构
本题我把回溯问题抽象为树形结构, 可以直观的看出其搜索的过程: for循环横向遍历, 递归纵向遍历, 回溯不断调整结果集。
因为回溯法解决的都是在集合中递归查找子集, 集合的大小就构成了树的宽度, 递归的深度, 都构成的树的深度。
递归就要有终止条件, 所以必然是一棵高度有限的树 (N叉树)

回溯三部曲
    1. 回溯函数模板, 返回值以及参数
    2. 回溯函数终止条件
    3. 回溯搜索的遍历过程

for循环横向遍历,递归纵向遍历,回溯不断调整结果集
大家可以从图中看出for循环可以理解是横向遍历, backtracking (递归) 就是纵向遍历, 这样就把这棵树全遍历完了, 
一般来说, 搜索叶子节点就是找的其中一个结果了。优化回溯算法**只有剪枝**一种方法

如何判断call递归函数的时候要不要index的值, 如果要的话是i+1还是i? 
    如果是组合问题
        同一个集合中取
            如果不可以有重复的元素, 需要index, 需要i+1 那么每一层的元素都是从上层的元素开始选取, 也就是从i+1开始选取, 这样就不会出现重复的组合。
            如果可以有重复的元素, 需要index, 需要i
        不同集合中取组合
            那么就不需要考虑i+1还是i了, 因为每个集合之间是互不影响的。

    如果是排列问题
        那么每一层的元素都是从0开始选取, 也就是从i开始选取, 这样就会出现重复的排列。不需要index, 不需要i+1

***剪枝去重只需要同层去重, 所以只需要在for循环里面考虑去重条件***
所以整个for循环都可以用一个set(), array()或者dict()来存储已经出现过的元素. 
call递归函数后, 进入下一层可以再新建一个set(), array()或者dict()来存储已经出现过的元素, 这样就可以避免重复的元素进入下一层递归。


回溯算法能解决如下问题:
组合问题: N个数里面按一定规则找出k个数的集合
排列问题: N个数按一定规则全排列, 有几种排列方式
切割问题: 一个字符串按一定规则有几种切割方式
子集问题: 一个N个数的集合里有多少符合条件的子集
棋盘问题: N皇后, 解数独等等
"""
# 以下在计算空间复杂度的时候我都把系统栈（不是数据结构里的栈）所占空间算进去。

# 子集问题分析：
# 时间复杂度：O(2^n)，因为每一个元素的状态无外乎取与不取，所以时间复杂度为O(2^n)
# 空间复杂度：O(n)，递归深度为n，所以系统栈所用空间为O(n)，每一层递归所用的空间都是常数级别，
    # 注意代码里的result和path都是全局变量，就算是放在参数里，传的也是引用，并不会新申请内存空间，最终空间复杂度为O(n)
    # path和result：
    # path的大小最多为n（当选择所有元素时）。
    # result最终会存储所有2^n个子集，但其空间不计入空间复杂度分析（属于输出空间）。
    # 关键点在于：path在递归过程中是共享的（通过引用传递或全局变量），不会在每个递归层都复制一份。

# 排列问题分析：
# 时间复杂度：O(n!)，这个可以从排列的树形图中很明显发现，每一层节点为n，第二层每一个分支都延伸了n-1个分支，
# 再往下又是n-2个分支，所以一直到叶子节点一共就是 n * n-1 * n-2 * ..... 1 = n!。
# 空间复杂度：O(n)，和子集问题同理。

# 组合问题分析：
# 时间复杂度: O(C(n, k))，最坏情况下是 O(2^n / sqrt(n))，而不是简单的 O(2^n)。
# 空间复杂度: O(k) 最大递归深度为 k（生成 k 个元素的组合），因此空间复杂度为 O(k)。如果 k 可以接近 n（如 k = n），则空间复杂度为 O(n)。

# N皇后问题分析：
# 时间复杂度：O(n!) ，其实如果看树形图的话，直觉上是O(n^n)，但皇后之间不能见面所以在搜索的过程中是有剪枝的，
# 最差也就是O（n!），n!表示n * (n-1) * .... * 1。
# 空间复杂度：O(n)，和子集问题同理。

# 解数独问题分析：
# 时间复杂度：O(9^m) , m是'.'的数目。
# 空间复杂度：O(n^2)，递归的深度是n^2

# 一般说道回溯算法的复杂度，都说是指数级别的时间复杂度，这也算是一个概括吧！

# 回溯算法模板：
# 用for循环来横向遍历,递归来进行纵向遍历
# for循环可以理解为选树的root，然后从root开始递归整个树里面的路径
vector<vector<int>> result; // 存放符合条件结果的集合
vector<int> path; // 用来存放符合条件结果

void backtracking(参数) { # 1.回溯函数模板返回值以及参数
    # 2.回溯函数终止条件
    if (终止条件) {
        存放结果;
        return;
    }

    # 3.回溯搜索的遍历过程
    for (选择: 本层集合中元素（树中节点孩子的数量就是集合的大小）) { 
        处理节点;
        backtracking(路径, 选择列表); // 递归
        回溯, 撤销处理结果
    }
}


#1 (Medium) 第77题.组合
# 给定两个整数 n 和 k, 返回 1 ... n 中所有可能的 k 个数的组合。
# 示例:
# 输入: n = 4, k = 2
# 输出:
# [[2,4], [3,4], [2,3], [1,2], [1,3], [1,4],]
# *** 剪枝优化
# 剪枝精髓是：for循环在寻找起点的时候要有一个范围，如果这个起点到集合终止之间的元素已经不够 题目要求的k个元素了，就没有必要搜索了。
# Time Complexity: O(nCk) 其中nCk是组合数
# Space Complexity: O(nCk) 其中nCk是组合数
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        res=[]  #存放符合条件结果的集合
        path=[]  #用来存放符合条件结果

        def backtrack(n, k, startIndex):
            if len(path) == k:
                res.append(path[:]) #self.path.copy()  # shallow copy
                return

            # for i in range(startIndex, n+1):
            for i in range(startIndex, n- (k-len(path)) +2):  #优化的地方
                path.append(i)  #处理节点
                backtrack(n, k, i+1)  #递归
                path.pop()  #回溯, 撤销处理的节点

        backtrack(n, k, 1)
        return res
# 未剪枝优化
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        result = []  # 存放结果集
        self.backtracking(n, k, 1, [], result)
        return result

    def backtracking(self, n, k, startIndex, path, result):
        if len(path) == k:
            result.append(path[:])
            return
        for i in range(startIndex, n + 1):  # 需要优化的地方
            path.append(i)  # 处理节点
            self.backtracking(n, k, i + 1, path, result)
            path.pop()  # 回溯，撤销处理的节点
# # 剪枝优化
# class Solution:
#     def combine(self, n: int, k: int) -> List[List[int]]:
#         result = []  # 存放结果集
#         self.backtracking(n, k, 1, [], result)
#         return result
#     def backtracking(self, n, k, startIndex, path, result):
#         if len(path) == k:
#             result.append(path[:])
#             return
#         for i in range(startIndex, n - (k - len(path)) + 2):  # 优化的地方
#             path.append(i)  # 处理节点
#             self.backtracking(n, k, i + 1, path, result)
#             path.pop()  # 回溯，撤销处理的节点


#2 (Medium) 216.组合总和III
    # 找出所有相加之和为 n 的 k 个数的组合。组合中只允许含有 1 - 9 的正整数, 并且每种组合中不存在重复的数字。
    # 示例 1: 输入: k = 3, n = 7 输出: [[1,2,4]]
    # 示例 2: 输入: k = 3, n = 9 输出: [[1,2,6], [1,3,5], [2,3,4]]
# 本题的剪枝会好想一些，即：已选元素总和如果已经大于n（题中要求的和）了，那么往后遍历就没有意义了，直接剪掉。
# Time Complexity: O(9Ck) 其中9Ck是组合数
# Space Complexity: O(9Ck) 其中9Ck是组合数
class Solution:
    def __init__(self):
        self.res = []
        self.path = []
        self.sum_now = 0

    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        self.backtracking(k, n, 1)
        return self.res

    def backtracking(self, k: int, n: int, start_num: int):
        # 剪枝
        if self.sum_now > n:  # 剪枝
            return
        # Base Case
        if len(self.path) == k:  # len(path)==k时不管sum是否等于n都会返回
            if self.sum_now == n:
                self.res.append(self.path[:]) #self.path.copy()  # shallow copy
            return
        # 单层递归逻辑
        for i in range(start_num, 10 - (k - len(self.path)) + 1):
            self.path.append(i)
            self.sum_now += i
            self.backtracking(k, n, i + 1)
            self.path.pop()
            self.sum_now -= i
# class Solution:
#     def combinationSum3(self, k: int, n: int) -> List[List[int]]:
#         result = []  # 存放结果集
#         self.backtracking(n, k, 0, 1, [], result)
#         return result

#     def backtracking(self, targetSum, k, currentSum, startIndex, path, result):
#         if currentSum > targetSum:  # 剪枝操作
#             return  # 如果path的长度等于k但currentSum不等于targetSum，则直接返回
#         if len(path) == k:
#             if currentSum == targetSum:
#                 result.append(path[:])
#             return
#         for i in range(startIndex, 9 - (k - len(path)) + 2):  # 剪枝
#             currentSum += i  # 处理
#             path.append(i)  # 处理
#             self.backtracking(targetSum, k, currentSum, i + 1, path, result)  # 注意i+1调整startIndex
#             currentSum -= i  # 回溯
#             path.pop()  # 回溯


#X3 (Medium) 17.电话号码的字母组合
    # 给定一个仅包含数字 2-9 的字符串, 返回所有它能表示的字母组合。
    # 示例: 输入："23" 输出：["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
# 因为本题每一个数字代表的是不同集合，也就是求不同集合之间的组合，而回溯算法：求组合问题！ 和回溯算法：求组合总和！ 都是是求同一个集合中的组合！
# 回溯
# class Solution:
#     def __init__(self):
#         self.answers: List[str] = []
#         self.answer: str = ''
#         self.letter_map = {
#             '2': 'abc',
#             '3': 'def',
#             '4': 'ghi',
#             '5': 'jkl',
#             '6': 'mno',
#             '7': 'pqrs',
#             '8': 'tuv',
#             '9': 'wxyz'
#         }

#     def letterCombinations(self, digits: str) -> List[str]:
#         self.answers.clear()
#         if not digits: return []
#         self.backtracking(digits, 0)
#         return self.answers
    
#     def backtracking(self, digits: str, index: int) -> None:
#         # 回溯函数没有返回值
#         # Base Case
#         if index == len(digits):    # 当遍历穷尽后的下一层时
#             self.answers.append(self.answer)
#             return 
#         # 单层递归逻辑  
#         letters: str = self.letter_map[digits[index]]
#         for letter in letters:
#             self.answer += letter   # 处理
#             self.backtracking(digits, index + 1)    # 递归至下一层
#             self.answer = self.answer[:-1]  # 回溯
# # 回溯简化
# class Solution:
#     def __init__(self):
#         self.answers: List[str] = []
#         self.letter_map = {
#             '2': 'abc',
#             '3': 'def',
#             '4': 'ghi',
#             '5': 'jkl',
#             '6': 'mno',
#             '7': 'pqrs',
#             '8': 'tuv',
#             '9': 'wxyz'
#         }

#     def letterCombinations(self, digits: str) -> List[str]:
#         self.answers.clear()
#         if not digits: return []
#         self.backtracking(digits, 0, '')
#         return self.answers
    
#     def backtracking(self, digits: str, index: int, answer: str) -> None:
#         # 回溯函数没有返回值
#         # Base Case
#         if index == len(digits):    # 当遍历穷尽后的下一层时
#             self.answers.append(answer)
#             return 
#         # 单层递归逻辑  
#         letters: str = self.letter_map[digits[index]]
#         for letter in letters:
#             self.backtracking(digits, index + 1, answer + letter)    # 递归至下一层 + 回溯
# 回溯
class Solution:
    def __init__(self):
        self.letterMap = [
            "",     # 0
            "",     # 1
            "abc",  # 2
            "def",  # 3
            "ghi",  # 4
            "jkl",  # 5
            "mno",  # 6
            "pqrs", # 7
            "tuv",  # 8
            "wxyz"  # 9
        ]
        self.result = []
        self.s = ""

    def letterCombinations(self, digits):
        if len(digits) == 0:
            return self.result
        self.backtracking(digits, 0)
        return self.result

    def backtracking(self, digits, index):
        if index == len(digits):
            self.result.append(self.s)
            return

        digit = int(digits[index])    # 将索引处的数字转换为整数
        letters = self.letterMap[digit]    # 获取对应的字符集
        # 遍历字符集
        for i in range(len(letters)):
            self.s += letters[i]    # 处理字符
            self.backtracking(digits, index + 1)    # 递归调用，注意索引加1，处理下一个数字
            self.s = self.s[:-1]    # 回溯，删除最后添加的字符
# *** 回溯精简（版本一）
# Time Complexity: O(4^n) 其中n是输入数字的长度，因为每个数字最多对应4个字母
# Space Complexity: O(n) 其中n是输入数字的长度，因为递归深度最多为n
class Solution:
    def __init__(self):
        self.letterMap = [
            "",     # 0
            "",     # 1
            "abc",  # 2
            "def",  # 3
            "ghi",  # 4
            "jkl",  # 5
            "mno",  # 6
            "pqrs", # 7
            "tuv",  # 8
            "wxyz"  # 9
        ]
        self.result = []
    # 回溯
    def letterCombinations(self, digits):
        if len(digits) == 0:
            return self.result
        self.getCombinations(digits, 0, "")
        return self.result    
    # 回溯函数
    def getCombinations(self, digits, index, s):
        if index == len(digits):
            self.result.append(s)
            return
        
        digit = int(digits[index])
        letters = self.letterMap[digit]
        for letter in letters:
            self.getCombinations(digits, index + 1, s + letter)
# 回溯精简（版本二）
# class Solution:
#     def __init__(self):
#         self.letterMap = [
#             "",     # 0
#             "",     # 1
#             "abc",  # 2
#             "def",  # 3
#             "ghi",  # 4
#             "jkl",  # 5
#             "mno",  # 6
#             "pqrs", # 7
#             "tuv",  # 8
#             "wxyz"  # 9
#         ]
#     def letterCombinations(self, digits):
#         result = []
#         if len(digits) == 0:
#             return result
#         self.getCombinations(digits, 0, "", result)
#         return result    
#     def getCombinations(self, digits, index, s, result):
#         if index == len(digits):
#             result.append(s)
#             return
#         digit = int(digits[index])
#         letters = self.letterMap[digit]
#         for letter in letters:
#             self.getCombinations(digits, index + 1, s + letter, result)
# # 回溯优化使用列表
# class Solution:
#     def __init__(self):
#         self.letterMap = [
#             "",     # 0
#             "",     # 1
#             "abc",  # 2
#             "def",  # 3
#             "ghi",  # 4
#             "jkl",  # 5
#             "mno",  # 6
#             "pqrs", # 7
#             "tuv",  # 8
#             "wxyz"  # 9
#         ]
#     def letterCombinations(self, digits):
#         result = []
#         if len(digits) == 0:
#             return result
#         self.getCombinations(digits, 0, [], result)
#         return result    
#     def getCombinations(self, digits, index, path, result):
#         if index == len(digits):
#             result.append(''.join(path))
#             return
#         digit = int(digits[index])
#         letters = self.letterMap[digit]
#         for letter in letters:
#             path.append(letter)
#             self.getCombinations(digits, index + 1, path, result)
#             path.pop()

# Phone Keypad Combinations
    # You are given a string containing digits from 2 to 9 inclusive. Each digit maps 
    # to a set of letters as on a traditional phone keypad:
    # Return all possible letter combinations the input digits could represent.
    # Example:
    # Input: digits = '69'
    # Output: ['mw', 'mx', 'my', 'mz', 'nw', 'nx', 'ny', 'nz', 'ow', 'ox', 'oy', 'oz']
from typing import Dict, List

def phone_keypad_combinations(digits: str) -> List[str]:
    keypad_map = {
        '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
        '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
    }
    res = []
    backtrack(0, [], digits, keypad_map, res)
    return res

def backtrack(i: int, curr_combination: List[str], digits: str, keypad_map: Dict[str, str], res: List[str]) -> None:
    # Termination condition: if all digits have been considered, add the
    # current combination to the output list.
    if len(curr_combination) == len(digits):
        res.append("".join(curr_combination))
        return
    for letter in keypad_map[digits[i]]:
       # Add the current letter.
        curr_combination.append(letter)
        # Recursively explore all paths that branch from this combination.
        backtrack(i + 1, curr_combination, digits, keypad_map, res)
        # Backtrack by removing the letter we just added.
        curr_combination.pop()


#X4 (Medium) 39.组合总和
    # 给定一个无重复元素的数组 candidates 和一个目标数 target , 找出 candidates 中所有可以使数字和为 target 的组合。
    # 示例 1： 输入：candidates = [2,3,6,7], target = 7, 所求解集为： [ [7], [2,2,3] ]
    # 示例 2： 输入：candidates = [2,3,5], target = 8, 所求解集为： [ [2,2,2,2], [2,3,3], [3,5] ]
# 如果是一个集合来求组合的话，就需要startIndex. 例如：77.组合 ，216.组合总和III 。
# 如果是多个集合取组合，各个集合之间相互不影响，那么就不用startIndex，例如：17.电话号码的字母组合
# *** 回溯 + 剪枝
# Time Complexity: O(2^n) 其中n是candidates的长度，因为每个元素可以选择或不选择
# Space Complexity: O(n) 其中n是candidates的长度，因为递归深度最多为n
class Solution:
    def __init__(self):
        self.path = []
        self.paths = []

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        '''
        因为本题没有组合数量限制, 所以只要元素总和大于target就算结束
        '''
        # self.path.clear()
        # self.paths.clear()

        # 为了剪枝需要提前进行排序
        # 对总集合排序之后, 如果下一层的sum（就是本层的 sum + candidates[i]）已经大于target, 就可以结束本轮for循环的遍历。
        candidates.sort()
        self.backtracking(candidates, target, 0, 0)
        return self.paths

    def backtracking(self, candidates: List[int], target: int, sum_: int, start_index: int) -> None:
        # Base Case
        if sum_ == target:
            self.paths.append(self.path[:]) # 因为是shallow copy, 所以不能直接传入self.path
            return
        # 单层递归逻辑 
        # 如果本层 sum + condidates[i] > target, 就提前结束遍历, 剪枝
        # 对总集合排序之后, 如果下一层的sum（就是本层的 sum + candidates[i]）已经大于target, 就可以结束本轮for循环的遍历。
        for i in range(start_index, len(candidates)):
            if sum_ + candidates[i] > target: 
                return

            sum_ += candidates[i]
            self.path.append(candidates[i])

            self.backtracking(candidates, target, sum_, i)  # 因为无限制重复选取, 所以不是i+1

            sum_ -= candidates[i]   # 回溯
            self.path.pop()        # 回溯
# 回溯（版本一）
# class Solution:
#     def combinationSum(self, candidates, target):
#         result = []
#         self.backtracking(candidates, target, 0, 0, [], result)
#         return result
#     def backtracking(self, candidates, target, total, startIndex, path, result):
#         if total > target:
#             return
#         if total == target:
#             result.append(path[:])
#             return

#         for i in range(startIndex, len(candidates)):
#             total += candidates[i]
#             path.append(candidates[i])
#             self.backtracking(candidates, target, total, i, path, result)  # 不用i+1了，表示可以重复读取当前的数
#             total -= candidates[i]
#             path.pop()
# # 回溯剪枝（版本一）
# class Solution:
#     def combinationSum(self, candidates, target):
#         result = []
#         candidates.sort()  # 需要排序
#         self.backtracking(candidates, target, 0, 0, [], result)
#         return result
#     def backtracking(self, candidates, target, total, startIndex, path, result):
#         if total == target:
#             result.append(path[:])
#             return

#         for i in range(startIndex, len(candidates)):
#             if total + candidates[i] > target:
#                 continue
#             total += candidates[i]
#             path.append(candidates[i])
#             self.backtracking(candidates, target, total, i, path, result)
#             total -= candidates[i]
#             path.pop()
# 回溯（版本二）
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        result =[]
        self.backtracking(candidates, target, 0, [], result)
        return result

    def backtracking(self, candidates, target, startIndex, path, result):
        if target == 0:
            result.append(path[:])
            return
        # 对于sum已经大于target的情况，其实是依然进入了下一层递归，只是下一层递归结束判断的时候，会判断sum > target的话就返回
        if target < 0:
            return
        for i in range(startIndex, len(candidates)):
            path.append(candidates[i])
            self.backtracking(candidates, target - candidates[i], i, path, result)
            path.pop()
# *** 回溯剪枝（版本二）
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        result =[]
        candidates.sort()
        self.backtracking(candidates, target, 0, [], result)
        return result

    def backtracking(self, candidates, target, startIndex, path, result):
        if target == 0:
            result.append(path[:])
            return

        for i in range(startIndex, len(candidates)):
            # 其实如果已经知道下一层的sum会大于target，就没有必要进入下一层递归了。
            # 对总集合排序之后，如果下一层的sum（就是本层的 sum + candidates[i]）已经大于target，就可以结束本轮for循环的遍历。
            if target - candidates[i]  < 0:
                break
            path.append(candidates[i])
            self.backtracking(candidates, target - candidates[i], i, path, result)
            path.pop()

# Combinations of a Sum
    # Given an integer array and a target value, find all unique combinations in the array 
    # where the numbers in each combination sum to the target. Each number in the array 
    # may be used an unlimited number of times in the combination.
    # Example:
    # Input: nums = [1, 2, 3], target = 4
    # Output: [[1, 1, 1, 1], [1, 1, 2], [1, 3], [2, 2]]
def combinations_of_sum_k(nums: List[int], target: int) -> List[List[int]]:
    res = []
    dfs([], 0, nums, target, res)
    return res

def dfs(combination: List[int], start_index: int, nums: List[int], target: int,
        res: List[List[int]]) -> None:
    # Termination condition: If the target is equal to 0, we found a combination 
    # that sums to 'k'.
    if target == 0:
        res.append(combination[:])
        return
    # Termination condition: If the target is less than 0, no more valid 
    # combinations can be created by adding it to the current combination.
    if target < 0:
        return
    # Starting from start_index, explore all combinations after adding nums[i].
    for i in range(start_index, len(nums)):
        # Add the current number to create a new combination.
        combination.append(nums[i])
        # Recursively explore all paths that branch from this new combination.
        dfs(combination, i, nums, target - nums[i], res)
        # Backtrack by removing the number we just added.
        combination.pop()


#5 (Medium) 40.组合总和II
    # 给定一个数组 candidates 和一个目标数 target , 找出 candidates 中所有可以使数字和为 target 的组合。
    # 本题的难点在于区别2中：集合（数组candidates）有重复元素, 但还不能有重复的组合。
    # 示例 1: 输入: candidates = [10,1,2,7,6,1,5], target = 8, 所求解集为: [ [1, 7], [1, 2, 5], [2, 6], [1, 1, 6] ]
    # 示例 2: 输入: candidates = [2,5,2,1,2], target = 5, 所求解集为: [   [1,2,2],   [5] ]
# *** 回溯优化，不用used
# Time Complexity: O(2^n) 其中n是candidates的长度，因为每个元素可以选择或不选择
# Space Complexity: O(n) 其中n是candidates的长度，因为递归深度最多为n
class Solution:
    def __init__(self):
        self.paths = []
        self.path = []

    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        '''
        类似于求三数之和, 求四数之和, 为了避免重复组合, 需要提前进行数组排序
        '''
        self.paths.clear()
        self.path.clear()
        # 必须提前进行数组排序, 避免重复
        candidates.sort()
        self.backtracking(candidates, target, 0, 0)
        return self.paths

    def backtracking(self, candidates: List[int], target: int, sum_: int, start_index: int) -> None:
        # Base Case
        if sum_ == target:
            self.paths.append(self.path[:])
            return
        
        # 单层递归逻辑
        for i in range(start_index, len(candidates)):
            # 剪枝, 同39.组合总和
            if sum_ + candidates[i] > target:
                return
            
            # 跳过同一树层使用过的元素
            if i > start_index and candidates[i] == candidates[i-1]:
                continue
            
            sum_ += candidates[i]
            self.path.append(candidates[i])
            self.backtracking(candidates, target, sum_, i+1) # 用i+1，表示不可以重复读取当前的数
            self.path.pop()             # 回溯, 为了下一轮for loop
            sum_ -= candidates[i]       # 回溯, 为了下一轮for loop
# 回溯
# class Solution:
#     def combinationSum2(self, candidates, target):
#         result = []
#         candidates.sort()
#         self.backtracking(candidates, target, 0, 0, [], result)
#         return result
#     def backtracking(self, candidates, target, total, startIndex, path, result):
#         if total == target:
#             result.append(path[:])
#             return

#         for i in range(startIndex, len(candidates)):
#             if i > startIndex and candidates[i] == candidates[i - 1]:
#                 continue

#             if total + candidates[i] > target:
#                 break

#             total += candidates[i]
#             path.append(candidates[i])
#             self.backtracking(candidates, target, total, i + 1, path, result)
#             total -= candidates[i]
#             path.pop()
# # 回溯 使用used
# class Solution:
#     def combinationSum2(self, candidates, target):
#         used = [False] * len(candidates)
#         result = []
#         candidates.sort()
#         self.backtracking(candidates, target, 0, 0, used, [], result)
#         return result

#     def backtracking(self, candidates, target, total, startIndex, used, path, result):
#         if total == target:
#             result.append(path[:])
#             return

#         for i in range(startIndex, len(candidates)):
#             # 对于相同的数字，只选择第一个未被使用的数字，跳过其他相同数字
#             if i > startIndex and candidates[i] == candidates[i - 1] and not used[i - 1]:
#                 continue

#             if total + candidates[i] > target:
#                 break

#             total += candidates[i]
#             path.append(candidates[i])
#             used[i] = True
#             self.backtracking(candidates, target, total, i + 1, used, path, result)
#             used[i] = False
#             total -= candidates[i]
#             path.pop()
# # 回溯优化
# class Solution:
#     def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
#         candidates.sort()
#         results = []
#         self.combinationSumHelper(candidates, target, 0, [], results)
#         return results

#     def combinationSumHelper(self, candidates, target, index, path, results):
#         if target == 0:
#             results.append(path[:])
#             return
#         for i in range(index, len(candidates)):
#             if i > index and candidates[i] == candidates[i - 1]:
#                 continue  
#             if candidates[i] > target:
#                 break  
#             path.append(candidates[i])
#             self.combinationSumHelper(candidates, target - candidates[i], i + 1, path, results)
#             path.pop()


#6 (Medium) 131.分割回文串
    # 给定一个字符串 s, 将 s 分割成一些子串, 使每个子串都是回文串。
    # 示例: 输入: "aab" 输出: [ ["aa","b"], ["a","a","b"] ]
# 回溯+正反序判断回文串
class Solution:
    def __init__(self):
        self.paths = []
        self.path = []

    def partition(self, s: str) -> List[List[str]]:
        '''
        递归用于纵向遍历
        for循环用于横向遍历
        当切割线迭代至字符串末尾, 说明找到一种方法
        类似组合问题, 为了不重复切割同一位置, 需要start_index来做标记下一轮递归的起始位置(切割线)
        '''
        self.path.clear()
        self.paths.clear()
        self.backtracking(s, 0)
        return self.paths

    def backtracking(self, s: str, start_index: int) -> None:
        # Base Case
        if start_index >= len(s):
            self.paths.append(self.path[:])
            return
        
        # 单层递归逻辑
        for i in range(start_index, len(s)):
            # 此次比其他组合题目多了一步判断：
            # 判断被截取的这一段子串([start_index, i])是否为回文串
            temp = s[start_index:i+1]
            if temp == temp[::-1]:  # 若反序和正序相同, 意味着这是回文串
                self.path.append(temp)
                self.backtracking(s, i+1)   # 递归纵向遍历：从下一处进行切割, 判断其余是否仍为回文串
                self.path.pop()
            else:
                continue
# 回溯+函数判断回文串
class Solution:
    def __init__(self):
        self.paths = []
        self.path = []

    def partition(self, s: str) -> List[List[str]]:
        '''
        递归用于纵向遍历
        for循环用于横向遍历
        当切割线迭代至字符串末尾, 说明找到一种方法
        类似组合问题, 为了不重复切割同一位置, 需要start_index来做标记下一轮递归的起始位置(切割线)
        '''
        self.path.clear()
        self.paths.clear()
        self.backtracking(s, 0)
        return self.paths

    def backtracking(self, s: str, start_index: int) -> None:
        # Base Case
        if start_index >= len(s):
            self.paths.append(self.path[:])
            return
        
        # 单层递归逻辑
        for i in range(start_index, len(s)):
            # 此次比其他组合题目多了一步判断：
            # 判断被截取的这一段子串([start_index, i])是否为回文串
            if self.is_palindrome(s, start_index, i):
                self.path.append(s[start_index:i+1])
                self.backtracking(s, i+1)   # 递归纵向遍历：从下一处进行切割, 判断其余是否仍为回文串
                self.path.pop()             # 回溯
            else:
                continue    

    def is_palindrome(self, s: str, start: int, end: int) -> bool:
        i: int = start        
        j: int = end
        while i < j:
            if s[i] != s[j]:
                return False
            i += 1
            j -= 1
        return True
# 回溯 基本版
# class Solution:
#     def partition(self, s: str) -> List[List[str]]:
#         '''
#         递归用于纵向遍历
#         for循环用于横向遍历
#         当切割线迭代至字符串末尾，说明找到一种方法
#         类似组合问题，为了不重复切割同一位置，需要start_index来做标记下一轮递归的起始位置(切割线)
#         '''
#         result = []
#         self.backtracking(s, 0, [], result)
#         return result

#     def backtracking(self, s, start_index, path, result ):
#         # Base Case
#         if start_index == len(s):
#             result.append(path[:])
#             return
#         # 单层递归逻辑
#         for i in range(start_index, len(s)):
#             # 此次比其他组合题目多了一步判断：
#             # 判断被截取的这一段子串([start_index, i])是否为回文串
#             if self.is_palindrome(s, start_index, i):
#                 path.append(s[start_index:i+1])
#                 self.backtracking(s, i+1, path, result)   # 递归纵向遍历：从下一处进行切割，判断其余是否仍为回文串
#                 path.pop()             # 回溯
#     def is_palindrome(self, s: str, start: int, end: int) -> bool:
#         i: int = start        
#         j: int = end
#         while i < j:
#             if s[i] != s[j]:
#                 return False
#             i += 1
#             j -= 1
#         return True 
# # 回溯+优化判定回文函数
# class Solution:

#     def partition(self, s: str) -> List[List[str]]:
#         result = []
#         self.backtracking(s, 0, [], result)
#         return result

#     def backtracking(self, s, start_index, path, result ):
#         # Base Case
#         if start_index == len(s):
#             result.append(path[:])
#             return
        
#         # 单层递归逻辑
#         for i in range(start_index, len(s)):
#             # 若反序和正序相同，意味着这是回文串
#             if s[start_index: i + 1] == s[start_index: i + 1][::-1]:
#                 path.append(s[start_index:i+1])
#                 self.backtracking(s, i+1, path, result)   # 递归纵向遍历：从下一处进行切割，判断其余是否仍为回文串
#                 path.pop()             # 回溯
# *** 回溯+高效判断回文子串，动态规划
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        result = []
        isPalindrome = [[False] * len(s) for _ in range(len(s))]  # 初始化isPalindrome矩阵
        self.computePalindrome(s, isPalindrome)
        self.backtracking(s, 0, [], result, isPalindrome)
        return result

    def backtracking(self, s, startIndex, path, result, isPalindrome):
        if startIndex >= len(s):
            result.append(path[:])
            return
        for i in range(startIndex, len(s)):
            if isPalindrome[startIndex][i]:   # 是回文子串
                substring = s[startIndex:i + 1]
                path.append(substring)
                self.backtracking(s, i + 1, path, result, isPalindrome)  # 寻找i+1为起始位置的子串
                path.pop()           # 回溯过程，弹出本次已经添加的子串

    def computePalindrome(self, s, isPalindrome):
        for i in range(len(s) - 1, -1, -1):  # 需要倒序计算，保证在i行时，i+1行已经计算好了
            for j in range(i, len(s)):
                if j == i:
                    isPalindrome[i][j] = True
                elif j - i == 1:
                    isPalindrome[i][j] = (s[i] == s[j])
                else:
                    isPalindrome[i][j] = (s[i] == s[j] and isPalindrome[i+1][j-1])
# 回溯+使用all函数判断回文子串
# class Solution:
#     def partition(self, s: str) -> List[List[str]]:
#         result = []
#         self.partition_helper(s, 0, [], result)
#         return result

#     def partition_helper(self, s, start_index, path, result):
#         if start_index == len(s):
#             result.append(path[:])
#             return

#         for i in range(start_index + 1, len(s) + 1):
#             sub = s[start_index:i]
#             if self.isPalindrome(sub):
#                 path.append(sub)
#                 self.partition_helper(s, i, path, result)
#                 path.pop()

#     def isPalindrome(self, s):
#         return all(s[i] == s[len(s) - 1 - i] for i in range(len(s) // 2))


#7 (Medium) 93.复原IP地址
    # 给定一个只包含数字的字符串, 复原它并返回所有可能的 IP 地址格式。
    # 输入：s = "25525511135"
    # 输出：["255.255.11.135","255.255.111.35"]
class Solution:
    def __init__(self):
        self.result = []

    def restoreIpAddresses(self, s: str) -> List[str]:
        '''
        本质切割问题使用回溯搜索法, 本题只能切割三次, 所以纵向递归总共四层
        因为不能重复分割, 所以需要start_index来记录下一层递归分割的起始位置
        添加变量point_num来记录逗号的数量[0,3]
        '''
        self.result.clear()
        if len(s) > 12: return []
        self.backtracking(s, 0, 0)
        return self.result

    def backtracking(self, s: str, start_index: int, point_num: int) -> None:
        # Base Case
        if point_num == 3:
            if self.is_valid(s, start_index, len(s)-1):
                self.result.append(s[:])
            return
        # 单层递归逻辑
        for i in range(start_index, len(s)):
            # [start_index, i]就是被截取的子串
            if self.is_valid(s, start_index, i):
                s = s[:i+1] + '.' + s[i+1:]
                self.backtracking(s, i+2, point_num+1)  # 在填入.后, 下一子串起始后移2位
                s = s[:i+1] + s[i+2:]    # 回溯
            else:
                # 若当前被截取的子串大于255或者大于三位数, 直接结束本层循环
                break
    
    def is_valid(self, s: str, start: int, end: int) -> bool:
        if start > end: return False
        # 若数字是0开头, 不合法
        if s[start] == '0' and start != end:
            return False
        if not 0 <= int(s[start:end+1]) <= 255:
            return False
        return True
# 回溯（版本一）
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        result = []
        self.backtracking(s, 0, 0, "", result)
        return result

    def backtracking(self, s, start_index, point_num, current, result):
        if point_num == 3:  # 逗点数量为3时，分隔结束
            if self.is_valid(s, start_index, len(s) - 1):  # 判断第四段子字符串是否合法
                current += s[start_index:]  # 添加最后一段子字符串
                result.append(current)
            return

        for i in range(start_index, len(s)):
            if self.is_valid(s, start_index, i):  # 判断 [start_index, i] 这个区间的子串是否合法
                sub = s[start_index:i + 1]
                self.backtracking(s, i + 1, point_num + 1, current + sub + '.', result)
            else:
                break

    def is_valid(self, s, start, end):
        if start > end:
            return False
        if s[start] == '0' and start != end:  # 0开头的数字不合法
            return False
        num = 0
        for i in range(start, end + 1):
            if not s[i].isdigit():  # 遇到非数字字符不合法
                return False
            num = num * 10 + int(s[i])
            if num > 255:  # 如果大于255了不合法
                return False
        return True
# *** 回溯+剪枝（版本二）
# Time Complexity: O(1) 因为IP地址的长度是固定的，最多只有12个字符
# Space Complexity: O(1) 因为IP地址的长度是固定的，最多只有12个字符
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        results = []
        self.backtracking(s, 0, [], results)
        return results

    def backtracking(self, s, index, path, results):
        if index == len(s) and len(path) == 4:
            results.append('.'.join(path))
            return

        if len(path) > 4:  # 剪枝
            return

        for i in range(index, min(index + 3, len(s))):
            if self.is_valid(s, index, i):
                sub = s[index:i+1]
                path.append(sub)
                self.backtracking(s, i+1, path, results) # i+1是因为下一段子串的起始位置
                path.pop()

    def is_valid(self, s, start, end):
        if start > end:
            return False
        if s[start] == '0' and start != end:  # 0开头的数字不合法
            return False
        num = int(s[start:end+1])
        return 0 <= num <= 255


#X8 (Medium) 78.子集
    # 给定一组不含重复元素的整数数组 nums, 返回该数组所有可能的子集（幂集）。
    # 说明：解集不能包含重复的子集。
    # 示例: 输入: nums = [1,2,3] 输出: [ [3],   [1],   [2],   [1,2,3],   [1,3],   [2,3],   [1,2],   [] ]
# 但是要清楚子集问题和组合问题、分割问题的的区别, 子集是收集树形结构中树的所有节点的结果。
# 而组合问题、分割问题是收集树形结构中叶子节点的结果。
# class Solution:
#     def __init__(self):
#         self.path: List[int] = []
#         self.paths: List[List[int]] = []

#     def subsets(self, nums: List[int]) -> List[List[int]]:
#         self.paths.clear()
#         self.path.clear()
#         self.backtracking(nums, 0)
#         return self.paths

#     def backtracking(self, nums: List[int], start_index: int) -> None:
#         # 收集子集, 要先于终止判断(因为要收集所以node, 而不只是leaf, 所以要base case前把path放入paths)
#         self.paths.append(self.path[:])
#         # Base Case
#         if start_index == len(nums):
#             return

#         # 单层递归逻辑
#         for i in range(start_index, len(nums)):
#             self.path.append(nums[i])
#             self.backtracking(nums, i+1)
#             self.path.pop()     # 回溯
# Time Complexity: O(2^n) 其中n是nums的长度，因为每个元素可以选择或不选择
# Space Complexity: O(n) 其中n是nums的长度，因为递归深度最多为n
class Solution:
    def subsets(self, nums):
        result = []
        path = []
        self.backtracking(nums, 0, path, result)
        return result

    def backtracking(self, nums, startIndex, path, result):
        result.append(path[:])  # 收集子集，要放在终止添加的上面，否则会漏掉自己
        # if startIndex >= len(nums):  # 终止条件可以不加
        #     return
        # 从startIndex而不是从0开始,是因为子集不能重复
        for i in range(startIndex, len(nums)):
            path.append(nums[i])
            self.backtracking(nums, i + 1, path, result) # 注意从i+1开始，元素不重复取
            path.pop()

# Find All Subsets
    # Return all possible subsets of a given set of unique integers. 
    # Each subset can be ordered in any way, and the subsets can be returned in any order.
    # Example:
    # Input: nums = [4, 5, 6]
    # Output: [[], [4], [4, 5], [4, 5, 6], [4, 6], [5], [5, 6], [6]]
def find_all_subsets(nums: List[int]) -> List[List[int]]:
    res = []
    backtrack(0, [], nums, res)
    return res

def backtrack(i: int, curr_subset: List[int], nums: List[int], res: List[List[int]]) -> None:
    # Base case: if all elements have been considered, add the
    # current subset to the output.
    if i == len(nums):
        res.append(curr_subset[:])
        return
    # Include the current element and recursively explore all paths
    # that branch from this subset.
    curr_subset.append(nums[i])
    backtrack(i + 1, curr_subset, nums, res)
    # Exclude the current element and recursively explore all paths
    # that branch from this subset.
    curr_subset.pop()
    backtrack(i + 1, curr_subset, nums, res)


#9 (Medium) 90.子集II
    # 给定一个可能包含重复元素的整数数组 nums, 返回该数组所有可能的子集（幂集）。
    # 输入: [1,2,2]
    # 输出: [ [2], [1], [1,2,2], [2,2], [1,2], [] ]
# class Solution:
#     def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
#         res = []
#         path = []
#         nums.sort() # 去重需要先对数组进行排序

#         def backtracking(nums, startIndex):
#             # 终止条件
#             res.append(path[:])
#             if startIndex == len(nums):
#                 return
            
#             # for循环
#             for i in range(startIndex, len(nums)):
#                 # 数层去重
#                 if i > startIndex and nums[i] == nums[i-1]: # 去重
#                     continue
#                 path.append(nums[i])
#                 backtracking(nums, i+1)
#                 path.pop()
        
#         backtracking(nums, 0)
#         return res
# 回溯 利用used数组去重
class Solution:
    def subsetsWithDup(self, nums):
        result = []
        path = []
        used = [False] * len(nums)
        nums.sort()  # 去重需要排序
        self.backtracking(nums, 0, used, path, result)
        return result

    def backtracking(self, nums, startIndex, used, path, result):
        result.append(path[:])  # 收集子集
        for i in range(startIndex, len(nums)):
            # used[i - 1] == True，说明同一树枝 nums[i - 1] 使用过 (在递归中使用过)
            # used[i - 1] == False，说明同一树层 nums[i - 1] 使用过 (在for循环中使用过)
            # 而我们要对同一树层使用过的元素进行跳过
            if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
                continue
            path.append(nums[i])
            used[i] = True
            self.backtracking(nums, i + 1, used, path, result) # i + 1是因为不允许重复使用同一元素
            used[i] = False
            path.pop()
# 回溯 利用集合去重
# class Solution:
#     def subsetsWithDup(self, nums):
#         result = []
#         path = []
#         nums.sort()  # 去重需要排序
#         self.backtracking(nums, 0, path, result)
#         return result

#     def backtracking(self, nums, startIndex, path, result):
#         result.append(path[:])  # 收集子集
#         uset = set()
#         for i in range(startIndex, len(nums)):
#             if nums[i] in uset:
#                 continue
#             uset.add(nums[i])
#             path.append(nums[i])
#             self.backtracking(nums, i + 1, path, result)
#             path.pop()
# *** 回溯 利用递归的时候下一个startIndex是i+1而不是0去重
# Time Complexity: O(2^n) 其中n是nums的长度，因为每个元素可以选择或不选择
# Space Complexity: O(n) 其中n是nums的长度，因为递归深度最多为n
class Solution:
    def subsetsWithDup(self, nums):
        result = []
        path = []
        nums.sort()  # 去重需要排序
        self.backtracking(nums, 0, path, result)
        return result

    def backtracking(self, nums, startIndex, path, result):
        result.append(path[:])  # 收集子集
        for i in range(startIndex, len(nums)):
            # 而我们要对同一树层使用过的元素进行跳过, i > startIndex表示同一树层，已经使用过i=startIndex,使用过nums[i-1]
            if i > startIndex and nums[i] == nums[i - 1]:
                continue
            path.append(nums[i])
            self.backtracking(nums, i + 1, path, result)
            path.pop()


#10 (Medium) 491.递增子序列
    # 给定一个整型数组, 你的任务是找到所有该数组的递增子序列, 递增子序列的长度至少是2。
    # 输入: [4, 6, 7, 7]
    # 输出: [[4, 6], [4, 7], [4, 6, 7], [4, 6, 7, 7], [6, 7], [6, 7, 7], [7,7], [4,7,7]]
# 本题求子序列, 很明显一个元素不能重复使用, 所以需要startIndex, 调整下一层递归的起始位置。
# 回溯
# class Solution:
#     def __init__(self):
#         self.paths = []
#         self.path = []

#     def findSubsequences(self, nums: List[int]) -> List[List[int]]:
#         '''
#         本题求自增子序列, 所以不能改变原数组顺序
#         '''
#         self.backtracking(nums, 0)
#         return self.paths

#     def backtracking(self, nums: List[int], start_index: int):
#         # 收集结果, 同78.子集, 仍要置于终止条件之前
#         if len(self.path) >= 2:
#             # 本题要求所有的节点
#             self.paths.append(self.path[:])
        
#         # Base Case（可忽略）
#         if start_index == len(nums):
#             return

#         # 单层递归逻辑
#         # 深度遍历中每一层都会有一个全新的usage_list用于记录本层元素是否重复使用
#         usage_list = set()
#         # 同层横向遍历
#         for i in range(start_index, len(nums)):
#             # 若当前元素值小于前一个时（非递增）或者曾用过, 跳入下一循环
#             if (self.path and nums[i] < self.path[-1]) or nums[i] in usage_list:
#                 continue
#             usage_list.add(nums[i])
#             self.path.append(nums[i])
#             self.backtracking(nums, i+1)
#             self.path.pop() 
# # 回溯+哈希表去重
# class Solution:
#     def __init__(self):
#         self.paths = []
#         self.path = []

#     def findSubsequences(self, nums: List[int]) -> List[List[int]]:
#         '''
#         本题求自增子序列, 所以不能改变原数组顺序
#         '''
#         self.backtracking(nums, 0)
#         return self.paths

#     def backtracking(self, nums: List[int], start_index: int):
#         # 收集结果, 同78.子集, 仍要置于终止条件之前
#         if len(self.path) >= 2:
#             # 本题要求所有的节点
#             self.paths.append(self.path[:])
        
        # # Base Case（可忽略）
        # # Implicit Return at Function End:
        # # Python functions automatically return None when they reach the end
        # # After the for loop completes, the function simply ends and returns control to the previous level
        # # 终止条件
        # # 这里可以不加终止条件, 因为在每次递归中都会检查path的长度
        # if start_index == len(nums):
        #     return

#         # 单层递归逻辑
#         # 深度遍历中每一层都会有一个全新的usage_list用于记录本层元素是否重复使用
#         usage_list = [False] * 201  # 使用列表去重, 题中取值范围[-100, 100]
#         # 同层横向遍历
#         for i in range(start_index, len(nums)):
#             # 若当前元素值小于前一个时（非递增）或者曾用过, 跳入下一循环
#             if (self.path and nums[i] < self.path[-1]) or usage_list[nums[i]+100] == True:
#                 continue
#             usage_list[nums[i]+100] = True
#             self.path.append(nums[i])
#             self.backtracking(nums, i+1)
#             self.path.pop()
# 回溯 利用set去重
class Solution:
    def findSubsequences(self, nums):
        result = []
        path = []
        self.backtracking(nums, 0, path, result)
        return result
    
    def backtracking(self, nums, startIndex, path, result):
        if len(path) > 1:
            result.append(path[:])  # 注意要使用切片将当前路径的副本加入结果集
            # 注意这里不要加return，要取树上的节点
        
        uset = set()  # 使用集合对本层元素进行去重
        for i in range(startIndex, len(nums)):
            if (path and nums[i] < path[-1]) or nums[i] in uset:
                continue
            
            uset.add(nums[i])  # 记录这个元素在本层用过了，本层后面不能再用了
            path.append(nums[i])
            self.backtracking(nums, i + 1, path, result)
            path.pop()
# *** 回溯 利用哈希表去重
# Time Complexity: O(2^n) 其中n是nums的长度，因为每个元素可以选择或不选择
# Space Complexity: O(n) 其中n是nums的长度，因为递归深度最多为n
class Solution:
    def findSubsequences(self, nums):
        result = []
        path = []
        self.backtracking(nums, 0, path, result)
        return result

    def backtracking(self, nums, startIndex, path, result):
        if len(path) > 1:
            result.append(path[:])  # 注意要使用切片将当前路径的副本加入结果集
        
        used = [0] * 201  # 使用数组来进行去重操作，题目说数值范围[-100, 100]
        for i in range(startIndex, len(nums)):
            if (path and nums[i] < path[-1]) or used[nums[i] + 100] == 1:
                continue  # 如果当前元素小于上一个元素，或者已经使用过当前元素，则跳过当前元素
            
            used[nums[i] + 100] = 1  # 标记当前元素已经使用过
            path.append(nums[i])  # 将当前元素加入当前递增子序列
            self.backtracking(nums, i + 1, path, result) # i + 1是因为不允许重复使用同一元素
            path.pop()


#X11 (Medium) 46.全排列
    # 给定一个 没有重复 数字的序列, 返回其所有可能的全排列。
    # 输入: [1,2,3]
    # 输出: [ [1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1] ]
# 回溯
# class Solution:
#     def __init__(self):
#         self.path = []
#         self.paths = []

#     def permute(self, nums: List[int]) -> List[List[int]]:
#         '''
#         因为本题排列是有序的, 这意味着同一层的元素可以重复使用, 但同一树枝上不能重复使用(usage_list)
#         所以处理排列问题每层都需要从头搜索, 故不再使用start_index
#         '''
#         usage_list = [False] * len(nums)
#         self.backtracking(nums, usage_list)
#         return self.paths

#     def backtracking(self, nums: List[int], usage_list: List[bool]) -> None:
#         # Base Case本题求叶子节点
#         if len(self.path) == len(nums):
#             self.paths.append(self.path[:])
#             return

#         # 单层递归逻辑
#         for i in range(0, len(nums)):  # 从头开始搜索
#             # 若遇到self.path里已收录的元素, 跳过
#             if usage_list[i] == True:
#                 continue
#             usage_list[i] = True
#             self.path.append(nums[i])
#             self.backtracking(nums, usage_list)     # 纵向传递使用信息, 去重
#             self.path.pop()
#             usage_list[i] = False
# # 回溯+丢掉usage_list
# class Solution:
#     def __init__(self):
#         self.path = []
#         self.paths = []

#     def permute(self, nums: List[int]) -> List[List[int]]:
#         '''
#         因为本题排列是有序的, 这意味着同一层的元素可以重复使用, 但同一树枝上不能重复使用
#         所以处理排列问题每层都需要从头搜索, 故不再使用start_index
#         '''
#         self.backtracking(nums)
#         return self.paths

#     def backtracking(self, nums: List[int]) -> None:
#         # Base Case本题求叶子节点
#         if len(self.path) == len(nums):
#             self.paths.append(self.path[:])
#             return

#         # 单层递归逻辑
#         for i in range(0, len(nums)):  # 从头开始搜索
#             # 若遇到self.path里已收录的元素, 跳过
#             if nums[i] in self.path:
#                 continue
#             self.path.append(nums[i])
#             self.backtracking(nums)
#             self.path.pop()
# *** 回溯 使用used
# 使用used因为每次都是从第一个item遍历，这样才能保证所有item都考虑到了。因此没有用start index
# 使用start index是因为每次都从当前的start index开始遍历，这样就可以避免重复考虑了, 但是全排列需要考虑所有的item，所以不能用start index
# Time Complexity: O(n!)
# Space Complexity: O(n)
class Solution:
    def permute(self, nums):
        result = []
        self.backtracking(nums, [], [False] * len(nums), result)
        return result

    def backtracking(self, nums, path, used, result):
        # Base Case 需要设置条件的才需要return, 如果是遍历所有可能同时记录, 那不需要return, 
        # 因为implicit return at function end (for loop done)
        if len(path) == len(nums):
            result.append(path[:])
            return

        for i in range(len(nums)):
            if used[i]:
                continue
            used[i] = True
            path.append(nums[i])
            self.backtracking(nums, path, used, result)
            path.pop()
            used[i] = False

# Find All Permutations
    # Return all possible permutations of a given array of unique integers. They can be returned in any order.
    # Example:
    # Input: nums = [4, 5, 6]
    # Output: [[4, 5, 6], [4, 6, 5], [5, 4, 6], [5, 6, 4],
    #          [6, 4, 5], [6, 5, 4]]
from typing import List, Set

def find_all_permutations(nums: List[int]) -> List[List[int]]:
    res = []
    backtrack(nums, [], set(), res)
    return res

def backtrack(nums: List[int], candidate: List[int], used: Set[int], res: List[List[int]]) -> None:
    # If the current candidate is a complete permutation, add it to the
    # result.
    if len(candidate) == len(nums):
        res.append(candidate[:])
        return

    for num in nums:
        if num not in used:
            # Add 'num' to the current permutation and mark it as used.
            candidate.append(num)
            used.add(num)
            # Recursively explore all branches using the updated
            # permutation candidate.
            backtrack(nums, candidate, used, res)
            # Backtrack by reversing the changes made.
            candidate.pop()
            used.remove(num)


#12 (Medium) 47.全排列 II
    # 给定一个可包含重复数字的序列 nums , 按任意顺序 返回所有不重复的全排列。
    # 输入：nums = [1,1,2]
    # 输出： [[1,1,2], [1,2,1], [2,1,1]]
# 还要强调的是去重一定要对元素进行排序, 这样我们才方便通过相邻的节点来判断是否重复使用了。
# class Solution:
#     def permuteUnique(self, nums: List[int]) -> List[List[int]]:
#         # res用来存放结果
#         if not nums: return []
#         res = []
#         used = [0] * len(nums)
#         def backtracking(nums, used, path):
#             # 终止条件
#             if len(path) == len(nums):
#                 res.append(path.copy())
#                 return
#             for i in range(len(nums)):
#                 if not used[i]:
#                     if i>0 and nums[i] == nums[i-1] and not used[i-1]:
#                         continue
#                     used[i] = 1
#                     path.append(nums[i])
#                     backtracking(nums, used, path)
#                     path.pop()
#                     used[i] = 0
#         # 记得给nums排序
#         backtracking(sorted(nums),used,[])
#         return res
# Time Complexity: O(n!)
# Space Complexity: O(n)
class Solution:
    def permuteUnique(self, nums):
        nums.sort()  # 排序
        result = []
        # used数组可是全局变量，每层与每层之间公用一个used数组，所以空间复杂度是O(n + n)，最终空间复杂度还是O(n)。
        self.backtracking(nums, [], [False] * len(nums), result)
        return result

    def backtracking(self, nums, path, used, result):
        if len(path) == len(nums):
            result.append(path[:])
            return

        for i in range(len(nums)):
            # used[i]用来检测是不是用过当前元素
            if (i > 0 and nums[i] == nums[i - 1] and not used[i - 1]) or used[i]:
                continue
            used[i] = True
            path.append(nums[i])
            self.backtracking(nums, path, used, result)
            path.pop()
            used[i] = False

# 使用set来对本层去重的代码实现
# 两种写法的性能分析
    # 需要注意的是：使用set去重的版本相对于used数组的版本效率都要低很多，大家在leetcode上提交，能明显发现。
    # 原因在回溯算法：递增子序列 中也分析过，主要是因为程序运行的时候对unordered_set 频繁的insert，
    # unordered_set需要做哈希映射（也就是把key通过hash function映射为唯一的哈希值）相对费时间，
    # 而且insert的时候其底层的符号表也要做相应的扩充，也是费时的。

    # 而使用used数组在时间复杂度上几乎没有额外负担！
    # 使用set去重，不仅时间复杂度高了，空间复杂度也高了，在本周小结！（回溯算法系列三） 中分析过，
    # 组合，子集，排列问题的空间复杂度都是O(n)，但如果使用set去重，空间复杂度就变成了O(n^2)，
    # 因为每一层递归都有一个set集合，系统栈空间是n，每一个空间都有set集合。
    # 那有同学可能疑惑 用used数组也是占用O(n)的空间啊？
    # used数组可是全局变量，每层与每层之间公用一个used数组，所以空间复杂度是O(n + n)，最终空间复杂度还是O(n)。
# 90.子集II
# class Solution:
#     def subsetsWithDup(self, nums):
#         nums.sort()  # 去重需要排序
#         result = []
#         self.backtracking(nums, 0, [], result)
#         return result

#     def backtracking(self, nums, startIndex, path, result):
#         result.append(path[:])
#         used = set()
#         for i in range(startIndex, len(nums)):
#             if nums[i] in used:
#                 continue
#             used.add(nums[i])
#             path.append(nums[i])
#             self.backtracking(nums, i + 1, path, result)
#             path.pop()
# # 40. 组合总和 II
# class Solution:
#     def combinationSum2(self, candidates, target):
#         candidates.sort()
#         result = []
#         self.backtracking(candidates, target, 0, 0, [], result)
#         return result

#     def backtracking(self, candidates, target, sum, startIndex, path, result):
#         if sum == target:
#             result.append(path[:])
#             return
#         used = set()
#         for i in range(startIndex, len(candidates)):
#             if sum + candidates[i] > target:
#                 break
#             if candidates[i] in used:
#                 continue
#             used.add(candidates[i])
#             sum += candidates[i]
#             path.append(candidates[i])
#             self.backtracking(candidates, target, sum, i + 1, path, result)
#             sum -= candidates[i]
#             path.pop()
# # 47. 全排列 II
# class Solution:
#     def permuteUnique(self, nums):
#         nums.sort()  # 排序
#         result = []
#         self.backtracking(nums, [False] * len(nums), [], result)
#         return result

#     def backtracking(self, nums, used, path, result):
#         if len(path) == len(nums):
#             result.append(path[:])
#             return
#         used_set = set()
#         for i in range(len(nums)):
#             if nums[i] in used_set:
#                 continue
#             if not used[i]:
#                 used_set.add(nums[i])
#                 used[i] = True
#                 path.append(nums[i])
#                 self.backtracking(nums, used, path, result)
#                 path.pop()
#                 used[i] = False


#13 ??? (Hard) 332.重新安排行程
    # 给定一个机票的字符串二维数组 [from, to], 子数组中的两个成员分别表示飞机出发和降落的机场地点, 对该行程进行重新规划排序。所有这些机票都属于一个从 JFK（肯尼迪国际机场）出发的先生, 所以该行程必须从 JFK 开始。
    # 输入：[["MUC", "LHR"], ["JFK", "MUC"], ["SFO", "SJC"], ["LHR", "SFO"]]
    # 输出：["JFK", "MUC", "LHR", "SFO", "SJC"]
# class Solution:
#     def findItinerary(self, tickets: List[List[str]]) -> List[str]:
#         # defaultdic(list) 是为了方便直接append
#         tickets_dict = defaultdict(list)
#         for item in tickets:
#             tickets_dict[item[0]].append(item[1])
# 	# 给每一个机场的到达机场排序，小的在前面，在回溯里首先被pop(0）出去
# 	# 这样最先找的的path就是排序最小的答案，直接返回
#         for airport in tickets_dict: tickets_dict[airport].sort()
#         '''
#         tickets_dict里面的内容是这样的
#         {'JFK': ['ATL', 'SFO'], 'SFO': ['ATL'], 'ATL': ['JFK', 'SFO']})
#         '''
#         path = ["JFK"]
#         def backtracking(start_point):
#             # 终止条件
#             if len(path) == len(tickets) + 1:
#                 return True
#             for _ in tickets_dict[start_point]:
#                 #必须及时删除，避免出现死循环
#                 end_point = tickets_dict[start_point].pop(0)
#                 path.append(end_point)
#                 # 只要找到一个就可以返回了
#                 # print(backtracking(end_point))
#                 if backtracking(end_point):
#                     return True
#                 path.pop()
#                 tickets_dict[start_point].append(end_point)

#         backtracking("JFK")
#         return path
# 回溯 使用used数组
class Solution:
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        tickets.sort() # 先排序，这样一旦找到第一个可行路径，一定是字母排序最小的
        used = [0] * len(tickets)
        path = ['JFK']
        results = []
        self.backtracking(tickets, used, path, 'JFK', results)
        return results[0]
    
    def backtracking(self, tickets, used, path, cur, results):
        if len(path) == len(tickets) + 1:  # 终止条件：路径长度等于机票数量+1
            results.append(path[:])  # 将当前路径添加到结果列表
            return True
        
        for i, ticket in enumerate(tickets):  # 遍历机票列表
            if ticket[0] == cur and used[i] == 0:  # 找到起始机场为cur且未使用过的机票
                used[i] = 1  # 标记该机票为已使用
                path.append(ticket[1])  # 将到达机场添加到路径中

                if self.backtracking(tickets, used, path, ticket[1], results):  # 递归搜索
                    return True  # 只要找到一个可行路径就返回，不继续搜索
                
                path.pop()  # 回溯，移除最后添加的到达机场
                used[i] = 0  # 标记该机票为未使用
        return False
# 回溯 使用字典
# from collections import defaultdict
class Solution:
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        targets = collections.defaultdict(list)  # 构建机场字典
        for ticket in tickets:
            targets[ticket[0]].append(ticket[1])
        for airport in targets:
            targets[airport].sort()  # 对目的地列表进行排序

        path = ["JFK"]  # 起始机场为"JFK"
        self.backtracking(targets, path, len(tickets))
        return path

    def backtracking(self, targets, path, ticketNum):
        if len(path) == ticketNum + 1:
            return True  # 找到有效行程

        airport = path[-1]  # 当前机场
        destinations = targets[airport]  # 当前机场可以到达的目的地列表
        for i, dest in enumerate(destinations):
            targets[airport].pop(i)  # 标记已使用的机票
            path.append(dest)  # 添加目的地到路径
            if self.backtracking(targets, path, ticketNum):
                return True  # 找到有效行程
            targets[airport].insert(i, dest)  # 回溯，恢复机票
            path.pop()  # 移除目的地
        return False  # 没有找到有效行程
# 回溯 使用字典 逆序
# from collections import defaultdict
class Solution:
    def findItinerary(self, tickets):
        targets = collections.defaultdict(list)  # 创建默认字典，用于存储机场映射关系
        for ticket in tickets:
            targets[ticket[0]].append(ticket[1])  # 将机票输入到字典中
        
        for key in targets:
            targets[key].sort(reverse=True)  # 对到达机场列表进行字母逆序排序
        
        result = []
        self.backtracking("JFK", targets, result)  # 调用回溯函数开始搜索路径
        return result[::-1]  # 返回逆序的行程路径
    
    def backtracking(self, airport, targets, result):
        while targets[airport]:  # 当机场还有可到达的机场时
            next_airport = targets[airport].pop()  # 弹出下一个机场
            self.backtracking(next_airport, targets, result)  # 递归调用回溯函数进行深度优先搜索
        result.append(airport)  # 将当前机场添加到行程路径中


#X14 (Hard) 第51题.N皇后
    # n 皇后问题 研究的是如何将 n 个皇后放置在 n×n 的棋盘上, 并且使皇后彼此之间不能相互攻击。
    # 给你一个整数 n ，返回所有不同的 n 皇后问题 的解决方案。
    # 每一种解法包含一个不同的 n 皇后问题 的棋子放置方案，该方案中 'Q' 和 '.' 分别代表了皇后和空位。
    # 示例 1：
    # 输入：n = 4
    # 输出：[[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
    # 解释：如上图所示, 4 皇后问题存在两个不同的解法。
    # 示例 2：
    # 输入：n = 1
    # 输出：[["Q"]]
# 如果从来没有接触过N皇后问题的同学看着这样的题会感觉无从下手，可能知道要用回溯法，但也不知道该怎么去搜。
# 这里我明确给出了棋盘的宽度就是for循环的长度，递归的深度就是棋盘的高度，这样就可以套进回溯法的模板里了。
# Time Complexity: O(n!) 其中n是棋盘的宽度，因为每一行都需要尝试放置皇后，最多有n个位置可以放置
# Space Complexity: O(n^2) 其中n是棋盘的宽度，因为需要存储棋盘的状态
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        if not n: return []
        board = [['.'] * n for _ in range(n)]
        res = []

        # 检查当前位置是否可以放置皇后
        def isVaild(board, row, col):
            #判断同一列是否冲突
            for i in range(len(board)):
                if board[i][col] == 'Q':
                    return False
            # 判断左上角是否冲突
            i = row -1
            j = col -1
            while i>=0 and j>=0:
                if board[i][j] == 'Q':
                    return False
                i -= 1
                j -= 1
            # 判断右上角是否冲突
            i = row - 1
            j = col + 1
            while i>=0 and j < len(board):
                if board[i][j] == 'Q':
                    return False
                i -= 1
                j += 1
            return True

        # 回溯
        # row表示当前在哪一行摆放皇后
        # n表示棋盘的宽度
        def backtracking(board, row, n):
            # 如果走到最后一行, 说明已经找到一个解
            if row == n:
                temp_res = []
                for temp in board:
                    temp_str = "".join(temp)
                    temp_res.append(temp_str)
                res.append(temp_res)
                return
            # 遍历每一列
            for col in range(n):
                if not isVaild(board, row, col):
                    continue
                board[row][col] = 'Q'
                backtracking(board, row+1, n)
                board[row][col] = '.'
        
        # 从第一行开始回溯
        backtracking(board, 0, n)
        return res

class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        result = []  # 存储最终结果的二维字符串数组

        chessboard = ['.' * n for _ in range(n)]  # 初始化棋盘
        self.backtracking(n, 0, chessboard, result)  # 回溯求解
        return [[''.join(row) for row in solution] for solution in result]  # 返回结果集

    def backtracking(self, n: int, row: int, chessboard: List[str], result: List[List[str]]) -> None:
        if row == n:
            result.append(chessboard[:])  # 棋盘填满，将当前解加入结果集
            return

        for col in range(n):
            if self.isValid(row, col, chessboard):
                chessboard[row] = chessboard[row][:col] + 'Q' + chessboard[row][col+1:]  # 放置皇后
                self.backtracking(n, row + 1, chessboard, result)  # 递归到下一行
                chessboard[row] = chessboard[row][:col] + '.' + chessboard[row][col+1:]  # 回溯，撤销当前位置的皇后

    def isValid(self, row: int, col: int, chessboard: List[str]) -> bool:
        # 检查列
        for i in range(row):
            if chessboard[i][col] == 'Q':
                return False  # 当前列已经存在皇后，不合法

        # 检查 45 度角是否有皇后
        i, j = row - 1, col - 1
        while i >= 0 and j >= 0:
            if chessboard[i][j] == 'Q':
                return False  # 左上方向已经存在皇后，不合法
            i -= 1
            j -= 1

        # 检查 135 度角是否有皇后
        i, j = row - 1, col + 1
        while i >= 0 and j < len(chessboard):
            if chessboard[i][j] == 'Q':
                return False  # 右上方向已经存在皇后，不合法
            i -= 1
            j += 1

        return True  # 当前位置合法

# N Queens
    # There is a chessboard of size n x n. Your goal is to place n queens on the board 
    # such that no two queens attack each other. Return the number of distinct 
    # configurations where this is possible.
# *** 回溯 用set()验证validity
from typing import Set

res = 0
def n_queens(n: int) -> int:
    dfs(0, set(), set(), set(), n)
    return res

def dfs(r: int, diagonals_set: Set[int], anti_diagonals_set: Set[int], cols_set: Set[int], n: int) -> None:
    global res
    # Termination condition: If we have reached the end of the rows,
    # we've placed all 'n' queens.
    if r == n:
        res += 1
        return
    # Try placing a queen in each column of the current row 'r'.
        # Iterate through all columns in the current row.
    for c in range(n):
        # For any cell (r, c) on a diagonal: Moving down-right: r increases by 1, c increases by 1 → r - c stays the same.
        curr_diagonal = r - c # Calculate the diagonal index for the current position.
        # For any cell (r, c) on an anti-diagonal: Moving down-left: r increases by 1, c decreases by 1 → r + c stays the same.
        curr_anti_diagonal = r + c # Calculate the anti-diagonal index for the current position.

        # If there are queens on the current column, diagonal or anti−diagonal, skip this square.
        if (c in cols_set or curr_diagonal in diagonals_set or curr_anti_diagonal in anti_diagonals_set):
            continue
        
        # Place the queen by marking the current column, diagonal, and
        # anti −diagonal as occupied.
        cols_set.add(c)
        diagonals_set.add(curr_diagonal)
        anti_diagonals_set.add(curr_anti_diagonal)
        # Recursively move to the next row to continue placing queens.
        dfs(r + 1, diagonals_set, anti_diagonals_set, cols_set, n)
        # Backtrack by removing the current column, diagonal, and
        # anti −diagonal from the hash sets.
        cols_set.remove(c)
        diagonals_set.remove(curr_diagonal)
        anti_diagonals_set.remove(curr_anti_diagonal)


#15 (Hard) 37.解数独
    # 编写一个程序, 通过填充空格来解决数独问题。
    # 一个数独的解法需遵循如下规则： 
    # 数字 1-9 在每一行只能出现一次。 
    # 数字 1-9 在每一列只能出现一次。 
    # 数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。 
    # 空白格用 '.' 表示。
# 在树形图中可以看出我们需要的是一个二维的递归（也就是两个for循环嵌套着递归）
# 一个for循环遍历棋盘的行，一个for循环遍历棋盘的列，一行一列确定下来之后，递归遍历这个位置放9个数字的可能性！
# Time Complexity: O(9^(n^2)) 其中n是数独的大小，因为每个空格最多可以填入9个数字
# Space Complexity: O(n^2) 其中n是数独的大小，因为需要存储数独的状态
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        self.backtracking(board)

    def backtracking(self, board: List[List[str]]) -> bool:
        # 若有解, 返回True；若无解, 返回False
        for i in range(len(board)): # 遍历行
            for j in range(len(board[0])):  # 遍历列
                # 若空格内已有数字, 跳过
                if board[i][j] != '.': continue
                for k in range(1, 10):  
                    if self.is_valid(i, j, k, board):
                        board[i][j] = str(k)
                        if self.backtracking(board): return True
                        board[i][j] = '.'
                # 若数字1-9都不能成功填入空格, 返回False无解
                return False
        return True # 有解

    def is_valid(self, row: int, col: int, val: int, board: List[List[str]]) -> bool:
        # 判断同一行是否冲突
        for i in range(9):
            if board[row][i] == str(val):
                return False
        # 判断同一列是否冲突
        for j in range(9):
            if board[j][col] == str(val):
                return False
        # 判断同一九宫格是否有冲突
        start_row = (row // 3) * 3
        start_col = (col // 3) * 3
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if board[i][j] == str(val):
                    return False
        return True
        
# 回溯 使用set()验证validity
from typing import List, Set

class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        # Initialize sets for rows, columns, and boxes
        rows = [set() for _ in range(9)]
        cols = [set() for _ in range(9)]
        boxes = [set() for _ in range(9)]
        
        # Populate the sets with existing numbers
        for i in range(9):
            for j in range(9):
                if board[i][j] != '.':
                    num = int(board[i][j])
                    rows[i].add(num)
                    cols[j].add(num)
                    box_idx = (i // 3) * 3 + j // 3
                    boxes[box_idx].add(num)
        
        self.backtracking(board, rows, cols, boxes)

    def backtracking(self, board: List[List[str]], rows: List[Set[int]], cols: List[Set[int]], boxes: List[Set[int]]) -> bool:
        for i in range(9):
            for j in range(9):
                if board[i][j] != '.':
                    continue
                
                for num in range(1, 10):
                    box_idx = (i // 3) * 3 + j // 3
                    
                    # Check if the number is valid using sets
                    if num in rows[i] or num in cols[j] or num in boxes[box_idx]:
                        continue
                    
                    # Place the number
                    board[i][j] = str(num)
                    rows[i].add(num)
                    cols[j].add(num)
                    boxes[box_idx].add(num)
                    
                    if self.backtracking(board, rows, cols, boxes):
                        return True
                    
                    # Backtrack
                    board[i][j] = '.'
                    rows[i].remove(num)
                    cols[j].remove(num)
                    boxes[box_idx].remove(num)
                
                return False
        return True