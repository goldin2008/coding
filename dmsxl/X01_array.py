"""

"""
#1 704. 二分查找
    # 给定一个 n 个元素有序的（升序）整型数组 nums 和一个目标值 target  ，写一个函数搜索 nums 中的 target，如果目标值存在返回下标，否则返回 -1。
# 第一种写法, 我们定义 target 是在一个在左闭右闭的区间里, 也就是[left, right] （这个很重要非常重要）。
# 区间的定义这就决定了二分法的代码应该如何写, 因为定义target在[left, right]区间, 所以有如下两点:
# 1. while (left <= right) 要使用 <= , 因为left == right是有意义的, 所以使用 <=
# 2. if (nums[middle] > target) right 要赋值为 middle - 1, 因为当前这个nums[middle]一定不是target, 那么接下来要查找的左区间结束下标位置就是 middle - 1
# （版本一）左闭右闭区间
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1  # 定义target在左闭右闭的区间里, [left, right]

        while left <= right:
            middle = left + (right - left) // 2
            
            if nums[middle] > target:
                right = middle - 1  # target在左区间, 所以[left, middle - 1]
            elif nums[middle] < target:
                left = middle + 1  # target在右区间, 所以[middle + 1, right]
            else:
                return middle  # 数组中找到目标值, 直接返回下标
        return -1  # 未找到目标值
# （版本二）左闭右开区间
# class Solution:
#     def search(self, nums: List[int], target: int) -> int:
#         left, right = 0, len(nums)  # 定义target在左闭右开的区间里，即：[left, right)

#         while left < right:  # 因为left == right的时候，在[left, right)是无效的空间，所以使用 <
#             middle = left + (right - left) // 2

#             if nums[middle] > target:
#                 right = middle  # target 在左区间，在[left, middle)中
#             elif nums[middle] < target:
#                 left = middle + 1  # target 在右区间，在[middle + 1, right)中
#             else:
#                 return middle  # 数组中找到目标值，直接返回下标
#         return -1  # 未找到目标值


#2 27. 移除元素
    # 给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。
    # 不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并原地修改输入数组。
    # 元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。
    # 示例 1: 给定 nums = [3,2,2,3], val = 3, 函数应该返回新的长度 2, 并且 nums 中的前两个元素均为 2。 你不需要考虑数组中超出新长度后面的元素。
    # 示例 2: 给定 nums = [0,1,2,2,3,0,4,2], val = 2, 函数应该返回新的长度 5, 并且 nums 中的前五个元素为 0, 1, 3, 0, 4。
    # 你不需要考虑数组中超出新长度后面的元素。
# （版本一）快慢指针法
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        # 快慢指针
        fast = 0  # 快指针
        slow = 0  # 慢指针
        size = len(nums)
        while fast < size:  # 不加等于是因为，a = size 时，nums[a] 会越界
            # slow 用来收集不等于 val 的值，如果 fast 对应值不等于 val，则把它与 slow 替换
            if nums[fast] != val:
                nums[slow] = nums[fast]
                slow += 1
            fast += 1
        return slow
# （版本二）暴力法
# class Solution:
#     def removeElement(self, nums: List[int], val: int) -> int:
#         i, l = 0, len(nums)
#         while i < l:
#             if nums[i] == val: # 找到等于目标值的节点
#                 for j in range(i+1, l): # 移除该元素，并将后面元素向前平移
#                     nums[j - 1] = nums[j]
#                 l -= 1
#                 i -= 1
#             i += 1
#         return l


#3 977.有序数组的平方
    # 给你一个按 非递减顺序 排序的整数数组 nums，返回 每个数字的平方 组成的新数组，要求也按 非递减顺序 排序。
    # 示例 1：
    # 输入：nums = [-4,-1,0,3,10]
    # 输出：[0,1,9,16,100]
    # 解释：平方后，数组变为 [16,1,0,9,100]，排序后，数组变为 [0,1,9,16,100]
    # 示例 2：
    # 输入：nums = [-7,-3,2,3,11]
    # 输出：[4,9,9,49,121]
# （版本一）双指针法
class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        l, r, i = 0, len(nums)-1, len(nums)-1
        res = [float('inf')] * len(nums) # 需要提前定义列表，存放结果
        while l <= r:
            if nums[l] ** 2 < nums[r] ** 2: # 左右边界进行对比，找出最大值
                res[i] = nums[r] ** 2
                r -= 1 # 右指针往左移动
            else:
                res[i] = nums[l] ** 2
                l += 1 # 左指针往右移动
            i -= 1 # 存放结果的指针需要往前平移一位
        return res
# （版本二）暴力排序法
# class Solution:
#     def sortedSquares(self, nums: List[int]) -> List[int]:
#         for i in range(len(nums)):
#             nums[i] *= nums[i]
#         nums.sort()
#         return nums
# （版本三）暴力排序法+列表推导法
# class Solution:
#     def sortedSquares(self, nums: List[int]) -> List[int]:
#         return sorted(x*x for x in nums)


#4 209.长度最小的子数组
    # 给定一个含有 n 个正整数的数组和一个正整数 s ，找出该数组中满足其和 ≥ s 的长度最小的 连续 子数组，并返回其长度。如果不存在符合条件的子数组，返回 0。
    # 示例：
    # 输入：s = 7, nums = [2,3,1,2,4,3]
    # 输出：2
    # 解释：子数组 [4,3] 是该条件下的长度最小的子数组。
# 904.水果成篮
# 76.最小覆盖子串
# （版本一）滑动窗口法
class Solution:
    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        l = len(nums)
        left = 0
        right = 0
        min_len = float('inf')
        cur_sum = 0 #当前的累加值
        
        while right < l:
            cur_sum += nums[right]
            
            while cur_sum >= s: # 当前累加值大于目标值
                min_len = min(min_len, right - left + 1)
                cur_sum -= nums[left]
                left += 1
            
            right += 1
        
        return min_len if min_len != float('inf') else 0
# （版本二）暴力法
# class Solution:
#     def minSubArrayLen(self, s: int, nums: List[int]) -> int:
#         l = len(nums)
#         min_len = float('inf')
        
#         for i in range(l):
#             cur_sum = 0
#             for j in range(i, l):
#                 cur_sum += nums[j]
#                 if cur_sum >= s:
#                     min_len = min(min_len, j - i + 1)
#                     break
        
#         return min_len if min_len != float('inf') else 0


#5 59.螺旋矩阵II
    # 给定一个正整数 n，生成一个包含 1 到 n^2 所有元素，且元素按顺时针顺序螺旋排列的正方形矩阵。
    # 示例:
    # 输入: 3 输出: [ [ 1, 2, 3 ], [ 8, 9, 4 ], [ 7, 6, 5 ] ]
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        nums = [[0] * n for _ in range(n)]
        startx, starty = 0, 0               # 起始点
        loop, mid = n // 2, n // 2          # 迭代次数、n为奇数时，矩阵的中心点
        count = 1                           # 计数

        for offset in range(1, loop + 1) :      # 每循环一层偏移量加1，偏移量从1开始
            for i in range(starty, n - offset) :    # 从左至右，左闭右开
                nums[startx][i] = count
                count += 1
            for i in range(startx, n - offset) :    # 从上至下
                nums[i][n - offset] = count
                count += 1
            for i in range(n - offset, starty, -1) : # 从右至左
                nums[n - offset][i] = count
                count += 1
            for i in range(n - offset, startx, -1) : # 从下至上
                nums[i][starty] = count
                count += 1                
            startx += 1         # 更新起始点
            starty += 1

        if n % 2 != 0 :			# n为奇数时，填充中心点
            nums[mid][mid] = count 
        return nums

# Prefix Sums
#X6 Easy Sum Between Range
# Given an integer array, write a function which returns the sum of values between two indexes.
# Example:
# Input: nums = [3, -7, 6, 0, -2, 5],
#        [sum_range(0, 3), sum_range(2, 4), sum_range(2, 2)]
# Output: [2, 4, 6]
class SumBetweenRange:
    def __init__(self, nums: List[int]):
        self.prefix_sum = [nums[0]]
        for i in range(1, len(nums)):
            self.prefix_sum.append(self.prefix_sum[-1] + nums[i])

    def sum_range(self, i: int, j: int) -> int:
        if i == 0:
            return self.prefix_sum[j]
        return self.prefix_sum[j] - self.prefix_sum[i - 1]
#X7 Medium K-Sum Subarrays
# Find the number of subarrays in an integer array that sum to k.
# Example:
# Input: nums = [1, 2, -1, 1, 2], k = 3
# Output: 3
def k_sum_subarrays(nums: List[int], k: int) -> int:
    n = len(nums)
    count = 0
    # Populate the prefix sum array, setting its first element to 0.
    prefix_sum = [0]
    for i in range(0, n):
        prefix_sum.append(prefix_sum[-1] + nums[i])
    # Loop through all valid pairs of prefix sum values to find all 
    # subarrays that sum to 'k'.
    for j in range(1, n + 1):
        for i in range(1, j + 1):
            if prefix_sum[j] - prefix_sum[i - 1] == k:
                count += 1
    return count

def k_sum_subarrays_optimized(nums: List[int], k: int) -> int:
    count = 0
    # Initialize the map with 0 to handle subarrays that sum to 'k' 
    # from the start of the array.
    prefix_sum_map = {0: 1}
    curr_prefix_sum = 0
    for num in nums:
        # Update the running prefix sum by adding the current number.
        curr_prefix_sum += num
        # If a subarray with sum 'k' exists, increment 'count' by the 
        # number of times it has been found.
        if curr_prefix_sum - k in prefix_sum_map:
            count += prefix_sum_map[curr_prefix_sum - k]
        # Update the frequency of 'curr_prefix_sum' in the hash map.
        freq = prefix_sum_map.get(curr_prefix_sum, 0)
        prefix_sum_map[curr_prefix_sum] = freq + 1
    return count

#X8 Medium Product Array Without Current Element
# Given an array of integers, return an array res so that res[i] is equal to the 
# product of all the elements of the input array except nums[i] itself.
# Example:
# Input: nums = [2, 3, 1, 4, 5]
# Output: [60, 40, 120, 30, 24]
# Explanation: The output value at index 0 is the product of all numbers 
# except nums[0] (3⋅1⋅4⋅5 = 60). The same logic applies to the rest of the output.
def product_array_without_current_element(nums: List[int]) -> List[int]:
    n = len(nums)
    res = [1] * n
    # Populate the output with the running left product.
    for i in range(1, n):
        res[i] = res[i - 1] * nums[i - 1]
    # Multiply the output with the running right product, from right to 
    # left.
    right_product = 1
    for i in range(n - 1, -1, -1):
        res[i] *= right_product
        right_product *= nums[i]
    return res

# Intervals
#X9 Medium Merge Overlapping Intervals
# Merge an array of intervals so there are no overlapping intervals, 
# and return the resultant merged intervals.
# Example:
# Input: intervals = [[3, 4], [7, 8], [2, 5], [6, 7], [1, 4]]
# Output: [[1, 5], [6, 8]]
from ds import Interval
from typing import List
"""
Definition of Interval:
class Interval:
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end
"""
def merge_overlapping_intervals(intervals: List[Interval]) -> List[Interval]:
    intervals.sort(key=lambda x: x.start)
    merged = [intervals[0]]
    for B in intervals[1:]:
        A = merged[-1]
        # If A and B don't overlap, add B to the merged list.
        if A.end < B.start:
            merged.append(B)
        # If they do overlap, merge A with B.
        else:
            merged[-1] = Interval(A.start, max(A.end, B.end))
    return merged

#X10 Medium Identify All Interval Overlaps
# Return an array of all overlaps between two arrays of intervals; intervals1 and intervals2. 
# Each individual interval array is sorted by start value, and contains no overlapping 
# intervals within itself.
# Example:
# Input: intervals1 = [[1, 4], [5, 6], [9, 10]],
#        intervals2 = [[2, 7], [8, 9]]
# Output: [[2, 4], [5, 6], [9, 9]]
# Constraints:
# For every index i in intervals1, intervals1[i].start < intervals1[i].end.
# For every index j in intervals2, intervals2[j].start < intervals2[j].end.
def identify_all_interval_overlaps(intervals1: List[Interval], intervals2: List[Interval]) -> List[Interval]:
    overlaps = []
    i = j = 0
    while i < len(intervals1) and j < len(intervals2):
        # Set A to the interval that starts first and B to the other 
        # interval.
        if intervals1[i].start <= intervals2[j].start:
            A, B = intervals1[i], intervals2[j]
        else:
            A, B = intervals2[j], intervals1[i]
        # If there's an overlap, add the overlap.
        if A.end >= B.start:
            overlaps.append(Interval(B.start, min(A.end, B.end)))
        # Advance the pointer associated with the interval that ends 
        # first.
        if intervals1[i].end < intervals2[j].end:
            i += 1
        else:
            j += 1
    return overlaps

#X11 Medium Largest Overlap of Intervals
# Given an array of intervals, determine the maximum number of intervals that overlap 
# at any point. Each interval is half-open, meaning it includes the start point but 
# excludes the end point.
# Example:
# Input: intervals = [[1, 3], [5, 7], [2, 6], [4, 8]]
# Output: 3
# Constraints:
# The input will contain at least one interval.
# For every index i in the list, intervals[i].start < intervals[i].end.
def largest_overlap_of_intervals(intervals: List[Interval]) -> int:
    points = []
    for interval in intervals:
        points.append((interval.start, 'S'))
        points.append((interval.end, 'E'))
    # Sort in chronological order. If multiple points occur at the same 
    # time, ensure end points are prioritized before start points.
    points.sort(key=lambda x: (x[0], x[1]))
    active_intervals = 0
    max_overlaps = 0
    for time, point_type in points:
        if point_type == 'S':
            active_intervals += 1
        else:
            active_intervals -= 1
        max_overlaps = max(max_overlaps, active_intervals)
    return max_overlaps