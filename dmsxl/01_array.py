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
