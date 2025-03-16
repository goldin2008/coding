"""


"""
#1 739. 每日温度
    # 请根据每日 气温 列表，重新生成一个列表。对应位置的输出为：要想观测到更高的气温，至少需要等待的天数。如果气温在这之后都不会升高，请在该位置用 0 来代替。
    # 例如，给定一个列表 temperatures = [73, 74, 75, 71, 69, 72, 76, 73]，你的输出应该是 [1, 1, 4, 2, 1, 1, 0, 0]。
    # 提示：气温 列表长度的范围是 [1, 30000]。每个气温的值的均为华氏度，都是在 [30, 100] 范围内的整数。
# 首先想到的当然是暴力解法，两层for循环，把至少需要等待的天数就搜出来了。时间复杂度是O(n^2)
# 通常是一维数组，要寻找任一个元素的右边或者左边第一个比自己大或者小的元素的位置，此时我们就要想到可以用单调栈了。时间复杂度为O(n)。
# 情况一：当前遍历的元素T[i]小于栈顶元素T[st.top()]的情况
# 情况二：当前遍历的元素T[i]等于栈顶元素T[st.top()]的情况
# 情况三：当前遍历的元素T[i]大于栈顶元素T[st.top()]的情况
# 未精简版本
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        answer = [0]*len(temperatures)
        stack = [0]
        for i in range(1,len(temperatures)):
            # 情况一和情况二
            if temperatures[i]<=temperatures[stack[-1]]:
                stack.append(i)
            # 情况三
            else:
                while len(stack) != 0 and temperatures[i]>temperatures[stack[-1]]:
                    answer[stack[-1]]=i-stack[-1]
                    stack.pop()
                stack.append(i)
            
        return answer
# *** 精简版本
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        answer = [0]*len(temperatures)
        stack = []
        for i in range(len(temperatures)):
            while len(stack)>0 and temperatures[i] > temperatures[stack[-1]]:
                answer[stack[-1]] = i - stack[-1]
                stack.pop()
            stack.append(i)
        return answer


#2 496.下一个更大元素 I
    # 给你两个 没有重复元素 的数组 nums1 和 nums2 ，其中nums1 是 nums2 的子集。
    # 请你找出 nums1 中每个元素在 nums2 中的下一个比其大的值。
    # nums1 中数字 x 的下一个更大元素是指 x 在 nums2 中对应位置的右边的第一个比 x 大的元素。如果不存在，对应位置输出 -1 。
    # 示例 1:
    # 输入: nums1 = [4,1,2], nums2 = [1,3,4,2].
    # 输出: [-1,3,-1]
    # 解释:
    # 对于 num1 中的数字 4 ，你无法在第二个数组中找到下一个更大的数字，因此输出 -1 。
    # 对于 num1 中的数字 1 ，第二个数组中数字1右边的下一个较大数字是 3 。
    # 对于 num1 中的数字 2 ，第二个数组中没有下一个更大的数字，因此输出 -1 。
    # 示例 2:
    # 输入: nums1 = [2,4], nums2 = [1,2,3,4].
    # 输出: [3,-1]
    # 解释:
    # 对于 num1 中的数字 2 ，第二个数组中的下一个较大数字是 3 。
    # 对于 num1 中的数字 4 ，第二个数组中没有下一个更大的数字，因此输出-1 。
    # 提示：
    # 1 <= nums1.length <= nums2.length <= 1000
    # 0 <= nums1[i], nums2[i] <= 10^4
    # nums1和nums2中所有整数 互不相同
    # nums1 中的所有整数同样出现在 nums2 中
class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        result = [-1]*len(nums1)
        stack = [0]
        for i in range(1,len(nums2)):
            # 情况一情况二
            if nums2[i]<=nums2[stack[-1]]:
                stack.append(i)
            # 情况三
            else:
                while len(stack)!=0 and nums2[i]>nums2[stack[-1]]:
                    if nums2[stack[-1]] in nums1:
                        index = nums1.index(nums2[stack[-1]])
                        result[index]=nums2[i]
                    stack.pop()                 
                stack.append(i)
        return result
# *** 精简版本
class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        result = [-1]*len(nums1)
        stack = [0]
        for i in range(1,len(nums2)):
            # 情况三
            while len(stack)!=0 and nums2[i]>nums2[stack[-1]]:
                if nums2[stack[-1]] in nums1:
                    index = nums1.index(nums2[stack[-1]])
                    result[index]=nums2[i]
                stack.pop()
            
            # 情况一情况二  
            stack.append(i)
        return result


#3 503.下一个更大元素II
    # 给定一个循环数组（最后一个元素的下一个元素是数组的第一个元素），输出每个元素的下一个更大元素。数字 x 的下一个更大的元素是按数组遍历顺序，这个数字之后的第一个比它更大的数，这意味着你应该循环地搜索它的下一个更大的数。如果不存在，则输出 -1。
    # 示例 1:
    # 输入: [1,2,1]
    # 输出: [2,-1,2]
    # 解释: 第一个 1 的下一个更大的数是 2；数字 2 找不到下一个更大的数；第二个 1 的下一个最大的数需要循环搜索，结果也是 2。
    # 提示:
    # 1 <= nums.length <= 10^4
    # -10^9 <= nums[i] <= 10^9
# 方法 1:
class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        dp = [-1] * len(nums)
        stack = []
        for i in range(len(nums)*2):
            while(len(stack) != 0 and nums[i%len(nums)] > nums[stack[-1]]):
                    dp[stack[-1]] = nums[i%len(nums)]
                    stack.pop()
            stack.append(i%len(nums))
        return dp
# 方法 2:
class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        stack = []
        # 创建答案数组
        ans = [-1] * len(nums1)
        for i in range(len(nums2)):
            while len(stack) > 0 and nums2[i] > nums2[stack[-1]]:
                # 判断 num1 是否有 nums2[stack[-1]]。如果没有这个判断会出现指针异常
                if nums2[stack[-1]] in nums1:
                    # 锁定 num1 检索的 index
                    index = nums1.index(nums2[stack[-1]])
                    # 更新答案数组
                    ans[index] = nums2[i]
                # 弹出小元素
                # 这个代码一定要放在 if 外面。否则单调栈的逻辑就不成立了
                stack.pop()
            stack.append(i)
        return ans


#4 42. 接雨水
    # 给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。
    # 示例 1：
    # 输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
    # 输出：6
    # 解释：上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。
    # 示例 2：
    # 输入：height = [4,2,0,3,2,5]
    # 输出：9
# 通常是一维数组，要寻找任一个元素的右边或者左边第一个比自己大或者小的元素的位置，此时我们就要想到可以用单调栈了。
# 而接雨水这道题目，我们正需要寻找一个元素，右边最大元素以及左边最大元素，来计算雨水面积。
# 暴力解法
# class Solution:
#     def trap(self, height: List[int]) -> int:
#         res = 0
#         for i in range(len(height)):
#             if i == 0 or i == len(height)-1: continue
#             lHight = height[i-1]
#             rHight = height[i+1]
#             for j in range(i-1):
#                 if height[j] > lHight:
#                     lHight = height[j]
#             for k in range(i+2,len(height)):
#                 if height[k] > rHight:
#                     rHight = height[k]
#             res1 = min(lHight,rHight) - height[i]
#             if res1 > 0:
#                 res += res1
#         return res
# # 双指针
# class Solution:
#     def trap(self, height: List[int]) -> int:
#         leftheight, rightheight = [0]*len(height), [0]*len(height)

#         leftheight[0]=height[0]
#         for i in range(1,len(height)):
#             leftheight[i]=max(leftheight[i-1],height[i])
#         rightheight[-1]=height[-1]
#         for i in range(len(height)-2,-1,-1):
#             rightheight[i]=max(rightheight[i+1],height[i])

#         result = 0
#         for i in range(0,len(height)):
#             summ = min(leftheight[i],rightheight[i])-height[i]
#             result += summ
#         return result
# 单调栈
# 从栈头（元素从栈头弹出）到栈底的顺序应该是从小到大的顺序。
# 因为一旦发现添加的柱子高度大于栈头元素了，此时就出现凹槽了，
# 栈头元素就是凹槽底部的柱子，栈头第二个元素就是凹槽左边的柱子，而添加的元素就是凹槽右边的柱子。
# 情况一：当前遍历的元素（柱子）高度小于栈顶元素的高度 height[i] < height[st.top()]
# 情况二：当前遍历的元素（柱子）高度等于栈顶元素的高度 height[i] == height[st.top()]
# 情况三：当前遍历的元素（柱子）高度大于栈顶元素的高度 height[i] > height[st.top()]
# *** 单调栈非压缩版 
class Solution:
    def trap(self, height: List[int]) -> int:
        # 单调栈
        '''
        单调栈是按照 行 的方向来计算雨水
        从栈顶到栈底的顺序：从小到大
        通过三个元素来接水：栈顶，栈顶的下一个元素，以及即将入栈的元素
        雨水高度是 min(凹槽左边高度, 凹槽右边高度) - 凹槽底部高度
        雨水的宽度是 凹槽右边的下标 - 凹槽左边的下标 - 1 (因为只求中间宽度)
        '''
        # stack储存index，用于计算对应的柱子高度
        stack = [0]
        result = 0
        for i in range(1, len(height)):
            # 情况一
            if height[i] < height[stack[-1]]:
                stack.append(i)

            # 情况二
            # 当当前柱子高度和栈顶一致时，左边的一个是不可能存放雨水的，所以保留右侧新柱子
            # 需要使用最右边的柱子来计算宽度
            elif height[i] == height[stack[-1]]:
                stack.pop()
                stack.append(i)

            # 情况三
            else:
                # 抛出所有较低的柱子
                while stack and height[i] > height[stack[-1]]:
                    # 栈顶就是中间的柱子：储水槽，就是凹槽的地步
                    mid_height = height[stack[-1]]
                    stack.pop()
                    if stack:
                        right_height = height[i]
                        left_height = height[stack[-1]]
                        # 两侧的较矮一方的高度 - 凹槽底部高度
                        h = min(right_height, left_height) - mid_height
                        # 凹槽右侧下标 - 凹槽左侧下标 - 1: 只求中间宽度
                        w = i - stack[-1] - 1
                        # 体积：高乘宽
                        result += h * w
                stack.append(i)
        return result
# 单调栈压缩版        
class Solution:
    def trap(self, height: List[int]) -> int:
        stack = [0]
        result = 0
        for i in range(1, len(height)):
            while stack and height[i] > height[stack[-1]]:
                mid_height = stack.pop()
                if stack:
                    # 雨水高度是 min(凹槽左侧高度, 凹槽右侧高度) - 凹槽底部高度
                    h = min(height[stack[-1]], height[i]) - height[mid_height]
                    # 雨水宽度是 凹槽右侧的下标 - 凹槽左侧的下标 - 1
                    w = i - stack[-1] - 1
                    # 累计总雨水体积
                    result += h * w
            stack.append(i)
        return result


#5 84.柱状图中最大的矩形
    # 给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。
    # 求在该柱状图中，能够勾勒出来的矩形的最大面积。
# 暴力解法（leetcode超时）
# class Solution:
#     def largestRectangleArea(self, heights: List[int]) -> int:
#         # 从左向右遍历：以每一根柱子为主心骨（当前轮最高的参照物），迭代直到找到左侧和右侧各第一个矮一级的柱子
#         res = 0

#         for i in range(len(heights)):
#             left = i
#             right = i
#             # 向左侧遍历：寻找第一个矮一级的柱子
#             for _ in range(left, -1, -1):
#                 if heights[left] < heights[i]:
#                     break
#                 left -= 1
#             # 向右侧遍历：寻找第一个矮一级的柱子
#             for _ in range(right, len(heights)):
#                 if heights[right] < heights[i]:
#                     break
#                 right += 1
                
#             width = right - left - 1
#             height = heights[i]
#             res = max(res, width * height)

#         return res

# # 双指针 
# class Solution:
#     def largestRectangleArea(self, heights: List[int]) -> int:
#         size = len(heights)
#         # 两个DP数列储存的均是下标index
#         min_left_index = [0] * size
#         min_right_index = [0] * size
#         result = 0

#         # 记录每个柱子的左侧第一个矮一级的柱子的下标
#         min_left_index[0] = -1  # 初始化防止while死循环
#         for i in range(1, size):
#             # 以当前柱子为主心骨，向左迭代寻找次级柱子
#             temp = i - 1
#             while temp >= 0 and heights[temp] >= heights[i]:
#                 # 当左侧的柱子持续较高时，尝试这个高柱子自己的次级柱子（DP
#                 temp = min_left_index[temp]
#             # 当找到左侧矮一级的目标柱子时
#             min_left_index[i] = temp
        
#         # 记录每个柱子的右侧第一个矮一级的柱子的下标
#         min_right_index[size-1] = size  # 初始化防止while死循环
#         for i in range(size-2, -1, -1):
#             # 以当前柱子为主心骨，向右迭代寻找次级柱子
#             temp = i + 1
#             while temp < size and heights[temp] >= heights[i]:
#                 # 当右侧的柱子持续较高时，尝试这个高柱子自己的次级柱子（DP
#                 temp = min_right_index[temp]
#             # 当找到右侧矮一级的目标柱子时
#             min_right_index[i] = temp
        
#         for i in range(size):
#             area = heights[i] * (min_right_index[i] - min_left_index[i] - 1)
#             result = max(area, result)
#         return result
# 单调栈
# 本地单调栈的解法和接雨水的题目是遥相呼应的。
# 为什么这么说呢，42. 接雨水 是找每个柱子左右两边第一个大于该柱子高度的柱子，而本题是找每个柱子左右两边第一个小于该柱子的柱子。
# 这里就涉及到了单调栈很重要的性质，就是单调栈里的顺序，是从小到大还是从大到小。
# 在题解42. 接雨水 中我讲解了接雨水的单调栈从栈头（元素从栈头弹出）到栈底的顺序应该是从小到大的顺序。
# 那么因为本题是要找每个柱子左右两边第一个小于该柱子的柱子，所以从栈头（元素从栈头弹出）到栈底的顺序应该是从大到小的顺序！
# 只有栈里从大到小的顺序，才能保证栈顶元素找到左右两边第一个小于栈顶元素的柱子。
# 所以本题单调栈的顺序正好与接雨水反过来。
# 此时大家应该可以发现其实就是栈顶和栈顶的下一个元素以及要入栈的三个元素组成了我们要求最大面积的高度和宽度

# 情况一：当前遍历的元素heights[i]大于栈顶元素heights[st.top()]的情况
# 情况二：当前遍历的元素heights[i]等于栈顶元素heights[st.top()]的情况
# 情况三：当前遍历的元素heights[i]小于栈顶元素heights[st.top()]的情况
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        # Monotonic Stack
        '''
        找每个柱子左右侧的第一个高度值小于该柱子的柱子
        单调栈：栈顶到栈底：从大到小（每插入一个新的小数值时，都要弹出先前的大数值）
        栈顶，栈顶的下一个元素，即将入栈的元素：这三个元素组成了最大面积的高度和宽度
        情况一: 当前遍历的元素heights[i]大于栈顶元素的情况
        情况二: 当前遍历的元素heights[i]等于栈顶元素的情况
        情况三: 当前遍历的元素heights[i]小于栈顶元素的情况
        '''

        # 输入数组首尾各补上一个0（与42.接雨水不同的是，本题原首尾的两个柱子可以作为核心柱进行最大面积尝试
        heights.insert(0, 0)
        heights.append(0)
        stack = [0]
        result = 0
        for i in range(1, len(heights)):
            # 情况一
            if heights[i] > heights[stack[-1]]:
                stack.append(i)
            # 情况二
            elif heights[i] == heights[stack[-1]]:
                stack.pop()
                stack.append(i)
            # 情况三
            else:
                # 抛出所有较高的柱子
                while stack and heights[i] < heights[stack[-1]]:
                    # 栈顶就是中间的柱子，主心骨
                    mid_index = stack[-1]
                    stack.pop()
                    if stack:
                        left_index = stack[-1]
                        right_index = i
                        width = right_index - left_index - 1
                        height = heights[mid_index]
                        result = max(result, width * height)
                stack.append(i)
        return result

# 单调栈精简
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        heights.insert(0, 0)
        heights.append(0)
        stack = [0]
        result = 0
        for i in range(1, len(heights)):
            while stack and heights[i] < heights[stack[-1]]:
                mid_height = heights[stack[-1]]
                stack.pop()
                if stack:
                    # area = width * height
                    area = (i - stack[-1] - 1) * mid_height
                    result = max(area, result)
            stack.append(i)
        return result