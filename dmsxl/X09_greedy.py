"""
贪心算法一般分为如下四步：

将问题分解为若干个子问题
找出适合的贪心策略
求解每一个子问题的最优解
将局部最优解堆叠成全局最优解
"""
#1 (Easy) 455.分发饼干
    # 假设你是一位很棒的家长，想要给你的孩子们一些小饼干。但是，每个孩子最多只能给一块饼干。
    # 对每个孩子 i，都有一个胃口值  g[i]，这是能让孩子们满足胃口的饼干的最小尺寸；并且每块饼干 j，都有一个尺寸 s[j] 。如果 s[j] >= g[i]，我们可以将这个饼干 j 分配给孩子 i ，这个孩子会得到满足。你的目标是尽可能满足越多数量的孩子，并输出这个最大数值。
    # 示例  1:
    # 输入: g = [1,2,3], s = [1,1]
    # 输出: 1 解释:你有三个孩子和两块小饼干，3 个孩子的胃口值分别是：1,2,3。虽然你有两块小饼干，由于他们的尺寸都是 1，你只能让胃口值是 1 的孩子满足。所以你应该输出 1。
    # 示例  2:
    # 输入: g = [1,2], s = [1,2,3]
    # 输出: 2
    # 解释:你有两个孩子和三块小饼干，2 个孩子的胃口值分别是 1,2。你拥有的饼干数量和尺寸都足以让所有孩子满足。所以你应该输出 2.
# 注意不可以小胃口优先
# # 贪心 大饼干优先
# class Solution:
#     def findContentChildren(self, g, s):
#         g.sort()  # 将孩子的贪心因子排序
#         s.sort()  # 将饼干的尺寸排序
#         index = len(s) - 1  # 饼干数组的下标，从最后一个饼干开始
#         result = 0  # 满足孩子的数量
#         for i in range(len(g)-1, -1, -1):  # 遍历胃口，从最后一个孩子开始
#             if index >= 0 and s[index] >= g[i]:  # 遍历饼干
#                 result += 1
#                 index -= 1
#         return result
# 贪心 小饼干优先
class Solution:
    def findContentChildren(self, g, s):
        g.sort()  # 将孩子的贪心因子排序
        s.sort()  # 将饼干的尺寸排序
        index = 0
        for i in range(len(s)):  # 遍历饼干
            if index < len(g) and g[index] <= s[i]:  # 如果当前孩子的贪心因子小于等于当前饼干尺寸
                index += 1  # 满足一个孩子，指向下一个孩子
        return index  # 返回满足的孩子数目


#2 (Medium) 376.摆动序列
    # 如果连续数字之间的差严格地在正数和负数之间交替，则数字序列称为摆动序列。第一个差（如果存在的话）可能是正数或负数。少于两个元素的序列也是摆动序列。
    # 例如， [1,7,4,9,2,5] 是一个摆动序列，因为差值 (6,-3,5,-7,3)  是正负交替出现的。相反, [1,4,7,2,5]  和  [1,7,4,5,5] 不是摆动序列，第一个序列是因为它的前两个差值都是正数，第二个序列是因为它的最后一个差值为零。
    # 给定一个整数序列，返回作为摆动序列的最长子序列的长度。 通过从原始序列中删除一些（也可以不删除）元素来获得子序列，剩下的元素保持其原始顺序。
    # 示例 1:
    # 输入: [1,7,4,9,2,5]
    # 输出: 6
    # 解释: 整个序列均为摆动序列。
    # 示例 2:
    # 输入: [1,17,5,10,13,15,10,5,16,8]
    # 输出: 7
    # 解释: 这个序列包含几个长度为 7 摆动序列，其中一个可为[1,17,10,13,10,16,8]。
    # 示例 3:
    # 输入: [1,2,3,4,5,6,7,8,9]
    # 输出: 2
# 考虑用动态规划的思想来解决这个问题。
# 很容易可以发现，对于我们当前考虑的这个数，要么是作为山峰（即 nums[i] > nums[i-1]），要么是作为山谷（即 nums[i] < nums[i - 1]）。
# 设 dp 状态dp[i][0]，表示考虑前 i 个数，第 i 个数作为山峰的摆动子序列的最长长度
# 设 dp 状态dp[i][1]，表示考虑前 i 个数，第 i 个数作为山谷的摆动子序列的最长长度
# 则转移方程为：
# dp[i][0] = max(dp[i][0], dp[j][1] + 1)，其中0 < j < i且nums[j] < nums[i]，表示将 nums[i]接到前面某个山谷后面，作为山峰。
# dp[i][1] = max(dp[i][1], dp[j][0] + 1)，其中0 < j < i且nums[j] > nums[i]，表示将 nums[i]接到前面某个山峰后面，作为山谷。
# 初始状态：
# 由于一个数可以接到前面的某个数后面，也可以以自身为子序列的起点，所以初始状态为：dp[0][0] = dp[0][1] = 1。
# 贪心（版本一）
# class Solution:
#     def wiggleMaxLength(self, nums):
#         if len(nums) <= 1:
#             return len(nums)  # 如果数组长度为0或1，则返回数组长度
#         curDiff = 0  # 当前一对元素的差值
#         preDiff = 0  # 前一对元素的差值
#         result = 1  # 记录峰值的个数，初始为1（默认最右边的元素被视为峰值）
#         for i in range(len(nums) - 1):
#             curDiff = nums[i + 1] - nums[i]  # 计算下一个元素与当前元素的差值
#             # 如果遇到一个峰值
#             if (preDiff <= 0 and curDiff > 0) or (preDiff >= 0 and curDiff < 0):
#                 result += 1  # 峰值个数加1
#                 preDiff = curDiff  # 注意这里，只在摆动变化的时候更新preDiff
#         return result  # 返回最长摆动子序列的长度
# # 贪心（版本二）
# class Solution:
#     def wiggleMaxLength(self, nums: List[int]) -> int:
#         if len(nums) <= 1:
#             return len(nums)  # 如果数组长度为0或1，则返回数组长度
#         preDiff,curDiff ,result  = 0,0,1  #题目里nums长度大于等于1，当长度为1时，其实到不了for循环里去，所以不用考虑nums长度
#         for i in range(len(nums) - 1):
#             curDiff = nums[i + 1] - nums[i]
#             if curDiff * preDiff <= 0 and curDiff !=0:  #差值为0时，不算摆动
#                 result += 1
#                 preDiff = curDiff  #如果当前差值和上一个差值为一正一负时，才需要用当前差值替代上一个差值
#         return result
# # 动态规划（版本一）
# class Solution:
#     def wiggleMaxLength(self, nums: List[int]) -> int:
#         # 0 i 作为波峰的最大长度
#         # 1 i 作为波谷的最大长度
#         # dp是一个列表，列表中每个元素是长度为 2 的列表
#         dp = []
#         for i in range(len(nums)):
#             # 初始为[1, 1]
#             dp.append([1, 1])
#             for j in range(i):
#                 # nums[i] 为波谷
#                 if nums[j] > nums[i]:
#                     dp[i][1] = max(dp[i][1], dp[j][0] + 1)
#                 # nums[i] 为波峰
#                 if nums[j] < nums[i]:
#                     dp[i][0] = max(dp[i][0], dp[j][1] + 1)
#         return max(dp[-1][0], dp[-1][1])
# 动态规划（版本二）
class Solution:
    def wiggleMaxLength(self, nums):
        # 0 i 作为波峰的最大长度
        # 1 i 作为波谷的最大长度
        # dp是一个列表，列表中每个元素是长度为 2 的列表
        dp = [[0, 0] for _ in range(len(nums))]  # 创建二维dp数组，用于记录摆动序列的最大长度
        dp[0][0] = dp[0][1] = 1  # 初始条件，序列中的第一个元素默认为峰值，最小长度为1
        for i in range(1, len(nums)):
            dp[i][0] = dp[i][1] = 1  # 初始化当前位置的dp值为1
            for j in range(i):
                if nums[j] > nums[i]:
                    dp[i][1] = max(dp[i][1], dp[j][0] + 1)  # 如果前一个数比当前数大，可以形成一个上升峰值，更新dp[i][1]
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i][0] = max(dp[i][0], dp[j][1] + 1)  # 如果前一个数比当前数小，可以形成一个下降峰值，更新dp[i][0]
        return max(dp[-1][0], dp[-1][1])  # 返回最大的摆动序列长度
# 动态规划（版本三）优化
# class Solution:
#     def wiggleMaxLength(self, nums):
#         if len(nums) <= 1:
#             return len(nums)  # 如果数组长度为0或1，则返回数组长度
        
#         up = down = 1  # 记录上升和下降摆动序列的最大长度
#         for i in range(1, len(nums)):
#             if nums[i] > nums[i-1]:
#                 up = down + 1  # 如果当前数比前一个数大，则可以形成一个上升峰值
#             elif nums[i] < nums[i-1]:
#                 down = up + 1  # 如果当前数比前一个数小，则可以形成一个下降峰值
        
#         return max(up, down)  # 返回上升和下降摆动序列的最大长度
class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        preC,curC,res = 0,0,1  #题目里nums长度大于等于1，当长度为1时，其实到不了for循环里去，所以不用考虑nums长度
        for i in range(len(nums) - 1):
            curC = nums[i + 1] - nums[i]
            if curC * preC <= 0 and curC !=0:  #差值为0时，不算摆动
                res += 1
                preC = curC  #如果当前差值和上一个差值为一正一负时，才需要用当前差值替代上一个差值
        return res


#3 (Medium) 53.最大子序和
    # 给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
    # 示例:
    # 输入: [-2,1,-3,4,-1,2,1,-5,4]
    # 输出: 6
    # 解释:  连续子数组  [4,-1,2,1] 的和最大，为  6。
# 暴力法
# class Solution:
#     def maxSubArray(self, nums):
#         result = float('-inf')  # 初始化结果为负无穷大
#         count = 0
#         for i in range(len(nums)):  # 设置起始位置
#             count = 0
#             for j in range(i, len(nums)):  # 从起始位置i开始遍历寻找最大值
#                 count += nums[j]
#                 result = max(count, result)  # 更新最大值
#         return result
class Solution:
    def maxSubArray(self, nums):
        # result = float('-inf')  # 初始化结果为负无穷大
        result = nums[0]  # 初始化结果为负无穷大
        count = 0
        for i in range(len(nums)):
            count += nums[i]
            if count > result:  # 取区间累计的最大值（相当于不断确定最大子序终止位置）
                result = count
            if count <= 0:  # 相当于重置最大子序起始位置，因为遇到负数一定是拉低总和
                count = 0
        return result


#4 (Medium) 122.买卖股票的最佳时机II
    # 给定一个数组，它的第  i 个元素是一支给定股票第 i 天的价格。
    # 设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。
    # 注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
    # 示例 1:
    # 输入: [7,1,5,3,6,4]
    # 输出: 7
    # 解释: 在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4。随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6-3 = 3 。
    # 示例 2:
    # 输入: [1,2,3,4,5]
    # 输出: 4
    # 解释: 在第 1 天（股票价格 = 1）的时候买入，在第 5 天 （股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。注意你不能在第 1 天和第 2 天接连购买股票，之后再将它们卖出。因为这样属于同时参与了多笔交易，你必须在再次购买前出售掉之前的股票。
    # 示例  3:
    # 输入: [7,6,4,3,1]
    # 输出: 0
    # 解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。
# 第一天当然没有利润，至少要第二天才会有利润，所以利润的序列比股票序列少一天！
# 从图中可以发现，其实我们需要收集每天的正利润就可以，收集正利润的区间，就是股票买卖的区间，而我们只需要关注最终利润，不需要记录区间。
# 那么只收集正利润就是贪心所贪的地方！
# 贪心
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        result = 0
        for i in range(1, len(prices)):
            result += max(prices[i] - prices[i - 1], 0)
        return result
# 动态规划
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        length = len(prices)
        dp = [[0] * 2 for _ in range(length)]
        dp[0][0] = -prices[0]
        dp[0][1] = 0
        for i in range(1, length):
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] - prices[i]) #注意这里是和121. 买卖股票的最佳时机唯一不同的地方
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] + prices[i])
        return dp[-1][1]


#X5 (Medium) 55.跳跃游戏
    # 给定一个非负整数数组，你最初位于数组的第一个位置。
    # 数组中的每个元素代表你在该位置可以跳跃的最大长度。
    # 判断你是否能够到达最后一个位置。
    # 示例  1:
    # 输入: [2,3,1,1,4]
    # 输出: true
    # 解释: 我们可以先跳 1 步，从位置 0 到达 位置 1, 然后再从位置 1 跳 3 步到达最后一个位置。
    # 示例  2:
    # 输入: [3,2,1,0,4]
    # 输出: false
    # 解释: 无论怎样，你总会到达索引为 3 的位置。但该位置的最大跳跃长度是 0 ， 所以你永远不可能到达最后一个位置。
# 这道题目关键点在于：不用拘泥于每次究竟跳几步，而是看覆盖范围，覆盖范围内一定是可以跳过来的，不用管是怎么跳的。
# 贪心算法局部最优解：每次取最大跳跃步数（取最大覆盖范围），整体最优解：最后得到整体最大覆盖范围，看是否能到终点。
# i 每次移动只能在 cover 的范围内移动，每移动一个元素，cover 得到该元素数值（新的覆盖范围）的补充，让 i 继续移动下去。
# 而 cover 每次只取 max(该元素数值补充后的范围, cover 本身范围)。
# 如果 cover 大于等于了终点下标，直接 return true 就可以了。
## *** while循环
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        cover = 0
        if len(nums) == 1: return True
        i = 0
        # python不支持动态修改for循环中变量,使用while循环代替
        while i <= cover:
            cover = max(i + nums[i], cover)
            if cover >= len(nums) - 1: return True
            i += 1
        return False
## for循环
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        cover = 0
        if len(nums) == 1: return True
        for i in range(len(nums)):
            if i <= cover:
                cover = max(i + nums[i], cover)
                if cover >= len(nums) - 1: return True
        return False
    
# Jump to the End
    # You are given an integer array in which you're originally positioned at index 0. 
    # Each number in the array represents the maximum jump distance from the current index. 
    # Determine if it's possible to reach the end of the array.
def jump_to_the_end(nums: List[int]) -> bool:
    # Set the initial destination to the last index in the array.
    destination = len(nums) - 1
    # Traverse the array in reverse to see if the destination can be 
    # reached by earlier indexes.
    for i in range(len(nums) - 1, -1, -1):
        # If we can reach the destination from the current index,
        # set this index as the new destination.
        if i + nums[i] >= destination:
            destination = i
    # If the destination is index 0, we can jump to the end from index 
    # 0.
    return destination == 0


#6 (Medium) 45.跳跃游戏II
    # 给定一个非负整数数组，你最初位于数组的第一个位置。
    # 数组中的每个元素代表你在该位置可以跳跃的最大长度。
    # 你的目标是使用最少的跳跃次数到达数组的最后一个位置。
    # 示例:
    # 输入: [2,3,1,1,4]
    # 输出: 2
    # 解释: 跳到最后一个位置的最小跳跃数是 2。从下标为 0 跳到下标为 1 的位置，跳  1  步，然后跳  3  步到达数组的最后一个位置。
    # 说明: 假设你总是可以到达数组的最后一个位置。
# 真正解题的时候，要从覆盖范围出发，不管怎么跳，覆盖范围内一定是可以跳到的，以最小的步数增加覆盖范围，覆盖范围一旦覆盖了终点，得到的就是最少步数！
# 这里需要统计两个覆盖范围，当前这一步的最大覆盖和下一步最大覆盖。
# 理解本题的关键在于：以最小的步数增加最大的覆盖范围，直到覆盖范围覆盖了终点，这个范围内最少步数一定可以跳到，不用管具体是怎么跳的，不纠结于一步究竟跳一个单位还是两个单位。
# class Solution:
#     def jump(self, nums: List[int]) -> int:
#         if len(nums) == 1: return 0
#         ans = 0
#         curDistance = 0
#         nextDistance = 0
#         for i in range(len(nums)):
#             nextDistance = max(i + nums[i], nextDistance)
#             if i == curDistance: 
#                 if curDistance != len(nums) - 1:
#                     ans += 1
#                     curDistance = nextDistance
#                     if nextDistance >= len(nums) - 1: break
#         return ans
# 贪心（版本一）
# class Solution:
#     def jump(self, nums):
#         if len(nums) == 1:
#             return 0
        
#         cur_distance = 0  # 当前覆盖最远距离下标
#         ans = 0  # 记录走的最大步数
#         next_distance = 0  # 下一步覆盖最远距离下标
        
#         for i in range(len(nums)):
#             next_distance = max(nums[i] + i, next_distance)  # 更新下一步覆盖最远距离下标
#             if i == cur_distance:  # 遇到当前覆盖最远距离下标
#                 ans += 1  # 需要走下一步
#                 cur_distance = next_distance  # 更新当前覆盖最远距离下标（相当于加油了）
#                 if next_distance >= len(nums) - 1:  # 当前覆盖最远距离达到数组末尾，不用再做ans++操作，直接结束
#                     break
        
#         return ans
# # # 贪心（版本二）
# class Solution:
#     def jump(self, nums):
#         cur_distance = 0  # 当前覆盖的最远距离下标
#         ans = 0  # 记录走的最大步数
#         next_distance = 0  # 下一步覆盖的最远距离下标
        
#         for i in range(len(nums) - 1):  # 注意这里是小于len(nums) - 1，这是关键所在
#             next_distance = max(nums[i] + i, next_distance)  # 更新下一步覆盖的最远距离下标
#             if i == cur_distance:  # 遇到当前覆盖的最远距离下标
#                 cur_distance = next_distance  # 更新当前覆盖的最远距离下标
#                 ans += 1
#         return ans
# *** 贪心（版本三） 类似‘55-跳跃游戏’写法
class Solution:
    def jump(self, nums) -> int:
        if len(nums)==1:  # 如果数组只有一个元素，不需要跳跃，步数为0
            return 0
        
        i = 0  # 当前位置
        count = 0  # 步数计数器
        cover = 0  # 当前能够覆盖的最远距离
        
        while i <= cover:  # 当前位置小于等于当前能够覆盖的最远距离时循环
            for i in range(i, cover+1):  # 遍历从当前位置到当前能够覆盖的最远距离之间的所有位置
                cover = max(nums[i]+i, cover)  # 更新当前能够覆盖的最远距离
                if cover >= len(nums)-1:  # 如果当前能够覆盖的最远距离达到或超过数组的最后一个位置，直接返回步数+1
                    return count+1
            count += 1  # 每一轮遍历结束后，步数+1
# 动态规划
# class Solution:
#     def jump(self, nums: List[int]) -> int:
#         result = [10**4+1] * len(nums)  # 初始化结果数组，初始值为一个较大的数
#         result[0] = 0  # 起始位置的步数为0

#         for i in range(len(nums)):  # 遍历数组
#             for j in range(nums[i] + 1):  # 在当前位置能够跳跃的范围内遍历
#                 if i + j < len(nums):  # 确保下一跳的位置不超过数组范围
#                     result[i + j] = min(result[i + j], result[i] + 1)  # 更新到达下一跳位置的最少步数

#         return result[-1]  # 返回到达最后一个位置的最少步数


#7 (Easy) 1005.K次取反后最大化的数组和
    # 给定一个整数数组 A，我们只能用以下方法修改该数组：我们选择某个索引 i 并将 A[i] 替换为 -A[i]，然后总共重复这个过程 K 次。（我们可以多次选择同一个索引 i。）
    # 以这种方式修改数组后，返回数组可能的最大和。
    # 示例 1：
    # 输入：A = [4,2,3], K = 1
    # 输出：5
    # 解释：选择索引 (1,) ，然后 A 变为 [4,-2,3]。
    # 示例 2：
    # 输入：A = [3,-1,0,2], K = 3
    # 输出：6
    # 解释：选择索引 (1, 2, 2) ，然后 A 变为 [3,1,0,2]。
    # 示例 3：
    # 输入：A = [2,-3,-1,5,-4], K = 2
    # 输出：13
    # 解释：选择索引 (1, 4) ，然后 A 变为 [2,3,-1,5,4]。
# 题目中限定了数据范围是正负一百，所以可以使用桶排序.这样时间复杂度就可以优化为$O(n)$了。但可能代码要复杂一些了。
class Solution:
    def largestSumAfterKNegations(self, A: List[int], K: int) -> int:
        A.sort(key=lambda x: abs(x), reverse=True)  # 第一步：按照绝对值降序排序数组A

        for i in range(len(A)):  # 第二步：执行K次取反操作
            if A[i] < 0 and K > 0:
                A[i] *= -1
                K -= 1

        if K % 2 == 1:  # 第三步：如果K还有剩余次数，将绝对值最小的元素取反
            A[-1] *= -1

        result = sum(A)  # 第四步：计算数组A的元素和
        return result    


#X8 (Medium) 134. 加油站
    # 在一条环路上有 N 个加油站，其中第 i 个加油站有汽油 gas[i] 升。
    # 你有一辆油箱容量无限的的汽车，从第 i 个加油站开往第 i+1 个加油站需要消耗汽油 cost[i] 升。你从其中的一个加油站出发，开始时油箱为空。
    # 如果你可以绕环路行驶一周，则返回出发时加油站的编号，否则返回 -1。
    # 说明:
    # 如果题目有解，该答案即为唯一答案。
    # 输入数组均为非空数组，且长度相同。
    # 输入数组中的元素均为非负数。
    # 示例 1: 输入:
    # gas = [1,2,3,4,5]
    # cost = [3,4,5,1,2]
    # 输出: 3 解释:
    # 从 3 号加油站(索引为 3 处)出发，可获得 4 升汽油。此时油箱有 = 0 + 4 = 4 升汽油
    # 开往 4 号加油站，此时油箱有 4 - 1 + 5 = 8 升汽油
    # 开往 0 号加油站，此时油箱有 8 - 2 + 1 = 7 升汽油
    # 开往 1 号加油站，此时油箱有 7 - 3 + 2 = 6 升汽油
    # 开往 2 号加油站，此时油箱有 6 - 4 + 3 = 5 升汽油
    # 开往 3 号加油站，你需要消耗 5 升汽油，正好足够你返回到 3 号加油站。
    # 因此，3 可为起始索引。
    # 示例 2: 输入:
    # gas = [2,3,4]
    # cost = [3,4,3]
    # 输出: -1
    # 解释: 你不能从 0 号或 1 号加油站出发，因为没有足够的汽油可以让你行驶到下一个加油站。我们从 2 号加油站出发，可以获得 4 升汽油。 此时油箱有 = 0 + 4 = 4 升汽油。开往 0 号加油站，此时油箱有 4 - 3 + 2 = 3 升汽油。开往 1 号加油站，此时油箱有 3 - 3 + 3 = 3 升汽油。你无法返回 2 号加油站，因为返程需要消耗 4 升汽油，但是你的油箱只有 3 升汽油。因此，无论怎样，你都不可能绕环路行驶一周。
# 暴力法
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        for i in range(len(cost)):
            rest = gas[i] - cost[i]  # 记录剩余油量
            index = (i + 1) % len(cost)  # 下一个加油站的索引

            while rest > 0 and index != i:  # 模拟以i为起点行驶一圈（如果有rest==0，那么答案就不唯一了）
                rest += gas[index] - cost[index]  # 更新剩余油量
                index = (index + 1) % len(cost)  # 更新下一个加油站的索引

            if rest >= 0 and index == i:  # 如果以i为起点跑一圈，剩余油量>=0，并且回到起始位置
                return i  # 返回起始位置i

        return -1  # 所有起始位置都无法环绕一圈，返回-1
# 贪心（版本一）
# class Solution:
#     def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
#         curSum = 0  # 当前累计的剩余油量
#         minFuel = float('inf')  # 从起点出发，油箱里的油量最小值
        
#         for i in range(len(gas)):
#             rest = gas[i] - cost[i]
#             curSum += rest
#             if curSum < minFuel:
#                 minFuel = curSum
        
#         if curSum < 0:
#             return -1  # 情况1：整个行程的总消耗大于总供给，无法完成一圈
        
#         if minFuel >= 0:
#             return 0  # 情况2：从起点出发到任何一个加油站时油箱的剩余油量都不会小于0，可以从起点出发完成一圈
        
#         for i in range(len(gas) - 1, -1, -1):
#             rest = gas[i] - cost[i]
#             minFuel += rest
#             if minFuel >= 0:
#                 return i  # 情况3：找到一个位置使得从该位置出发油箱的剩余油量不会小于0，返回该位置的索引
#         return -1  # 无法完成一圈
# *** 贪心（版本二）
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        curSum = 0  # 当前累计的剩余油量
        totalSum = 0  # 总剩余油量
        start = 0  # 起始位置
        
        for i in range(len(gas)):
            curSum += gas[i] - cost[i]
            totalSum += gas[i] - cost[i]
            
            if curSum < 0:  # 当前累计剩余油量curSum小于0
                start = i + 1  # 起始位置更新为i+1
                curSum = 0  # curSum重新从0开始累计
        
        if totalSum < 0:
            return -1  # 总剩余油量totalSum小于0，说明无法环绕一圈
        return start

# Gas Stations
    # There's a circular route which contains gas stations. At each station, you can fill your car with a 
    # certain amount of gas, and moving from that station to the next one consumes some fuel.
    # Find the index of the gas station you would need to start at, in order to complete the circuit 
    # without running out of gas. Assume your car starts with an empty tank. If it's not possible to 
    # complete the circuit, return -1. If it's possible, assume only one solution exists.
    # Example:
    # Input: gas = [2, 5, 1, 3], cost = [3, 2, 1, 4]
    # Output: 1
def gas_stations(gas: List[int], cost: List[int]) -> int:
    # If the total gas is less than the total cost, completing the
    # circuit is impossible.
    if sum(gas) < sum(cost):
        return -1
    start = tank = 0
    for i in range(len(gas)):
        tank += gas[i] - cost[i]
        # If our tank has negative gas, we cannot continue through the 
        # circuit from the current start point, nor from any station 
        # before or including the current station 'i'.
        if tank < 0:
            # Set the next station as the new start point and reset the 
            # tank.
            start, tank = i + 1, 0
    return start


#X9 (Hard) 135. 分发糖果
    # 老师想给孩子们分发糖果，有 N 个孩子站成了一条直线，老师会根据每个孩子的表现，预先给他们评分。
    # 你需要按照以下要求，帮助老师给这些孩子分发糖果：
    # 每个孩子至少分配到 1 个糖果。
    # 相邻的孩子中，评分高的孩子必须获得更多的糖果。
    # 那么这样下来，老师至少需要准备多少颗糖果呢？
    # 示例 1:
    # 输入: [1,0,2]
    # 输出: 5
    # 解释: 你可以分别给这三个孩子分发 2、1、2 颗糖果。
    # 示例 2:
    # 输入: [1,2,2]
    # 输出: 4
    # 解释: 你可以分别给这三个孩子分发 1、2、1 颗糖果。第三个孩子只得到 1 颗糖果，这已满足上述两个条件。
# 这在leetcode上是一道困难的题目，其难点就在于贪心的策略，如果在考虑局部的时候想两边兼顾，就会顾此失彼。
# 那么本题我采用了两次贪心的策略：
# 一次是从左到右遍历，只比较右边孩子评分比左边大的情况。
# 一次是从右到左遍历，只比较左边孩子评分比右边大的情况。
# 这样从局部最优推出了全局最优，即：相邻的孩子中，评分高的孩子获得更多的糖果。
class Solution:
    def candy(self, ratings: List[int]) -> int:
        candyVec = [1] * len(ratings)
        
        # 从前向后遍历，处理右侧比左侧评分高的情况
        for i in range(1, len(ratings)):
            if ratings[i] > ratings[i - 1]:
                candyVec[i] = candyVec[i - 1] + 1
        
        # 从后向前遍历，处理左侧比右侧评分高的情况
        for i in range(len(ratings) - 2, -1, -1):
            if ratings[i] > ratings[i + 1]:
                candyVec[i] = max(candyVec[i], candyVec[i + 1] + 1)
        
        # 统计结果
        result = sum(candyVec)
        return result

# Candies
    # You teach a class of children sitting in a row, each of whom has a rating based on their performance. 
    # You want to distribute candies to the children while abiding by the following rules:
    # Each child must receive at least one candy.
    # If two children sit next to each other, the child with the higher rating must receive more candies.
    # Determine the minimum number of candies you need to distribute to satisfy these conditions.
    # Example 1:
    # Input: ratings = [4, 3, 2, 4, 5, 1]
    # Output: 12
def candies(ratings: List[int]) -> int:
    n = len(ratings)
    # Ensure each child starts with 1 candy.
    candies = [1] * n
    # First pass: for each child, ensure the child has more candies  
    # than their left-side neighbor if the current child's rating is 
    # higher.
    for i in range(1, n):
        if ratings[i] > ratings[i - 1]:
            candies[i] = candies[i - 1] + 1
    # Second pass: for each child, ensure the child has more candies 
    # than their right-side neighbor if the current child's rating is 
    # higher.
    for i in range(n - 2, -1, -1):
        if ratings[i] > ratings[i + 1]:
            # If the current child already has more candies than their 
            # right-side neighbor, keep the higher amount.
            candies[i] = max(candies[i], candies[i + 1] + 1)
    return sum(candies)


#10 (Easy) 860.柠檬水找零
    # 在柠檬水摊上，每一杯柠檬水的售价为 5 美元。
    # 顾客排队购买你的产品，（按账单 bills 支付的顺序）一次购买一杯。
    # 每位顾客只买一杯柠檬水，然后向你付 5 美元、10 美元或 20 美元。你必须给每个顾客正确找零，也就是说净交易是每位顾客向你支付 5 美元。
    # 注意，一开始你手头没有任何零钱。
    # 如果你能给每位顾客正确找零，返回 true ，否则返回 false 。
    # 示例 1：
    # 输入：[5,5,5,10,20]
    # 输出：true
    # 解释：
    # 前 3 位顾客那里，我们按顺序收取 3 张 5 美元的钞票。
    # 第 4 位顾客那里，我们收取一张 10 美元的钞票，并返还 5 美元。
    # 第 5 位顾客那里，我们找还一张 10 美元的钞票和一张 5 美元的钞票。
    # 由于所有客户都得到了正确的找零，所以我们输出 true。
    # 示例 2：
    # 输入：[5,5,10]
    # 输出：true
    # 示例 3：
    # 输入：[10,10]
    # 输出：false
    # 示例 4：
    # 输入：[5,5,10,10,20]
    # 输出：false
    # 解释：
    # 前 2 位顾客那里，我们按顺序收取 2 张 5 美元的钞票。
    # 对于接下来的 2 位顾客，我们收取一张 10 美元的钞票，然后返还 5 美元。
    # 对于最后一位顾客，我们无法退回 15 美元，因为我们现在只有两张 10 美元的钞票。
    # 由于不是每位顾客都得到了正确的找零，所以答案是 false。
# 有如下三种情况：
# 情况一：账单是5，直接收下。
# 情况二：账单是10，消耗一个5，增加一个10
# 情况三：账单是20，优先消耗一个10和一个5，如果不够，再消耗三个5
class Solution:
    def lemonadeChange(self, bills: List[int]) -> bool:
        five = 0
        ten = 0
        # twenty = 0
        
        for bill in bills:
            # 情况一：收到5美元
            if bill == 5:
                five += 1
            
            # 情况二：收到10美元
            if bill == 10:
                if five <= 0:
                    return False
                ten += 1
                five -= 1
            
            # 情况三：收到20美元
            if bill == 20:
                # 先尝试使用10美元和5美元找零
                if five > 0 and ten > 0:
                    five -= 1
                    ten -= 1
                    #twenty += 1
                # 如果无法使用10美元找零，则尝试使用三张5美元找零
                elif five >= 3:
                    five -= 3
                    #twenty += 1
                else:
                    return False
        return True
    

#11 (Medium) 406.根据身高重建队列
    # 假设有打乱顺序的一群人站成一个队列，数组 people 表示队列中一些人的属性（不一定按顺序）。每个 people[i] = [hi, ki] 表示第 i 个人的身高为 hi ，前面 正好 有 ki 个身高大于或等于 hi 的人。
    # 请你重新构造并返回输入数组 people 所表示的队列。返回的队列应该格式化为数组 queue ，其中 queue[j] = [hj, kj] 是队列中第 j 个人的属性（queue[0] 是排在队列前面的人）。
    # 示例 1：
    # 输入：people = [[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]
    # 输出：[[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]]
    # 解释：
    # 编号为 0 的人身高为 5 ，没有身高更高或者相同的人排在他前面。
    # 编号为 1 的人身高为 7 ，没有身高更高或者相同的人排在他前面。
    # 编号为 2 的人身高为 5 ，有 2 个身高更高或者相同的人排在他前面，即编号为 0 和 1 的人。
    # 编号为 3 的人身高为 6 ，有 1 个身高更高或者相同的人排在他前面，即编号为 1 的人。
    # 编号为 4 的人身高为 4 ，有 4 个身高更高或者相同的人排在他前面，即编号为 0、1、2、3 的人。
    # 编号为 5 的人身高为 7 ，有 1 个身高更高或者相同的人排在他前面，即编号为 1 的人。
    # 因此 [[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]] 是重新构造后的队列。
    # 示例 2：
    # 输入：people = [[6,0],[5,0],[4,0],[3,2],[2,2],[1,4]]
    # 输出：[[4,0],[5,0],[2,2],[3,2],[1,4],[6,0]]
# 本题有两个维度，h和k，看到这种题目一定要想如何确定一个维度，然后再按照另一个维度重新排列。
# 其实如果大家认真做了135. 分发糖果，就会发现和此题有点点的像。
# 在135. 分发糖果 我就强调过一次，遇到两个维度权衡的时候，一定要先确定一个维度，再确定另一个维度。
# 如果两个维度一起考虑一定会顾此失彼。
class Solution:
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
    	# 先按照h维度的身高顺序从高到低排序。确定第一个维度
        # lambda返回的是一个元组：当-x[0](维度h）相同时, 再根据x[1]（维度k）从小到大排序
        people.sort(key=lambda x: (-x[0], x[1]))
        que = []
	
	# 根据每个元素的第二个维度k, 贪心算法, 进行插入
        # people已经排序过了：同一高度时k值小的排前面。
        for p in people:
            que.insert(p[1], p)
        return que


#12 (Medium) 452.用最少数量的箭引爆气球
    # 在二维空间中有许多球形的气球。对于每个气球，提供的输入是水平方向上，气球直径的开始和结束坐标。由于它是水平的，所以纵坐标并不重要，因此只要知道开始和结束的横坐标就足够了。开始坐标总是小于结束坐标。
    # 一支弓箭可以沿着 x 轴从不同点完全垂直地射出。在坐标 x 处射出一支箭，若有一个气球的直径的开始和结束坐标为 xstart，xend， 且满足  xstart ≤ x ≤ xend，则该气球会被引爆。可以射出的弓箭的数量没有限制。 弓箭一旦被射出之后，可以无限地前进。我们想找到使得所有气球全部被引爆，所需的弓箭的最小数量。
    # 给你一个数组 points ，其中 points [i] = [xstart,xend] ，返回引爆所有气球所必须射出的最小弓箭数。
    # 示例 1：
    # 输入：points = [[10,16],[2,8],[1,6],[7,12]]
    # 输出：2
    # 解释：对于该样例，x = 6 可以射爆 [2,8],[1,6] 两个气球，以及 x = 11 射爆另外两个气球
    # 示例 2：
    # 输入：points = [[1,2],[3,4],[5,6],[7,8]]
    # 输出：4
    # 示例 3：
    # 输入：points = [[1,2],[2,3],[3,4],[4,5]]
    # 输出：2
    # 示例 4：
    # 输入：points = [[1,2]]
    # 输出：1
    # 示例 5：
    # 输入：points = [[2,3],[2,3]]
    # 输出：1
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        if len(points) == 0: return 0
        points.sort(key=lambda x: x[0])
        result = 1
        for i in range(1, len(points)):
            if points[i][0] > points[i - 1][1]: # 气球i和气球i-1不挨着, 注意这里不是>=
                result += 1     
            else:
                points[i][1] = min(points[i - 1][1], points[i][1]) # 更新重叠气球最小右边界
        return result
# class Solution: # 不改变原数组
#     def findMinArrowShots(self, points: List[List[int]]) -> int:
#         points.sort(key = lambda x: x[0])
#         sl,sr = points[0][0],points[0][1]
#         count = 1
#         for i in points:
#             if i[0]>sr:
#                 count+=1
#                 sl,sr = i[0],i[1]
#             else:
#                 sl = max(sl,i[0])
#                 sr = min(sr,i[1])
#         return count


#13 (Medium) 435.无重叠区间
    # 给定一个区间的集合，找到需要移除区间的最小数量，使剩余区间互不重叠。
    # 注意: 可以认为区间的终点总是大于它的起点。 区间 [1,2] 和 [2,3] 的边界相互“接触”，但没有相互重叠。
    # 示例 1:
    # 输入: [ [1,2], [2,3], [3,4], [1,3] ]
    # 输出: 1
    # 解释: 移除 [1,3] 后，剩下的区间没有重叠。
    # 示例 2:
    # 输入: [ [1,2], [1,2], [1,2] ]
    # 输出: 2
    # 解释: 你需要移除两个 [1,2] 来使剩下的区间没有重叠。
    # 示例 3:
    # 输入: [ [1,2], [2,3] ]
    # 输出: 0
    # 解释: 你不需要移除任何区间，因为它们已经是无重叠的了。
# class Solution:
#     def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
#         if len(intervals) == 0: return 0
#         intervals.sort(key=lambda x: x[1])
#         count = 1 # 记录非交叉区间的个数
#         end = intervals[0][1] # 记录区间分割点
#         for i in range(1, len(intervals)):
#             if end <= intervals[i][0]:
#                 count += 1
#                 end = intervals[i][1]
#         return len(intervals) - count
# # 贪心 基于左边界
# class Solution:
#     def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
#         if not intervals:
#             return 0
        
#         intervals.sort(key=lambda x: x[0])  # 按照左边界升序排序
#         count = 0  # 记录重叠区间数量
        
#         for i in range(1, len(intervals)):
#             if intervals[i][0] < intervals[i - 1][1]:  # 存在重叠区间
#                 intervals[i][1] = min(intervals[i - 1][1], intervals[i][1])  # 更新重叠区间的右边界
#                 count += 1
        
#         return count    
# *** 贪心 基于左边界 把452.用最少数量的箭引爆气球代码稍做修改
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        if not intervals:
            return 0
        
        intervals.sort(key=lambda x: x[0])  # 按照左边界升序排序
        
        result = 1  # 不重叠区间数量，初始化为1，因为至少有一个不重叠的区间
        
        for i in range(1, len(intervals)):
            if intervals[i][0] >= intervals[i - 1][1]:  # 没有重叠
                result += 1
            else:  # 重叠情况
                intervals[i][1] = min(intervals[i - 1][1], intervals[i][1])  # 更新重叠区间的右边界
        
        return len(intervals) - result


#14 ??? (Medium) 763.划分字母区间
    # 字符串 S 由小写字母组成。我们要把这个字符串划分为尽可能多的片段，同一字母最多出现在一个片段中。返回一个表示每个字符串片段的长度的列表。
    # 示例：
    # 输入：S = "ababcbacadefegdehijhklij"
    # 输出：[9,7,8] 解释： 划分结果为 "ababcbaca", "defegde", "hijhklij"。 每个字母最多出现在一个片段中。 像 "ababcbacadefegde", "hijhklij" 的划分是错误的，因为划分的片段数较少。
    # 提示：
    # S的长度在[1, 500]之间。
    # S只包含小写字母 'a' 到 'z' 。
# 这里提供一种与452.用最少数量的箭引爆气球 、435.无重叠区间 相同的思路。
# 统计字符串中所有字符的起始和结束位置，记录这些区间(实际上也就是435.无重叠区间 题目里的输入)，将区间按左边界从小到大排序，找到边界将区间划分成组，互不重叠。找到的边界就是答案。
# class Solution:
#     def partitionLabels(self, s: str) -> List[int]:
#         hash = [0] * 26
#         for i in range(len(s)):
#             hash[ord(s[i]) - ord('a')] = i
#         result = []
#         left = 0
#         right = 0
#         for i in range(len(s)):
#             right = max(right, hash[ord(s[i]) - ord('a')])
#             if i == right:
#                 result.append(right - left + 1)
#                 left = i + 1
#         return result
# 贪心（版本一）
class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        last_occurrence = {}  # 存储每个字符最后出现的位置
        for i, ch in enumerate(s):
            last_occurrence[ch] = i

        result = []
        start = 0
        end = 0
        for i, ch in enumerate(s):
            end = max(end, last_occurrence[ch])  # 找到当前字符出现的最远位置
            if i == end:  # 如果当前位置是最远位置，表示可以分割出一个区间
                result.append(end - start + 1)
                start = i + 1
        return result
# *** 贪心（版本二）
# 与452.用最少数量的箭引爆气球 、435.无重叠区间 相同的思路。
class Solution:
    def partitionLabels(self, s):
        res = []
        hash = self.countLabels(s)
        hash.sort(key=lambda x: x[0])  # 按左边界从小到大排序
        leftBoard = 0
        rightBoard = hash[0][1]  # 记录最大右边界
        for i in range(1, len(hash)):
            if hash[i][0] > rightBoard:  # 出现分割点
                res.append(rightBoard - leftBoard + 1)
                leftBoard = hash[i][0]
            rightBoard = max(rightBoard, hash[i][1])
        res.append(rightBoard - leftBoard + 1)  # 最右端
        return res
    def countLabels(self, s):
        # 初始化一个长度为26的区间列表，初始值为负无穷
        hash = [[float('-inf'), float('-inf')] for _ in range(26)]
        hash_filter = []
        for i in range(len(s)):
            if hash[ord(s[i]) - ord('a')][0] == float('-inf'):
                hash[ord(s[i]) - ord('a')][0] = i
            hash[ord(s[i]) - ord('a')][1] = i
        for i in range(len(hash)):
            if hash[i][0] != float('-inf'):
                hash_filter.append(hash[i])
        return hash_filter


#15 (Medium) 56.合并区间
    # 给出一个区间的集合，请合并所有重叠的区间。
    # 示例 1:
    # 输入: intervals = [[1,3],[2,6],[8,10],[15,18]]
    # 输出: [[1,6],[8,10],[15,18]]
    # 解释: 区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
    # 示例 2:
    # 输入: intervals = [[1,4],[4,5]]
    # 输出: [[1,5]]
    # 解释: 区间 [1,4] 和 [4,5] 可被视为重叠区间。
    # 注意：输入类型已于2019年4月15日更改。 请重置默认代码定义以获取新方法签名。
# 452. 用最少数量的箭引爆气球 和 435. 无重叠区间 都是一个套路。
# 这几道题都是判断区间重叠，区别就是判断区间重叠后的逻辑，本题是判断区间重贴后要进行区间合并。
class Solution:
    def merge(self, intervals):
        result = []
        if len(intervals) == 0:
            return result  # 区间集合为空直接返回

        intervals.sort(key=lambda x: x[0])  # 按照区间的左边界进行排序
        result.append(intervals[0])  # 第一个区间可以直接放入结果集中

        for i in range(1, len(intervals)):
            if result[-1][1] >= intervals[i][0]:  # 发现重叠区间
                # 合并区间，只需要更新结果集最后一个区间的右边界，因为根据排序，左边界已经是最小的
                # 通过合并，让区间越来越大
                result[-1][1] = max(result[-1][1], intervals[i][1])
            else:
                result.append(intervals[i])  # 区间不重叠
        return result


#16 (Medium) 738.单调递增的数字
    # 给定一个非负整数 N，找出小于或等于 N 的最大的整数，同时这个整数需要满足其各个位数上的数字是单调递增。
    # （当且仅当每个相邻位数上的数字 x 和 y 满足 x <= y 时，我们称这个整数是单调递增的。）
    # 示例 1:
    # 输入: N = 10
    # 输出: 9
    # 示例 2:
    # 输入: N = 1234
    # 输出: 1234
    # 示例 3:
    # 输入: N = 332
    # 输出: 299
# class Solution:
#     def monotoneIncreasingDigits(self, n: int) -> int:
#         a = list(str(n))
#         for i in range(len(a)-1,0,-1):
#             if int(a[i]) < int(a[i-1]):
#                 a[i-1] = str(int(a[i-1]) - 1)
#                 a[i:] = '9' * (len(a) - i)  #python不需要设置flag值, 直接按长度给9就好了
#         return int("".join(a)) 
# 暴力
# class Solution:
#     def checkNum(self, num):
#         max_digit = 10
#         while num:
#             digit = num % 10
#             if max_digit >= digit:
#                 max_digit = digit
#             else:
#                 return False
#             num //= 10
#         return True

#     def monotoneIncreasingDigits(self, N):
#         for i in range(N, 0, -1):
#             if self.checkNum(i):
#                 return i
#         return 0
# 贪心（版本一）
# class Solution:
#     def monotoneIncreasingDigits(self, N: int) -> int:
#         # 将整数转换为字符串
#         strNum = str(N)
#         # flag用来标记赋值9从哪里开始
#         # 设置为字符串长度，为了防止第二个for循环在flag没有被赋值的情况下执行
#         flag = len(strNum)
        
#         # 从右往左遍历字符串
#         for i in range(len(strNum) - 1, 0, -1):
#             # 如果当前字符比前一个字符小，说明需要修改前一个字符
#             if strNum[i - 1] > strNum[i]:
#                 flag = i  # 更新flag的值，记录需要修改的位置
#                 # 将前一个字符减1，以保证递增性质
#                 strNum = strNum[:i - 1] + str(int(strNum[i - 1]) - 1) + strNum[i:]
        
#         # 将flag位置及之后的字符都修改为9，以保证最大的递增数字
#         for i in range(flag, len(strNum)):
#             strNum = strNum[:i] + '9' + strNum[i + 1:]
        
#         # 将最终的字符串转换回整数并返回
#         return int(strNum)
# 贪心（版本二）
# class Solution:
#     def monotoneIncreasingDigits(self, N: int) -> int:
#         # 将整数转换为字符串
#         strNum = list(str(N))

#         # 从右往左遍历字符串
#         for i in range(len(strNum) - 1, 0, -1):
#             # 如果当前字符比前一个字符小，说明需要修改前一个字符
#             if strNum[i - 1] > strNum[i]:
#                 strNum[i - 1] = str(int(strNum[i - 1]) - 1)  # 将前一个字符减1
#                 # 将修改位置后面的字符都设置为9，因为修改前一个字符可能破坏了递增性质
#                 for j in range(i, len(strNum)):
#                     strNum[j] = '9'

#         # 将列表转换为字符串，并将字符串转换为整数并返回
#         return int(''.join(strNum))
# 贪心（版本三）
class Solution:
    def monotoneIncreasingDigits(self, N: int) -> int:
        # 将整数转换为字符串
        strNum = list(str(N))
        # 从右往左遍历字符串
        for i in range(len(strNum) - 1, 0, -1):
            # 如果当前字符比前一个字符小，说明需要修改前一个字符
            if strNum[i - 1] > strNum[i]:
                strNum[i - 1] = str(int(strNum[i - 1]) - 1)  # 将前一个字符减1
                # 将修改位置后面的字符都设置为9，因为修改前一个字符可能破坏了递增性质
                strNum[i:] = '9' * (len(strNum) - i)
        # 将列表转换为字符串，并将字符串转换为整数并返回
        return int(''.join(strNum))
# 贪心（版本四）精简
# class Solution:
#     def monotoneIncreasingDigits(self, N: int) -> int:
#         strNum = str(N)        
#         for i in range(len(strNum) - 1, 0, -1):
#             # 如果当前字符比前一个字符小，说明需要修改前一个字符
#             if strNum[i - 1] > strNum[i]:
#                 # 将前一个字符减1，以保证递增性质
#                 # 使用字符串切片操作将修改后的前面部分与后面部分进行拼接
#                 strNum = strNum[:i - 1] + str(int(strNum[i - 1]) - 1) + '9' * (len(strNum) - i)       
#         return int(strNum)


#17 ??? (Hard) 968.监控二叉树
    # 给定一个二叉树，我们在树的节点上安装摄像头。
    # 节点上的每个摄影头都可以监视其父对象、自身及其直接子对象。
    # 计算监控树的所有节点所需的最小摄像头数量。
# 来看看这个状态应该如何转移，先来看看每个节点可能有几种状态：
# 有如下三种：
# 该节点无覆盖
# 本节点有摄像头
# 本节点有覆盖
# 我们分别有三个数字来表示：
# 0：该节点无覆盖
# 1：本节点有摄像头
# 2：本节点有覆盖
# 大家应该找不出第四个节点的状态了。
class Solution:
    def minCameraCover(self, root: TreeNode) -> int:
        # Greedy Algo:
        # 从下往上安装摄像头：跳过leaves这样安装数量最少, 局部最优 -> 全局最优
        # 先给leaves的父节点安装, 然后每隔两层节点安装一个摄像头, 直到Head
        # 0: 该节点未覆盖
        # 1: 该节点有摄像头
        # 2: 该节点有覆盖
        
        result = 0
        # 从下往上遍历：后序（左右中）
        def traversal(curr: TreeNode) -> int:
            nonlocal result
            
            if not curr: return 2
            left = traversal(curr.left)
            right = traversal(curr.right)

            # Case 1:
            # 左右节点都有覆盖
            if left == 2 and right == 2: 
                return 0

            # Case 2:
                # left == 0 && right == 0 左右节点无覆盖
                # left == 1 && right == 0 左节点有摄像头, 右节点无覆盖
                # left == 0 && right == 1 左节点有无覆盖, 右节点摄像头
                # left == 0 && right == 2 左节点无覆盖, 右节点覆盖
                # left == 2 && right == 0 左节点覆盖, 右节点无覆盖
            elif left == 0 or right == 0: 
                result += 1
                return 1

            # Case 3:
                # left == 1 && right == 2 左节点有摄像头, 右节点有覆盖
                # left == 2 && right == 1 左节点有覆盖, 右节点有摄像头
                # left == 1 && right == 1 左右节点都有摄像头
            elif left == 1 or right == 1:
                return 2
            
            # 其他情况前段代码均已覆盖

        if traversal(root) == 0:
            result += 1
            
        return result
# 贪心（版本一）
class Solution:
         # Greedy Algo:
        # 从下往上安装摄像头：跳过leaves这样安装数量最少，局部最优 -> 全局最优
        # 先给leaves的父节点安装，然后每隔两层节点安装一个摄像头，直到Head
        # 0: 该节点未覆盖
        # 1: 该节点有摄像头
        # 2: 该节点有覆盖
    def minCameraCover(self, root: TreeNode) -> int:
        # 定义递归函数
        result = [0]  # 用于记录摄像头的安装数量
        if self.traversal(root, result) == 0:
            result[0] += 1

        return result[0]
    def traversal(self, cur: TreeNode, result: List[int]) -> int:
        if not cur:
            return 2

        left = self.traversal(cur.left, result)
        right = self.traversal(cur.right, result)

        # 情况1: 左右节点都有覆盖
        if left == 2 and right == 2:
            return 0

        # 情况2:
        # left == 0 && right == 0 左右节点无覆盖
        # left == 1 && right == 0 左节点有摄像头，右节点无覆盖
        # left == 0 && right == 1 左节点无覆盖，右节点有摄像头
        # left == 0 && right == 2 左节点无覆盖，右节点覆盖
        # left == 2 && right == 0 左节点覆盖，右节点无覆盖
        if left == 0 or right == 0:
            result[0] += 1
            return 1

        # 情况3:
        # left == 1 && right == 2 左节点有摄像头，右节点有覆盖
        # left == 2 && right == 1 左节点有覆盖，右节点有摄像头
        # left == 1 && right == 1 左右节点都有摄像头
        if left == 1 or right == 1:
            return 2
# 贪心（版本二）利用elif精简代码
class Solution:
         # Greedy Algo:
        # 从下往上安装摄像头：跳过leaves这样安装数量最少，局部最优 -> 全局最优
        # 先给leaves的父节点安装，然后每隔两层节点安装一个摄像头，直到Head
        # 0: 该节点未覆盖
        # 1: 该节点有摄像头
        # 2: 该节点有覆盖
    def minCameraCover(self, root: TreeNode) -> int:
        # 定义递归函数
        result = [0]  # 用于记录摄像头的安装数量
        if self.traversal(root, result) == 0:
            result[0] += 1

        return result[0]

        
    def traversal(self, cur: TreeNode, result: List[int]) -> int:
        if not cur:
            return 2

        left = self.traversal(cur.left, result)
        right = self.traversal(cur.right, result)

        # 情况1: 左右节点都有覆盖
        if left == 2 and right == 2:
            return 0

        # 情况2:
        # left == 0 && right == 0 左右节点无覆盖
        # left == 1 && right == 0 左节点有摄像头，右节点无覆盖
        # left == 0 && right == 1 左节点无覆盖，右节点有摄像头
        # left == 0 && right == 2 左节点无覆盖，右节点覆盖
        # left == 2 && right == 0 左节点覆盖，右节点无覆盖
        elif left == 0 or right == 0:
            result[0] += 1
            return 1

        # 情况3:
        # left == 1 && right == 2 左节点有摄像头，右节点有覆盖
        # left == 2 && right == 1 左节点有覆盖，右节点有摄像头
        # left == 1 && right == 1 左右节点都有摄像头
        else:
            return 2