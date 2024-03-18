"""
===================================================================================================
动态规划方法总结:

1. 确定dp数组 (dp table) 以及下标的含义
2. 确定递推公式
3. dp数组如何初始化
4. 确定遍历顺序
5. 举例推导dp数组

01背包:
二维数组, 外层内层遍历可以交换, 都是从小到大遍历
一维数组, 外层物品从小到大, 内层容量从大到小, 遍历不可交换
一维dp数组的背包在遍历顺序上和二维dp数组实现的01背包其实是有很大差异的, 需要注意!

完全背包:
纯完全背包（求是否能装满背包）
先遍历物品, 再遍历背包 与 先遍历背包, 再遍历物品都是可以的, 都要从小到大去遍历

非纯完全背包（求装满背包有几种方法）
一维数组 (组合问题), 外层物品, 内层容量, 都是从小到大, 遍历不可交换
一维数组 (排列问题), 外层容量, 内层物品, 都是从小到大, 遍历不可交换
(DP方法求的是排列总和, 而且仅仅是求排列总和的个数, 并不是把所有的排列都列出来。
如果要把排列都列出来的话, 只能使用回溯算法爆搜。)
如果求最小数, 那么两层循环的先后顺序就无所谓了

递推公式场景:
问能否能装满背包（或者最多装多少）: dp[j] = max(dp[j], dp[j - nums[i]] + nums[i])
问背包装满最大价值: dp[j] = max(dp[j], dp[j - weight[i]] + value[i])
问装满背包所有物品的最小个数: dp[j] = min(dp[j - coins[i]] + 1, dp[j])
问装满背包有几种方法: dp[j] += dp[j - nums[i]]
===================================================================================================
"""

"""
01背包: 每件物品只能用一次
完全背包: 商品可以重复多次放入
多重背包
分组背包

对于背包问题其实状态都是可以压缩的。
在使用二维数组的时候, 递推公式: 
dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight[i]] + value[i]);
其实可以发现如果把dp[i - 1]那一层拷贝到dp[i]上, 表达式完全可以是: 
dp[i][j] = max(dp[i][j], dp[i][j - weight[i]] + value[i]);
与其把dp[i - 1]这一层拷贝到dp[i]上, 不如只用一个一维数组了, 只用dp[j]（一维数组, 也可以理解是一个滚动数组）。
这就是滚动数组的由来, 需要满足的条件是上一层可以重复利用, 直接拷贝到当前层。
读到这里估计大家都忘了 dp[i][j]里的i和j表达的是什么了, i是物品, j是背包容量。
dp[i][j] 表示从下标为[0-i]的物品里任意取, 放进容量为j的背包, 价值总和最大是多少

dp[j]可以通过dp[j - weight[i]]推导出来, dp[j - weight[i]]表示容量为j - weight[i]的背包所背的最大价值。
dp[j - weight[i]] + value[i] 表示 容量为 j - 物品i重量 的背包 加上 物品i的价值。
也就是容量为j的背包, 放入物品i了之后的价值即:dp[j]）、
此时dp[j]有两个选择, 一个是取自己dp[j] 相当于 二维dp数组中的dp[i-1][j], 即不放物品i
一个是取dp[j - weight[i]] + value[i], 即放物品i, 指定是取最大的, 毕竟是求最大价值, 
dp[j] = max(dp[j], dp[j - weight[i]] + value[i]);


01背包二维dp数组在遍历顺序上, 外层遍历物品 , 内层遍历背包容量 和 外层遍历背包容量 , 内层遍历物品 都是可以的!
如果使用一维dp数组, 物品遍历的for循环放在外层, 遍历背包的for循环放在内层, 且内层for循环倒序遍历!
二维dp遍历的时候, 背包容量是从小到大, 而一维dp遍历的时候, 背包是从大到小。

1. 倒序遍历是为了保证物品i只被放入一次!
2. 为什么二维dp数组历的时候不用倒序呢?
因为对于二维dp, dp[i][j]都是通过上一层即dp[i - 1][j]计算而来, 本层的dp[i][j]并不会被覆盖!
3. 对于一维dp, 倒序遍历的原因是, 本质上还是一个对二维数组的遍历, 并且右下角的值依赖上一层左上角的值, 
因此需要保证左边的值仍然是上一层的, 从右向左覆盖。
4. 两个嵌套for循环的顺序, 代码中是先遍历物品嵌套遍历背包容量, 那可不可以先遍历背包容量嵌套遍历物品呢？不可以!
因为一维dp的写法, 背包容量一定是要倒序遍历, 如果遍历背包容量放在上一层, 那么每个dp[j]就只会放入一个物品, 即:背包里只放入了一个物品。
5. 一维dp数组在推导的时候一定是取价值最大的数, 首先dp[0]一定是0
如果题目给的价值都是正整数那么非0下标都初始化为0就可以了, 如果题目给的价值有负数, 那么非0下标就要初始化为负无穷。
因为重量都不会是负数, 所以dp[j]都初始化为0就可以了, 
这样在递归公式dp[j] = max(dp[j], dp[j - weights[i]] + values[i])中dp[j]才不会初始值所覆盖。
因为如果按照二维dp数组那样来初始化第一row, 那么背包容量后序遍历的时候, 同一个物品i会放两次。
"""

"""
遍历顺序
#01背包
在动态规划:关于01背包问题,你该了解这些！ (opens new window)中我们讲解二维dp数组01背包先遍历物品还是先遍历背包都是可以的,且第二层for循环是从小到大遍历。
和动态规划:关于01背包问题,你该了解这些！（滚动数组） (opens new window)中,我们讲解一维dp数组01背包只能先遍历物品再遍历背包容量,且第二层for循环是从大到小遍历。
一维dp数组的背包在遍历顺序上和二维dp数组实现的01背包其实是有很大差异的,大家需要注意！

完全背包
说完01背包,再看看完全背包。

在动态规划:关于完全背包,你该了解这些！ (opens new window)中,讲解了纯完全背包的一维dp数组实现,先遍历物品还是先遍历背包都是可以的,且第二层for循环是从小到大遍历。
但是仅仅是纯完全背包的遍历顺序是这样的,题目稍有变化,两个for循环的先后顺序就不一样了。

如果求组合数就是外层for循环遍历物品,内层for遍历背包。
如果求排列数就是外层for遍历背包,内层for循环遍历物品。

相关题目如下:
求组合数:动态规划:518.零钱兑换II(opens new window)
求排列数:动态规划:377. 组合总和 Ⅳ (opens new window)、动态规划:70. 爬楼梯进阶版（完全背包）(opens new window)
如果求最小数,那么两层for循环的先后顺序就无所谓了,相关题目如下:
求最小数:动态规划:322. 零钱兑换 (opens new window)、动态规划:279.完全平方数(opens new window)
对于背包问题,其实递推公式算是容易的,难是难在遍历顺序上,如果把遍历顺序搞透,才算是真正理解了。
"""

# 背包递推公式
# 问能否能装满背包（或者最多装多少）:dp[j] = max(dp[j], dp[j - nums[i]] + nums[i]); ,对应题目如下:

# 动态规划:416.分割等和子集(opens new window)
# 动态规划:1049.最后一块石头的重量 II(opens new window)
# 问装满背包有几种方法:dp[j] += dp[j - nums[i]] ,对应题目如下:

# 动态规划:494.目标和(opens new window)
# 动态规划:518. 零钱兑换 II(opens new window)
# 动态规划:377.组合总和Ⅳ(opens new window)
# 动态规划:70. 爬楼梯进阶版（完全背包）(opens new window)
# 问背包装满最大价值:dp[j] = max(dp[j], dp[j - weight[i]] + value[i]); ,对应题目如下:

# 动态规划:474.一和零(opens new window)
# 问装满背包所有物品的最小个数:dp[j] = min(dp[j - coins[i]] + 1, dp[j]); ,对应题目如下:

# 动态规划:322.零钱兑换(opens new window)
# 动态规划:279.完全平方数


# 二维dp
def test_2_wei_bag_problem1(bag_size, weight, value) -> int: 
	rows, cols = len(weight), bag_size + 1
	dp = [[0 for _ in range(cols)] for _ in range(rows)]
    
	# 初始化dp数组. 
	for i in range(rows): 
		dp[i][0] = 0
	first_item_weight, first_item_value = weight[0], value[0]
	for j in range(1, cols): 	
		if first_item_weight <= j: 
			dp[0][j] = first_item_value

	# 更新dp数组: 先遍历物品, 再遍历背包. 
	for i in range(1, len(weight)): 
		cur_weight, cur_val = weight[i], value[i]
		for j in range(1, cols): 
			if cur_weight > j: # 说明背包装不下当前物品. 
				dp[i][j] = dp[i - 1][j] # 所以不装当前物品. 
			else: 
				# 定义dp数组: dp[i][j] 前i个物品里, 放进容量为j的背包, 价值总和最大是多少。
				dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - cur_weight]+ cur_val)

	print(dp)

if __name__ == "__main__": 
	bag_size = 4
	weight = [1, 3, 4]
	value = [15, 20, 30]
	test_2_wei_bag_problem1(bag_size, weight, value)

# 一维dp
def test_1_wei_bag_problem():
    weight = [1, 3, 4]
    value = [15, 20, 30]
    bag_weight = 4
    # 初始化: 全为0
    dp = [0] * (bag_weight + 1)

    # 先遍历物品, 再遍历背包容量
    for i in range(len(weight)):
        for j in range(bag_weight, weight[i] - 1, -1):
            # 递归公式
            dp[j] = max(dp[j], dp[j - weight[i]] + value[i])

    print(dp)

test_1_wei_bag_problem()


"""
01背包中二维dp数组的两个for遍历的先后循序是可以颠倒了, 
一维dp数组的两个for循环先后循序一定是先遍历物品, 再遍历背包容量。
在完全背包中, 对于一维dp数组来说, 其实两个for循环嵌套顺序是无所谓的!
"""
# 先遍历物品, 再遍历背包
def test_complete_pack1():
    weight = [1, 3, 4]
    value = [15, 20, 30]
    bag_weight = 4

    dp = [0]*(bag_weight + 1)

    for i in range(len(weight)):
        for j in range(weight[i], bag_weight + 1):
            dp[j] = max(dp[j], dp[j - weight[i]] + value[i])
    
    print(dp[bag_weight])

# 先遍历背包, 再遍历物品
def test_complete_pack2():
    weight = [1, 3, 4]
    value = [15, 20, 30]
    bag_weight = 4

    dp = [0]*(bag_weight + 1)

    for j in range(bag_weight + 1):
        for i in range(len(weight)):
            if j >= weight[i]: dp[j] = max(dp[j], dp[j - weight[i]] + value[i])
    
    print(dp[bag_weight])

if __name__ == '__main__':
    test_complete_pack1()
    test_complete_pack2()


"""
完全背包和01背包问题唯一不同的地方就是, 每种物品有无限件。
我们知道01背包内嵌的循环是从大到小遍历, 为了保证每个物品仅被添加一次。
而完全背包的物品是可以添加多次的, 所以要从小到大去遍历

内外循环遍历顺序:
01背包中: 二维dp数组的两个for遍历的先后循序是可以颠倒了, 一维dp数组的两个for循环先后循序一定是先遍历物品, 再遍历背包容量。
完全背包中: 对于一维dp数组来说, 其实两个for循环嵌套顺序是无所谓的!

如果求组合数就是外层for循环遍历物品, 内层for遍历背包。
如果求排列数就是外层for遍历背包, 内层for循环遍历物品。

又可以出一道面试题了, 就是纯完全背包, 要求先用二维dp数组实现, 然后再用一维dp数组实现, 
最后在问, 两个for循环的先后是否可以颠倒？为什么？ 这个简单的完全背包问题, 估计就可以难住不少候选人了。
"""
# 01背包的核心代码
for (int i = 0; i < weight.size(); i++) { // 遍历物品
    for(int j = bagWeight; j >= weight[i]; j--) { // 遍历背包容量
        dp[j] = max(dp[j], dp[j - weight[i]] + value[i]);
    }
}

# 完全背包核心代码
# // 先遍历物品, 再遍历背包
for(int i = 0; i < weight.size(); i++) { // 遍历物品
    for(int j = weight[i]; j <= bagWeight ; j++) { // 遍历背包容量
        dp[j] = max(dp[j], dp[j - weight[i]] + value[i]);

    }
}

# // 先遍历背包, 再遍历物品
for(int j = 0; j <= bagWeight; j++) { // 遍历背包容量
    for(int i = 0; i < weight.size(); i++) { // 遍历物品
        if (j - weight[i] >= 0) dp[j] = max(dp[j], dp[j - weight[i]] + value[i]);
    }
    cout << endl;
}




"""
5.最长回文子串 Longest Palindromic Substring
132. 分割回文串 II
300.最长上升子序列
647. 回文子串
673.最长递增子序列的个数

115. Distinct Subsequences
392. Is Subsequence
516. Longest Palindromic Subsequence
673. Number of Longest Increasing Subsequence
2407. Longest Increasing Subsequence II

53. Maximum Subarray
209. Minimum Size Subarray Sum
560. Subarray Sum Equals K

1162. As Far from Land as Possible

72. Edit Distance
"""
#1 509. 斐波那契数
# 斐波那契数, 通常用 F(n) 表示, 形成的序列称为 斐波那契数列 。该数列由 0 和 1 开始, 后面的每一项数字都是前面两项数字的和。也就是: F(0) = 0, F(1) = 1 F(n) = F(n - 1) + F(n - 2), 其中 n > 1 给你n , 请计算 F(n) 。
# 递归实现
class Solution:
    def fib(self, n: int) -> int:
        if n < 2:
            return n
        return self.fib(n - 1) + self.fib(n - 2)
# 动态规划（版本一）
class Solution:
    def fib(self, n: int) -> int:
       
        # 排除 Corner Case
        if n == 0:
            return 0
        
        # 创建 dp table 
        dp = [0] * (n + 1)

        # 初始化 dp 数组
        dp[0] = 0
        dp[1] = 1

        # 遍历顺序: 由前向后。因为后面要用到前面的状态
        for i in range(2, n + 1):

            # 确定递归公式/状态转移公式
            dp[i] = dp[i - 1] + dp[i - 2]
        
        # 返回答案
        return dp[n]
# *** 动态规划（版本二）
class Solution:
    def fib(self, n: int) -> int:
        if n <= 1:
            return n
        
        dp = [0, 1]
        
        for i in range(2, n + 1):
            total = dp[0] + dp[1]
            dp[0] = dp[1]
            dp[1] = total
        
        return dp[1]
# 动态规划（版本三）
# class Solution:
#     def fib(self, n: int) -> int:
#         if n <= 1:
#             return n
        
#         prev1, prev2 = 0, 1
        
#         for _ in range(2, n + 1):
#             curr = prev1 + prev2
#             prev1, prev2 = prev2, curr
#         return prev2


#2 70. 爬楼梯
# 假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
# 每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
# 注意:给定 n 是一个正整数。
# 动态规划（版本一）
# 空间复杂度为O(n)版本
class Solution:
    def climbStairs(self, n: int) -> int:
        if n <= 1:
            return n
        
        dp = [0] * (n + 1)
        dp[1] = 1
        dp[2] = 2
        
        for i in range(3, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        
        return dp[n]
# 动态规划（版本二）
# 空间复杂度为O(3)版本
class Solution:
    def climbStairs(self, n: int) -> int:
        if n <= 1:
            return n
        
        dp = [0] * 3
        dp[1] = 1
        dp[2] = 2
        
        for i in range(3, n + 1):
            total = dp[1] + dp[2]
            dp[1] = dp[2]
            dp[2] = total
        return dp[2]
# 这道题目还可以继续深化,就是一步一个台阶,两个台阶,三个台阶,直到 m个台阶,有多少种方法爬到n阶楼顶。
class Solution {
public:
    int climbStairs(int n) {
        vector<int> dp(n + 1, 0);
        dp[0] = 1;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) { // 把m换成2,就可以AC爬楼梯这道题
                if (i - j >= 0) dp[i] += dp[i - j];
            }
        }
        return dp[n];
    }
};


#3 746. 使用最小花费爬楼梯
# 数组的每个下标作为一个阶梯, 第 i 个阶梯对应着一个非负数的体力花费值 cost[i]（下标从 0 开始）。
# 每当你爬上一个阶梯你都要花费对应的体力值, 一旦支付了相应的体力值, 你就可以选择向上爬一个阶梯或者爬两个阶梯。
# 请你找出达到楼层顶部的最低花费。在开始时, 你可以选择从下标为 0 或 1 的元素作为初始阶梯。
# 示例 1:输入:cost = [10, 15, 20] 输出:15 解释:最低花费是从 cost[1] 开始, 然后走两步即可到阶梯顶, 一共花费 15 。
# 示例 2:输入:cost = [1, 100, 1, 1, 1, 100, 1, 1, 100, 1] 输出:6 解释:最低花费方式是从 cost[0] 开始, 逐个经过那些 1 , 跳过 cost[3] , 一共花费 6 。
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        dp = [0] * (len(cost))
        dp[0] = cost[0]
        dp[1] = cost[1]
        for i in range(2, len(cost)):
            dp[i] = min(dp[i - 1], dp[i - 2]) + cost[i]
        return min(dp[len(cost) - 1], dp[len(cost) - 2])
# 动态规划（版本一）
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        dp = [0] * (len(cost) + 1)
        dp[0] = 0  # 初始值,表示从起点开始不需要花费体力
        dp[1] = 0  # 初始值,表示经过第一步不需要花费体力
        
        for i in range(2, len(cost) + 1):
            # 在第i步,可以选择从前一步（i-1）花费体力到达当前步,或者从前两步（i-2）花费体力到达当前步
            # 选择其中花费体力较小的路径,加上当前步的花费,更新dp数组
            dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])
        
        return dp[len(cost)]  # 返回到达楼顶的最小花费
# 动态规划（版本二）
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        dp0 = 0  # 初始值,表示从起点开始不需要花费体力
        dp1 = 0  # 初始值,表示经过第一步不需要花费体力
        
        for i in range(2, len(cost) + 1):
            # 在第i步,可以选择从前一步（i-1）花费体力到达当前步,或者从前两步（i-2）花费体力到达当前步
            # 选择其中花费体力较小的路径,加上当前步的花费,得到当前步的最小花费
            dpi = min(dp1 + cost[i - 1], dp0 + cost[i - 2])
            
            dp0 = dp1  # 更新dp0为前一步的值,即上一次循环中的dp1
            dp1 = dpi  # 更新dp1为当前步的最小花费
        
        return dp1  # 返回到达楼顶的最小花费
# # 动态规划（版本三）
# class Solution:
#     def minCostClimbingStairs(self, cost: List[int]) -> int:
#         dp = [0] * len(cost)
#         dp[0] = cost[0]  # 第一步有花费
#         dp[1] = cost[1]
#         for i in range(2, len(cost)):
#             dp[i] = min(dp[i - 1], dp[i - 2]) + cost[i]
#         # 注意最后一步可以理解为不用花费,所以取倒数第一步,第二步的最少值
#         return min(dp[-1], dp[-2])
# # 动态规划（版本四）
# class Solution:
#     def minCostClimbingStairs(self, cost: List[int]) -> int:
#         n = len(cost)
#         prev_1 = cost[0]  # 前一步的最小花费
#         prev_2 = cost[1]  # 前两步的最小花费
#         for i in range(2, n):
#             current = min(prev_1, prev_2) + cost[i]  # 当前位置的最小花费
#             prev_1, prev_2 = prev_2, current  # 更新前一步和前两步的最小花费
#         return min(prev_1, prev_2)  # 最后一步可以理解为不用花费,取倒数第一步和第二步的最少值


#4 62.不同路径
# 一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。
# 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。
# 问总共有多少条不同的路径？
# 输入:m = 3, n = 7
# 输出:28
# class Solution: # 动态规划
#     def uniquePaths(self, m: int, n: int) -> int:
#         dp = [[1 for i in range(n)] for j in range(m)]
#         for i in range(1, m):
#             for j in range(1, n):
#                 dp[i][j] = dp[i][j - 1] + dp[i - 1][j]
#         return dp[m - 1][n - 1]
# # 递归
# class Solution:
#     def uniquePaths(self, m: int, n: int) -> int:
#         if m == 1 or n == 1:
#             return 1
#         return self.uniquePaths(m - 1, n) + self.uniquePaths(m, n - 1)
# 动态规划（版本一）
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        # 创建一个二维列表用于存储唯一路径数
        dp = [[0] * n for _ in range(m)]
        
        # 设置第一行和第一列的基本情况
        for i in range(m):
            dp[i][0] = 1
        for j in range(n):
            dp[0][j] = 1
        
        # 计算每个单元格的唯一路径数
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        
        # 返回右下角单元格的唯一路径数
        return dp[m - 1][n - 1]
# 动态规划（版本二）
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        # 创建一个一维列表用于存储每列的唯一路径数
        dp = [1] * n
        
        # 计算每个单元格的唯一路径数
        for j in range(1, m):
            for i in range(1, n):
                dp[i] += dp[i - 1]
        
        # 返回右下角单元格的唯一路径数
        return dp[n - 1]
# 数论
# class Solution:
#     def uniquePaths(self, m: int, n: int) -> int:
#         numerator = 1  # 分子
#         denominator = m - 1  # 分母
#         count = m - 1  # 计数器,表示剩余需要计算的乘积项个数
#         t = m + n - 2  # 初始乘积项
#         while count > 0:
#             numerator *= t  # 计算乘积项的分子部分
#             t -= 1  # 递减乘积项
#             while denominator != 0 and numerator % denominator == 0:
#                 numerator //= denominator  # 约简分子
#                 denominator -= 1  # 递减分母
#             count -= 1  # 计数器减1,继续下一项的计算
#         return numerator  # 返回最终的唯一路径数


#5 63. 不同路径 II
# 一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。
# 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。
# 现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？
# 网格中的障碍物和空位置分别用 1 和 0 来表示。
# 输入:obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
# 输出:2
"""
但就算是做过62.不同路径, 在做本题也会有感觉遇到障碍无从下手。
其实只要考虑到, 遇到障碍dp[i][j]保持0就可以了。
也有一些小细节, 例如: 初始化的部分, 很容易忽略了障碍之后应该都是0的情况。
"""
# class Solution:
#     def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
#         # 构造一个DP table
#         row = len(obstacleGrid)
#         col = len(obstacleGrid[0])
#         dp = [[0 for _ in range(col)] for _ in range(row)]

#         dp[0][0] = 1 if obstacleGrid[0][0] != 1 else 0
#         if dp[0][0] == 0: return 0  # 如果第一个格子就是障碍, return 0
#         # 第一行
#         for i in range(1, col):
#             if obstacleGrid[0][i] != 1:
#                 dp[0][i] = dp[0][i-1]

#         # 第一列
#         for i in range(1, row):
#             if obstacleGrid[i][0] != 1:
#                 dp[i][0] = dp[i-1][0]
#         print(dp)

#         for i in range(1, row):
#             for j in range(1, col):
#                 if obstacleGrid[i][j] != 1:
#                     dp[i][j] = dp[i-1][j] + dp[i][j-1]
#         return dp[-1][-1]
# 动态规划（版本一）二维数组
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid):
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        if obstacleGrid[m - 1][n - 1] == 1 or obstacleGrid[0][0] == 1:
            return 0
        dp = [[0] * n for _ in range(m)]
        for i in range(m):
            if obstacleGrid[i][0] == 0:  # 遇到障碍物时,直接退出循环,后面默认都是0
                dp[i][0] = 1
            else:
                break
        for j in range(n):
            if obstacleGrid[0][j] == 0:
                dp[0][j] = 1
            else:
                break
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] == 1:
                    continue
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[m - 1][n - 1]
# 动态规划（版本二）
# class Solution:
#     def uniquePathsWithObstacles(self, obstacleGrid):
#         m = len(obstacleGrid)  # 网格的行数
#         n = len(obstacleGrid[0])  # 网格的列数
        
#         if obstacleGrid[m - 1][n - 1] == 1 or obstacleGrid[0][0] == 1:
#             # 如果起点或终点有障碍物,直接返回0
#             return 0
        
#         dp = [[0] * n for _ in range(m)]  # 创建一个二维列表用于存储路径数
        
#         # 设置起点的路径数为1
#         dp[0][0] = 1 if obstacleGrid[0][0] == 0 else 0
        
#         # 计算第一列的路径数
#         for i in range(1, m):
#             if obstacleGrid[i][0] == 0:
#                 dp[i][0] = dp[i - 1][0]
        
#         # 计算第一行的路径数
#         for j in range(1, n):
#             if obstacleGrid[0][j] == 0:
#                 dp[0][j] = dp[0][j - 1]
        
#         # 计算其他位置的路径数
#         for i in range(1, m):
#             for j in range(1, n):
#                 if obstacleGrid[i][j] == 1:
#                     continue
#                 dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
#         return dp[m - 1][n - 1]  # 返回终点的路径数
# 动态规划（版本三）
# class Solution:
#     def uniquePathsWithObstacles(self, obstacleGrid):
#         if obstacleGrid[0][0] == 1:
#             return 0
        
#         dp = [0] * len(obstacleGrid[0])  # 创建一个一维列表用于存储路径数
        
#         # 初始化第一行的路径数
#         for j in range(len(dp)):
#             if obstacleGrid[0][j] == 1:
#                 dp[j] = 0
#             elif j == 0:
#                 dp[j] = 1
#             else:
#                 dp[j] = dp[j - 1]

#         # 计算其他行的路径数
#         for i in range(1, len(obstacleGrid)):
#             for j in range(len(dp)):
#                 if obstacleGrid[i][j] == 1:
#                     dp[j] = 0
#                 elif j != 0:
#                     dp[j] = dp[j] + dp[j - 1]
        
#         return dp[-1]  # 返回最后一个元素,即终点的路径数
# *** 动态规划（版本四）一维数组
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid):
        if obstacleGrid[0][0] == 1:
            return 0
        
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        
        dp = [0] * n  # 创建一个一维列表用于存储路径数
        
        # 初始化第一行的路径数
        for j in range(n):
            if obstacleGrid[0][j] == 1:
                break
            dp[j] = 1

        # 计算其他行的路径数
        for i in range(1, m):
            if obstacleGrid[i][0] == 1:
                dp[0] = 0
            for j in range(1, n):
                if obstacleGrid[i][j] == 1:
                    dp[j] = 0
                else:
                    dp[j] += dp[j - 1]
        
        return dp[-1]  # 返回最后一个元素,即终点的路径数
# 动态规划（版本五）
# class Solution:
#     def uniquePathsWithObstacles(self, obstacleGrid):
#         if obstacleGrid[0][0] == 1:
#             return 0
        
#         m, n = len(obstacleGrid), len(obstacleGrid[0])
        
#         dp = [0] * n  # 创建一个一维列表用于存储路径数
        
#         # 初始化第一行的路径数
#         for j in range(n):
#             if obstacleGrid[0][j] == 1:
#                 break
#             dp[j] = 1

#         # 计算其他行的路径数
#         for i in range(1, m):
#             if obstacleGrid[i][0] == 1:
#                 dp[0] = 0
#             for j in range(1, n):
#                 if obstacleGrid[i][j] == 1:
#                     dp[j] = 0
#                     continue
                
#                 dp[j] += dp[j - 1]
#         return dp[-1]  # 返回最后一个元素,即终点的路径数


#6 343. 整数拆分
# 给定一个正整数 n, 将其拆分为至少两个正整数的和, 并使这些整数的乘积最大化。 返回你可以获得的最大乘积。
# 示例 1:
# 输入: 2
# 输出: 1
# 解释: 2 = 1 + 1, 1 × 1 = 1。
# 示例 2:
# 输入: 10
# 输出: 36
# 解释: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36。
# 说明: 你可以假设 n 不小于 2 且不大于 58
# XXX dp[i] = max(dp[i], dp[i - j] * dp[j])
    # dp[i] = max(dp[i], max((i - j) * j, dp[i - j] * j));
    # 也可以这么理解,j * (i - j) 是单纯的把整数拆分为两个数相乘,而j * dp[i - j]是拆分成两个以及两个以上的个数相乘。
    # 那么在取最大值的时候,为什么还要比较dp[i]呢？因为在递推公式推导的过程中,每次计算dp[i],取最大的而已。

class Solution:
    def integerBreak(self, n: int) -> int:
        dp = [0] * (n + 1)
        dp[2] = 1
        for i in range(3, n + 1):
            # 假设对正整数 i 拆分出的第一个正整数是 j（1 <= j < i）, 则有以下两种方案:
            # 1) 将 i 拆分成 j 和 i−j 的和, 且 i−j 不再拆分成多个正整数, 此时的乘积是 j * (i-j)
            # 2) 将 i 拆分成 j 和 i−j 的和, 且 i−j 继续拆分成多个正整数, 此时的乘积是 j * dp[i-j]
            for j in range(1, i - 1):
                dp[i] = max(dp[i], max(j * (i - j), j * dp[i - j]))
        return dp[n]
# 动态规划（版本一）
# class Solution:
#          # 假设对正整数 i 拆分出的第一个正整数是 j（1 <= j < i）,则有以下两种方案:
#         # 1) 将 i 拆分成 j 和 i−j 的和,且 i−j 不再拆分成多个正整数,此时的乘积是 j * (i-j)
#         # 2) 将 i 拆分成 j 和 i−j 的和,且 i−j 继续拆分成多个正整数,此时的乘积是 j * dp[i-j]
#     def integerBreak(self, n):
#         dp = [0] * (n + 1)   # 创建一个大小为n+1的数组来存储计算结果
#         dp[2] = 1  # 初始化dp[2]为1,因为当n=2时,只有一个切割方式1+1=2,乘积为1
       
#         # 从3开始计算,直到n
#         for i in range(3, n + 1):
#             # 遍历所有可能的切割点
#                 # 因为拆分一个数n 使之乘积最大,那么一定是拆分成m个近似相同的子数相乘才是最大的。
#                 # 只不过我们不知道m究竟是多少而已,但可以明确的是m一定大于等于2,既然m大于等于2,也就是 最差也应该是拆成两个相同的 可能是最大值。
#                 # 那么 j 遍历,只需要遍历到 n/2 就可以,后面就没有必要遍历了,一定不是最大值。
#             for j in range(1, i // 2 + 1):

#                 # 计算切割点j和剩余部分(i-j)的乘积,并与之前的结果进行比较取较大值
                
#                 dp[i] = max(dp[i], (i - j) * j, dp[i - j] * j)
#         return dp[n]  # 返回最终的计算结果
# 动态规划（版本二）
# class Solution:
#     def integerBreak(self, n):
#         if n <= 3:
#             return 1 * (n - 1)  # 对于n小于等于3的情况,返回1 * (n - 1)

#         dp = [0] * (n + 1)  # 创建一个大小为n+1的数组来存储最大乘积结果
#         dp[1] = 1  # 当n等于1时,最大乘积为1
#         dp[2] = 2  # 当n等于2时,最大乘积为2
#         dp[3] = 3  # 当n等于3时,最大乘积为3

#         # 从4开始计算,直到n
#         for i in range(4, n + 1):
#             # 遍历所有可能的切割点
#             for j in range(1, i // 2 + 1):
#                 # 计算切割点j和剩余部分(i - j)的乘积,并与之前的结果进行比较取较大值
#                 dp[i] = max(dp[i], dp[i - j] * dp[j])
#         return dp[n]  # 返回整数拆分的最大乘积结果
# 贪心
# class Solution:
#     def integerBreak(self, n):
#         if n == 2:  # 当n等于2时,只有一种拆分方式:1+1=2,乘积为1
#             return 1
#         if n == 3:  # 当n等于3时,只有一种拆分方式:1+1+1=3,乘积为1
#             return 2
#         if n == 4:  # 当n等于4时,有两种拆分方式:2+2=4和1+1+1+1=4,乘积都为4
#             return 4
#         result = 1
#         while n > 4:
#             result *= 3  # 每次乘以3,因为3的乘积比其他数字更大
#             n -= 3  # 每次减去3
#         result *= n  # 将剩余的n乘以最后的结果
#         return result


#7 ??? 96.不同的二叉搜索树
# 给定一个整数 n, 求以 1 ... n 为节点组成的二叉搜索树有多少种？
# dp[i] += dp[j - 1] * dp[i - j]
class Solution:
    def numTrees(self, n: int) -> int:
        dp = [0] * (n + 1)  # 创建一个长度为n+1的数组,初始化为0
        dp[0] = 1  # 当n为0时,只有一种情况,即空树,所以dp[0] = 1
        for i in range(1, n + 1):  # 遍历从1到n的每个数字
            for j in range(1, i + 1):  # 对于每个数字i,计算以i为根节点的二叉搜索树的数量
                dp[i] += dp[j - 1] * dp[i - j]  # 利用动态规划的思想,累加左子树和右子树的组合数量
        return dp[n]  # 返回以1到n为节点的二叉搜索树的总数量


#8 01 背包
# 有n件物品和一个最多能背重量为w 的背包。第i件物品的重量是weight[i], 得到的价值是value[i] 。每件物品只能用一次, 求解将哪些物品装入背包里物品价值总和最大。
# dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight[i]] + value[i])
# 大家可以看出, 虽然两个for循环遍历的次序不同, 但是dp[i][j]所需要的数据就是左上角, 根本不影响dp[i][j]公式的推导！
# 其实背包问题里, 两个for循环的先后循序是非常有讲究的, 理解遍历顺序其实比理解推导公式难多了。
def test_2_wei_bag_problem1(bag_size, weight, value) -> int: 
	rows, cols = len(weight), bag_size + 1
	dp = [[0 for _ in range(cols)] for _ in range(rows)]
    
	# 初始化dp数组. 
	for i in range(rows): 
		dp[i][0] = 0
	first_item_weight, first_item_value = weight[0], value[0]
	for j in range(1, cols): 	
		if first_item_weight <= j: 
			dp[0][j] = first_item_value

	# 更新dp数组: 先遍历物品, 再遍历背包. 
	for i in range(1, len(weight)): 
		cur_weight, cur_val = weight[i], value[i]
		for j in range(1, cols): 
			if cur_weight > j: # 说明背包装不下当前物品. 
				dp[i][j] = dp[i - 1][j] # 所以不装当前物品. 
			else: 
				# 定义dp数组: dp[i][j] 前i个物品里, 放进容量为j的背包, 价值总和最大是多少。
				dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - cur_weight]+ cur_val)
	print(dp)

if __name__ == "__main__": 
	bag_size = 4
	weight = [1, 3, 4]
	value = [15, 20, 30]
	test_2_wei_bag_problem1(bag_size, weight, value)

# 无参数版
# def test_2_wei_bag_problem1():
#     weight = [1, 3, 4]
#     value = [15, 20, 30]
#     bagweight = 4

#     # 二维数组
#     dp = [[0] * (bagweight + 1) for _ in range(len(weight))]

#     # 初始化
#     for j in range(weight[0], bagweight + 1):
#         dp[0][j] = value[0]

#     # weight数组的大小就是物品个数
#     for i in range(1, len(weight)):  # 遍历物品
#         for j in range(bagweight + 1):  # 遍历背包容量
#             if j < weight[i]:
#                 dp[i][j] = dp[i - 1][j]
#             else:
#                 dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight[i]] + value[i])

#     print(dp[len(weight) - 1][bagweight])

# test_2_wei_bag_problem1()
# 有参数版
# def test_2_wei_bag_problem1(weight, value, bagweight):
#     # 二维数组
#     dp = [[0] * (bagweight + 1) for _ in range(len(weight))]

#     # 初始化
#     for j in range(weight[0], bagweight + 1):
#         dp[0][j] = value[0]

#     # weight数组的大小就是物品个数
#     for i in range(1, len(weight)):  # 遍历物品
#         for j in range(bagweight + 1):  # 遍历背包容量
#             if j < weight[i]:
#                 dp[i][j] = dp[i - 1][j]
#             else:
#                 dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight[i]] + value[i])

#     return dp[len(weight) - 1][bagweight]

# if __name__ == "__main__":

#     weight = [1, 3, 4]
#     value = [15, 20, 30]
#     bagweight = 4
#     result = test_2_wei_bag_problem1(weight, value, bagweight)
#     print(result)


#9 01 背包(滚动数组)
# 把i相关的部分去掉
# dp[j] = max(dp[j], dp[j - weight[i]] + value[i])
# 二维dp遍历的时候, 背包容量是从小到大, 而一维dp遍历的时候, 背包是从大到小。
# 倒序遍历是为了保证物品i只被放入一次！。但如果一旦正序遍历了, 那么物品0就会被重复加入多次！
# 无参版
def test_1_wei_bag_problem():
    weight = [1, 3, 4]
    value = [15, 20, 30]
    bag_weight = 4
    # 初始化: 全为0
    dp = [0] * (bag_weight + 1)

    # 先遍历物品, 再遍历背包容量
    for i in range(len(weight)): # 遍历物品
        for j in range(bag_weight, weight[i] - 1, -1): # 遍历背包容量
            dp[j] = max(dp[j], dp[j - weight[i]] + value[i])

    print(dp)

test_1_wei_bag_problem()
# 有参版
# def test_1_wei_bag_problem(weight, value, bagWeight):
#     # 初始化
#     dp = [0] * (bagWeight + 1)
#     for i in range(len(weight)):  # 遍历物品
#         for j in range(bagWeight, weight[i] - 1, -1):  # 遍历背包容量
#             dp[j] = max(dp[j], dp[j - weight[i]] + value[i])

#     return dp[bagWeight]


# if __name__ == "__main__":

#     weight = [1, 3, 4]
#     value = [15, 20, 30]
#     bagweight = 4

#     result = test_1_wei_bag_problem(weight, value, bagweight)
#     print(result)


#10 416. 分割等和子集
# 给定一个只包含正整数的非空数组。是否可以将这个数组分割成两个子集, 使得两个子集的元素和相等。
# 注意: 每个数组中的元素不会超过 100 数组的大小不会超过 200
# 示例 1: 输入: [1, 5, 11, 5] 输出: true 解释: 数组可以分割成 [1, 5, 5] 和 [11].
# 一维DP
# class Solution:
#     def canPartition(self, nums: List[int]) -> bool:
#         target = sum(nums)
#         if target % 2 == 1: return False
#         target //= 2
#         # // 总和不会大于20000, 背包最大只需要其中一半, 所以10001大小就可以了
#         dp = [0] * 10001
#         for i in range(len(nums)):
#             for j in range(target, nums[i] - 1, -1):
#                 dp[j] = max(dp[j], dp[j - nums[i]] + nums[i])
#         return target == dp[target]
# 卡哥版
# class Solution:
#     def canPartition(self, nums: List[int]) -> bool:
#         _sum = 0

#         # dp[i]中的i表示背包内总和
#         # 题目中说:每个数组中的元素不会超过 100,数组的大小不会超过 200
#         # 总和不会大于20000,背包最大只需要其中一半,所以10001大小就可以了
#         dp = [0] * 10001
#         for num in nums:
#             _sum += num
#         # 也可以使用内置函数一步求和
#         # _sum = sum(nums)
#         if _sum % 2 == 1:
#             return False
#         target = _sum // 2

#         # 开始 0-1背包
#         for num in nums:
#             for j in range(target, num - 1, -1):  # 每一个元素一定是不可重复放入,所以从大到小遍历
#                 dp[j] = max(dp[j], dp[j - num] + num)

#         # 集合中的元素正好可以凑成总和target
#         if dp[target] == target:
#             return True
#         return False
# *** 一维DP 卡哥版(简化版)
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        if sum(nums) % 2 != 0:
            return False
        target = sum(nums) // 2
        dp = [0] * (target + 1)
        for num in nums:
            for j in range(target, num-1, -1):
                dp[j] = max(dp[j], dp[j-num] + num)
        return dp[-1] == target
# 二维DP版
# class Solution:
#     def canPartition(self, nums: List[int]) -> bool:
        
#         total_sum = sum(nums)

#         if total_sum % 2 != 0:
#             return False

#         target_sum = total_sum // 2
#         dp = [[False] * (target_sum + 1) for _ in range(len(nums) + 1)]

#         # 初始化第一行（空子集可以得到和为0）
#         for i in range(len(nums) + 1):
#             dp[i][0] = True

#         for i in range(1, len(nums) + 1):
#             for j in range(1, target_sum + 1):
#                 if j < nums[i - 1]:
#                     # 当前数字大于目标和时,无法使用该数字
#                     dp[i][j] = dp[i - 1][j]
#                 else:
#                     # 当前数字小于等于目标和时,可以选择使用或不使用该数字
#                     dp[i][j] = dp[i - 1][j] or dp[i - 1][j - nums[i - 1]]

#         return dp[len(nums)][target_sum]
# 一维DP版
# class Solution:
#     def canPartition(self, nums: List[int]) -> bool:

#         total_sum = sum(nums)

#         if total_sum % 2 != 0:
#             return False

#         target_sum = total_sum // 2
#         dp = [False] * (target_sum + 1)
#         dp[0] = True

#         for num in nums:
#             # 从target_sum逆序迭代到num,步长为-1
#             for i in range(target_sum, num - 1, -1):
#                 dp[i] = dp[i] or dp[i - num]
#         return dp[target_sum]


#11 1049.最后一块石头的重量II
# 卡哥版
# class Solution:
#     def lastStoneWeightII(self, stones: List[int]) -> int:
#         dp = [0] * 15001
#         total_sum = sum(stones)
#         target = total_sum // 2

#         for stone in stones:  # 遍历物品
#             for j in range(target, stone - 1, -1):  # 遍历背包
#                 dp[j] = max(dp[j], dp[j - stone] + stone)

#         return total_sum - dp[target] - dp[target]
# 卡哥版（简化版）
class Solution:
    def lastStoneWeightII(self, stones):
        total_sum = sum(stones)
        target = total_sum // 2
        dp = [0] * (target + 1)
        for stone in stones: # 遍历物品
            for j in range(target, stone - 1, -1): # 遍历背包
                dp[j] = max(dp[j], dp[j - stone] + stone)
        return total_sum - 2* dp[-1]
# 二维DP版
# class Solution:
#     def lastStoneWeightII(self, stones: List[int]) -> int:
#         total_sum = sum(stones)
#         target = total_sum // 2
        
#         # 创建二维dp数组,行数为石头的数量加1,列数为target加1
#         # dp[i][j]表示前i个石头能否组成总重量为j
#         dp = [[False] * (target + 1) for _ in range(len(stones) + 1)]
        
#         # 初始化第一列,表示总重量为0时,前i个石头都能组成
#         for i in range(len(stones) + 1):
#             dp[i][0] = True
        
#         for i in range(1, len(stones) + 1):
#             for j in range(1, target + 1):
#                 # 如果当前石头重量大于当前目标重量j,则无法选择该石头
#                 if stones[i - 1] > j:
#                     dp[i][j] = dp[i - 1][j]
#                 else:
#                     # 可选择该石头或不选择该石头
#                     dp[i][j] = dp[i - 1][j] or dp[i - 1][j - stones[i - 1]]
        
#         # 找到最大的重量i,使得dp[len(stones)][i]为True
#         # 返回总重量减去两倍的最接近总重量一半的重量
#         for i in range(target, -1, -1):
#             if dp[len(stones)][i]:
#                 return total_sum - 2 * i
#         return 0
# 一维DP版
# class Solution:
#     def lastStoneWeightII(self, stones):
#         total_sum = sum(stones)
#         target = total_sum // 2
#         dp = [False] * (target + 1)
#         dp[0] = True

#         for stone in stones:
#             for j in range(target, stone - 1, -1):
#                 # 判断当前重量是否可以通过选择之前的石头得到或选择当前石头和之前的石头得到
#                 dp[j] = dp[j] or dp[j - stone]

#         for i in range(target, -1, -1):
#             if dp[i]:
#                 # 返回剩余石头的重量,即总重量减去两倍的最接近总重量一半的重量
#                 return total_sum - 2 * i
#         return 0


#12 494.目标和
    # 大家也可以记住,在求装满背包有几种方法的情况下,递推公式一般为:dp[j] += dp[j - nums[i]];
# 回溯版
# class Solution:
#     def backtracking(self, candidates, target, total, startIndex, path, result):
#         if total == target:
#             result.append(path[:])  # 将当前路径的副本添加到结果中
#         # 如果 sum + candidates[i] > target,则停止遍历
#         for i in range(startIndex, len(candidates)):
#             if total + candidates[i] > target:
#                 break
#             total += candidates[i]
#             path.append(candidates[i])
#             self.backtracking(candidates, target, total, i + 1, path, result)
#             total -= candidates[i]
#             path.pop()

#     def findTargetSumWays(self, nums: List[int], target: int) -> int:
#         total = sum(nums)
#         if target > total:
#             return 0  # 此时没有方案
#         if (target + total) % 2 != 0:
#             return 0  # 此时没有方案,两个整数相加时要注意数值溢出的问题
#         bagSize = (target + total) // 2  # 转化为组合总和问题,bagSize就是目标和

#         # 以下是回溯法代码
#         result = []
#         nums.sort()  # 需要对nums进行排序
#         self.backtracking(nums, bagSize, 0, 0, [], result)
#         return len(result)
# # 二维DP
# class Solution:
#     def findTargetSumWays(self, nums: List[int], target: int) -> int:
#         total_sum = sum(nums)  # 计算nums的总和
#         if abs(target) > total_sum:
#             return 0  # 此时没有方案
#         if (target + total_sum) % 2 == 1:
#             return 0  # 此时没有方案
#         target_sum = (target + total_sum) // 2  # 目标和

#         # 创建二维动态规划数组,行表示选取的元素数量,列表示累加和
#         dp = [[0] * (target_sum + 1) for _ in range(len(nums) + 1)]

#         # 初始化状态
#         dp[0][0] = 1

#         # 动态规划过程
#         for i in range(1, len(nums) + 1):
#             for j in range(target_sum + 1):
#                 dp[i][j] = dp[i - 1][j]  # 不选取当前元素
#                 if j >= nums[i - 1]:
#                     dp[i][j] += dp[i - 1][j - nums[i - 1]]  # 选取当前元素

#         return dp[len(nums)][target_sum]  # 返回达到目标和的方案数
# 一维DP
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        total_sum = sum(nums)  # 计算nums的总和
        if abs(target) > total_sum:
            return 0  # 此时没有方案
        if (target + total_sum) % 2 == 1:
            return 0  # 此时没有方案
        target_sum = (target + total_sum) // 2  # 目标和
        dp = [0] * (target_sum + 1)  # 创建动态规划数组,初始化为0
        dp[0] = 1  # 当目标和为0时,只有一种方案,即什么都不选
        for num in nums:
            for j in range(target_sum, num - 1, -1):
                dp[j] += dp[j - num]  # 状态转移方程,累加不同选择方式的数量
        return dp[target_sum]  # 返回达到目标和的方案数


#13 474.一和零
# DP（版本一）
# class Solution:
#     def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
#         dp = [[0] * (n + 1) for _ in range(m + 1)]  # 创建二维动态规划数组,初始化为0
#         for s in strs:  # 遍历物品
#             zeroNum = s.count('0')  # 统计0的个数
#             oneNum = len(s) - zeroNum  # 统计1的个数
#             for i in range(m, zeroNum - 1, -1):  # 遍历背包容量且从后向前遍历
#                 for j in range(n, oneNum - 1, -1):
#                     dp[i][j] = max(dp[i][j], dp[i - zeroNum][j - oneNum] + 1)  # 状态转移方程
#         return dp[m][n]
# DP（版本二）
class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        dp = [[0] * (n + 1) for _ in range(m + 1)]  # 创建二维动态规划数组,初始化为0
        # 遍历物品
        for s in strs:
            ones = s.count('1')  # 统计字符串中1的个数
            zeros = s.count('0')  # 统计字符串中0的个数
            # 遍历背包容量且从后向前遍历
            for i in range(m, zeros - 1, -1):
                for j in range(n, ones - 1, -1):
                    dp[i][j] = max(dp[i][j], dp[i - zeros][j - ones] + 1)  # 状态转移方程
        return dp[m][n]


#14 动态规划:完全背包理论基础
    # 我们知道01背包内嵌的循环是从大到小遍历,为了保证每个物品仅被添加一次。
    # 而完全背包的物品是可以添加多次的,所以要从小到大去遍历
    # 01背包中二维dp数组的两个for遍历的先后循序是可以颠倒了
    # 一维dp数组的两个for循环先后循序一定是先遍历物品,再遍历背包容量。
    # 在完全背包中,对于一维dp数组来说,其实两个for循环嵌套顺序是无所谓的！
# 先遍历物品,再遍历背包（无参版）
def test_CompletePack():
    weight = [1, 3, 4]
    value = [15, 20, 30]
    bagWeight = 4
    dp = [0] * (bagWeight + 1)
    for i in range(len(weight)):  # 遍历物品
        for j in range(weight[i], bagWeight + 1):  # 遍历背包容量
            dp[j] = max(dp[j], dp[j - weight[i]] + value[i])
    print(dp[bagWeight])

test_CompletePack()
# 先遍历物品,再遍历背包（有参版）
def test_CompletePack(weight, value, bagWeight):
    dp = [0] * (bagWeight + 1)
    for i in range(len(weight)):  # 遍历物品
        for j in range(weight[i], bagWeight + 1):  # 遍历背包容量
            dp[j] = max(dp[j], dp[j - weight[i]] + value[i])
    return dp[bagWeight]

if __name__ == "__main__":
    weight = [1, 3, 4]
    value = [15, 20, 30]
    bagWeight = 4
    result = test_CompletePack(weight, value, bagWeight)
    print(result)
# 先遍历背包,再遍历物品（无参版）
def test_CompletePack():
    weight = [1, 3, 4]
    value = [15, 20, 30]
    bagWeight = 4

    dp = [0] * (bagWeight + 1)

    for j in range(bagWeight + 1):  # 遍历背包容量
        for i in range(len(weight)):  # 遍历物品
            if j - weight[i] >= 0:
                dp[j] = max(dp[j], dp[j - weight[i]] + value[i])

    print(dp[bagWeight])

test_CompletePack()
# 先遍历背包,再遍历物品（有参版）
def test_CompletePack(weight, value, bagWeight):
    dp = [0] * (bagWeight + 1)
    for j in range(bagWeight + 1):  # 遍历背包容量
        for i in range(len(weight)):  # 遍历物品
            if j - weight[i] >= 0:
                dp[j] = max(dp[j], dp[j - weight[i]] + value[i])
    return dp[bagWeight]


if __name__ == "__main__":
    weight = [1, 3, 4]
    value = [15, 20, 30]
    bagWeight = 4
    result = test_CompletePack(weight, value, bagWeight)
    print(result)


#15 518.零钱兑换II
    # 在求装满背包有几种方案的时候,认清遍历顺序是非常关键的。
    # 如果求组合数就是外层for循环遍历物品,内层for遍历背包。
    # 如果求排列数就是外层for遍历背包,内层for循环遍历物品。
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        dp = [0]*(amount + 1)
        dp[0] = 1
        # 遍历物品
        for i in range(len(coins)):
            # 遍历背包
            for j in range(coins[i], amount + 1):
                dp[j] += dp[j - coins[i]]
        return dp[amount]


#16 377. 组合总和 Ⅳ
    # 如果把遍历nums（物品）放在外循环,遍历target的作为内循环的话
    # 举一个例子:计算dp[4]的时候,结果集只有 {1,3} 这样的集合,不会有{3,1}这样的集合
    # 因为nums遍历放在外层,3只能出现在1后面！
    # 所以本题遍历顺序最终遍历顺序:target（背包）放在外循环,将nums（物品）放在内循环,内循环从前到后遍历。
# 卡哥版
# class Solution:
#     def combinationSum4(self, nums: List[int], target: int) -> int:
#         dp = [0] * (target + 1)
#         dp[0] = 1
#         for i in range(1, target + 1):  # 遍历背包
#             for j in range(len(nums)):  # 遍历物品
#                 if i - nums[j] >= 0:
#                     dp[i] += dp[i - nums[j]]
#         return dp[target]
# 优化版
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        dp = [0] * (target + 1)  # 创建动态规划数组,用于存储组合总数
        dp[0] = 1  # 初始化背包容量为0时的组合总数为1

        for i in range(1, target + 1):  # 遍历背包容量
            for j in nums:  # 遍历物品列表
                if i >= j:  # 当背包容量大于等于当前物品重量时
                    dp[i] += dp[i - j]  # 更新组合总数

        return dp[-1]  # 返回背包容量为target时的组合总数
# 70. 爬楼梯（进阶版）
# 假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
# 每次你可以爬至多m (1 <= m < n)个台阶。你有多少种不同的方法可以爬到楼顶呢？
# 注意:给定 n 是一个正整数。
# 输入描述:输入共一行,包含两个正整数,分别表示n, m
# 输出描述:输出一个整数,表示爬到楼顶的方法数。
# 输入示例:3 2
# 输出示例:3
# 提示:
# 当 m = 2,n = 3 时,n = 3 这表示一共有三个台阶,m = 2 代表你每次可以爬一个台阶或者两个台阶。
# 此时你有三种方法可以爬到楼顶。
# 1 阶 + 1 阶 + 1 阶段
# 1 阶 + 2 阶
# 2 阶 + 1 阶
# 时间复杂度: O(n * m)
# 空间复杂度: O(n)


#17 322. 零钱兑换
# 先遍历物品 后遍历背包
# class Solution:
#     def coinChange(self, coins: List[int], amount: int) -> int:
#         dp = [float('inf')] * (amount + 1)  # 创建动态规划数组,初始值为正无穷大
#         dp[0] = 0  # 初始化背包容量为0时的最小硬币数量为0

#         for coin in coins:  # 遍历硬币列表,相当于遍历物品
#             for i in range(coin, amount + 1):  # 遍历背包容量
#                 if dp[i - coin] != float('inf'):  # 如果dp[i - coin]不是初始值,则进行状态转移
#                     dp[i] = min(dp[i - coin] + 1, dp[i])  # 更新最小硬币数量

#         if dp[amount] == float('inf'):  # 如果最终背包容量的最小硬币数量仍为正无穷大,表示无解
#             return -1
#         return dp[amount]  # 返回背包容量为amount时的最小硬币数量
# 先遍历背包 后遍历物品
# class Solution:
#     def coinChange(self, coins: List[int], amount: int) -> int:
#         dp = [float('inf')] * (amount + 1)  # 创建动态规划数组,初始值为正无穷大
#         dp[0] = 0  # 初始化背包容量为0时的最小硬币数量为0

#         for i in range(1, amount + 1):  # 遍历背包容量
#             for j in range(len(coins)):  # 遍历硬币列表,相当于遍历物品
#                 if i - coins[j] >= 0 and dp[i - coins[j]] != float('inf'):  # 如果dp[i - coins[j]]不是初始值,则进行状态转移
#                     dp[i] = min(dp[i - coins[j]] + 1, dp[i])  # 更新最小硬币数量

#         if dp[amount] == float('inf'):  # 如果最终背包容量的最小硬币数量仍为正无穷大,表示无解
#             return -1
#         return dp[amount]  # 返回背包容量为amount时的最小硬币数量
# 先遍历物品 后遍历背包（优化版）
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0

        for coin in coins:
            for i in range(coin, amount + 1): # 进行优化,从能装得下的背包开始计算,则不需要进行比较
                # 更新凑成金额 i 所需的最少硬币数量
                dp[i] = min(dp[i], dp[i - coin] + 1)

        return dp[amount] if dp[amount] != float('inf') else -1
# 先遍历背包 后遍历物品（优化版）
# class Solution:
#     def coinChange(self, coins: List[int], amount: int) -> int:
#         dp = [float('inf')] * (amount + 1)
#         dp[0] = 0

#         for i in range(1, amount + 1):  # 遍历背包容量
#             for coin in coins:  # 遍历物品
#                 if i - coin >= 0:
#                     # 更新凑成金额 i 所需的最少硬币数量
#                     dp[i] = min(dp[i], dp[i - coin] + 1)
#         return dp[amount] if dp[amount] != float('inf') else -1


#18 279.完全平方数
# 先遍历背包, 再遍历物品
# class Solution:
#     def numSquares(self, n: int) -> int:
#         dp = [float('inf')] * (n + 1)
#         dp[0] = 0

#         for i in range(1, n + 1):  # 遍历背包
#             for j in range(1, int(i ** 0.5) + 1):  # 遍历物品
#                 # 更新凑成数字 i 所需的最少完全平方数数量
#                 dp[i] = min(dp[i], dp[i - j * j] + 1)

#         return dp[n]
# 先遍历物品, 再遍历背包
class Solution:
    def numSquares(self, n: int) -> int:
        dp = [float('inf')] * (n + 1)
        dp[0] = 0

        for i in range(1, int(n ** 0.5) + 1):  # 遍历物品
            for j in range(i * i, n + 1):  # 遍历背包
                # 更新凑成数字 j 所需的最少完全平方数数量
                dp[j] = min(dp[j - i * i] + 1, dp[j])

        return dp[n]
# 其他版本
# class Solution:
#     def numSquares(self, n: int) -> int:
#         # 创建动态规划数组,初始值为最大值
#         dp = [float('inf')] * (n + 1)
#         # 初始化已知情况
#         dp[0] = 0

#         # 遍历背包容量
#         for i in range(1, n + 1):
#             # 遍历完全平方数作为物品
#             j = 1
#             while j * j <= i:
#                 # 更新最少完全平方数的数量
#                 dp[i] = min(dp[i], dp[i - j * j] + 1)
#                 j += 1

#         # 返回结果
#         return dp[n]


#19 139.单词拆分
    # 求组合数:动态规划:518.零钱兑换II (opens new window)
    # 求排列数:动态规划:377. 组合总和 Ⅳ (opens new window)、动态规划:70. 爬楼梯进阶版（完全背包） (opens new window)
    # 求最小数:动态规划:322. 零钱兑换 (opens new window)、动态规划:279.完全平方数
# 回溯
# class Solution:
#     def backtracking(self, s: str, wordSet: set[str], startIndex: int) -> bool:
#         # 边界情况:已经遍历到字符串末尾,返回True
#         if startIndex >= len(s):
#             return True

#         # 遍历所有可能的拆分位置
#         for i in range(startIndex, len(s)):
#             word = s[startIndex:i + 1]  # 截取子串
#             if word in wordSet and self.backtracking(s, wordSet, i + 1):
#                 # 如果截取的子串在字典中,并且后续部分也可以被拆分成单词,返回True
#                 return True

#         # 无法进行有效拆分,返回False
#         return False

#     def wordBreak(self, s: str, wordDict: List[str]) -> bool:
#         wordSet = set(wordDict)  # 转换为哈希集合,提高查找效率
#         return self.backtracking(s, wordSet, 0)
# DP（版本一）
# class Solution:
#     def wordBreak(self, s: str, wordDict: List[str]) -> bool:
#         wordSet = set(wordDict)
#         n = len(s)
#         dp = [False] * (n + 1)  # dp[i] 表示字符串的前 i 个字符是否可以被拆分成单词
#         dp[0] = True  # 初始状态,空字符串可以被拆分成单词

#         for i in range(1, n + 1): # 遍历背包
#             for j in range(i): # 遍历单词
#                 if dp[j] and s[j:i] in wordSet:
#                     dp[i] = True  # 如果 s[0:j] 可以被拆分成单词,并且 s[j:i] 在单词集合中存在,则 s[0:i] 可以被拆分成单词
#                     break
#         return dp[n]
# DP（版本二）
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        dp = [False]*(len(s) + 1)
        dp[0] = True
        # 遍历背包
        for j in range(1, len(s) + 1):
            # 遍历单词
            for word in wordDict:
                if j >= len(word):
                    dp[j] = dp[j] or (dp[j - len(word)] and word == s[j - len(word):j])
        return dp[len(s)]


#20 198.打家劫舍
# 1维DP
class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums) == 0:  # 如果没有房屋,返回0
            return 0
        if len(nums) == 1:  # 如果只有一个房屋,返回其金额
            return nums[0]

        # 创建一个动态规划数组,用于存储最大金额
        dp = [0] * len(nums)
        dp[0] = nums[0]  # 将dp的第一个元素设置为第一个房屋的金额
        dp[1] = max(nums[0], nums[1])  # 将dp的第二个元素设置为第一二个房屋中的金额较大者

        # 遍历剩余的房屋
        for i in range(2, len(nums)):
            # 对于每个房屋,选择抢劫当前房屋和抢劫前一个房屋的最大金额
            # 注意这里是考虑，并不是一定要偷i-1,i-2房，这是很多同学容易混淆的点
            dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])

        return dp[-1]  # 返回最后一个房屋中可抢劫的最大金额
# 2维DP
# class Solution:
#     def rob(self, nums: List[int]) -> int:
#         if not nums:  # 如果没有房屋,返回0
#             return 0

#         n = len(nums)
#         dp = [[0, 0] for _ in range(n)]  # 创建二维动态规划数组,dp[i][0]表示不抢劫第i个房屋的最大金额,dp[i][1]表示抢劫第i个房屋的最大金额

#         dp[0][1] = nums[0]  # 抢劫第一个房屋的最大金额为第一个房屋的金额

#         for i in range(1, n):
#             dp[i][0] = max(dp[i-1][0], dp[i-1][1])  # 不抢劫第i个房屋,最大金额为前一个房屋抢劫和不抢劫的最大值
#             dp[i][1] = dp[i-1][0] + nums[i]  # 抢劫第i个房屋,最大金额为前一个房屋不抢劫的最大金额加上当前房屋的金额

#         return max(dp[n-1][0], dp[n-1][1])  # 返回最后一个房屋中可抢劫的最大金额
# *** 优化版
class Solution:
    def rob(self, nums: List[int]) -> int:
        if not nums:  # 如果没有房屋,返回0
            return 0

        prev_max = 0  # 上一个房屋的最大金额
        curr_max = 0  # 当前房屋的最大金额

        for num in nums:
            temp = curr_max  # 临时变量保存当前房屋的最大金额
            curr_max = max(prev_max + num, curr_max)  # 更新当前房屋的最大金额
            prev_max = temp  # 更新上一个房屋的最大金额
        return curr_max  # 返回最后一个房屋中可抢劫的最大金额


#21 213.打家劫舍II
class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0
        if len(nums) == 1:
            return nums[0]
        
        result1 = self.robRange(nums, 0, len(nums) - 2)  # 情况二
        result2 = self.robRange(nums, 1, len(nums) - 1)  # 情况三
        return max(result1, result2)
    # 198.打家劫舍的逻辑
    def robRange(self, nums: List[int], start: int, end: int) -> int:
        if end == start:
            return nums[start]
        
        prev_max = nums[start]
        curr_max = max(nums[start], nums[start + 1])
        
        for i in range(start + 2, end + 1):
            temp = curr_max
            curr_max = max(prev_max + nums[i], curr_max)
            prev_max = temp
        return curr_max
# # 2维DP
# class Solution:
#     def rob(self, nums: List[int]) -> int:
#         if len(nums) < 3:
#             return max(nums)

#         # 情况二:不抢劫第一个房屋
#         result1 = self.robRange(nums[:-1])

#         # 情况三:不抢劫最后一个房屋
#         result2 = self.robRange(nums[1:])

#         return max(result1, result2)

#     def robRange(self, nums):
#         dp = [[0, 0] for _ in range(len(nums))]
#         dp[0][1] = nums[0]

#         for i in range(1, len(nums)):
#             dp[i][0] = max(dp[i - 1])
#             dp[i][1] = dp[i - 1][0] + nums[i]

#         return max(dp[-1])
# # 优化版
# class Solution:
#     def rob(self, nums: List[int]) -> int:
#         if not nums:  # 如果没有房屋,返回0
#             return 0

#         if len(nums) == 1:  # 如果只有一个房屋,返回该房屋的金额
#             return nums[0]

#         # 情况二:不抢劫第一个房屋
#         prev_max = 0  # 上一个房屋的最大金额
#         curr_max = 0  # 当前房屋的最大金额
#         for num in nums[1:]:
#             temp = curr_max  # 临时变量保存当前房屋的最大金额
#             curr_max = max(prev_max + num, curr_max)  # 更新当前房屋的最大金额
#             prev_max = temp  # 更新上一个房屋的最大金额
#         result1 = curr_max

#         # 情况三:不抢劫最后一个房屋
#         prev_max = 0  # 上一个房屋的最大金额
#         curr_max = 0  # 当前房屋的最大金额
#         for num in nums[:-1]:
#             temp = curr_max  # 临时变量保存当前房屋的最大金额
#             curr_max = max(prev_max + num, curr_max)  # 更新当前房屋的最大金额
#             prev_max = temp  # 更新上一个房屋的最大金额
#         result2 = curr_max
#         return max(result1, result2)


#22 337.打家劫舍 III
# 暴力递归
# class Solution:
#     def rob(self, root: TreeNode) -> int:
#         if root is None:
#             return 0
#         if root.left is None and root.right  is None:
#             return root.val
#         # 偷父节点
#         val1 = root.val
#         if root.left:
#             val1 += self.rob(root.left.left) + self.rob(root.left.right)
#         if root.right:
#             val1 += self.rob(root.right.left) + self.rob(root.right.right)
#         # 不偷父节点
#         val2 = self.rob(root.left) + self.rob(root.right)
#         return max(val1, val2)
# # 记忆化递归
# class Solution:
#     memory = {}
#     def rob(self, root: TreeNode) -> int:
#         if root is None:
#             return 0
#         if root.left is None and root.right  is None:
#             return root.val
#         if self.memory.get(root) is not None:
#             return self.memory[root]
#         # 偷父节点
#         val1 = root.val
#         if root.left:
#             val1 += self.rob(root.left.left) + self.rob(root.left.right)
#         if root.right:
#             val1 += self.rob(root.right.left) + self.rob(root.right.right)
#         # 不偷父节点
#         val2 = self.rob(root.left) + self.rob(root.right)
#         self.memory[root] = max(val1, val2)
#         return max(val1, val2)
# 动态规划
class Solution:
    def rob(self, root: Optional[TreeNode]) -> int:
        # dp数组（dp table）以及下标的含义:
        # 1. 下标为 0 记录 **不偷该节点** 所得到的的最大金钱
        # 2. 下标为 1 记录 **偷该节点** 所得到的的最大金钱
        dp = self.traversal(root)
        return max(dp)

    # 要用后序遍历, 因为要通过递归函数的返回值来做下一步计算
    def traversal(self, node):
        
        # 递归终止条件,就是遇到了空节点,那肯定是不偷的
        if not node:
            return (0, 0)

        left = self.traversal(node.left)
        right = self.traversal(node.right)

        # 不偷当前节点, 偷子节点
        val_0 = max(left[0], left[1]) + max(right[0], right[1])

        # 偷当前节点, 不偷子节点
        val_1 = node.val + left[0] + right[0]
        return (val_0, val_1)


#23 121. 买卖股票的最佳时机
    # dp数组的含义：
    # dp[i][0] 表示第i天持有股票所得现金。
    # dp[i][1] 表示第i天不持有股票所得最多现金
# 贪心法
# class Solution:
#     def maxProfit(self, prices: List[int]) -> int:
#         low = float("inf")
#         result = 0
#         for i in range(len(prices)):
#             low = min(low, prices[i]) #取最左最小价格
#             result = max(result, prices[i] - low) #直接取最大区间利润
#         return result
# *** 动态规划:版本一
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        length = len(prices)
        if len == 0:
            return 0
        dp = [[0] * 2 for _ in range(length)]
        dp[0][0] = -prices[0]
        dp[0][1] = 0
        for i in range(1, length):
            dp[i][0] = max(dp[i-1][0], -prices[i])
            dp[i][1] = max(dp[i-1][1], prices[i] + dp[i-1][0])
        return dp[-1][1]
# *** 动态规划:版本二
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        length = len(prices)
        dp = [[0] * 2 for _ in range(2)] #注意这里只开辟了一个2 * 2大小的二维数组
        dp[0][0] = -prices[0]
        dp[0][1] = 0
        for i in range(1, length):
            dp[i % 2][0] = max(dp[(i-1) % 2][0], -prices[i])
            dp[i % 2][1] = max(dp[(i-1) % 2][1], prices[i] + dp[(i-1) % 2][0])
        return dp[(length-1) % 2][1]
# *** 动态规划:版本三
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        length = len(prices)
        dp0, dp1 = -prices[0], 0 #注意这里只维护两个常量,因为dp0的更新不受dp1的影响
        for i in range(1, length):
            dp1 = max(dp1, dp0 + prices[i])
            dp0 = max(dp0, -prices[i])
        return dp1


#24 122.买卖股票的最佳时机II
# *** 版本一
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
# *** 版本二
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        length = len(prices)
        dp = [[0] * 2 for _ in range(2)] #注意这里只开辟了一个2 * 2大小的二维数组
        dp[0][0] = -prices[0]
        dp[0][1] = 0
        for i in range(1, length):
            dp[i % 2][0] = max(dp[(i-1) % 2][0], dp[(i-1) % 2][1] - prices[i])
            dp[i % 2][1] = max(dp[(i-1) % 2][1], dp[(i-1) % 2][0] + prices[i])
        return dp[(length-1) % 2][1]


#25 123.买卖股票的最佳时机III
# *** 版本一
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) == 0:
            return 0
        dp = [[0] * 5 for _ in range(len(prices))]
        dp[0][1] = -prices[0]
        dp[0][3] = -prices[0]
        for i in range(1, len(prices)):
            dp[i][0] = dp[i-1][0]
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i])
            dp[i][2] = max(dp[i-1][2], dp[i-1][1] + prices[i])
            dp[i][3] = max(dp[i-1][3], dp[i-1][2] - prices[i])
            dp[i][4] = max(dp[i-1][4], dp[i-1][3] + prices[i])
        return dp[-1][4]
# 版本二
# class Solution:
#     def maxProfit(self, prices: List[int]) -> int:
#         if len(prices) == 0:
#             return 0
#         dp = [0] * 5 
#         dp[1] = -prices[0]
#         dp[3] = -prices[0]
#         for i in range(1, len(prices)):
#             dp[1] = max(dp[1], dp[0] - prices[i])
#             dp[2] = max(dp[2], dp[1] + prices[i])
#             dp[3] = max(dp[3], dp[2] - prices[i])
#             dp[4] = max(dp[4], dp[3] + prices[i])
#         return dp[4]


#26 188.买卖股票的最佳时机IV
# *** 版本一
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        if len(prices) == 0:
            return 0
        dp = [[0] * (2*k+1) for _ in range(len(prices))]
        for j in range(1, 2*k, 2):
            dp[0][j] = -prices[0]
        for i in range(1, len(prices)):
            for j in range(0, 2*k-1, 2):
                dp[i][j+1] = max(dp[i-1][j+1], dp[i-1][j] - prices[i])
                dp[i][j+2] = max(dp[i-1][j+2], dp[i-1][j+1] + prices[i])
        return dp[-1][2*k]
# 版本二
# class Solution:
#     def maxProfit(self, k: int, prices: List[int]) -> int:
#         if len(prices) == 0: return 0
#         dp = [0] * (2*k + 1)
#         for i in range(1,2*k,2):
#             dp[i] = -prices[0]
#         for i in range(1,len(prices)):
#             for j in range(1,2*k + 1):
#                 if j % 2:
#                     dp[j] = max(dp[j],dp[j-1]-prices[i])
#                 else:
#                     dp[j] = max(dp[j],dp[j-1]+prices[i])
#         return dp[2*k]


#27 309.最佳买卖股票时机含冷冻期
# 版本一
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        if n == 0:
            return 0
        dp = [[0] * 4 for _ in range(n)]  # 创建动态规划数组,4个状态分别表示持有股票、不持有股票且处于冷冻期、不持有股票且不处于冷冻期、不持有股票且当天卖出后处于冷冻期
        dp[0][0] = -prices[0]  # 初始状态:第一天持有股票的最大利润为买入股票的价格
        for i in range(1, n):
            dp[i][0] = max(dp[i-1][0], max(dp[i-1][3], dp[i-1][1]) - prices[i])  # 当前持有股票的最大利润等于前一天持有股票的最大利润或者前一天不持有股票且不处于冷冻期的最大利润减去当前股票的价格
            dp[i][1] = max(dp[i-1][1], dp[i-1][3])  # 当前不持有股票且处于冷冻期的最大利润等于前一天持有股票的最大利润加上当前股票的价格
            dp[i][2] = dp[i-1][0] + prices[i]  # 当前不持有股票且不处于冷冻期的最大利润等于前一天不持有股票的最大利润或者前一天处于冷冻期的最大利润
            dp[i][3] = dp[i-1][2]  # 当前不持有股票且当天卖出后处于冷冻期的最大利润等于前一天不持有股票且不处于冷冻期的最大利润
        return max(dp[n-1][3], dp[n-1][1], dp[n-1][2])  # 返回最后一天不持有股票的最大利润
# 版本二
# class Solution:
#     def maxProfit(self, prices: List[int]) -> int:
#         n = len(prices)
#         if n < 2:
#             return 0

#         # 定义三种状态的动态规划数组
#         dp = [[0] * 3 for _ in range(n)]
#         dp[0][0] = -prices[0]  # 持有股票的最大利润
#         dp[0][1] = 0           # 不持有股票,且处于冷冻期的最大利润
#         dp[0][2] = 0           # 不持有股票,不处于冷冻期的最大利润

#         for i in range(1, n):
#             # 当前持有股票的最大利润等于前一天持有股票的最大利润或者前一天不持有股票且不处于冷冻期的最大利润减去当前股票的价格
#             dp[i][0] = max(dp[i-1][0], dp[i-1][2] - prices[i])
#             # 当前不持有股票且处于冷冻期的最大利润等于前一天持有股票的最大利润加上当前股票的价格
#             dp[i][1] = dp[i-1][0] + prices[i]
#             # 当前不持有股票且不处于冷冻期的最大利润等于前一天不持有股票的最大利润或者前一天处于冷冻期的最大利润
#             dp[i][2] = max(dp[i-1][2], dp[i-1][1])

#         # 返回最后一天不持有股票的最大利润
#         return max(dp[-1][1], dp[-1][2])


#28 714.买卖股票的最佳时机含手续费
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        n = len(prices)
        dp = [[0] * 2 for _ in range(n)]
        dp[0][0] = -prices[0] #持股票
        for i in range(1, n):
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] - prices[i])
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] + prices[i] - fee)
        return max(dp[-1][0], dp[-1][1])


#29 300.最长递增子序列
# DP
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if len(nums) <= 1:
            return len(nums)
        dp = [1] * len(nums)
        result = 1
        for i in range(1, len(nums)):
            for j in range(0, i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
            result = max(result, dp[i]) #取长的子序列
        return result
# 贪心
# class Solution:
#     def lengthOfLIS(self, nums: List[int]) -> int:
#         if len(nums) <= 1:
#             return len(nums)
        
#         tails = [nums[0]]  # 存储递增子序列的尾部元素
#         for num in nums[1:]:
#             if num > tails[-1]:
#                 tails.append(num)  # 如果当前元素大于递增子序列的最后一个元素,直接加入到子序列末尾
#             else:
#                 # 使用二分查找找到当前元素在递增子序列中的位置,并替换对应位置的元素
#                 left, right = 0, len(tails) - 1
#                 while left < right:
#                     mid = (left + right) // 2
#                     if tails[mid] < num:
#                         left = mid + 1
#                     else:
#                         right = mid
#                 tails[left] = num
#         return len(tails)  # 返回递增子序列的长度


#30 674. 最长连续递增序列
# DP
class Solution:
    def findLengthOfLCIS(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0
        result = 1
        dp = [1] * len(nums)
        for i in range(len(nums)-1):
            if nums[i+1] > nums[i]: #连续记录
                dp[i+1] = dp[i] + 1
            result = max(result, dp[i+1])
        return result
# DP(优化版)
# class Solution:
#     def findLengthOfLCIS(self, nums: List[int]) -> int:
#         if not nums:
#             return 0

#         max_length = 1
#         current_length = 1

#         for i in range(1, len(nums)):
#             if nums[i] > nums[i - 1]:
#                 current_length += 1
#                 max_length = max(max_length, current_length)
#             else:
#                 current_length = 1

#         return max_length
# # 贪心
# class Solution:
#     def findLengthOfLCIS(self, nums: List[int]) -> int:
#         if len(nums) == 0:
#             return 0
#         result = 1 #连续子序列最少也是1
#         count = 1
#         for i in range(len(nums)-1):
#             if nums[i+1] > nums[i]: #连续记录
#                 count += 1
#             else: #不连续,count从头开始
#                 count = 1
#             result = max(result, count)
#         return result


#31 718. 最长重复子数组
# 2维DP
# class Solution:
#     def findLength(self, nums1: List[int], nums2: List[int]) -> int:
#         # 创建一个二维数组 dp,用于存储最长公共子数组的长度
#         dp = [[0] * (len(nums2) + 1) for _ in range(len(nums1) + 1)]
#         # 记录最长公共子数组的长度
#         result = 0

#         # 遍历数组 nums1
#         for i in range(1, len(nums1) + 1):
#             # 遍历数组 nums2
#             for j in range(1, len(nums2) + 1):
#                 # 如果 nums1[i-1] 和 nums2[j-1] 相等
#                 if nums1[i - 1] == nums2[j - 1]:
#                     # 在当前位置上的最长公共子数组长度为前一个位置上的长度加一
#                     dp[i][j] = dp[i - 1][j - 1] + 1
#                 # 更新最长公共子数组的长度
#                 if dp[i][j] > result:
#                     result = dp[i][j]

#         # 返回最长公共子数组的长度
#         return result
# 1维DP
class Solution:
    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        # 创建一个一维数组 dp,用于存储最长公共子数组的长度
        dp = [0] * (len(nums2) + 1)
        # 记录最长公共子数组的长度
        result = 0

        # 遍历数组 nums1
        for i in range(1, len(nums1) + 1):
            # 用于保存上一个位置的值
            prev = 0
            # 遍历数组 nums2
            for j in range(1, len(nums2) + 1):
                # 保存当前位置的值,因为会在后面被更新
                current = dp[j]
                # 如果 nums1[i-1] 和 nums2[j-1] 相等
                if nums1[i - 1] == nums2[j - 1]:
                    # 在当前位置上的最长公共子数组长度为上一个位置的长度加一
                    dp[j] = prev + 1
                    # 更新最长公共子数组的长度
                    if dp[j] > result:
                        result = dp[j]
                else:
                    # 如果不相等,将当前位置的值置为零
                    dp[j] = 0
                # 更新 prev 变量为当前位置的值,供下一次迭代使用
                prev = current

        # 返回最长公共子数组的长度
        return result
# 2维DP 扩展
# class Solution:
#     def findLength(self, nums1: List[int], nums2: List[int]) -> int:
#         # 创建一个二维数组 dp,用于存储最长公共子数组的长度
#         dp = [[0] * (len(nums2) + 1) for _ in range(len(nums1) + 1)]
#         # 记录最长公共子数组的长度
#         result = 0

#         # 对第一行和第一列进行初始化
#         for i in range(len(nums1)):
#             if nums1[i] == nums2[0]:
#                 dp[i + 1][1] = 1
#         for j in range(len(nums2)):
#             if nums1[0] == nums2[j]:
#                 dp[1][j + 1] = 1

#         # 填充dp数组
#         for i in range(1, len(nums1) + 1):
#             for j in range(1, len(nums2) + 1):
#                 if nums1[i - 1] == nums2[j - 1]:
#                     # 如果 nums1[i-1] 和 nums2[j-1] 相等,则当前位置的最长公共子数组长度为左上角位置的值加一
#                     dp[i][j] = dp[i - 1][j - 1] + 1
#                 if dp[i][j] > result:
#                     # 更新最长公共子数组的长度
#                     result = dp[i][j]

#         # 返回最长公共子数组的长度
#         return result


#32 1143.最长公共子序列
# 2维DP
# class Solution:
#     def longestCommonSubsequence(self, text1: str, text2: str) -> int:
#         # 创建一个二维数组 dp,用于存储最长公共子序列的长度
#         dp = [[0] * (len(text2) + 1) for _ in range(len(text1) + 1)]
        
#         # 遍历 text1 和 text2,填充 dp 数组
#         for i in range(1, len(text1) + 1):
#             for j in range(1, len(text2) + 1):
#                 if text1[i - 1] == text2[j - 1]:
#                     # 如果 text1[i-1] 和 text2[j-1] 相等,则当前位置的最长公共子序列长度为左上角位置的值加一
#                     dp[i][j] = dp[i - 1][j - 1] + 1
#                 else:
#                     # 如果 text1[i-1] 和 text2[j-1] 不相等,则当前位置的最长公共子序列长度为上方或左方的较大值
#                     dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
#         # 返回最长公共子序列的长度
#         return dp[len(text1)][len(text2)]
# 1维DP
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m, n = len(text1), len(text2)
        dp = [0] * (n + 1)  # 初始化一维DP数组
        
        for i in range(1, m + 1):
            prev = 0  # 保存上一个位置的最长公共子序列长度
            for j in range(1, n + 1):
                curr = dp[j]  # 保存当前位置的最长公共子序列长度
                if text1[i - 1] == text2[j - 1]:
                    # 如果当前字符相等,则最长公共子序列长度加一
                    dp[j] = prev + 1
                else:
                    # 如果当前字符不相等,则选择保留前一个位置的最长公共子序列长度中的较大值
                    dp[j] = max(dp[j], dp[j - 1])
                prev = curr  # 更新上一个位置的最长公共子序列长度
        
        return dp[n]  # 返回最后一个位置的最长公共子序列长度作为结果


#33 1035.不相交的线
class Solution:
    def maxUncrossedLines(self, A: List[int], B: List[int]) -> int:
        dp = [[0] * (len(B)+1) for _ in range(len(A)+1)]
        for i in range(1, len(A)+1):
            for j in range(1, len(B)+1):
                if A[i-1] == B[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[-1][-1]


#34 53. 最大子序和
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        dp = [0] * len(nums)
        dp[0] = nums[0]
        result = dp[0]
        for i in range(1, len(nums)):
            dp[i] = max(dp[i-1] + nums[i], nums[i]) #状态转移公式
            result = max(result, dp[i]) #result 保存dp[i]的最大值
        return result


#35 392.判断子序列
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        dp = [[0] * (len(t)+1) for _ in range(len(s)+1)]
        for i in range(1, len(s)+1):
            for j in range(1, len(t)+1):
                if s[i-1] == t[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = dp[i][j-1]
        if dp[-1][-1] == len(s):
            return True
        return False


#36 115.不同的子序列
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        dp = [[0] * (len(t)+1) for _ in range(len(s)+1)]
        for i in range(len(s)):
            dp[i][0] = 1
        for j in range(1, len(t)):
            dp[0][j] = 0
        for i in range(1, len(s)+1):
            for j in range(1, len(t)+1):
                if s[i-1] == t[j-1]:
                    dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
                else:
                    dp[i][j] = dp[i-1][j]
        return dp[-1][-1]

class SolutionDP2:
    """
    既然dp[i]只用到dp[i - 1]的状态,
    我们可以通过缓存dp[i - 1]的状态来对dp进行压缩,
    减少空间复杂度。
    （原理等同同于滚动数组）
    """
    
    def numDistinct(self, s: str, t: str) -> int:
        n1, n2 = len(s), len(t)
        if n1 < n2:
            return 0

        dp = [0 for _ in range(n2 + 1)]
        dp[0] = 1

        for i in range(1, n1 + 1):
            # 必须深拷贝
            # 不然prev[i]和dp[i]是同一个地址的引用
            prev = dp.copy()
            # 剪枝,保证s的长度大于等于t
            # 因为对于任意i,i > n1, dp[i] = 0
            # 没必要跟新状态。 
            end = i if i < n2 else n2
            for j in range(1, end + 1):
                if s[i - 1] == t[j - 1]:
                    dp[j] = prev[j - 1] + prev[j]
                else:
                    dp[j] = prev[j]
        return dp[-1]


#37 583. 两个字符串的删除操作
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        dp = [[0] * (len(word2)+1) for _ in range(len(word1)+1)]
        for i in range(len(word1)+1):
            dp[i][0] = i
        for j in range(len(word2)+1):
            dp[0][j] = j
        for i in range(1, len(word1)+1):
            for j in range(1, len(word2)+1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j-1] + 2, dp[i-1][j] + 1, dp[i][j-1] + 1)
        return dp[-1][-1]


#38 72. 编辑距离
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        dp = [[0] * (len(word2)+1) for _ in range(len(word1)+1)]
        for i in range(len(word1)+1):
            dp[i][0] = i
        for j in range(len(word2)+1):
            dp[0][j] = j
        for i in range(1, len(word1)+1):
            for j in range(1, len(word2)+1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1
        return dp[-1][-1]


#39 647. 回文子串
# 动态规划
class Solution:
    def countSubstrings(self, s: str) -> int:
        dp = [[False] * len(s) for _ in range(len(s))]
        result = 0
        for i in range(len(s)-1, -1, -1): #注意遍历顺序
            for j in range(i, len(s)):
                if s[i] == s[j]:
                    if j - i <= 1: #情况一 和 情况二
                        result += 1
                        dp[i][j] = True
                    elif dp[i+1][j-1]: #情况三
                        result += 1
                        dp[i][j] = True
        return result
# 动态规划:简洁版
class Solution:
    def countSubstrings(self, s: str) -> int:
        dp = [[False] * len(s) for _ in range(len(s))]
        result = 0
        for i in range(len(s)-1, -1, -1): #注意遍历顺序
            for j in range(i, len(s)):
                if s[i] == s[j] and (j - i <= 1 or dp[i+1][j-1]): 
                    result += 1
                    dp[i][j] = True
        return result
# 双指针法
class Solution:
    def countSubstrings(self, s: str) -> int:
        result = 0
        for i in range(len(s)):
            result += self.extend(s, i, i, len(s)) #以i为中心
            result += self.extend(s, i, i+1, len(s)) #以i和i+1为中心
        return result
    
    def extend(self, s, i, j, n):
        res = 0
        while i >= 0 and j < n and s[i] == s[j]:
            i -= 1
            j += 1
            res += 1
        return res


#40 516.最长回文子序列
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        dp = [[0] * len(s) for _ in range(len(s))]
        for i in range(len(s)):
            dp[i][i] = 1
        for i in range(len(s)-1, -1, -1):
            for j in range(i+1, len(s)):
                if s[i] == s[j]:
                    dp[i][j] = dp[i+1][j-1] + 2
                else:
                    dp[i][j] = max(dp[i+1][j], dp[i][j-1])
        return dp[0][-1]
