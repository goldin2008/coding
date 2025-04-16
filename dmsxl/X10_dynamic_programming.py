"""
===================================================================================================
动态规划方法总结:

1. 确定dp数组 (dp table) 以及下标的含义
2. 确定递推公式
3. dp数组如何初始化
4. 确定遍历顺序
5. 举例推导dp数组

总结:
2D:从小到大遍历,内外层可换
1D:只有从小到大遍历,才可能内层外层变换(比如完全背包)(取决于组合还是排列,如果是最大最小值,那都可以)
如果包含从大到小遍历(比如01背包的1D数组),那只能先物品后容量,原因是需要每一行都填充以后才能到下一行,这样用上一行的元素计算下一行的元素

*** 01背包:
递归公式: 2D dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight[i]] + value[i]);
         1D dp[j] = max(dp[j], dp[j - weight[i]] + value[i])
二维数组, 外层内层遍历**可以交换**, 都是**从小到大**遍历
一维数组, 外层物品**从小到大**, 内层容量**从大到小**, ***遍历不可交换***
一维dp数组不可以先遍历背包容量嵌套遍历物品,因为一维dp的写法,背包容量一定是要倒序遍历 (原因上面已经讲了)
如果遍历背包容量放在上一层,那么每个dp[j]就只会放入一个物品,即:背包里只放入了一个物品。
倒序遍历的原因是,本质上还是一个对二维数组的遍历,并且右下角的值依赖上一层左上角的值,因此需要保证左边的值仍然是上一层的,从右向左覆盖。
如果先容量再物品,最大容量的时候,遍历每一个物品,这个时候不符合1D数组的关系,因为1D数组是一行一行的,这个时候遍历每个物品,上一行还没有完全完成,
当遍历到下一个物品的时候,不能依靠上一行物品的数组来计算,因为都是依靠上一行左面的元素.这时是从最大容量遍历,左面元素还没有ready
一维dp数组的背包在遍历顺序上和二维dp数组实现的01背包其实是有很大差异的, 需要注意!

*** 完全背包:
纯完全背包(求是否能装满背包)
01背包和完全背包唯一不同就是体现在遍历顺序上,我们知道01背包内嵌的循环是从大到小遍历(1D),为了保证每个物品仅被添加一次。
而完全背包的物品是可以添加多次的,所以要从小到大去遍历(1D)
(求最大价值)先遍历物品, 再遍历背包 与 先遍历背包, 再遍历物品都是可以的, 都要**从小到大**去遍历
(求几种方法)遍历方式跟是排列还是组合问题相关

我们先来看 外层for循环遍历物品(钱币),内层for遍历背包(金钱总额)的情况。
假设:coins[0] = 1,coins[1] = 5。
那么就是先把1加入计算,然后再把5加入计算,得到的方法数量只有{1, 5}这种情况。而不会出现{5, 1}的情况。
所以这种遍历顺序中dp[j]里计算的是组合数！
如果把两个for交换顺序
背包容量的每一个值,都是经过 1 和 5 的计算,包含了{1, 5} 和 {5, 1}两种情况。

*** 01背包,完全背包区别
01背包中二维dp数组的两个for遍历的先后循序是可以颠倒了,一维dp数组的两个for循环先后循序一定是先遍历物品,再遍历背包容量。
在完全背包中,对于一维dp数组来说,其实两个for循环嵌套顺序是无所谓的!

非纯完全背包(求装满背包有几种方法)
一维数组 (组合问题), 外层物品, 内层容量, 都是从小到大, 遍历不可交换
一维数组 (排列问题), 外层容量, 内层物品, 都是从小到大, 遍历不可交换
(DP方法求的是排列总和, 而且仅仅是求排列总和的个数, 并不是把所有的排列都列出来。如果要把排列都列出来的话, 只能使用回溯算法爆搜。)
如果求最小数, 那么两层循环的先后顺序就无所谓了

递推公式场景:
都用的1D数组
问背包装满最大价值: dp[j] = max(dp[j], dp[j - weight[i]] + value[i])
问能否能装满背包(或者最多装多少): dp[j] = max(dp[j], dp[j - nums[i]] + nums[i])
问装满背包所有物品的最小个数: dp[j] = min(dp[j], dp[j - coins[i]] + 1)
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
与其把dp[i - 1]这一层拷贝到dp[i]上, 不如只用一个一维数组了, 只用dp[j](一维数组, 也可以理解是一个滚动数组)。
这就是滚动数组的由来, 需要满足的条件是上一层可以重复利用, 直接拷贝到当前层。
读到这里估计大家都忘了 dp[i][j]里的i和j表达的是什么了, i是物品, j是背包容量。
dp[i][j] 表示从下标为[0-i]的物品里任意取, 放进容量为j的背包, 价值总和最大是多少

dp[j]可以通过dp[j - weight[i]]推导出来, dp[j - weight[i]]表示容量为j - weight[i]的背包所背的最大价值。
dp[j - weight[i]] + value[i] 表示 容量为 j - 物品i重量 的背包 加上 物品i的价值。
也就是容量为j的背包, 放入物品i了之后的价值即:dp[j])、
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
在动态规划:关于01背包问题,你该了解这些! 中我们讲解二维dp数组01背包先遍历物品还是先遍历背包都是可以的,且第二层for循环是从小到大遍历。
和动态规划:关于01背包问题,你该了解这些!(滚动数组) 中,我们讲解一维dp数组01背包只能先遍历物品再遍历背包容量,且第二层for循环是从大到小遍历。
一维dp数组的背包在遍历顺序上和二维dp数组实现的01背包其实是有很大差异的,大家需要注意!

完全背包
说完01背包,再看看完全背包。

在动态规划:关于完全背包,你该了解这些! 中,讲解了纯完全背包的一维dp数组实现,先遍历物品还是先遍历背包都是可以的,且第二层for循环是从小到大遍历。
但是仅仅是纯完全背包的遍历顺序是这样的,题目稍有变化,两个for循环的先后顺序就不一样了。

如果求组合数就是外层for循环遍历物品,内层for遍历背包。
如果求排列数就是外层for遍历背包,内层for循环遍历物品。

相关题目如下:
求组合数:动态规划:518.零钱兑换II
求排列数:动态规划:377. 组合总和 Ⅳ 、动态规划:70. 爬楼梯进阶版(完全背包)
如果求最小数,那么两层for循环的先后顺序就无所谓了,相关题目如下:
求最小数:动态规划:322. 零钱兑换 、动态规划:279.完全平方数
对于背包问题,其实递推公式算是容易的,难是难在遍历顺序上,如果把遍历顺序搞透,才算是真正理解了。
"""

# 背包递推公式
# 问能否能装满背包(或者最多装多少):dp[j] = max(dp[j], dp[j - nums[i]] + nums[i]); ,对应题目如下:

# 动态规划:416.分割等和子集
# 动态规划:1049.最后一块石头的重量 II
# 问装满背包有几种方法:dp[j] += dp[j - nums[i]] ,对应题目如下:

# 动态规划:494.目标和
# 动态规划:518. 零钱兑换 II
# 动态规划:377.组合总和Ⅳ
# 动态规划:70. 爬楼梯进阶版(完全背包)
# 问背包装满最大价值:dp[j] = max(dp[j], dp[j - weight[i]] + value[i]); ,对应题目如下:

# 动态规划:474.一和零
# 问装满背包所有物品的最小个数:dp[j] = min(dp[j - coins[i]] + 1, dp[j]); ,对应题目如下:

# 动态规划:322.零钱兑换
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
最后在问, 两个for循环的先后是否可以颠倒? 为什么? 这个简单的完全背包问题, 估计就可以难住不少候选人了。
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
动态规划基础
关于动态规划,你该了解这些!
动态规划:斐波那契数
动态规划:爬楼梯
动态规划:使用最小花费爬楼梯
动态规划:不同路径
动态规划:不同路径还不够,要有障碍!
动态规划:整数拆分,你要怎么拆？
动态规划:不同的二叉搜索树

背包问题系列
动态规划:关于01背包问题,你该了解这些!
动态规划:关于01背包问题,你该了解这些!(滚动数组)
动态规划:分割等和子集可以用01背包!
动态规划:最后一块石头的重量 II
动态规划:目标和!
动态规划:一和零!
动态规划:关于完全背包,你该了解这些!
动态规划:给你一些零钱,你要怎么凑？
动态规划:Carl称它为排列总和!
动态规划:以前我没得选,现在我选择再爬一次!
动态规划: 给我个机会,我再兑换一次零钱
动态规划:一样的套路,再求一次完全平方数
动态规划:单词拆分
动态规划:关于多重背包,你该了解这些!
听说背包问题很难？ 这篇总结篇来拯救你了

打家劫舍系列
动态规划:开始打家劫舍!
动态规划:继续打家劫舍!
动态规划:还要打家劫舍!

股票系列
动态规划:买卖股票的最佳时机
动态规划:本周我们都讲了这些(系列六)
动态规划:买卖股票的最佳时机II
动态规划:买卖股票的最佳时机III
动态规划:买卖股票的最佳时机IV
动态规划:最佳买卖股票时机含冷冻期
动态规划:本周我们都讲了这些(系列七)
动态规划:买卖股票的最佳时机含手续费
动态规划:股票系列总结篇

子序列系列
动态规划:最长递增子序列
动态规划:最长连续递增序列
动态规划:最长重复子数组
动态规划:最长公共子序列
动态规划:不相交的线
动态规划:最大子序和
动态规划:判断子序列
动态规划:不同的子序列
动态规划:两个字符串的删除操作
动态规划:编辑距离
为了绝杀编辑距离,我做了三步铺垫,你都知道么？
动态规划:回文子串
动态规划:最长回文子序列

关于动规,还有 树形DP(打家劫舍系列里有一道),数位DP,区间DP ,概率型DP,博弈型DP,状态压缩dp等等等,这些我就不去做讲解了,面试中出现的概率非常低。
能把本篇中列举的题目都研究通透的话,你的动规水平就已经非常高了。 对付面试已经足够!
"""


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

"""
1. 动态规划基础 (7)

关于动态规划,你该了解这些!
动态规划:斐波那契数
动态规划:爬楼梯
动态规划:使用最小花费爬楼梯
动态规划:不同路径
动态规划:不同路径还不够,要有障碍!
动态规划:整数拆分,你要怎么拆？
动态规划:不同的二叉搜索树
"""
#1 (Easy) 509.斐波那契数
    # 斐波那契数, 通常用 F(n) 表示, 形成的序列称为 斐波那契数列 。
    # 该数列由 0 和 1 开始, 后面的每一项数字都是前面两项数字的和。
    # 也就是: F(0) = 0, F(1) = 1 F(n) = F(n - 1) + F(n - 2), 其中 n > 1 给你n , 请计算 F(n) 。
# 递归实现
class Solution:
    def fib(self, n: int) -> int:
        if n < 2:
            return n
        return self.fib(n - 1) + self.fib(n - 2)
# 动态规划(版本一)
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
# *** 动态规划(版本二)
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
# 动态规划(版本三)
# class Solution:
#     def fib(self, n: int) -> int:
#         if n <= 1:
#             return n
        
#         prev1, prev2 = 0, 1
        
#         for _ in range(2, n + 1):
#             curr = prev1 + prev2
#             prev1, prev2 = prev2, curr
#         return prev2


#X2 (Easy) 70.爬楼梯
    # 假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
    # 每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
    # 注意:给定 n 是一个正整数。
# 动态规划(版本一)
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
# 动态规划(版本二)
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

# Climbing Stairs
# Determine the number of distinct ways to climb a staircase of n steps by taking either 1 or 2 steps at a time.
def climbing_stairs_bottom_up(n: int) -> int:
    if n <= 2:
        return n
    dp = [0] * (n + 1)
    # Base cases.
    dp[1], dp[2] = 1, 2
    # Starting from step 3, calculate the number of ways to reach each 
    # step until the n-th step.
    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]

def climbing_stairs_bottom_up_optimized(n: int) -> int:
    if n <= 2:
        return n
    # Set 'one_step_before' and 'two_steps_before' as the base cases.
    one_step_before, two_steps_before = 2, 1
    for i in range(3, n + 1):
        # Calculate the number of ways to reach the current step.
        current = one_step_before + two_steps_before
        # Update the values for the next iteration.
        two_steps_before = one_step_before
        one_step_before = current
    return one_step_before


#3 ??? (Easy) 746.使用最小花费爬楼梯
    # 数组的每个下标作为一个阶梯, 第 i 个阶梯对应着一个非负数的体力花费值 cost[i](下标从 0 开始)。
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
# 动态规划(版本一)
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        dp = [0] * (len(cost) + 1)
        dp[0] = 0  # 初始值,表示从起点开始不需要花费体力
        dp[1] = 0  # 初始值,表示经过第一步不需要花费体力
        
        for i in range(2, len(cost) + 1):
            # 在第i步,可以选择从前一步(i-1)花费体力到达当前步,或者从前两步(i-2)花费体力到达当前步
            # 选择其中花费体力较小的路径,加上当前步的花费,更新dp数组
            dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])
        
        return dp[len(cost)]  # 返回到达楼顶的最小花费
# 动态规划(版本二)
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        dp0 = 0  # 初始值,表示从起点开始不需要花费体力
        dp1 = 0  # 初始值,表示经过第一步不需要花费体力
        
        for i in range(2, len(cost) + 1):
            # 在第i步,可以选择从前一步(i-1)花费体力到达当前步,或者从前两步(i-2)花费体力到达当前步
            # 选择其中花费体力较小的路径,加上当前步的花费,得到当前步的最小花费
            dpi = min(dp1 + cost[i - 1], dp0 + cost[i - 2])
            
            dp0 = dp1  # 更新dp0为前一步的值,即上一次循环中的dp1
            dp1 = dpi  # 更新dp1为当前步的最小花费
        
        return dp1  # 返回到达楼顶的最小花费
# # 动态规划(版本三)
# class Solution:
#     def minCostClimbingStairs(self, cost: List[int]) -> int:
#         dp = [0] * len(cost)
#         dp[0] = cost[0]  # 第一步有花费
#         dp[1] = cost[1]
#         for i in range(2, len(cost)):
#             dp[i] = min(dp[i - 1], dp[i - 2]) + cost[i]
#         # 注意最后一步可以理解为不用花费,所以取倒数第一步,第二步的最少值
#         return min(dp[-1], dp[-2])
# # 动态规划(版本四)
# class Solution:
#     def minCostClimbingStairs(self, cost: List[int]) -> int:
#         n = len(cost)
#         prev_1 = cost[0]  # 前一步的最小花费
#         prev_2 = cost[1]  # 前两步的最小花费
#         for i in range(2, n):
#             current = min(prev_1, prev_2) + cost[i]  # 当前位置的最小花费
#             prev_1, prev_2 = prev_2, current  # 更新前一步和前两步的最小花费
#         return min(prev_1, prev_2)  # 最后一步可以理解为不用花费,取倒数第一步和第二步的最少值


#4 (Medium) 62.不同路径
    # 一个机器人位于一个 m x n 网格的左上角 (起始点在下图中标记为 “Start” )。
    # 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角(在下图中标记为 “Finish” )。
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
# *** 动态规划(版本一) 2D
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
# 动态规划(版本二) 1D
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


#5 (Medium) 63.不同路径 II
    # 一个机器人位于一个 m x n 网格的左上角 (起始点在下图中标记为“Start” )。
    # 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角(在下图中标记为“Finish”)。
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
# 动态规划(版本一)二维数组
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
                break # breack will stop the for loop
        for j in range(n):
            if obstacleGrid[0][j] == 0:
                dp[0][j] = 1
            else:
                break
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] == 1:
                    continue # continue will go to the next iteration in loop
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[m - 1][n - 1]
# 动态规划(版本二)
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
# 动态规划(版本三)
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
# *** 动态规划(版本四)一维数组
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
# 动态规划(版本五)
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


#6 ??? (Medium) 343.整数拆分
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
            # 假设对正整数 i 拆分出的第一个正整数是 j(1 <= j < i), 则有以下两种方案:
            # 1) 将 i 拆分成 j 和 i−j 的和, 且 i−j 不再拆分成多个正整数, 此时的乘积是 j * (i-j)
            # 2) 将 i 拆分成 j 和 i−j 的和, 且 i−j 继续拆分成多个正整数, 此时的乘积是 j * dp[i-j]
            for j in range(1, i - 1):
                # dp[i] = max(dp[i], max(j * (i - j), j * dp[i - j]))
                dp[i] = max(dp[i], (i - j) * j, dp[i - j] * j)
        return dp[n]
# 动态规划(版本一)
# class Solution:
#          # 假设对正整数 i 拆分出的第一个正整数是 j(1 <= j < i),则有以下两种方案:
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
# 动态规划(版本二)
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


#7 (Medium) 96.不同的二叉搜索树
    # 给定一个整数 n, 求以 1 ... n 为节点组成的二叉搜索树有多少种？
# dp[i] += dp[j - 1] * dp[i - j]
# 在上面的分析中,其实已经看出其递推关系, dp[i] += dp[以j为头结点左子树节点数量] * dp[以j为头结点右子树节点数量]
# j相当于是头结点的元素,从1遍历到i为止。
# 所以递推公式:dp[i] += dp[j - 1] * dp[i - j]; ,j-1 为j为头结点左子树节点数量,i-j 为以j为头结点右子树节点数量
# 1, 2, ..., j, ..., i, in this list, left of j is j-1, right of j is i-j
class Solution:
    def numTrees(self, n: int) -> int:
        dp = [0] * (n + 1)  # 创建一个长度为n+1的数组,初始化为0
        dp[0] = 1  # 当n为0时,只有一种情况,即空树,所以dp[0] = 1
        for i in range(1, n + 1):  # 遍历从1到n的每个数字
            for j in range(1, i + 1):  # 对于每个数字i,计算以i为根节点的二叉搜索树的数量
                dp[i] += dp[j - 1] * dp[i - j]  # 利用动态规划的思想,累加左子树和右子树的组合数量
        return dp[n]  # 返回以1到n为节点的二叉搜索树的总数量


"""
2. 背包问题系列 (13)

动态规划:关于01背包问题,你该了解这些!
动态规划:关于01背包问题,你该了解这些!(滚动数组)
动态规划:分割等和子集可以用01背包!
动态规划:最后一块石头的重量 II
动态规划:目标和!
动态规划:一和零!
动态规划:关于完全背包,你该了解这些!
动态规划:给你一些零钱,你要怎么凑？
动态规划:Carl称它为排列总和!
动态规划:以前我没得选,现在我选择再爬一次!
动态规划: 给我个机会,我再兑换一次零钱
动态规划:一样的套路,再求一次完全平方数
动态规划:单词拆分
动态规划:关于多重背包,你该了解这些!
听说背包问题很难？ 这篇总结篇来拯救你了
"""
#X8 01 背包
    # 有n件物品和一个最多能背重量为w 的背包。第i件物品的重量是weight[i], 得到的价值是value[i] 。每件物品只能用一次, 
    # 求解将哪些物品装入背包里物品价值总和最大。
    # dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight[i]] + value[i])
    # 大家可以看出, 虽然两个for循环遍历的次序不同, 但是dp[i][j]所需要的数据就是左上角, 根本不影响dp[i][j]公式的推导!
    # 其实背包问题里, 两个for循环的先后循序是非常有讲究的, 理解遍历顺序其实比理解推导公式难多了。
# 0/1 Knapsack
# You are a thief planning to rob a store. However, you can only carry a knapsack with a 
# maximum capacity of cap units. Each item (i) in the store has a weight (weights[i]) and a value (values[i]).
# Find the maximum total value of items you can carry in your knapsack.
# Example:
# Input: cap = 7, weights = [5, 3, 4, 1], values = [70, 50, 40, 10]
# Output: 90
# Explanation: The most valuable combination of items that can fit in the knapsack together are items 1 and 2 . These items have a combined value of 50 + 40 = 90 and a total weight of 3 + 4 = 7 , which fits within the knapsack's capacity.
def knapsack(cap: int, weights: List[int], values: List[int]) -> int:
    n = len(values)
    # Base case: Set the first column and last row to 0 by
    # initializing the entire DP table to 0.
    dp = [[0 for x in range(cap + 1)] for x in range(n + 1)]
    # Populate the DP table.
    for i in range(n - 1, -1, -1):
        for c in range(1, cap + 1):
            # If the item 'i' fits in the current knapsack capacity, 
            # the maximum value at 'dp[i][c]' is the largest of either:
            # 1. The maximum value if we include item 'i'.
            # 2. The maximum value if we exclude item 'i'.
            if weights[i] <= c:
                dp[i][c] = max(values[i] + dp[i + 1][c - weights[i]], dp[i + 1][c])
            # If it doesn't fit, we have to exclude it.
            else:
                dp[i][c] = dp[i + 1][c]
    return dp[0][cap]

def knapsack_optimized(cap: int, weights: List[int], values: List[int]) -> int:
    n = len(values)
    # Initialize 'prev_row' as the DP values of the row below the 
    # current row.
    prev_row = [0] * (cap + 1)
    for i in range(n - 1, -1, -1):
        # Set the first cell of the 'curr_row' to 0 to set the base 
        # case for this row. This is done by initializing the entire 
        # row to 0.
        curr_row = [0] * (cap + 1)
        for c in range(1, cap + 1):
            # If item 'i' fits in the current knapsack capacity, the 
            # maximum value at 'curr_row[c]' is the largest of either:
            # 1. The maximum value if we include item 'i'.
            # 2. The maximum value if we exclude item 'i'.
            if weights[i] <= c:
                curr_row[c] = max(values[i] + prev_row[c - weights[i]], prev_row[c])
            # If item 'i' doesn't fit, we exclude it.
            else:
                curr_row[c] = prev_row[c]
        # Set 'prev_row' to 'curr_row' values for the next iteration.
        prev_row = curr_row
    return prev_row[cap]

def test_2_wei_bag_problem1(bag_size, weight, value) -> int: 
	rows, cols = len(weight), bag_size + 1
	# dp = [[0 for _ in range(cols)] for _ in range(rows)]
	dp = [[0]*len(cols) for _ in range(rows)]
    
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
    # 倒序遍历是为了保证物品i只被放入一次!。但如果一旦正序遍历了, 那么物品0就会被重复加入多次!
    # ************
    # 多次放入的根本原因是,新行的值依赖于上一行当前位置和前面位置。如果只用一维数组,从左到右的话,会覆盖前面位置
    # 此时前面位置的值是新行,而不是上一行的值,所以造成重复放入。根本原因还是没有找到上一行的对应值,造成计算错误
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


#10 (Medium) 416.分割等和子集
    # 给定一个只包含正整数的非空数组。是否可以将这个数组分割成两个子集, 使得两个子集的元素和相等。
    # 注意: 每个数组中的元素不会超过 100 数组的大小不会超过 200
    # 示例 1: 输入: [1, 5, 11, 5] 输出: true 解释: 数组可以分割成 [1, 5, 5] 和 [11].
# 这道题目是要找是否可以将这个数组分割成两个子集,使得两个子集的元素和相等。
# 那么只要找到集合里能够出现 sum / 2 的子集总和,就算是可以分割成两个相同元素和子集了。

# 即一个商品如果可以重复多次放入是完全背包,而只能放入一次是01背包,写法还是不一样的。
# 要明确本题中我们要使用的是01背包,因为元素我们只能用一次。
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
            # 如果使用一维dp数组,物品遍历的for循环放在外层,遍历背包的for循环放在内层,且内层for循环倒序遍历!
            for j in range(target, num-1, -1): # 每一个元素一定是不可重复放入,所以从大到小遍历
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

#         # 初始化第一行(空子集可以得到和为0)
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


#11 (Medium) 1049.最后一块石头的重量II
    # 有一堆石头,每块石头的重量都是正整数。
    # 每一回合,从中选出任意两块石头,然后将它们一起粉碎。假设石头的重量分别为 x 和 y,且 x <= y。那么粉碎的可能结果如下:
    # 如果 x == y,那么两块石头都会被完全粉碎；
    # 如果 x != y,那么重量为 x 的石头将会完全粉碎,而重量为 y 的石头新重量为 y-x。
    # 最后,最多只会剩下一块石头。返回此石头最小的可能重量。如果没有石头剩下,就返回 0。
    # 示例:
    # 输入:[2,7,4,1,8,1]
    # 输出:1
    # 解释:
    # 组合 2 和 4,得到 2,所以数组转化为 [2,7,1,8,1],
    # 组合 7 和 8,得到 1,所以数组转化为 [2,1,1,1],
    # 组合 2 和 1,得到 1,所以数组转化为 [1,1,1],
    # 组合 1 和 1,得到 0,所以数组转化为 [1],这就是最优值。
# 本题其实就是尽量让石头分成重量相同的两堆,相撞之后剩下的石头最小,这样就化解成01背包问题了。
# 是不是感觉和昨天讲解的416. 分割等和子集 (opens new window)非常像了。
# 本题物品的重量为stones[i],物品的价值也为stones[i]。
# 对应着01背包里的物品重量weight[i]和 物品价值value[i]。
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
# 卡哥版(简化版)
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


#12 (Medium) 494.目标和
    # 给定一个非负整数数组,a1, a2, ..., an, 和一个目标数,S。现在你有两个符号 + 和 -。对于数组中的任意一个整数,你都可以从 + 或 -中选择一个符号添加在前面。
    # 返回可以使最终数组和为目标数 S 的所有添加符号的方法数。
    # 示例:
    # 输入:nums: [1, 1, 1, 1, 1], S: 3
    # 输出:5
    # 解释:
    # -1+1+1+1+1 = 3
    # +1-1+1+1+1 = 3
    # +1+1-1+1+1 = 3
    # +1+1+1-1+1 = 3
    # +1+1+1+1-1 = 3
    # 一共有5种方法让最终目标和为3。
# 如果仅仅是求个数的话,就可以用dp,但回溯算法:39. 组合总和要求的是把所有组合列出来,还是要使用回溯法爆搜的。
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


#13 (Medium) 474.一和零
    # 给你一个二进制字符串数组 strs 和两个整数 m 和 n 。
    # 请你找出并返回 strs 的最大子集的大小,该子集中 最多 有 m 个 0 和 n 个 1 。
    # 示例 1:
    # 输入:strs = ["10", "0001", "111001", "1", "0"], m = 5, n = 3
    # 输出:4
    # 解释:最多有 5 个 0 和 3 个 1 的最大子集是 {"10","0001","1","0"} ,因此答案是 4 。 其他满足题意但较小的子集包括 {"0001","1"} 和 {"10","1","0"} 。{"111001"} 不满足题意,因为它含 4 个 1 ,大于 n 的值 3 。
    # 示例 2:
    # 输入:strs = ["10", "0", "1"], m = 1, n = 1
    # 输出:2
    # 解释:最大的子集是 {"0", "1"} ,所以答案是 2 。
# 本题中strs 数组里的元素就是物品,每个物品都是一个!
# 而m 和 n相当于是一个背包,两个维度的背包。
# 有同学可能想,那个遍历背包容量的两层for循环先后循序有没有什么讲究？
# 没讲究,都是物品重量的一个维度,先遍历哪个都行!
"""
# dp[i][j]:最多有i个0和j个1的strs的最大子集的大小为dp[i][j]。
dp[i][j] 可以由前一个strs里的字符串推导出来,strs里的字符串有zeroNum个0,oneNum个1。
dp[i][j] 就可以是 dp[i - zeroNum][j - oneNum] + 1。
然后我们在遍历的过程中,取dp[i][j]的最大值。
所以递推公式:dp[i][j] = max(dp[i][j], dp[i - zeroNum][j - oneNum] + 1);
此时大家可以回想一下01背包的递推公式:dp[j] = max(dp[j], dp[j - weight[i]] + value[i]);
对比一下就会发现,字符串的zeroNum和oneNum相当于物品的重量(weight[i]),字符串本身的个数相当于物品的价值(value[i])。
这就是一个典型的01背包! 只不过物品的重量有了两个维度而已。
"""
# DP(版本一)
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
# DP(版本二)
class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        dp = [[0] * (n + 1) for _ in range(m + 1)]  # 创建二维动态规划数组,初始化为0
        # 遍历物品,两个1D数组,所以需要从后往前遍历
        for s in strs:
            ones = s.count('1')  # 统计字符串中1的个数
            zeros = s.count('0')  # 统计字符串中0的个数
            # 遍历背包容量且从后向前遍历
            for i in range(m, zeros - 1, -1):
                for j in range(n, ones - 1, -1):
                    dp[i][j] = max(dp[i][j], dp[i - zeros][j - ones] + 1)  # 状态转移方程
        return dp[m][n]


#14 动态规划:完全背包理论基础
    # 我们知道01背包内嵌1D的循环是从大到小遍历,为了保证每个物品仅被添加一次。
    # 而完全背包的物品是可以添加多次的,所以要从小到大去遍历
    # 01背包中二维dp数组的两个for遍历的先后循序是可以颠倒了
    # 一维dp数组的两个for循环先后循序一定是先遍历物品,再遍历背包容量。
    # 在完全背包中,对于一维dp数组来说,其实两个for循环嵌套顺序是无所谓的!
# 先遍历物品,再遍历背包(无参版)
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
# 先遍历物品,再遍历背包(有参版)
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
# 先遍历背包,再遍历物品(无参版)
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
# 先遍历背包,再遍历物品(有参版)
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


#15 (Medium) 518.零钱兑换II
    # 给定不同面额的硬币和一个总金额。写出函数来计算可以凑成总金额的硬币组合数。假设每一种面额的硬币有无限个。
    # 示例 1:
    # 输入: amount = 5, coins = [1, 2, 5]
    # 输出: 4
    # 解释: 有四种方式可以凑成总金额:
    # 5=5
    # 5=2+2+1
    # 5=2+1+1+1
    # 5=1+1+1+1+1
    # 示例 2:
    # 输入: amount = 3, coins = [2]
    # 输出: 0
    # 解释: 只用面额2的硬币不能凑成总金额3。
    # 示例 3:
    # 输入: amount = 10, coins = [10]
    # 输出: 1
# 这是一道典型的背包问题,一看到钱币数量不限,就知道这是一个完全背包。
# 但本题和纯完全背包不一样,纯完全背包是凑成背包最大价值是多少,而本题是要求凑成总金额的物品组合个数！
# 组合不强调元素之间的顺序,排列强调元素之间的顺序。
# dp[j] 就是所有的dp[j - coins[i]](考虑coins[i]的情况)相加。
# 所以递推公式:dp[j] += dp[j - coins[i]];
# 这个递推公式大家应该不陌生了,我在讲解01背包题目的时候在这篇494. 
# 目标和中就讲解了,求装满背包有几种方法,公式都是:dp[j] += dp[j - nums[i]];

# 我在动态规划:关于完全背包,你该了解这些！中讲解了完全背包的两个for循环的先后顺序都是可以的。
# 但本题就不行了！
# 因为纯完全背包求得装满背包的最大价值是多少,和凑成总和的元素有没有顺序没关系,即:有顺序也行,没有顺序也行！
# 而本题要求凑成总和的组合数,元素之间明确要求没有顺序。
# 所以纯完全背包是能凑成总和就行,不用管怎么凑的。
# 本题是求凑出来的方案个数,且每个方案个数是为组合数。

# 我们先来看 外层for循环遍历物品(钱币),内层for遍历背包(金钱总额)的情况。
# 假设:coins[0] = 1,coins[1] = 5。
# 那么就是先把1加入计算,然后再把5加入计算,得到的方法数量只有{1, 5}这种情况。而不会出现{5, 1}的情况。
# 所以这种遍历顺序中dp[j]里计算的是组合数！
# 如果把两个for交换顺序
# 背包容量的每一个值,都是经过 1 和 5 的计算,包含了{1, 5} 和 {5, 1}两种情况。

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


#16 (Medium) 377.组合总和 Ⅳ
    # 给定一个由正整数组成且不存在重复数字的数组，找出和为给定目标正整数的组合的个数。
    # 示例:
    # nums = [1, 2, 3]
    # target = 4
    # 所有可能的组合为： (1, 1, 1, 1) (1, 1, 2) (1, 2, 1) (1, 3) (2, 1, 1) (2, 2) (3, 1)
    # 请注意，顺序不同的序列被视作不同的组合。
    # 因此输出为 7。
# 如果把遍历nums(物品)放在外循环,遍历target的作为内循环的话
# 举一个例子:计算dp[4]的时候,结果集只有 {1,3} 这样的集合,不会有{3,1}这样的集合
# 因为nums遍历放在外层,3只能出现在1后面!
# 所以本题遍历顺序最终遍历顺序:target(背包)放在外循环,将nums(物品)放在内循环,内循环从前到后遍历。
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


#17 (Easy) 70.爬楼梯(进阶版)
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
class Solution {
public:
    int climbStairs(int n) {
        vector<int> dp(n + 1, 0);
        dp[0] = 1;
        for (int i = 1; i <= n; i++) { // 遍历背包
            for (int j = 1; j <= m; j++) { // 遍历物品
                if (i - j >= 0) dp[i] += dp[i - j];
            }
        }
        return dp[n];
    }
};


#18 (Medium) 322.零钱兑换
    # 给定不同面额的硬币 coins 和一个总金额 amount。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回 -1。
    # 你可以认为每种硬币的数量是无限的。
    # 示例 1：
    # 输入：coins = [1, 2, 5], amount = 11
    # 输出：3
    # 解释：11 = 5 + 5 + 1
    # 示例 2：
    # 输入：coins = [2], amount = 3
    # 输出：-1
    # 示例 3：
    # 输入：coins = [1], amount = 0
    # 输出：0
    # 示例 4：
    # 输入：coins = [1], amount = 1
    # 输出：1
    # 示例 5：
    # 输入：coins = [1], amount = 2
    # 输出：2
# 本题求钱币最小个数，那么钱币有顺序和没有顺序都可以，都不影响钱币的最小个数。
# 所以本题并不强调集合是组合还是排列。
# 如果求组合数就是外层for循环遍历物品，内层for遍历背包。
# 如果求排列数就是外层for遍历背包，内层for循环遍历物品。
# 在动态规划专题我们讲过了求组合数是动态规划：518.零钱兑换II，求排列数是动态规划：377. 组合总和 Ⅳ。
# 本题的两个for循环的关系是：外层for循环遍历物品，内层for遍历背包或者外层for遍历背包，内层for循环遍历物品都是可以的！
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
# 先遍历物品 后遍历背包(优化版)
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0

        for coin in coins:
            for i in range(coin, amount + 1): # 进行优化,从能装得下的背包开始计算,则不需要进行比较
                # 更新凑成金额 i 所需的最少硬币数量
                dp[i] = min(dp[i], dp[i - coin] + 1)

        return dp[amount] if dp[amount] != float('inf') else -1
# 先遍历背包 后遍历物品(优化版)
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


#19 (Medium) 279.完全平方数
    # 给定正整数 n，找到若干个完全平方数（比如 1, 4, 9, 16, ...）使得它们的和等于 n。你需要让组成和的完全平方数的个数最少。
    # 给你一个整数 n ，返回和为 n 的完全平方数的 最少数量 。
    # 完全平方数 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，1、4、9 和 16 都是完全平方数，而 3 和 11 不是。
    # 示例 1：
    # 输入：n = 12
    # 输出：3
    # 解释：12 = 4 + 4 + 4
    # 示例 2：
    # 输入：n = 13
    # 输出：2
    # 解释：13 = 4 + 9
# 在动态规划：322. 零钱兑换中我们就深入探讨了这个问题，本题也是一样的，是求最小数！
# 所以本题外层for遍历背包，内层for遍历物品，还是外层for遍历物品，内层for遍历背包，都是可以的！
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


#20 ??? (Medium) 139.单词拆分
    # 给定一个非空字符串 s 和一个包含非空单词的列表 wordDict，判定 s 是否可以被空格拆分为一个或多个在字典中出现的单词。
    # 说明：
    # 拆分时可以重复使用字典中的单词。
    # 你可以假设字典中没有重复的单词。
    # 示例 1：
    # 输入: s = "leetcode", wordDict = ["leet", "code"]
    # 输出: true
    # 解释: 返回 true 因为 "leetcode" 可以被拆分成 "leet code"。
    # 示例 2：
    # 输入: s = "applepenapple", wordDict = ["apple", "pen"]
    # 输出: true
    # 解释: 返回 true 因为 "applepenapple" 可以被拆分成 "apple pen apple"。
    # 注意你可以重复使用字典中的单词。
    # 示例 3：
    # 输入: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
    # 输出: false
# dp[i] : 字符串长度为i的话，dp[i]为true，表示可以拆分为一个或多个在字典中出现的单词。
# 如果确定dp[j] 是true，且 [j, i] 这个区间的子串出现在字典里，那么dp[i]一定是true。（j < i ）。
# 所以递推公式是 if([j, i] 这个区间的子串出现在字典里 && dp[j]是true) 那么 dp[i] = true。
# 本题其实我们求的是排列数，为什么呢。 拿 s = "applepenapple", wordDict = ["apple", "pen"] 举例。
# "apple", "pen" 是物品，那么我们要求 物品的组合一定是 "apple" + "pen" + "apple" 才能组成 "applepenapple"。
# "apple" + "apple" + "pen" 或者 "pen" + "apple" + "apple" 是不可以的，那么我们就是强调物品之间顺序。
# 所以说，本题一定是 先遍历 背包，再遍历物品。

# 求组合数:动态规划:518.零钱兑换II 
# 求排列数:动态规划:377. 组合总和 Ⅳ 、动态规划:70. 爬楼梯进阶版(完全背包) 
# 求最小数:动态规划:322. 零钱兑换 、动态规划:279.完全平方数
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
# DP(版本一)
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
# DP(版本二)
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


"""
3. 打家劫舍系列 (3)

动态规划:开始打家劫舍!
动态规划:继续打家劫舍!
动态规划:还要打家劫舍!
"""
#X21 (Medium) 198.打家劫舍
    # 你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。
    # 给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。
    # 示例 1：
    # 输入：[1,2,3,1]
    # 输出：4
    # 解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。   偷窃到的最高金额 = 1 + 3 = 4 。
    # 示例 2：
    # 输入：[2,7,9,3,1]
    # 输出：12 解释：偷窃 1 号房屋 (金额 = 2), 偷窃 3 号房屋 (金额 = 9)，接着偷窃 5 号房屋 (金额 = 1)。   偷窃到的最高金额 = 2 + 9 + 1 = 12 。
# dp[i]：考虑下标i（包括i）以内的房屋，最多可以偷窃的金额为dp[i]。
# 决定dp[i]的因素就是第i房间偷还是不偷。
# 如果偷第i房间，那么dp[i] = dp[i - 2] + nums[i] ，即：第i-1房一定是不考虑的，找出 下标i-2（包括i-2）以内的房屋，最多可以偷窃的金额为dp[i-2] 加上第i房间偷到的钱。
# 如果不偷第i房间，那么dp[i] = dp[i - 1]，即考 虑i-1房，（注意这里是考虑，并不是一定要偷i-1房，这是很多同学容易混淆的点）
# 然后dp[i]取最大值，即dp[i] = max(dp[i - 2] + nums[i], dp[i - 1]);
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
            # 注意这里是考虑,并不是一定要偷i-1,i-2房,这是很多同学容易混淆的点
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
    
# Neighborhood Burglary
    # You plan to rob houses in a street where each house stores a certain amount of money. 
    # The neighborhood has a security system that sets off an alarm when two adjacent houses are robbed. 
    # Return the maximum amount of cash that can be stolen without triggering the alarms.
def neighborhood_burglary(houses: List[int]) -> int:
    # Handle the cases when the array is less than the size of 2 to 
    # avoid out-of-bounds errors when assigning the base case values.
    if not houses:
        return 0
    if len(houses) == 1:
        return houses[0]
    dp = [0] * len(houses)
    # Base case: when there's only one house, rob that house.
    dp[0] = houses[0]
    # Base case: when there are two houses, rob the one with the most 
    # money.
    dp[1] = max(houses[0], houses[1])
    # Fill in the rest of the DP array.
    for i in range(2, len(houses)):
        # 'dp[i]' = max(profit if we skip house 'i', profit if we rob 
        # house 'i').
        dp[i] = max(dp[i - 1], houses[i] + dp[i - 2])
    return dp[len(houses) - 1]


def neighborhood_burglary_optimized(houses: List[int]) -> int:
    if not houses:
        return 0
    if len(houses) == 1:
        return houses[0]
    # Initialize the variables with the base cases.
    prev_max_profit = max(houses[0], houses[1])
    prev_prev_max_profit = houses[0]
    for i in range(2, len(houses)):
        curr_max_profit = max(prev_max_profit, houses[i] + prev_prev_max_profit)
        # Update the values for the next iteration.
        prev_prev_max_profit = prev_max_profit
        prev_max_profit = curr_max_profit
    return prev_max_profit


#22 (Medium) 213.打家劫舍II
    # 你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都 围成一圈 ，这意味着第一个房屋和最后一个房屋是紧挨着的。同时，相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警 。
    # 给定一个代表每个房屋存放金额的非负整数数组，计算你 在不触动警报装置的情况下 ，能够偷窃到的最高金额。
    # 示例 1：
    # 输入：nums = [2,3,2]
    # 输出：3
    # 解释：你不能先偷窃 1 号房屋（金额 = 2），然后偷窃 3 号房屋（金额 = 2）, 因为他们是相邻的。
    # 示例 2：
    # 输入：nums = [1,2,3,1]
    # 输出：4
    # 解释：你可以先偷窃 1 号房屋（金额 = 1），然后偷窃 3 号房屋（金额 = 3）。偷窃到的最高金额 = 1 + 3 = 4 。
    # 示例 3：
    # 输入：nums = [0]
    # 输出：0
# 情况一：考虑不包含首尾元素
# 情况二：考虑包含首元素，不包含尾元素
# 情况三：考虑包含尾元素，不包含首元素
# 注意我这里用的是"考虑"，例如情况三，虽然是考虑包含尾元素，但不一定要选尾部元素！ 对于情况三，取nums[1] 和 nums[3]就是最大的。
# 而情况二 和 情况三 都包含了情况一了，所以只考虑情况二和情况三就可以了。
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
        # Don't miss this case
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


#23 (Medium) 337.打家劫舍 III
    # 在上次打劫完一条街道之后和一圈房屋后，小偷又发现了一个新的可行窃的地区。
    # 这个地区只有一个入口，我们称之为“根”。 除了“根”之外，每栋房子有且只有一个“父“房子与之相连。
    # 一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。 
    # 如果两个直接相连的房子在同一天晚上被打劫，房屋将自动报警。
    # 计算在不触动警报的情况下，小偷一晚能够盗取的最高金额。
# 这里我们要求一个节点 偷与不偷的两个状态所得到的金钱，那么返回值就是一个长度为2的数组。
# 所以dp数组（dp table）以及下标的含义：下标为0记录不偷该节点所得到的的最大金钱，下标为1记录偷该节点所得到的的最大金钱。
# 所以本题dp数组就是一个长度为2的数组！
# 首先明确的是使用后序遍历。 因为要通过递归函数的返回值来做下一步计算。
# 通过递归左节点，得到左节点偷与不偷的金钱。
# 通过递归右节点，得到右节点偷与不偷的金钱。
# 最后当前节点的状态就是{val0, val1}; 即：{不偷当前节点得到的最大金钱，偷当前节点得到的最大金钱}
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
        # dp数组(dp table)以及下标的含义:
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


"""
4. 股票系列 (6)

动态规划:买卖股票的最佳时机
动态规划:本周我们都讲了这些(系列六)
动态规划:买卖股票的最佳时机II
动态规划:买卖股票的最佳时机III
动态规划:买卖股票的最佳时机IV
动态规划:最佳买卖股票时机含冷冻期
动态规划:本周我们都讲了这些(系列七)
动态规划:买卖股票的最佳时机含手续费
动态规划:股票系列总结篇
"""
#24 (Easy) 121. 买卖股票的最佳时机
    # 给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。
    # 你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。
    # 返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。
    # 示例 1：
    # 输入：[7,1,5,3,6,4]
    # 输出：5
    # 解释：在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。
    # 示例 2：
    # 输入：prices = [7,6,4,3,1]
    # 输出：0
    # 解释：在这种情况下, 没有交易完成, 所以最大利润为 0。
# dp数组的含义:
# dp[i][0] 表示第i天持有股票所得现金。
# dp[i][1] 表示第i天不持有股票所得最多现金
# 注意这里说的是“持有”,“持有”不代表就是当天“买入”!也有可能是昨天就买入了,今天保持持有的状态
# 很多同学把“持有”和“买入”没区分清楚。
# 如果第i天持有股票即dp[i][0]， 那么可以由两个状态推出来
# 第i-1天就持有股票，那么就保持现状，所得现金就是昨天持有股票的所得现金 即：dp[i - 1][0]
# 第i天买入股票，所得现金就是买入今天的股票后所得现金即：-prices[i]
# 那么dp[i][0]应该选所得现金最大的，所以dp[i][0] = max(dp[i - 1][0], -prices[i]);
# 如果第i天不持有股票即dp[i][1]， 也可以由两个状态推出来
# 第i-1天就不持有股票，那么就保持现状，所得现金就是昨天不持有股票的所得现金 即：dp[i - 1][1]
# 第i天卖出股票，所得现金就是按照今天股票价格卖出后所得现金即：prices[i] + dp[i - 1][0]
# 同样dp[i][1]取最大的，dp[i][1] = max(dp[i - 1][1], prices[i] + dp[i - 1][0]);
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
# 动态规划:版本二
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
# 动态规划:版本三
# class Solution:
#     def maxProfit(self, prices: List[int]) -> int:
#         length = len(prices)
#         dp0, dp1 = -prices[0], 0 #注意这里只维护两个常量,因为dp0的更新不受dp1的影响
#         for i in range(1, length):
#             dp1 = max(dp1, dp0 + prices[i])
#             dp0 = max(dp0, -prices[i])
#         return dp1


#25 (Medium) 122.买卖股票的最佳时机II
    # 给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
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
    # 示例 3:
    # 输入: [7,6,4,3,1]
    # 输出: 0
    # 解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。
# 这里重申一下dp数组的含义：
# dp[i][0] 表示第i天持有股票所得现金。
# dp[i][1] 表示第i天不持有股票所得最多现金
# 如果第i天持有股票即dp[i][0]， 那么可以由两个状态推出来
# 第i-1天就持有股票，那么就保持现状，所得现金就是昨天持有股票的所得现金 即：dp[i - 1][0]
# 第i天买入股票，所得现金就是昨天不持有股票的所得现金减去 今天的股票价格 即：dp[i - 1][1] - prices[i]
# 注意这里和121. 买卖股票的最佳时机 (opens new window)唯一不同的地方，就是推导dp[i][0]的时候，第i天买入股票的情况。
# 在121. 买卖股票的最佳时机 (opens new window)中，因为股票全程只能买卖一次，所以如果买入股票，那么第i天持有股票即dp[i][0]一定就是 -prices[i]。
# 而本题，因为一只股票可以买卖多次，所以当第i天买入股票的时候，所持有的现金可能有之前买卖过的利润。
# 那么第i天持有股票即dp[i][0]，如果是第i天买入股票，所得现金就是昨天不持有股票的所得现金 减去 今天的股票价格 即：dp[i - 1][1] - prices[i]。
# 再来看看如果第i天不持有股票即dp[i][1]的情况， 依然可以由两个状态推出来
# 第i-1天就不持有股票，那么就保持现状，所得现金就是昨天不持有股票的所得现金 即：dp[i - 1][1]
# 第i天卖出股票，所得现金就是按照今天股票价格卖出后所得现金即：prices[i] + dp[i - 1][0]
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


#26 (Hard) 123.买卖股票的最佳时机III
    # 给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。
    # 设计一个算法来计算你所能获取的最大利润。你最多可以完成 两笔 交易。
    # 注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
    # 示例 1:
    # 输入：prices = [3,3,5,0,0,3,1,4]
    # 输出：6 解释：在第 4 天（股票价格 = 0）的时候买入，在第 6 天（股票价格 = 3）的时候卖出，这笔交易所能获得利润 = 3-0 = 3 。随后，在第 7 天（股票价格 = 1）的时候买入，在第 8 天 （股票价格 = 4）的时候卖出，这笔交易所能获得利润 = 4-1 = 3。
    # 示例 2：
    # 输入：prices = [1,2,3,4,5]
    # 输出：4 解释：在第 1 天（股票价格 = 1）的时候买入，在第 5 天 （股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4。注意你不能在第 1 天和第 2 天接连购买股票，之后再将它们卖出。因为这样属于同时参与了多笔交易，你必须在再次购买前出售掉之前的股票。
    # 示例 3：
    # 输入：prices = [7,6,4,3,1]
    # 输出：0 解释：在这个情况下, 没有交易完成, 所以最大利润为0。
    # 示例 4：
    # 输入：prices = [1] 输出：0
# 确定dp数组以及下标的含义
# 一天一共就有五个状态,
# 没有操作 (其实我们也可以不设置这个状态)
# 第一次持有股票
# 第一次不持有股票
# 第二次持有股票
# 第二次不持有股票
# dp[i][j]中 i表示第i天，j为 [0 - 4] 五个状态，dp[i][j]表示第i天状态j所剩最大现金。
# 需要注意：dp[i][1]，表示的是第i天，买入股票的状态，并不是说一定要第i天买入股票，这是很多同学容易陷入的误区。
# 例如 dp[i][1] ，并不是说 第i天一定买入股票，有可能 第 i-1天 就买入了，那么 dp[i][1] 延续买入股票的这个状态。
# 达到dp[i][1]状态，有两个具体操作：
# 操作一：第i天买入股票了，那么dp[i][1] = dp[i-1][0] - prices[i]
# 操作二：第i天没有操作，而是沿用前一天买入的状态，即：dp[i][1] = dp[i - 1][1]
# 那么dp[i][1]究竟选 dp[i-1][0] - prices[i]，还是dp[i - 1][1]呢？
# 一定是选最大的，所以 dp[i][1] = max(dp[i-1][0] - prices[i], dp[i - 1][1]);
# 同理dp[i][2]也有两个操作：
# 操作一：第i天卖出股票了，那么dp[i][2] = dp[i - 1][1] + prices[i]
# 操作二：第i天没有操作，沿用前一天卖出股票的状态，即：dp[i][2] = dp[i - 1][2]
# 所以dp[i][2] = max(dp[i - 1][1] + prices[i], dp[i - 1][2])
# 同理可推出剩下状态部分：
# dp[i][3] = max(dp[i - 1][3], dp[i - 1][2] - prices[i]);
# dp[i][4] = max(dp[i - 1][4], dp[i - 1][3] + prices[i]);
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


#27 (Hard) 188.买卖股票的最佳时机IV
    # 给定一个整数数组 prices ，它的第 i 个元素 prices[i] 是一支给定的股票在第 i 天的价格。
    # 设计一个算法来计算你所能获取的最大利润。你最多可以完成 k 笔交易。
    # 注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
    # 示例 1：
    # 输入：k = 2, prices = [2,4,1]
    # 输出：2 解释：在第 1 天 (股票价格 = 2) 的时候买入，在第 2 天 (股票价格 = 4) 的时候卖出，这笔交易所能获得利润 = 4-2 = 2。
    # 示例 2：
    # 输入：k = 2, prices = [3,2,6,5,0,3]
    # 输出：7 解释：在第 2 天 (股票价格 = 2) 的时候买入，在第 3 天 (股票价格 = 6) 的时候卖出, 这笔交易所能获得利润 = 6-2 = 4。随后，在第 5 天 (股票价格 = 0) 的时候买入，在第 6 天 (股票价格 = 3) 的时候卖出, 这笔交易所能获得利润 = 3-0 = 3 。
# 在动态规划：123.买卖股票的最佳时机III (opens new window)中，我是定义了一个二维dp数组，本题其实依然可以用一个二维dp数组。
# 使用二维数组 dp[i][j] ：第i天的状态为j，所剩下的最大现金是dp[i][j]
# j的状态表示为：
# 0 表示不操作
# 1 第一次买入
# 2 第一次卖出
# 3 第二次买入
# 4 第二次卖出
# .....
# 大家应该发现规律了吧 ，除了0以外，偶数就是卖出，奇数就是买入。
# 题目要求是至多有K笔交易，那么j的范围就定义为 2 * k + 1 就可以了。
# 还要强调一下：dp[i][1]，表示的是第i天，买入股票的状态，并不是说一定要第i天买入股票，这是很多同学容易陷入的误区。
# 达到dp[i][1]状态，有两个具体操作：
# 操作一：第i天买入股票了，那么dp[i][1] = dp[i - 1][0] - prices[i]
# 操作二：第i天没有操作，而是沿用前一天买入的状态，即：dp[i][1] = dp[i - 1][1]
# 选最大的，所以 dp[i][1] = max(dp[i - 1][0] - prices[i], dp[i - 1][1]);
# 同理dp[i][2]也有两个操作：
# 操作一：第i天卖出股票了，那么dp[i][2] = dp[i - 1][1] + prices[i]
# 操作二：第i天没有操作，沿用前一天卖出股票的状态，即：dp[i][2] = dp[i - 1][2]
# 所以dp[i][2] = max(dp[i - 1][1] + prices[i], dp[i - 1][2])
# *** 版本一 2D
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
# 版本二 1D
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        if len(prices) == 0: return 0
        dp = [0] * (2*k + 1)
        for i in range(1,2*k,2):
            dp[i] = -prices[0]
        for i in range(1,len(prices)):
            for j in range(1,2*k + 1):
                if j % 2:
                    dp[j] = max(dp[j],dp[j-1]-prices[i])
                else:
                    dp[j] = max(dp[j],dp[j-1]+prices[i])
        return dp[2*k]


#28 ??? (Medium) 309.最佳买卖股票时机含冷冻期
    # 给定一个整数数组，其中第 i 个元素代表了第 i 天的股票价格 。
    # 设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:
    # 你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
    # 卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。
    # 示例:
    # 输入: [1,2,3,0,2]
    # 输出: 3
    # 解释: 对应的交易状态为: [买入, 卖出, 冷冻期, 买入, 卖出]
# dp[i][j]，第i天状态为j，所剩的最多现金为dp[i][j]。
# 其实本题很多同学搞的比较懵，是因为出现冷冻期之后，状态其实是比较复杂度，例如今天买入股票、今天卖出股票、今天是冷冻期，都是不能操作股票的。
# 具体可以区分出如下四个状态：
# 状态一：持有股票状态（今天买入股票，或者是之前就买入了股票然后没有操作，一直持有）
# 不持有股票状态，这里就有两种卖出股票状态
# 状态二：保持卖出股票的状态（两天前就卖出了股票，度过一天冷冻期。或者是前一天就是卖出股票状态，一直没操作）
# 状态三：今天卖出股票
# 状态四：今天为冷冻期状态，但冷冻期状态不可持续，只有一天！
# 达到买入股票状态（状态一）即：dp[i][0]，有两个具体操作：
# 操作一：前一天就是持有股票状态（状态一），dp[i][0] = dp[i - 1][0]
# 操作二：今天买入了，有两种情况
# 前一天是冷冻期（状态四），dp[i - 1][3] - prices[i]
# 前一天是保持卖出股票的状态（状态二），dp[i - 1][1] - prices[i]
# 那么dp[i][0] = max(dp[i - 1][0], dp[i - 1][3] - prices[i], dp[i - 1][1] - prices[i]);
# 达到保持卖出股票状态（状态二）即：dp[i][1]，有两个具体操作：
# 操作一：前一天就是状态二
# 操作二：前一天是冷冻期（状态四）
# dp[i][1] = max(dp[i - 1][1], dp[i - 1][3]);
# 达到今天就卖出股票状态（状态三），即：dp[i][2] ，只有一个操作：
# 昨天一定是持有股票状态（状态一），今天卖出
# 即：dp[i][2] = dp[i - 1][0] + prices[i];
# 达到冷冻期状态（状态四），即：dp[i][3]，只有一个操作：
# 昨天卖出了股票（状态三）
# dp[i][3] = dp[i - 1][2];
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
# 优化
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        if n == 0:
            return 0
        dp = [[0] * 4 for _ in range(2)]  # 创建动态规划数组,4个状态分别表示持有股票、不持有股票且处于冷冻期、不持有股票且不处于冷冻期、不持有股票且当天卖出后处于冷冻期
        dp[0][0] = -prices[0]  # 初始状态:第一天持有股票的最大利润为买入股票的价格
        for i in range(1, n):
            dp[i%2][0] = max(dp[((i-1)%2)%2][0], max(dp[(i-1)%2][3], dp[(i-1)%2][1]) - prices[i])  # 当前持有股票的最大利润等于前一天持有股票的最大利润或者前一天不持有股票且不处于冷冻期的最大利润减去当前股票的价格
            dp[i%2][1] = max(dp[(i-1)%2][1], dp[(i-1)%2][3])  # 当前不持有股票且处于冷冻期的最大利润等于前一天持有股票的最大利润加上当前股票的价格
            dp[i%2][2] = dp[(i-1)%2][0] + prices[i]  # 当前不持有股票且不处于冷冻期的最大利润等于前一天不持有股票的最大利润或者前一天处于冷冻期的最大利润
            dp[i%2][3] = dp[(i-1)%2][2]  # 当前不持有股票且当天卖出后处于冷冻期的最大利润等于前一天不持有股票且不处于冷冻期的最大利润
        return max(dp[(n-1)%2][3], dp[(n-1)%2][1], dp[(n-1)%2][2])  # 返回最后一天不持有股票的最大利润
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


#29 (Medium) 714.买卖股票的最佳时机含手续费
    # 给定一个整数数组 prices，其中第 i 个元素代表了第 i 天的股票价格 ；非负整数 fee 代表了交易股票的手续费用。
    # 你可以无限次地完成交易，但是你每笔交易都需要付手续费。如果你已经购买了一个股票，在卖出它之前你就不能再继续购买股票了。
    # 返回获得利润的最大值。
    # 注意：这里的一笔交易指买入持有并卖出股票的整个过程，每笔交易你只需要为支付一次手续费。
    # 示例 1:
    # 输入: prices = [1, 3, 2, 8, 4, 9], fee = 2
    # 输出: 8
    # 解释: 能够达到的最大利润:
    # 在此处买入 prices[0] = 1
    # 在此处卖出 prices[3] = 8
    # 在此处买入 prices[4] = 4
    # 在此处卖出 prices[5] = 9
    # 总利润: ((8 - 1) - 2) + ((9 - 4) - 2) = 8.
# dp[i][0] 表示第i天持有股票所省最多现金。 dp[i][1] 表示第i天不持有股票所得最多现金
# 如果第i天持有股票即dp[i][0]， 那么可以由两个状态推出来
# 第i-1天就持有股票，那么就保持现状，所得现金就是昨天持有股票的所得现金 即：dp[i - 1][0]
# 第i天买入股票，所得现金就是昨天不持有股票的所得现金减去 今天的股票价格 即：dp[i - 1][1] - prices[i]
# 所以：dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] - prices[i]);
# 在来看看如果第i天不持有股票即dp[i][1]的情况， 依然可以由两个状态推出来
# 第i-1天就不持有股票，那么就保持现状，所得现金就是昨天不持有股票的所得现金 即：dp[i - 1][1]
# 第i天卖出股票，所得现金就是按照今天股票价格卖出后所得现金，注意这里需要有手续费了即：dp[i - 1][0] + prices[i] - fee
# 所以：dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] + prices[i] - fee);
# 注意:这里的一笔交易指买入持有并卖出股票的整个过程,每笔交易你只需要为支付一次手续费。
# 卖出的时刻才需要交fee
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        n = len(prices)
        dp = [[0] * 2 for _ in range(n)]
        dp[0][0] = -prices[0] #持股票
        for i in range(1, n):
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] - prices[i])
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] + prices[i] - fee)
        return max(dp[-1][0], dp[-1][1])


"""
5. 子序列系列 (12)

动态规划:最长递增子序列
动态规划:最长连续递增序列
动态规划:最长重复子数组
动态规划:最长公共子序列
动态规划:不相交的线
动态规划:最大子序和
动态规划:判断子序列
动态规划:不同的子序列
动态规划:两个字符串的删除操作
动态规划:编辑距离
为了绝杀编辑距离,我做了三步铺垫,你都知道么？
动态规划:回文子串
动态规划:最长回文子序列
"""
#30 (Medium) 300.最长递增子序列 (不连续)
    # 给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。
    # 子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。
    # 示例 1：
    # 输入：nums = [10,9,2,5,3,7,101,18]
    # 输出：4
    # 解释：最长递增子序列是 [2,3,7,101]，因此长度为 4 。
    # 示例 2：
    # 输入：nums = [0,1,0,3,2,3]
    # 输出：4
    # 示例 3：
    # 输入：nums = [7,7,7,7,7,7,7]
    # 输出：1
# 子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序
# dp[i]表示i之前包括i的以nums[i]结尾的最长递增子序列的长度
# 为什么一定表示 “以nums[i]结尾的最长递增子序” ，因为我们在 做 递增比较的时候，如果比较 nums[j] 和 nums[i] 的大小，
# 那么两个递增子序列一定分别以nums[j]为结尾 和 nums[i]为结尾， 要不然这个比较就没有意义了，不是尾部元素的比较那么 如何算递增呢。
# 位置i的最长升序子序列等于j从0到i-1各个位置的最长升序子序列 + 1 的最大值。
# 所以：if (nums[i] > nums[j]) dp[i] = max(dp[i], dp[j] + 1)
# 注意这里不是要dp[i] 与 dp[j] + 1进行比较，而是我们要取dp[j] + 1的最大值。
# DP
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if len(nums) <= 1:
            return len(nums)
        # dp[i]表示i之前包括i的以nums[i]结尾的最长递增子序列的长度
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


#31 (Easy) 674.最长连续递增序列 (连续)
    # 给定一个未经排序的整数数组，找到最长且 连续递增的子序列，并返回该序列的长度。
    # 连续递增的子序列 可以由两个下标 l 和 r（l < r）确定，如果对于每个 l <= i < r，都有 nums[i] < nums[i + 1] ，那么子序列 [nums[l], nums[l + 1], ..., nums[r - 1], nums[r]] 就是连续递增子序列。
    # 示例 1：
    # 输入：nums = [1,3,5,4,7]
    # 输出：3
    # 解释：最长连续递增序列是 [1,3,5], 长度为3。尽管 [1,3,5,7] 也是升序的子序列, 但它不是连续的，因为 5 和 7 在原数组里被 4 隔开。
    # 示例 2：
    # 输入：nums = [2,2,2,2,2]
    # 输出：1
    # 解释：最长连续递增序列是 [2], 长度为1。
# dp[i]：以下标i为结尾的连续递增的子序列长度为dp[i]。
# 注意这里的定义，一定是以下标i为结尾，并不是说一定以下标0为起始位置。
# 如果 nums[i] > nums[i - 1]，那么以 i 为结尾的连续递增的子序列长度 一定等于 以i - 1为结尾的连续递增的子序列长度 + 1 。
# 即：dp[i] = dp[i - 1] + 1;
# 概括来说：不连续递增子序列的跟前0-i 个状态有关，连续递增的子序列只跟前一个状态有关
# DP
class Solution:
    def findLengthOfLCIS(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0
        result = 1
        # dp[i]：以下标i为结尾的连续递增的子序列长度为dp[i]。
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


#X32 *** (Medium) 718.最长重复子数组 (连续)
# 两个整数数组
    # 给两个整数数组 A 和 B ，返回两个数组中公共的、长度最长的子数组sub array的长度。
    # 示例：
    # 输入：
    # A: [1,2,3,2,1]
    # B: [3,2,1,4,7]
    # 输出：3
    # 解释：长度最长的公共子数组是 [3, 2, 1] 。
# dp[i][j] ：以下标i - 1为结尾的A，和以下标j - 1为结尾的B，最长重复子数组长度为dp[i][j]。 
# （特别注意： “以下标i - 1为结尾的A” 标明一定是 以A[i-1]为结尾的字符串 ）
# 此时细心的同学应该发现，那dp[0][0]是什么含义呢？总不能是以下标-1为结尾的A数组吧。
# 其实dp[i][j]的定义也就决定着，我们在遍历dp[i][j]的时候i 和 j都要从1开始。
# 那有同学问了，我就定义dp[i][j]为 以下标i为结尾的A，和以下标j 为结尾的B，最长重复子数组长度。不行么？
# 行倒是行！ 但实现起来就麻烦一点，需要单独处理初始化部分，在本题解下面的拓展内容里，我给出了 第二种 
# dp数组的定义方式所对应的代码和讲解，大家比较一下就了解了。
# 根据dp[i][j]的定义，dp[i][j]的状态只能由dp[i - 1][j - 1]推导出来。
# 即当A[i - 1] 和B[j - 1]相等的时候，dp[i][j] = dp[i - 1][j - 1] + 1;
# 注意题目中说的子数组,其实就是连续子序列。
# 2维DP
class Solution:
    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        # 创建一个二维数组 dp,用于存储最长公共子数组的长度
        # dp[i][j] ：以下标i - 1为结尾的A，和以下标j - 1为结尾的B，最长重复子数组长度为dp[i][j]
        dp = [[0] * (len(nums2) + 1) for _ in range(len(nums1) + 1)]
        # 记录最长公共子数组的长度
        result = 0

        # 遍历数组 nums1
        for i in range(1, len(nums1) + 1):
            # 遍历数组 nums2
            for j in range(1, len(nums2) + 1):
                # 如果 nums1[i-1] 和 nums2[j-1] 相等
                if nums1[i - 1] == nums2[j - 1]:
                    # 在当前位置上的最长公共子数组长度为前一个位置上的长度加一
                    dp[i][j] = dp[i - 1][j - 1] + 1
                # 更新最长公共子数组的长度
                if dp[i][j] > result:
                    result = dp[i][j]

        # 返回最长公共子数组的长度
        return result
# 1维DP，内层遍历需要从后向前
class Solution:
    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        dp = [0] * (len(nums2) + 1)
        result = 0

        for i in range(1, len(nums1) + 1):
            # for j in range(1, len(nums2) + 1):
            for j in range(len(nums2), 0, -1):
                if nums1[i - 1] == nums2[j - 1]:
                    dp[j] = dp[j - 1] + 1
                else:
                    dp[j] = 0 #注意这里不相等的时候要有赋0的操作
                result = max(result, dp[j])

        # 返回最长公共子数组的长度
        return result
# *** 1维DP 用prev, curr指针，内层遍历就不用从后向前了
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


#X33 (Medium) 1143.最长公共子序列 (不连续)
# 两个字符串
    # 给定两个字符串 text1 和 text2，返回这两个字符串的最长公共子序列的长度。
    # 一个字符串的 子序列 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）
    # 后组成的新字符串。
    # 例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。两个字符串的「公共子序列」
    # 是这两个字符串所共同拥有的子序列。
    # 若这两个字符串没有公共子序列，则返回 0。
    # 示例 1:
    # 输入：text1 = "abcde", text2 = "ace"
    # 输出：3
    # 解释：最长公共子序列是 "ace"，它的长度为 3。
    # 示例 2:
    # 输入：text1 = "abc", text2 = "abc"
    # 输出：3
    # 解释：最长公共子序列是 "abc"，它的长度为 3。
    # 示例 3:
    # 输入：text1 = "abc", text2 = "def"
    # 输出：0
    # 解释：两个字符串没有公共子序列，返回 0。
# 本题和动态规划:718. 最长重复子数组 区别在于这里不要求是连续的了,
# 但要有相对顺序,即:"ace" 是 "abcde" 的子序列,但 "aec" 不是 "abcde" 的子序列。
# dp[i][j]：长度为[0, i - 1]的字符串text1与长度为[0, j - 1]的字符串text2的最长公共子序列为dp[i][j]
# 有同学会问：为什么要定义长度为[0, i - 1]的字符串text1，定义为长度为[0, i]的字符串text1不香么？
# 这样定义是为了后面代码实现方便，如果非要定义为长度为[0, i]的字符串text1也可以，
# 我在 动态规划：718. 最长重复子数组中的「拓展」里 详细讲解了区别所在，其实就是简化了dp数组第一行和第一列的初始化逻辑。

# 主要就是两大情况： text1[i - 1] 与 text2[j - 1]相同，text1[i - 1] 与 text2[j - 1]不相同
# 如果text1[i - 1] 与 text2[j - 1]相同，那么找到了一个公共元素，所以dp[i][j] = dp[i - 1][j - 1] + 1;
# 如果text1[i - 1] 与 text2[j - 1]不相同，那就看看text1[0, i - 2]与text2[0, j - 1]的最长公共子序列 和 text1[0, i - 1]与text2[0, j - 2]的最长公共子序列，取最大的。
# 即：dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
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
        # dp[i][j]：长度为[0, i - 1]的字符串text1与长度为[0, j - 1]的字符串text2的最长公共子序列为dp[i][j]
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
                    # 这里dp[j]对应dp[i - 1][j], dp[j-1]对应dp[i][j - 1]
                    dp[j] = max(dp[j], dp[j - 1]) 
                prev = curr  # 更新上一个位置的最长公共子序列长度
        
        return dp[n]  # 返回最后一个位置的最长公共子序列长度作为结果

# Longest Common Subsequence
    # Given two strings, find the length of their longest common subsequence (LCS). 
    # A subsequence is a sequence of characters that can be derived from a string by 
    # deleting zero or more elements, without changing the order of the remaining elements.
    # Example:
    # Input: s1 = 'acabac', s2 = 'aebab'
    # Output: 3
def longest_common_subsequence(s1: str, s2: str) -> int:
    # Base case: Set the last row and last column to 0 by
    # initializing the entire DP table with 0s.
    dp = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
    # Populate the DP table.
    for i in range(len(s1) - 1, -1, -1):
        for j in range(len(s2) - 1, -1, -1):
            # If the characters match, the length of the LCS at
            # 'dp[i][j]' is  1 + the LCS length of the remaining
            # substrings.
            if s1[i] == s2[j]:
                dp[i][j] = 1 + dp[i + 1][j + 1]
            # If the characters don't match, the LCS length at
            # 'dp[i][j]' can be found by either:
            # 1. Excluding the current character of s1.
            # 2. Excluding the current character of s2.
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    return dp[0][0]

def longest_common_subsequence_optimized(s1: str, s2: str) -> int:
    # Initialize 'prev_row' as the DP values of the last row.
    prev_row = [0] * (len(s2) + 1)
    for i in range(len(s1) - 1, -1, -1):
        # Set the last cell of 'curr_row' to 0 to set the base case for 
        # this row. This is done by initializing the entire row to 0.
        curr_row = [0] * (len(s2) + 1)
        for j in range(len(s2) - 1, -1, -1):
            # If the characters match, the length of the LCS at
            # 'curr_row[j]' is 1 + the LCS length of the remaining
            # substrings ('prev_row[j + 1]').
            if s1[i] == s2[j]:
                curr_row[j] = 1 + prev_row[j + 1]
            # If the characters don't match, the LCS length at
            # 'curr_row[j]' can be found by either:
            # 1. Excluding the current character of s1 ('prev_row[j]').
            # 2. Excluding the current character of s2 
            # ('curr_row[j + 1]').
            else:
                curr_row[j] = max(prev_row[j], curr_row[j + 1])
        # Update 'prev_row' with 'curr_row' values for the next 
        # iteration.
        prev_row = curr_row
    return prev_row[0]


#34 (Medium) 1035.不相交的线 (不连续)
    # 我们在两条独立的水平线上按给定的顺序写下 A 和 B 中的整数。
    # 现在，我们可以绘制一些连接两个数字 A[i] 和 B[j] 的直线，只要 A[i] == B[j]，且我们绘制的直线不与任何其他连线（非水平线）相交。
    # 以这种方法绘制线条，并返回我们可以绘制的最大连线数。
# 本题说是求绘制的最大连线数，其实就是求两个字符串的最长公共子序列的长度！
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


#35 (Medium) 53.最大子序和 (连续)
    # 给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
    # 示例:
    # 输入: [-2,1,-3,4,-1,2,1,-5,4]
    # 输出: 6
    # 解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
# dp[i]：包括下标i（以nums[i]为结尾）的最大连续子序列和为dp[i]。
# dp[i]只有两个方向可以推出来：
# dp[i - 1] + nums[i]，即：nums[i]加入当前连续子序列和
# nums[i]，即：从头开始计算当前连续子序列和
# 一定是取最大的，所以dp[i] = max(dp[i - 1] + nums[i], nums[i]);
# 从递推公式可以看出来dp[i]是依赖于dp[i - 1]的状态，dp[0]就是递推公式的基础。
# dp[0]应该是多少呢?
# 根据dp[i]的定义，很明显dp[0]应为nums[0]即dp[0] = nums[0]。
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        # dp[i]：包括下标i（以nums[i]为结尾），所以长度为len(nums)
        dp = [0] * len(nums)
        dp[0] = nums[0]
        result = dp[0]
        for i in range(1, len(nums)):
            dp[i] = max(dp[i-1] + nums[i], nums[i]) #状态转移公式
            result = max(result, dp[i]) #result 保存dp[i]的最大值
        return result


#36 (Easy) 392.判断子序列
    # 给定字符串 s 和 t ，判断 s 是否为 t 的子序列。
    # 字符串的一个子序列是原始字符串删除一些（也可以不删除）字符而不改变剩余字符相对位置形成的新字符串。（例如，"ace"是"abcde"的一个子序列，而"aec"不是）。
    # 示例 1：
    # 输入：s = "abc", t = "ahbgdc"
    # 输出：true
    # 示例 2：
    # 输入：s = "axc", t = "ahbgdc"
    # 输出：false
# dp[i][j] 表示以下标i-1为结尾的字符串s，和以下标j-1为结尾的字符串t，相同子序列的长度为dp[i][j]。
# if (s[i - 1] == t[j - 1])，那么dp[i][j] = dp[i - 1][j - 1] + 1;，因为找到了一个相同的字符，相同子序列长度自然要在dp[i-1][j-1]的基础上加1（如果不理解，在回看一下dp[i][j]的定义）
# if (s[i - 1] != t[j - 1])，此时相当于t要删除元素，t如果把当前元素t[j - 1]删除，那么dp[i][j] 的数值就是 看s[i - 1]与 t[j - 2]的比较结果了，即：dp[i][j] = dp[i][j - 1];
# 其实这里 大家可以发现和 1143.最长公共子序列 (opens new window)的递推公式基本那就是一样的，区别就是 本题 如果删元素一定是字符串t，而 1143.最长公共子序列 是两个字符串都可以删元素。
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        # dp[i][j]对应的是i-1和j-1，所以要到i+1，j+1
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


#37 *** (Hard) 115.不同的子序列
    # 给定一个字符串 s 和一个字符串 t ，计算在 s 的子序列中 t 出现的个数。
    # 字符串的一个 子序列 是指，通过删除一些（也可以不删除）字符且不干扰剩余字符相对位置所组成的新字符串。（例如，"ACE" 是 "ABCDE" 的一个子序列，而 "AEC" 不是）
    # 题目数据保证答案符合 32 位带符号整数范围。
# 这道题目如果不是子序列,而是要求连续序列的,那就可以考虑用KMP。
# dp[i][j]：以i-1为结尾的s子序列中出现以j-1为结尾的t的个数为dp[i][j]。
# 这一类问题，基本是要分析两种情况
# s[i - 1] 与 t[j - 1]相等
# s[i - 1] 与 t[j - 1] 不相等
# 当s[i - 1] 与 t[j - 1]相等时，dp[i][j]可以有两部分组成。
# 一部分是用s[i - 1]来匹配，那么个数为dp[i - 1][j - 1]。即不需要考虑当前s子串和t子串的最后一位字母，所以只需要 dp[i-1][j-1]。
# 一部分是不用s[i - 1]来匹配，个数为dp[i - 1][j]。
# 这里可能有录友不明白了，为什么还要考虑 不用s[i - 1]来匹配，都相同了指定要匹配啊。
# 例如： s：bagg 和 t：bag ，s[3] 和 t[2]是相同的，但是字符串s也可以不用s[3]来匹配，即用s[0]s[1]s[2]组成的bag。
# 当然也可以用s[3]来匹配，即：s[0]s[1]s[3]组成的bag。
# 所以当s[i - 1] 与 t[j - 1]相等时，dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j];
# 当s[i - 1] 与 t[j - 1]不相等时，dp[i][j]只有一部分组成，不用s[i - 1]来匹配（就是模拟在s中删除这个元素），即：dp[i - 1][j]
# 所以递推公式为：dp[i][j] = dp[i - 1][j];
# 这里可能有录友还疑惑，为什么只考虑 “不用s[i - 1]来匹配” 这种情况， 不考虑 “不用t[j - 1]来匹配” 的情况呢。
# 这里大家要明确，我们求的是 s 中有多少个 t，而不是 求t中有多少个s，所以只考虑 s中删除元素的情况，即 不用s[i - 1]来匹配 的情况。
    
# 每次当初始化的时候，都要回顾一下dp[i][j]的定义，不要凭感觉初始化。
# dp[i][0]表示什么呢？
# dp[i][0] 表示：以i-1为结尾的s可以随便删除元素，出现空字符串的个数。
# 那么dp[i][0]一定都是1，因为也就是把以i-1为结尾的s，删除所有元素，出现空字符串的个数就是1。
# 再来看dp[0][j]，dp[0][j]：空字符串s可以随便删除元素，出现以j-1为结尾的字符串t的个数。
# 那么dp[0][j]一定都是0，s如论如何也变成不了t。
# 最后就要看一个特殊位置了，即：dp[0][0] 应该是多少。
# dp[0][0]应该是1，空字符串s，可以删除0个元素，变成空字符串t。
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        # dp[i][j]对应的是i-1和j-1，所以要到i+1，j+1
        dp = [[0] * (len(t)+1) for _ in range(len(s)+1)]
        for i in range(len(s)):
            dp[i][0] = 1
        for j in range(1, len(t)):
            dp[0][j] = 0
        for i in range(1, len(s)+1):
            for j in range(1, len(t)+1):
                if s[i-1] == t[j-1]:
                    # 当s[i - 1] 与 t[j - 1]相等时,dp[i][j]可以有两部分组成。
                    # 一部分是用s[i - 1]来匹配,那么个数为dp[i - 1][j - 1]。
                    # 即不需要考虑当前s子串和t子串的最后一位字母,所以只需要 dp[i-1][j-1]。
                    # 一部分是不用s[i - 1]来匹配,个数为dp[i - 1][j]。
                    dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
                else:
                    # 当s[i - 1] 与 t[j - 1]不相等时,dp[i][j]只有一部分组成,
                    # 不用s[i - 1]来匹配(就是模拟在s中删除这个元素),即:dp[i - 1][j]
                    dp[i][j] = dp[i-1][j]
        return dp[-1][-1]

class SolutionDP2:
    """
    既然dp[i]只用到dp[i - 1]的状态,
    我们可以通过缓存dp[i - 1]的状态来对dp进行压缩,
    减少空间复杂度。
    (原理等同同于滚动数组)
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


#38 (Medium) 583.两个字符串的删除操作
    # 给定两个单词 word1 和 word2，找到使得 word1 和 word2 相同所需的最小步数，每步可以删除任意一个字符串中的一个字符。
    # 示例：
    # 输入: "sea", "eat"
    # 输出: 2
    # 解释: 第一步将"sea"变为"ea"，第二步将"eat"变为"ea"
# dp[i][j]：以i-1为结尾的字符串word1，和以j-1位结尾的字符串word2，想要达到相等，所需要删除元素的最少次数。
# 当word1[i - 1] 与 word2[j - 1]相同的时候
# 当word1[i - 1] 与 word2[j - 1]不相同的时候
# 当word1[i - 1] 与 word2[j - 1]相同的时候，dp[i][j] = dp[i - 1][j - 1];
# 当word1[i - 1] 与 word2[j - 1]不相同的时候，有三种情况：
# 情况一：删word1[i - 1]，最少操作次数为dp[i - 1][j] + 1
# 情况二：删word2[j - 1]，最少操作次数为dp[i][j - 1] + 1
# 情况三：同时删word1[i - 1]和word2[j - 1]，操作的最少次数为dp[i - 1][j - 1] + 2
# 那最后当然是取最小值，所以当word1[i - 1] 与 word2[j - 1]不相同的时候，递推公式：dp[i][j] = min({dp[i - 1][j - 1] + 2, dp[i - 1][j] + 1, dp[i][j - 1] + 1});
# 因为 dp[i][j - 1] + 1 = dp[i - 1][j - 1] + 2，所以递推公式可简化为：dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1);

# 从递推公式中，可以看出来，dp[i][0] 和 dp[0][j]是一定要初始化的。
# dp[i][0]：word2为空字符串，以i-1为结尾的字符串word1要删除多少个元素，才能和word2相同呢，很明显dp[i][0] = i。
# dp[0][j]的话同理

# ??? 这里可能不少录友有点迷糊，从字面上理解 就是 当 同时删word1[i - 1]和word2[j - 1]，dp[i][j-1] 本来就不考虑 word2[j - 1]了，那么我在删 word1[i - 1]，是不是就达到两个元素都删除的效果，即 dp[i][j-1] + 1。
# 动态规划一
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        # dp[i][j]对应的是i-1和j-1，所以要到i+1，j+1
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
                    # 因为 dp[i][j - 1] + 1 = dp[i - 1][j - 1] + 2,
                    # 所以递推公式可简化为:dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1);
                    # 这里可能不少录友有点迷糊,从字面上理解 就是 当 同时删word1[i - 1]和word2[j - 1],
                    # dp[i][j-1] 本来就不考虑 word2[j - 1]了,那么我在删 word1[i - 1],
                    # 是不是就达到两个元素都删除的效果,即 dp[i][j-1] + 1。
                    dp[i][j] = min(dp[i-1][j-1] + 2, dp[i-1][j] + 1, dp[i][j-1] + 1)
                    # dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1)
        return dp[-1][-1]
# 动态规划二
    # 本题和动态规划:1143.最长公共子序列 基本相同,
    # 只要求出两个字符串的最长公共子序列长度即可,那么除了最长公共子序列之外的字符都是必须删除的,
    # 最后用两个字符串的总长度减去两个最长公共子序列的长度就是删除的最少步数。
class Solution:
    def minDistance(self, text1: str, text2: str) -> int:
        # 创建一个二维数组 dp,用于存储最长公共子序列的长度
        # dp[i][j]对应的是i-1和j-1，所以要到i+1，j+1
        dp = [[0] * (len(text2) + 1) for _ in range(len(text1) + 1)]
        
        # 遍历 text1 和 text2,填充 dp 数组
        for i in range(1, len(text1) + 1):
            for j in range(1, len(text2) + 1):
                if text1[i - 1] == text2[j - 1]:
                    # 如果 text1[i-1] 和 text2[j-1] 相等,则当前位置的最长公共子序列长度为左上角位置的值加一
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    # 如果 text1[i-1] 和 text2[j-1] 不相等,则当前位置的最长公共子序列长度为上方或左方的较大值
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        # 返回最长公共子序列的长度
        return len(text1)+len(text2)-dp[len(text1)][len(text2)]*2


#39 (Medium) 72.编辑距离
    # 给你两个单词 word1 和 word2，请你计算出将 word1 转换成 word2 所使用的最少操作数 。
    # 你可以对一个单词进行如下三种操作：
    # 插入一个字符
    # 删除一个字符
    # 替换一个字符
    # 示例 1：
    # 输入：word1 = "horse", word2 = "ros"
    # 输出：3
    # 解释： horse -> rorse (将 'h' 替换为 'r') rorse -> rose (删除 'r') rose -> ros (删除 'e')
    # 示例 2：
    # 输入：word1 = "intention", word2 = "execution"
    # 输出：5
    # 解释： intention -> inention (删除 't') inention -> enention (将 'i' 替换为 'e') enention -> exention (将 'n' 替换为 'x') exention -> exection (将 'n' 替换为 'c') exection -> execution (插入 'u')
# dp[i][j] 表示以下标i-1为结尾的字符串word1，和以下标j-1为结尾的字符串word2，最近编辑距离为dp[i][j]。
"""
    if (word1[i - 1] == word2[j - 1])
        不操作
    if (word1[i - 1] != word2[j - 1])
        增
        删
        换
"""
# if (word1[i - 1] == word2[j - 1]) 那么说明不用任何编辑，dp[i][j] 就应该是 dp[i - 1][j - 1]，即dp[i][j] = dp[i - 1][j - 1];
# if (word1[i - 1] != word2[j - 1])，此时就需要编辑了，如何编辑呢？
# 操作一：word1删除一个元素，那么就是以下标i - 2为结尾的word1 与 j-1为结尾的word2的最近编辑距离 再加上一个操作。
# 即 dp[i][j] = dp[i - 1][j] + 1;
# 操作二：word2删除一个元素，那么就是以下标i - 1为结尾的word1 与 j-2为结尾的word2的最近编辑距离 再加上一个操作。
# 即 dp[i][j] = dp[i][j - 1] + 1;
# 这里有同学发现了，怎么都是删除元素，添加元素去哪了。
# 操作三：替换元素，word1替换word1[i - 1]，使其与word2[j - 1]相同，此时不用增删加元素。
# 可以回顾一下，if (word1[i - 1] == word2[j - 1])的时候我们的操作 是 dp[i][j] = dp[i - 1][j - 1] 对吧。
# 那么只需要一次替换的操作，就可以让 word1[i - 1] 和 word2[j - 1] 相同。
# 所以 dp[i][j] = dp[i - 1][j - 1] + 1;
# 综上，当 if (word1[i - 1] != word2[j - 1]) 时取最小的，即：dp[i][j] = min({dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]}) + 1;
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        # dp[i][j]对应的是i-1和j-1，所以要到i+1，j+1
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
                    # if (word1[i - 1] != word2[j - 1])
                    # 操作一:word1删除一个元素 即 dp[i][j] = dp[i - 1][j] + 1
                    # 操作二:word2删除一个元素 即 dp[i][j] = dp[i][j - 1] + 1
                    # 操作三:替换元素,word1替换word1[i - 1],使其与word2[j - 1]相同
                    # 此时不用增删加元素。即 dp[i][j] = dp[i - 1][j - 1] + 1
                    # 综合三种操作,取最小就是
                    dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1
        return dp[-1][-1]


#X40 (Medium) 647.回文子串 (连续)
# 一个字符串
    # 给定一个字符串，你的任务是计算这个字符串中有多少个回文子串。
    # 具有不同开始位置或结束位置的子串，即使是由相同的字符组成，也会被视作不同的子串。
    # 示例 1：
    # 输入："abc"
    # 输出：3
    # 解释：三个回文子串: "a", "b", "c"
    # 示例 2：
    # 输入："aaa"
    # 输出：6
    # 解释：6个回文子串: "a", "a", "a", "aa", "aa", "aaa"
# 回文子串是要连续的,回文子序列可不是连续的!
# 647.回文子串
# 5.最长回文子串
# 布尔类型的dp[i][j]：表示区间范围[i,j] （注意是左闭右闭）的子串是否是回文子串，如果是dp[i][j]为true，否则为false。
# 在确定递推公式时，就要分析如下几种情况。
# 整体上是两种，就是s[i]与s[j]相等，s[i]与s[j]不相等这两种。
# 当s[i]与s[j]不相等，那没啥好说的了，dp[i][j]一定是false。
# 当s[i]与s[j]相等时，这就复杂一些了，有如下三种情况
# 情况一：下标i 与 j相同，同一个字符例如a，当然是回文子串
# 情况二：下标i 与 j相差为1，例如aa，也是回文子串
# 情况三：下标：i 与 j相差大于1的时候，例如cabac，此时s[i]与s[j]已经相同了，我们看i到j区间是不是回文子串就看aba是不是回文就可以了，那么aba的区间就是 i+1 与 j-1区间，这个区间是不是回文就看dp[i + 1][j - 1]是否为true。

# 遍历顺序可有有点讲究了。
# 首先从递推公式中可以看出，情况三是根据dp[i + 1][j - 1]是否为true，在对dp[i][j]进行赋值true的。
# dp[i + 1][j - 1] 在 dp[i][j]的左下角
# 如果这矩阵是从上到下，从左到右遍历，那么会用到没有计算过的dp[i + 1][j - 1]，也就是根据不确定是不是回文的区间[i+1,j-1]，来判断了[i,j]是不是回文，那结果一定是不对的。
# 所以一定要从下到上，从左到右遍历，这样保证dp[i + 1][j - 1]都是经过计算的。
# 有的代码实现是优先遍历列，然后遍历行，其实也是一个道理，都是为了保证dp[i + 1][j - 1]都是经过计算的。
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
# class Solution:
#     def countSubstrings(self, s: str) -> int:
#         dp = [[False] * len(s) for _ in range(len(s))]
#         result = 0
#         for i in range(len(s)-1, -1, -1): #注意遍历顺序
#             for j in range(i, len(s)):
#                 if s[i] == s[j] and (j - i <= 1 or dp[i+1][j-1]): 
#                     result += 1
#                     dp[i][j] = True
#         return result
# 双指针法
# 动态规划的空间复杂度是偏高的，我们再看一下双指针法。
# 首先确定回文串，就是找中心然后向两边扩散看是不是对称的就可以了。
# 在遍历中心点的时候，要注意中心点有两种情况。
# 一个元素可以作为中心点，两个元素也可以作为中心点。
# 那么有人同学问了，三个元素还可以做中心点呢。其实三个元素就可以由一个元素左右添加元素得到，四个元素则可以由两个元素左右添加元素得到。
# 所以我们在计算的时候，要注意一个元素为中心点和两个元素为中心点的情况。
# 这两种情况可以放在一起计算，但分别计算思路更清晰，我倾向于分别计算，代码如下
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

# Longest Palindrome in a String
    # Return the longest palindromic substring within a given string.
    # Example:
    # Input: s = 'abccbaba'
    # Output: 'abccba'
def longest_palindrome_in_a_string(s: str) -> str:
    n = len(s)
    if n == 0:
        return ""
    dp = [[False] * n for _ in range(n)]
    max_len = 1
    start_index = 0
    # Base case: a single character is always a palindrome.
    for i in range(n):
        dp[i][i] = True
    # Base case: a substring of length two is a palindrome if both  
    # characters are the same.
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            max_len = 2
            start_index = i
    # Find palindromic substrings of length 3 or greater.
    for substring_len in range(3, n + 1):
        # Iterate through each substring of length 'substring_len'.
        for i in range(n - substring_len + 1):
            j = i + substring_len - 1
            # If the first and last characters are the same, and the 
            # inner substring is a palindrome, then the current 
            # substring is a palindrome.
            if s[i] == s[j] and dp[i + 1][j - 1]:
                dp[i][j] = True
                max_len = substring_len
                start_index = i
    return s[start_index : start_index + max_len]

def longest_palindrome_in_a_string_expanding(s: str) -> str:
    n = len(s)
    start, max_len = 0, 0
    for center in range(n):
        # Check for odd-length palindromes.
        odd_start, odd_length = expand_palindrome(center, center, s)
        if odd_length > max_len:
            start = odd_start
            max_len = odd_length
        # Check for even-length palindromes.
        if center < n - 1 and s[center] == s[center + 1]:
            even_start, even_length = expand_palindrome(center, center + 1, s)
            if even_length > max_len:
                start = even_start
                max_len = even_length
    return s[start : start + max_len]

# Expands outward from the center of a base case to identify the start 
# index and length of the longest palindrome that extends from this 
# base case.
def expand_palindrome(left: int, right: int, s: str) -> Tuple[int, int]:
    while left > 0 and right < len(s) - 1 and s[left - 1] == s[right + 1]:
        left -= 1
        right += 1
    return left, right - left + 1


#X41 (Medium) 516.最长回文子序列 (不连续)
# 一个字符串
    # 给定一个字符串 s ，找到其中最长的回文子序列，并返回该序列的长度。可以假设 s 的最大长度为 1000 。
    # 示例 1: 输入: "bbbab" 输出: 4 一个可能的最长回文子序列为 "bbbb"。
    # 示例 2: 输入:"cbbd" 输出: 2 一个可能的最长回文子序列为 "bb"。
# 回文子串是要连续的，回文子序列可不是连续的！ 回文子串，回文子序列都是动态规划经典题目。
# 思路其实是差不多的，但本题要比求回文子串简单一点，因为情况少了一点。
# dp[i][j]：字符串s在[i, j]范围内最长的回文子序列的长度为dp[i][j]。
# 在判断回文子串的题目中，关键逻辑就是看s[i]与s[j]是否相同。
# 如果s[i]与s[j]相同，那么dp[i][j] = dp[i + 1][j - 1] + 2;
# 如果s[i]与s[j]不相同，说明s[i]和s[j]的同时加入 并不能增加[i,j]区间回文子序列的长度，那么分别加入s[i]、s[j]看看哪一个可以组成最长的回文子序列。
# 加入s[j]的回文子序列长度为dp[i + 1][j]。
# 加入s[i]的回文子序列长度为dp[i][j - 1]。
# 那么dp[i][j]一定是取最大的，即：dp[i][j] = max(dp[i + 1][j], dp[i][j - 1]);

# 首先要考虑当i 和j 相同的情况，从递推公式：dp[i][j] = dp[i + 1][j - 1] + 2; 可以看出 递推公式是计算不到 i 和j相同时候的情况。
# 所以需要手动初始化一下，当i与j相同，那么dp[i][j]一定是等于1的，即：一个字符的回文子序列长度就是1。
# 其他情况dp[i][j]初始为0就行，这样递推公式：dp[i][j] = max(dp[i + 1][j], dp[i][j - 1]); 中dp[i][j]才不会被初始值覆盖

# 从递归公式中，可以看出，dp[i][j] 依赖于 dp[i + 1][j - 1] ，dp[i + 1][j] 和 dp[i][j - 1]，
# 所以遍历i的时候一定要从下到上遍历，这样才能保证下一行的数据是经过计算的。
# j的话，可以正常从左向右遍历。
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
        return dp[0][-1] #注意最终状态的位置，因为是从下往上，所以最后在右上角

def longest_palindrome_in_a_string(s: str) -> str:
    n = len(s)
    if n == 0:
        return ""
    dp = [[False] * n for _ in range(n)]
    max_len = 1
    start_index = 0
    # Base case: a single character is always a palindrome.
    for i in range(n):
        dp[i][i] = True
    # Base case: a substring of length two is a palindrome if both  
    # characters are the same.
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            max_len = 2
            start_index = i
    # Find palindromic substrings of length 3 or greater.
    for substring_len in range(3, n + 1):
        # Iterate through each substring of length 'substring_len'.
        for i in range(n - substring_len + 1):
            j = i + substring_len - 1
            # If the first and last characters are the same, and the 
            # inner substring is a palindrome, then the current 
            # substring is a palindrome.
            if s[i] == s[j] and dp[i + 1][j - 1]:
                dp[i][j] = True
                max_len = substring_len
                start_index = i
    return s[start_index : start_index + max_len]

def longest_palindrome_in_a_string_expanding(s: str) -> str:
    n = len(s)
    start, max_len = 0, 0
    for center in range(n):
        # Check for odd-length palindromes.
        odd_start, odd_length = expand_palindrome(center, center, s)
        if odd_length > max_len:
            start = odd_start
            max_len = odd_length
        # Check for even-length palindromes.
        if center < n - 1 and s[center] == s[center + 1]:
            even_start, even_length = expand_palindrome(center, center + 1, s)
            if even_length > max_len:
                start = even_start
                max_len = even_length
    return s[start : start + max_len]

# Expands outward from the center of a base case to identify the start 
# index and length of the longest palindrome that extends from this 
# base case.
def expand_palindrome(left: int, right: int, s: str) -> Tuple[int, int]:
    while left > 0 and right < len(s) - 1 and s[left - 1] == s[right + 1]:
        left -= 1
        right += 1
    return left, right - left + 1


# ByteByteGo 101
#X42 Minimum Coin Combination
    # You are given an array of coin values and a target amount of money.
    # Return the minimum number of coins needed to total the target amount. If this isn't possible, return ‐1. 
    # You may assume there's an unlimited supply of each coin.
    # Example 1:
    # Input: coins = [1, 2, 3], target = 5
    # Output: 2
    # Explanation: Use one 2-dollar coin and one 3-dollar coin to make 5 dollars.
    # Example 2:
    # Input: coins = [2, 4], target = 5
    # Output: -1
from typing import List

def min_coin_combination_bottom_up(coins: List[int], target: int) -> int:
    # The DP array will store the minimum number of coins needed for 
    # each amount. Set each element to a large number initially.
    dp = [float('inf')] * (target + 1)
    # Base case: if the target is 0, then 0 coins are needed.
    dp[0] = 0
    # Update the DP array for all target amounts greater than 0.
    
    for t in range(1, target + 1):
        for coin in coins:
            if coin <= t:
                dp[t] = min(dp[t], 1 + dp[t - coin])
    return dp[target] if dp[target] != float('inf') else -1


#X43 Matrix Pathways
    # You are positioned at the top-left corner of a m × n matrix, and can only move 
    # downward or rightward through the matrix. Determine the number of unique pathways 
    # you can take to reach the bottom-right corner of the matrix.
def matrix_pathways(m: int, n: int) -> int:
    # Base cases: Set all cells in row 0 and column 0 to 1. We can
    # do this by initializing all cells in the DP table to 1.
    dp = [[1] * n for _ in range(m)]
    # Fill in the rest of the DP table.
    for r in range(1, m):
        for c in range(1, n):
            # Paths to current cell = paths from above + paths from 
            # left.
            dp[r][c] = dp[r - 1][c] + dp[r][c - 1]
    return dp[m - 1][n - 1]

def matrix_pathways_optimized(m: int, n: int) -> int:
    # Initialize 'prev_row' as the DP values of row 0, which are all 1s.
    prev_row = [1] * n
    # Iterate through the matrix starting from row 1.
    for r in range(1, m):
        # Set the first cell of 'curr_row' to 1. This is done by 
        # setting the entire row to 1.
        curr_row = [1] * n
        for c in range(1, n):
            # The number of unique paths to the current cell is the sum 
            # of the paths from the cell above it ('prev_row[c]') and 
            # the cell to the left ('curr_row[c - 1]').
            curr_row[c] = prev_row[c] + curr_row[c - 1]
        # Update 'prev_row' with 'curr_row' values for the next 
        # iteration.
        prev_row = curr_row
    # The last element in 'prev_row' stores the result for the 
    # bottom-right cell.
    return prev_row[n - 1]


#X44 Maximum Subarray Sum
    # Given an array of integers, return the sum of the subarray with the largest sum.
    # Example:
    # Input: nums = [3, 1, -6, 2, -1, 4, -9]
    # Output: 5
    # Explanation: subarray [2, -1, 4] has the largest sum of 5.
    # Constraints:
    # The input array contains at least one element.
from typing import List

def maximum_subarray_sum_dp(nums: List[int]) -> int:
    n = len(nums)
    if n == 0:
        return 0
    dp = [0] * n
    # Base case: the maximum subarray sum of an array with just one
    # element is that element.
    dp[0] = nums[0]
    max_sum = dp[0]
    # Populate the rest of the DP array.
    for i in range(1, n):
        # Determine the maximum subarray sum ending at the current 
        # index.
        dp[i] = max(dp[i - 1] + nums[i], nums[i])
        max_sum = max(max_sum, dp[i])
    return max_sum

def maximum_subarray_sum_dp_optimized(nums: List[int]) -> int:
    n = len(nums)
    if n == 0:
        return 0
    current_sum = nums[0]
    max_sum = nums[0]
    for i in range(1, n):
        current_sum = max(nums[i], current_sum + nums[i])
        max_sum = max(max_sum, current_sum)
    return max_sum


#X45 Largest Square in a Matrix
    # Determine the area of the largest square of 1's in a binary matrix.
from typing import List

def largest_square_in_a_matrix(matrix: List[List[int]]) -> int:
    if not matrix:
        return 0
    m, n = len(matrix), len(matrix[0])
    dp = [[0] * n for _ in range(m)]
    max_len = 0
    # Base case: If a cell in row 0 is 1, the largest square ending there has a
    # length of 1.
    for j in range(n):
        if matrix[0][j] == 1:
            dp[0][j] = 1
            max_len = 1
    # Base case: If a cell in column 0 is 1, the largest square ending there has
    # a length of 1.
    for i in range(m):
        if matrix[i][0] == 1:
            dp[i][0] = 1
            max_len = 1
    # Populate the rest of the DP table.
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][j] == 1:
                # The length of the largest square ending here is determined by 
                # the smallest square ending at the neighboring cells (left, 
                # top-left, top), plus 1 to include this cell.
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i - 1][j - 1], dp[i][j - 1])
            max_len = max(max_len, dp[i][j])
    return max_len ** 2

def largest_square_in_a_matrix_optimized(matrix: List[List[int]]) -> int:
    if not matrix:
        return 0
    m, n = len(matrix), len(matrix[0])
    prev_row = [0] * n
    max_len = 0
    # Iterate through the matrix.
    for i in range(m):
        curr_row = [0] * n
        for j in range(n):
            # Base cases: if we’re in row 0 or column 0, the largest square ending
            # here has a length of 1. This can be set by using the value in the
            # input matrix.
            if i == 0 or j == 0:
                curr_row[j] = matrix[i][j]
            else:
                if matrix[i][j] == 1:
                      # curr_row[j] = 1 + min(left, top-left, top)
                    curr_row[j] = 1 + min(curr_row[j - 1], prev_row[j - 1], prev_row[j])
            max_len = max(max_len, curr_row[j])
        # Update 'prev_row' with 'curr_row' values for the next iteration.
        prev_row, curr_row = curr_row, [0] * n
    return max_len ** 2


#45 (Medium) 1062. Longest Repeating Substring
    # Given a string s, return the length of the longest repeating substrings. If no repeating substring exists, return 0.
    # Example 1:
    # Input: s = "abcd"
    # Output: 0
    # Explanation: There is no repeating substring.
    # Example 2:
    # Input: s = "abbaba"
    # Output: 2
    # Explanation: The longest repeating substrings are "ab" and "ba", each of which occurs twice.
    # Example 3:
    # Input: s = "aabcaabdaab"
    # Output: 3
    # Explanation: The longest repeating substring is "aab", which occurs 3 times.
# Approach 5: Dynamic Programming
# Time complexity: O(n^2)
# Space complexity: O(n^2)
class Solution:
    def longestRepeatingSubstring(self, s: str) -> int:
        length = len(s)
        dp = [[0] * (length + 1) for _ in range(length + 1)]
        max_length = 0

        # Populate the DP array
        for i in range(1, length + 1):
            for j in range(i + 1, length + 1):
                # Check if the characters match and
                # update the DP value
                if s[i - 1] == s[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    # Update max_length
                    max_length = max(max_length, dp[i][j])
        return max_length