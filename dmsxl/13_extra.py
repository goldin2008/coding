"""

"""
#1 1365.有多少小于当前数字的数字
    # 给你一个数组 nums, 对于其中每个元素 nums[i], 请你统计数组中比它小的所有数字的数目。
    # 换而言之，对于每个 nums[i] 你必须计算出有效的 j 的数量，其中 j 满足 j != i 且 nums[j] < nums[i] 。
    # 以数组形式返回答案。
    # 示例 1：
    # 输入：nums = [8,1,2,2,3]
    # 输出：[4,0,1,1,3]
    # 解释： 对于 nums[0]=8 存在四个比它小的数字：（1，2，2 和 3）。
    # 对于 nums[1]=1 不存在比它小的数字。
    # 对于 nums[2]=2 存在一个比它小的数字：（1）。
    # 对于 nums[3]=2 存在一个比它小的数字：（1）。
    # 对于 nums[4]=3 存在三个比它小的数字：（1，2 和 2）。
    # 示例 2：
    # 输入：nums = [6,5,4,8]
    # 输出：[2,1,0,3]
    # 示例 3：
    # 输入：nums = [7,7,7,7]
    # 输出：[0,0,0,0]
# 可以排序之后加哈希，时间复杂度为$O(n\log n)$
class Solution:
    def smallerNumbersThanCurrent(self, nums: List[int]) -> List[int]:
        res = nums[:]
        hash = dict()
        res.sort() # 从小到大排序之后，元素下标就是小于当前数字的数字
        for i, num in enumerate(res):
            if num  not in hash.keys(): # 遇到了相同的数字，那么不需要更新该 number 的情况
                hash[num] = i       
        for i, num in enumerate(nums):
            res[i] = hash[num]
        return res


#2 941.有效的山脉数组
    # 给定一个整数数组 arr, 如果它是有效的山脉数组就返回 true, 否则返回 false。
    # 让我们回顾一下，如果 A 满足下述条件，那么它是一个山脉数组：
    # arr.length >= 3
    # 在 0 < i < arr.length - 1 条件下，存在 i 使得：
    # arr[0] < arr[1] < ... arr[i-1] < arr[i]
    # arr[i] > arr[i+1] > ... > arr[arr.length - 1]
    # 示例 1：
    # 输入：arr = [2,1]
    # 输出：false
    # 示例 2：
    # 输入：arr = [3,5,5]
    # 输出：false
    # 示例 3：
    # 输入：arr = [0,3,2,1]
    # 输出：true
class Solution:
    def validMountainArray(self, arr: List[int]) -> bool:
        left, right = 0, len(arr)-1
        
        while left < len(arr)-1 and arr[left+1] > arr[left]:
            left += 1
        
        while right > 0 and arr[right-1] > arr[right]:
            right -= 1
        
        return left == right and right != 0 and left != len(arr)-1


#3 1207.独一无二的出现次数
    # 给你一个整数数组 arr, 请你帮忙统计数组中每个数的出现次数。
    # 如果每个数的出现次数都是独一无二的, 就返回 true；否则返回 false。
    # 示例 1：
    # 输入：arr = [1,2,2,1,1,3]
    # 输出：true
    # 解释：在该数组中，1 出现了 3 次，2 出现了 2 次，3 只出现了 1 次。没有两个数的出现次数相同。
    # 示例 2：
    # 输入：arr = [1,2]
    # 输出：false
    # 示例 3：
    # 输入：arr = [-3,0,1,-3,1,1,1,-3,10,0]
    # 输出：true
    # 提示：
    # 1 <= arr.length <= 1000
    # -1000 <= arr[i] <= 1000
# 方法 1: 数组在哈西法的应用
class Solution:
    def uniqueOccurrences(self, arr: List[int]) -> bool:
        count = [0] * 2002
        for i in range(len(arr)):
            count[arr[i] + 1000] += 1 # 防止负数作为下标
        freq = [False] * 1002 # 标记相同频率是否重复出现
        for i in range(2001):
            if count[i] > 0:
                if freq[count[i]] == False:
                    freq[count[i]] = True
                else:
                    return False
        return True
# 方法 2： map 在哈西法的应用
class Solution:
    def uniqueOccurrences(self, arr: List[int]) -> bool:
        ref = dict()

        for i in range(len(arr)):
            ref[arr[i]] = ref.get(arr[i], 0) + 1

        value_list = sorted(ref.values())

        for i in range(len(value_list) - 1):
            if value_list[i + 1] == value_list[i]:
                return False 
        return True 
class Solution:
    def uniqueOccurrences(self, arr: List[int]) -> bool:
        res = [0] * 2001
        freq = [0] * 1001
        for a in arr:
            res[a+1000] += 1

        for r in res:
            if freq[r]==1 and r != 0:
                return False
            freq[r] += 1
        return True


#4 283. 移动零
    # 给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
    # 示例:
    # 输入: [0,1,0,3,12] 输出: [1,3,12,0,0] 说明:
    # 必须在原数组上操作，不能拷贝额外的数组。 尽量减少操作次数。
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        slow = 0
        for fast in range(len(nums)):
            if nums[fast] != 0:
                nums[slow] = nums[fast]
                slow += 1
        for i in range(slow, len(nums)):
            nums[i] = 0
# 交换前后变量，避免补零
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        slow, fast = 0, 0
        while fast < len(nums):
            if nums[fast] != 0:
                nums[slow], nums[fast] = nums[fast], nums[slow]
                slow += 1 # 保持[0, slow)区间是没有0的
            fast += 1


#5 189. 旋转数组
    # 给定一个数组, 将数组中的元素向右移动 k 个位置, 其中 k 是非负数。
    # 进阶：
    # 尽可能想出更多的解决方案，至少有三种不同的方法可以解决这个问题。 你可以使用空间复杂度为 O(1) 的 原地 算法解决这个问题吗？
    # 示例 1:
    # 输入: nums = [1,2,3,4,5,6,7], k = 3
    # 输出: [5,6,7,1,2,3,4]
    # 解释: 向右旋转 1 步: [7,1,2,3,4,5,6]。 向右旋转 2 步: [6,7,1,2,3,4,5]。 向右旋转 3 步: [5,6,7,1,2,3,4]。
    # 示例 2:
    # 输入：nums = [-1,-100,3,99], k = 2
    # 输出：[3,99,-1,-100]
    # 解释: 向右旋转 1 步: [99,-1,-100,3]。 向右旋转 2 步: [3,99,-1,-100]。



#6 724.寻找数组的中心下标
# 给你一个整数数组 nums , 请计算数组的 中心下标 。


#7 34. 在排序数组中查找元素的第一个和最后一个位置
# 给定一个按照升序排列的整数数组 nums, 和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。
# 如果数组中不存在目标值 target, 返回 [-1, -1]。


#8 922. 按奇偶排序数组II
# 给定一个非负整数数组 A,  A 中一半整数是奇数, 一半整数是偶数。


#9 35.搜索插入位置
# 给定一个排序数组和一个目标值, 在数组中找到目标值, 并返回其索引。如果目标值不存在于数组中, 返回它将会被按顺序插入的位置。


#10 24. 两两交换链表中的节点
# 给定一个链表, 两两交换其中相邻的节点, 并返回交换后的链表。


#11 234.回文链表
# 请判断一个链表是否为回文链表。


#12 143.重排链表


#13 141. 环形链表
# 给定一个链表, 判断链表中是否有环。


#14 205. 同构字符串
# 给定两个字符串 s 和 t, 判断它们是否是同构的。
# 如果 s 中的字符可以按某种映射关系替换得到 t , 那么这两个字符串是同构的。
# 每个出现的字符都应当映射到另一个字符, 同时不改变字符的顺序。不同字符不能映射到同一个字符上, 相同字符只能映射到同一个字符上, 字符可以映射到自己本身。


#15 1002. 查找常用字符
# 给你一个字符串数组 words , 请你找出所有在 words 的每个字符串中都出现的共用字符（ 包括重复字符）, 并以数组形式返回。你可以按 任意顺序 返回答案。


#16 925.长按键入
# 你的朋友正在使用键盘输入他的名字 name。偶尔, 在键入字符 c 时, 按键可能会被长按, 而字符可能被输入 1 次或多次。
# 你将会检查键盘输入的字符 typed。如果它对应的可能是你的朋友的名字（其中一些字符可能被长按）, 那么就返回 True。


#17 844.比较含退格的字符串
# 给定 S 和 T 两个字符串, 当它们分别被输入到空白的文本编辑器后, 判断二者是否相等, 并返回结果。 # 代表退格字符。


#18 129. 求根节点到叶节点数字之和


#19 1382.将二叉搜索树变平衡
# 给你一棵二叉搜索树, 请你返回一棵 平衡后 的二叉搜索树, 新生成的树应该与原来的树有着相同的节点值。
# 如果一棵二叉搜索树中, 每个节点的两棵子树高度差不超过 1 , 我们就称这棵二叉搜索树是 平衡的 。


#20 100. 相同的树
# 给定两个二叉树, 编写一个函数来检验它们是否相同。
# 如果两个树在结构上相同, 并且节点具有相同的值, 则认为它们是相同的。


#21 116. 填充每个节点的下一个右侧节点指针


#22 52. N皇后II


#23 649. Dota2 参议院


#24 1221. 分割平衡字符串


#25 5.最长回文子串
# 给你一个字符串 s, 找到 s 中最长的回文子串。


#26 132. 分割回文串 II
# 给你一个字符串 s, 请你将 s 分割成一些子串, 使每个子串都是回文。
# 返回符合要求的 最少分割次数


#27 673.最长递增子序列的个数
# 给定一个未排序的整数数组, 找到最长递增子序列的个数。


#28 841.钥匙和房间
# 有 N 个房间, 开始时你位于 0 号房间。每个房间有不同的号码：0, 1, 2, ..., N-1, 并且房间里可能有一些钥匙能使你进入下一个房间。


#29 127. 单词接龙
# 字典 wordList 中从单词 beginWord 和 endWord 的 转换序列 是一个按下述规格形成的序列：


#30 684.冗余连接


#31 685.冗余连接II


#32 657. 机器人能否返回原点
# 在二维平面上, 有一个机器人从原点 (0, 0) 开始。给出它的移动顺序, 判断这个机器人在完成移动后是否在 (0, 0) 处结束。


#33 31.下一个排列
# 实现获取 下一个排列 的函数, 算法需要将给定数字序列重新排列成字典序中下一个更大的排列。
# 如果不存在下一个更大的排列, 则将数字重新排列成最小的排列（即升序排列）。
# 必须 原地 修改, 只允许使用额外常数空间。


#34 463. 岛屿的周长


#35 1356. 根据数字二进制下 1 的数目排序