"""
数组
1365.有多少小于当前数字的数字
941.有效的山脉数组 （双指针）
1207.独一无二的出现次数 数组在哈希法中的经典应用
283.移动零 【数组】【双指针】
189.旋转数组
724.寻找数组的中心索引
34.在排序数组中查找元素的第一个和最后一个位置 （二分法）
922.按奇偶排序数组II
35.搜索插入位置

链表
24.两两交换链表中的节点
234.回文链表
143.重排链表【数组】【双向队列】【直接操作链表】
141.环形链表
160.相交链表

哈希表
205.同构字符串:【哈希表的应用】

字符串
925.长按键入 模拟匹配
0844.比较含退格的字符串【栈模拟】【空间更优的双指针】

二叉树
129.求根到叶子节点数字之和
1382.将二叉搜索树变平衡 构造平衡二叉搜索树
100.相同的树 同101.对称二叉树 一个思路
116.填充每个节点的下一个右侧节点指针

回溯算法
52.N皇后II

贪心
649.Dota2参议院 有难度
1221.分割平衡字符 简单贪心

动态规划
5.最长回文子串 和647.回文子串 差不多是一样的
132.分割回文串II 与647.回文子串和 5.最长回文子串 很像
673.最长递增子序列的个数

图论
463.岛屿的周长 （模拟）
841.钥匙和房间 【有向图】dfs, bfs都可以
127.单词接龙 广搜

并查集
684.冗余连接 【并查集基础题目】
685.冗余连接II【并查集的应用】

模拟
657.机器人能否返回原点
31.下一个排列

位运算
1356.根据数字二进制下1的数目排序
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
# 在字符串：剑指Offer58-II.左旋转字符串 (opens new window)中，我们提到，如下步骤就可以坐旋转字符串：
# 反转区间为前n的子串
# 反转区间为n到末尾的子串
# 反转整个字符串
# 本题是右旋转，其实就是反转的顺序改动一下，优先反转整个字符串，步骤如下：
# 反转整个字符串
# 反转区间为前k的子串
# 反转区间为k到末尾的子串
# 方法一：局部翻转 + 整体翻转
class Solution:
    def rotate(self, A: List[int], k: int) -> None:
        def reverse(i, j):
            while i < j:
                A[i], A[j] = A[j], A[i]
                i += 1
                j -= 1
        n = len(A)
        k %= n
        reverse(0, n - 1)
        reverse(0, k - 1)
        reverse(k, n - 1)
# 方法二：利用余数
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        copy = nums[:]

        for i in range(len(nums)):
            nums[(i + k) % len(nums)] = copy[i]
        
        return nums
        # 备注：这个方法会导致空间复杂度变成 O(n) 因为我们要创建一个 copy 数组。但是不失为一种思路。


#6 724.寻找数组的中心下标
    # 给你一个整数数组 nums , 请计算数组的 中心下标 。
    # 数组 中心下标 是数组的一个下标，其左侧所有元素相加的和等于右侧所有元素相加的和。
    # 如果中心下标位于数组最左端，那么左侧数之和视为 0 ，因为在下标的左侧不存在元素。这一点对于中心下标位于数组最右端同样适用。
    # 如果数组有多个中心下标，应该返回 最靠近左边 的那一个。如果数组不存在中心下标，返回 -1 。
    # 示例 1：
    # 输入：nums = [1, 7, 3, 6, 5, 6]
    # 输出：3
    # 解释：中心下标是 3。左侧数之和 sum = nums[0] + nums[1] + nums[2] = 1 + 7 + 3 = 11 ，右侧数之和 sum = nums[4] + nums[5] = 5 + 6 = 11 ，二者相等。
    # 示例 2：
    # 输入：nums = [1, 2, 3]
    # 输出：-1
    # 解释：数组中不存在满足此条件的中心下标。
    # 示例 3：
    # 输入：nums = [2, 1, -1]
    # 输出：0
    # 解释：中心下标是 0。左侧数之和 sum = 0 ，（下标 0 左侧不存在元素），右侧数之和 sum = nums[1] + nums[2] = 1 + -1 = 0 
class Solution:
    def pivotIndex(self, nums: List[int]) -> int:
        numSum = sum(nums) #数组总和
        leftSum = 0
        for i in range(len(nums)):
            if numSum - leftSum -nums[i] == leftSum: #左右和相等
                return i
            leftSum += nums[i]
        return -1


#7 34. 在排序数组中查找元素的第一个和最后一个位置
    # 给定一个按照升序排列的整数数组 nums, 和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。
    # 如果数组中不存在目标值 target, 返回 [-1, -1]。
    # 进阶：你可以设计并实现时间复杂度为 $O(\log n)$ 的算法解决此问题吗？
    # 示例 1：
    # 输入：nums = [5,7,7,8,8,10], target = 8
    # 输出：[3,4]
    # 示例 2：
    # 输入：nums = [5,7,7,8,8,10], target = 6
    # 输出：[-1,-1]
    # 示例 3：
    # 输入：nums = [], target = 0
    # 输出：[-1,-1]
# 寻找target在数组里的左右边界，有如下三种情况：
# 情况一：target 在数组范围的右边或者左边，例如数组{3, 4, 5}，target为2或者数组{3, 4, 5},target为6，此时应该返回{-1, -1}
# 情况二：target 在数组范围中，且数组中不存在target，例如数组{3,6,7},target为5，此时应该返回{-1, -1}
# 情况三：target 在数组范围中，且数组中存在target，例如数组{3,6,7},target为6，此时应该返回{1, 1}
# 这三种情况都考虑到，说明就想的很清楚了。
# 接下来，在去寻找左边界，和右边界了。
# 采用二分法来去寻找左右边界，为了让代码清晰，我分别写两个二分来寻找左边界和右边界。
# 刚刚接触二分搜索的同学不建议上来就想用一个二分来查找左右边界，很容易把自己绕进去，建议扎扎实实的写两个二分分别找左边界和右边界
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        def getRightBorder(nums:List[int], target:int) -> int:
            left, right = 0, len(nums)-1
            rightBoder = -2 # 记录一下rightBorder没有被赋值的情况
            while left <= right:
                middle = left + (right-left) // 2
                if nums[middle] > target:
                    right = middle - 1
                else: # 寻找右边界，nums[middle] == target的时候更新left
                    left = middle + 1
                    rightBoder = left
    
            return rightBoder
        
        def getLeftBorder(nums:List[int], target:int) -> int:
            left, right = 0, len(nums)-1 
            leftBoder = -2 # 记录一下leftBorder没有被赋值的情况
            while left <= right:
                middle = left + (right-left) // 2
                if nums[middle] >= target: #  寻找左边界，nums[middle] == target的时候更新right
                    right = middle - 1
                    leftBoder = right
                else:
                    left = middle + 1
            return leftBoder
        leftBoder = getLeftBorder(nums, target)
        rightBoder = getRightBorder(nums, target)
        # 情况一
        if leftBoder == -2 or rightBoder == -2: return [-1, -1]
        # 情况三
        if rightBoder -leftBoder >1: return [leftBoder + 1, rightBoder - 1]
        # 情况二
        return [-1, -1]
# 解法2
# 1、首先，在 nums 数组中二分查找 target；
# 2、如果二分查找失败，则 binarySearch 返回 -1，表明 nums 中没有 target。此时，searchRange 直接返回 {-1, -1}；
# 3、如果二分查找成功，则 binarySearch 返回 nums 中值为 target 的一个下标。然后，通过左右滑动指针，来找到符合题意的区间
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        def binarySearch(nums:List[int], target:int) -> int:
            left, right = 0, len(nums)-1
            while left<=right: # 不变量：左闭右闭区间
                middle = left + (right-left) // 2
                if nums[middle] > target:
                    right = middle - 1
                elif nums[middle] < target: 
                    left = middle + 1
                else:
                    return middle
            return -1
        index = binarySearch(nums, target)
        if index == -1:return [-1, -1] # nums 中不存在 target，直接返回 {-1, -1}
        # nums 中存在 target，则左右滑动指针，来找到符合题意的区间
        left, right = index, index
        # 向左滑动，找左边界
        while left -1 >=0 and nums[left - 1] == target: left -=1
        # 向右滑动，找右边界
        while right+1 < len(nums) and nums[right + 1] == target: right +=1
        return [left, right]
# 解法3
# 1、首先，在 nums 数组中二分查找得到第一个大于等于 target的下标（左边界）与第一个大于target的下标（右边界）；
# 2、如果左边界<= 右边界，则返回 [左边界, 右边界]。否则返回[-1, -1]
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        def binarySearch(nums:List[int], target:int, lower:bool) -> int:
            left, right = 0, len(nums)-1
            ans = len(nums)
            while left<=right: # 不变量：左闭右闭区间
                middle = left + (right-left) //2 
                # lower为True，执行前半部分，找到第一个大于等于 target的下标 ，否则找到第一个大于target的下标
                if nums[middle] > target or (lower and nums[middle] >= target): 
                    right = middle - 1
                    ans = middle
                else: 
                    left = middle + 1
            return ans

        leftBorder = binarySearch(nums, target, True) # 搜索左边界
        rightBorder = binarySearch(nums, target, False) -1  # 搜索右边界
        if leftBorder<= rightBorder and rightBorder< len(nums) and nums[leftBorder] == target and  nums[rightBorder] == target:
            return [leftBorder, rightBorder]
        return [-1, -1]
# 解法4
# 1、首先，在 nums 数组中二分查找得到第一个大于等于 target的下标leftBorder；
# 2、在 nums 数组中二分查找得到第一个大于等于 target+1的下标， 减1则得到rightBorder；
# 3、如果开始位置在数组的右边或者不存在target，则返回[-1, -1] 。否则返回[leftBorder, rightBorder]
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        def binarySearch(nums:List[int], target:int) -> int:
            left, right = 0, len(nums)-1
            while left<=right: # 不变量：左闭右闭区间
                middle = left + (right-left) //2 
                if nums[middle] >= target: 
                    right = middle - 1
                else: 
                    left = middle + 1
            return left  # 若存在target，则返回第一个等于target的值 

        leftBorder = binarySearch(nums, target) # 搜索左边界
        rightBorder = binarySearch(nums, target+1) -1  # 搜索右边界
        if leftBorder == len(nums) or nums[leftBorder]!= target: # 情况一和情况二
            return [-1, -1]
        return [leftBorder, rightBorder]


#8 922. 按奇偶排序数组II
    # 给定一个非负整数数组 A,  A 中一半整数是奇数, 一半整数是偶数。
    # 对数组进行排序，以便当 A[i] 为奇数时，i 也是奇数；当 A[i] 为偶数时， i 也是偶数。
    # 你可以返回任何满足上述条件的数组作为答案。
    # 示例：
    # 输入：[4,2,5,7]
    # 输出：[4,5,2,7]
    # 解释：[4,7,2,5]，[2,5,4,7]，[2,7,4,5] 也会被接受。
#方法2
# 以上代码我是建了两个辅助数组，而且A数组还相当于遍历了两次，用辅助数组的好处就是思路清晰，优化一下就是不用这两个辅助树，代码如下：
# 时间复杂度：O(n)
# 空间复杂度：O(n)
class Solution:
    def sortArrayByParityII(self, nums: List[int]) -> List[int]:
        result = [0]*len(nums)
        evenIndex = 0
        oddIndex = 1
        for i in range(len(nums)):
            if nums[i] % 2: #奇数
                result[oddIndex] = nums[i]
                oddIndex += 2
            else: #偶数
                result[evenIndex] = nums[i]
                evenIndex += 2
        return result

#方法3
# 当然还可以在原数组上修改，连result数组都不用了。
# 时间复杂度：O(n)
# 空间复杂度：O(1)
# 这里时间复杂度并不是O(n^2)，因为偶数位和奇数位都只操作一次，不是n/2 * n/2的关系，而是n/2 + n/2的关系！
class Solution:
    def sortArrayByParityII(self, nums: List[int]) -> List[int]:
        oddIndex = 1
        for i in range(0,len(nums),2): #步长为2
            if nums[i] % 2: #偶数位遇到奇数
                while  nums[oddIndex] % 2: #奇数位找偶数
                    oddIndex += 2
                nums[i], nums[oddIndex] = nums[oddIndex], nums[i]
        return nums


#9 35.搜索插入位置
    # 给定一个排序数组和一个目标值, 在数组中找到目标值, 并返回其索引。如果目标值不存在于数组中, 返回它将会被按顺序插入的位置。
    # 你可以假设数组中无重复元素。
    # 示例 1:
    # 输入: [1,3,5,6], 5
    # 输出: 2
    # 示例 2:
    # 输入: [1,3,5,6], 2
    # 输出: 1
    # 示例 3:
    # 输入: [1,3,5,6], 7
    # 输出: 4
    # 示例 4:
    # 输入: [1,3,5,6], 0
    # 输出: 0
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1

        while left <= right:
            middle = (left + right) // 2

            if nums[middle] < target:
                left = middle + 1
            elif nums[middle] > target:
                right = middle - 1
            else:
                return middle
        return right + 1


#10 24. 两两交换链表中的节点
# 给定一个链表, 两两交换其中相邻的节点, 并返回交换后的链表。
# 你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。
# 递归版本
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is None or head.next is None:
            return head

        # 待翻转的两个node分别是pre和cur
        pre = head
        cur = head.next
        next = head.next.next
        
        cur.next = pre  # 交换
        pre.next = self.swapPairs(next) # 将以next为head的后续链表两两交换
         
        return cur
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        dummy_head = ListNode(next=head)
        current = dummy_head
        
        # 必须有cur的下一个和下下个才能交换，否则说明已经交换结束了
        while current.next and current.next.next:
            temp = current.next # 防止节点修改
            temp1 = current.next.next.next
            
            current.next = current.next.next
            current.next.next = temp
            temp.next = temp1
            current = current.next.next
        return dummy_head.next


#11 234.回文链表
    # 请判断一个链表是否为回文链表。
    # 示例 1:
    # 输入: 1->2
    # 输出: false
    # 示例 2:
    # 输入: 1->2->2->1
    # 输出: true
#数组模拟
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        list=[]
        while head: 
            list.append(head.val)
            head=head.next
        l,r=0, len(list)-1
        while l<=r: 
            if list[l]!=list[r]:
                return False
            l+=1
            r-=1
        return True
#反转后半部分链表
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        fast = slow = head

        # find mid point which including (first) mid point into the first half linked list
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        node = None

        # reverse second half linked list
        while slow:
            slow.next, slow, node = node, slow.next, slow

        # compare reversed and original half; must maintain reversed linked list is shorter than 1st half
        while node:
            if node.val != head.val:
                return False
            node = node.next
            head = head.next
        return True


#12 143.重排链表
# 方法二 双向队列
class Solution:
    def reorderList(self, head: ListNode) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        d = collections.deque()
        tmp = head
        while tmp.next: # 链表除了首元素全部加入双向队列
            d.append(tmp.next)
            tmp = tmp.next
        tmp = head
        while len(d): # 一后一前加入链表
            tmp.next = d.pop()
            tmp = tmp.next
            if len(d):
                tmp.next = d.popleft()
                tmp = tmp.next
        tmp.next = None # 尾部置空
 
# 方法三 反转链表
class Solution:
    def reorderList(self, head: ListNode) -> None:
        if head == None or head.next == None:
            return True
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        right = slow.next # 分割右半边
        slow.next = None # 切断
        right = self.reverseList(right) #反转右半边
        left = head
        # 左半边一定比右半边长, 因此判断右半边即可
        while right:
            curLeft = left.next
            left.next = right
            left = curLeft

            curRight = right.next
            right.next = left
            right = curRight
    def reverseList(self, head: ListNode) -> ListNode:
        cur = head   
        pre = None
        while(cur!=None):
            temp = cur.next # 保存一下cur的下一个节点
            cur.next = pre # 反转
            pre = cur
            cur = temp
        return pre


#13 141. 环形链表
    # 给定一个链表, 判断链表中是否有环。
    # 如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。注意：pos 不作为参数进行传递，仅仅是为了标识链表的实际情况。
    # 如果链表中存在环，则返回 true 。 否则，返回 false 。
class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        if not head: return False
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if fast == slow:
                return True
        return False


# 160.相交链表
    # 同：160.链表相交
    # 给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。如果两个链表没有交点，返回 null 。
    # 图示两个链表在节点 c1 开始相交：
    # 题目数据 保证 整个链式结构中不存在环。
    # 注意，函数返回结果后，链表必须 保持其原始结构 
# （版本一）求长度，同时出发
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        lenA, lenB = 0, 0
        cur = headA
        while cur:         # 求链表A的长度
            cur = cur.next 
            lenA += 1
        cur = headB 
        while cur:         # 求链表B的长度
            cur = cur.next 
            lenB += 1
        curA, curB = headA, headB
        if lenA > lenB:     # 让curB为最长链表的头，lenB为其长度
            curA, curB = curB, curA
            lenA, lenB = lenB, lenA 
        for _ in range(lenB - lenA):  # 让curA和curB在同一起点上（末尾位置对齐）
            curB = curB.next 
        while curA:         #  遍历curA 和 curB，遇到相同则直接返回
            if curA == curB:
                return curA
            else:
                curA = curA.next 
                curB = curB.next
        return None 
# （版本二）求长度，同时出发 （代码复用）
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        lenA = self.getLength(headA)
        lenB = self.getLength(headB)
        
        # 通过移动较长的链表，使两链表长度相等
        if lenA > lenB:
            headA = self.moveForward(headA, lenA - lenB)
        else:
            headB = self.moveForward(headB, lenB - lenA)
        
        # 将两个头向前移动，直到它们相交
        while headA and headB:
            if headA == headB:
                return headA
            headA = headA.next
            headB = headB.next
        
        return None
    
    def getLength(self, head: ListNode) -> int:
        length = 0
        while head:
            length += 1
            head = head.next
        return length
    
    def moveForward(self, head: ListNode, steps: int) -> ListNode:
        while steps > 0:
            head = head.next
            steps -= 1
        return head
# （版本三）求长度，同时出发 （代码复用 + 精简）
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        dis = self.getLength(headA) - self.getLength(headB)
        
        # 通过移动较长的链表，使两链表长度相等
        if dis > 0:
            headA = self.moveForward(headA, dis)
        else:
            headB = self.moveForward(headB, abs(dis))
        
        # 将两个头向前移动，直到它们相交
        while headA and headB:
            if headA == headB:
                return headA
            headA = headA.next
            headB = headB.next
        
        return None
    
    def getLength(self, head: ListNode) -> int:
        length = 0
        while head:
            length += 1
            head = head.next
        return length
    
    def moveForward(self, head: ListNode, steps: int) -> ListNode:
        while steps > 0:
            head = head.next
            steps -= 1
        return head
# （版本四）等比例法
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        # 处理边缘情况
        if not headA or not headB:
            return None
        
        # 在每个链表的头部初始化两个指针
        pointerA = headA
        pointerB = headB
        
        # 遍历两个链表直到指针相交
        while pointerA != pointerB:
            # 将指针向前移动一个节点
            pointerA = pointerA.next if pointerA else headB
            pointerB = pointerB.next if pointerB else headA
        
        # 如果相交，指针将位于交点节点，如果没有交点，值为None
        return pointerA


#14 205. 同构字符串
    # 给定两个字符串 s 和 t, 判断它们是否是同构的。
    # 如果 s 中的字符可以按某种映射关系替换得到 t , 那么这两个字符串是同构的。
    # 每个出现的字符都应当映射到另一个字符, 同时不改变字符的顺序。不同字符不能映射到同一个字符上, 相同字符只能映射到同一个字符上, 字符可以映射到自己本身。
    # 示例 1:
    # 输入：s = "egg", t = "add"
    # 输出：true
    # 示例 2：
    # 输入：s = "foo", t = "bar"
    # 输出：false
    # 示例 3：
    # 输入：s = "paper", t = "title"
    # 输出：true
    # 提示：可以假设 s 和 t 长度相同。
class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        default_dict1 = defaultdict(str)
        default_dict2 = defaultdict(str)
    
        if len(s) != len(t): return false

        for i in range(len(s)):
            if not default_dict1[s[i]]:
                default_dict1[s[i]] = t[i]
            
            if not default_dict2[t[i]]:
                default_dict2[t[i]] = s[i]

            if default_dict1[s[i]] != t[i] or default_dict2[t[i]] != s[i]:
                return False
            
        return True


#15 1002. 查找常用字符
# 给你一个字符串数组 words , 请你找出所有在 words 的每个字符串中都出现的共用字符（ 包括重复字符）, 并以数组形式返回。你可以按 任意顺序 返回答案。


#16 925.长按键入
    # 你的朋友正在使用键盘输入他的名字 name。偶尔, 在键入字符 c 时, 按键可能会被长按, 而字符可能被输入 1 次或多次。
    # 你将会检查键盘输入的字符 typed。如果它对应的可能是你的朋友的名字（其中一些字符可能被长按）, 那么就返回 True。
    # 示例 1：
    # 输入：name = "alex", typed = "aaleex"
    # 输出：true
    # 解释：'alex' 中的 'a' 和 'e' 被长按。
    # 示例 2：
    # 输入：name = "saeed", typed = "ssaaedd"
    # 输出：false
    # 解释：'e' 一定需要被键入两次，但在 typed 的输出中不是这样。
    # 示例 3：
    # 输入：name = "leelee", typed = "lleeelee"
    # 输出：true
    # 示例 4：
    # 输入：name = "laiden", typed = "laiden"
    # 输出：true
    # 解释：长按名字中的字符并不是必要的。
class Solution:
    def isLongPressedName(self, name: str, typed: str) -> bool:
        i = j = 0
        while(i<len(name) and j<len(typed)):
        # If the current letter matches, move as far as possible
            if typed[j]==name[i]:
                while j+1<len(typed) and typed[j]==typed[j+1]:
                    j+=1
                    # special case when there are consecutive repeating letters
                    if i+1<len(name) and name[i]==name[i+1]:
                        i+=1
                else:
                    j+=1
                    i+=1
            else:
                return False
        return i == len(name) and j==len(typed)


#17 844.比较含退格的字符串
    # 给定 S 和 T 两个字符串, 当它们分别被输入到空白的文本编辑器后, 判断二者是否相等, 并返回结果。 # 代表退格字符。
    # 注意：如果对空文本输入退格字符，文本继续为空。
    # 示例 1：
    # 输入：S = "ab#c", T = "ad#c"
    # 输出：true
    # 解释：S 和 T 都会变成 “ac”。
    # 示例 2：
    # 输入：S = "ab##", T = "c#d#"
    # 输出：true
    # 解释：S 和 T 都会变成 “”。
    # 示例 3：
    # 输入：S = "a##c", T = "#a#c"
    # 输出：true
    # 解释：S 和 T 都会变成 “c”。
    # 示例 4：
    # 输入：S = "a#c", T = "b"
    # 输出：false
    # 解释：S 会变成 “c”，但 T 仍然是 “b”。
    #18 129. 求根节点到叶节点数字之和
class Solution:
    def get_string(self, s: str) -> str :
        bz = []
        for i in range(len(s)) :
            c = s[i]
            if c != '#' :
                bz.append(c) # 模拟入栈
            elif len(bz) > 0: # 栈非空才能弹栈
                bz.pop() # 模拟弹栈
        return str(bz)

    def backspaceCompare(self, s: str, t: str) -> bool:
        return self.get_string(s) == self.get_string(t)
        pass
# 双指针
class Solution:
    def backspaceCompare(self, s: str, t: str) -> bool:
        s_index, t_index = len(s) - 1, len(t) - 1
        s_backspace, t_backspace = 0, 0 # 记录s,t的#数量
        while s_index >= 0 or t_index >= 0: # 使用or，以防长度不一致
            while s_index >= 0: # 从后向前，消除s的#
                if s[s_index] == '#':
                    s_index -= 1
                    s_backspace += 1
                else:
                    if s_backspace > 0:
                        s_index -= 1
                        s_backspace -= 1
                    else:
                        break
            while t_index >= 0: # 从后向前，消除t的#
                if t[t_index] == '#':
                    t_index -= 1
                    t_backspace += 1
                else:
                    if t_backspace > 0:
                        t_index -= 1
                        t_backspace -= 1
                    else:
                        break
            if s_index >= 0 and t_index >= 0: # 后半部分#消除完了，接下来比较当前位的值
                if s[s_index] != t[t_index]:
                    return False
            elif s_index >= 0 or t_index >= 0: # 一个字符串找到了待比较的字符，另一个没有，返回False
                return False
            s_index -= 1
            t_index -= 1
        return True


# 129. 求根节点到叶节点数字之和
class Solution:
    def sumNumbers(self, root: TreeNode) -> int:
        res = 0
        path = []
        def backtrace(root):
            nonlocal res
            if not root: return # 节点空则返回
            path.append(root.val)
            if not root.left and not root.right: # 遇到了叶子节点
                res += get_sum(path)
            if root.left: # 左子树不空
                backtrace(root.left)
            if root.right: # 右子树不空
                backtrace(root.right)
            path.pop()

        def get_sum(arr):
            s = 0
            for i in range(len(arr)):
                s = s * 10 + arr[i]
            return s

        backtrace(root)
        return res


#19 1382.将二叉搜索树变平衡
    # 给你一棵二叉搜索树, 请你返回一棵 平衡后 的二叉搜索树, 新生成的树应该与原来的树有着相同的节点值。
    # 如果一棵二叉搜索树中, 每个节点的两棵子树高度差不超过 1 , 我们就称这棵二叉搜索树是 平衡的 。
    # 如果有多种构造方法，请你返回任意一种。
    # 输入：root = [1,null,2,null,3,null,4,null,null]
    # 输出：[2,1,3,null,null,null,4]
    # 解释：这不是唯一的正确答案，[3,1,4,null,2,null,null] 也是一个可行的构造方案。
    # 提示：
    # 树节点的数目在 1 到 10^4 之间。
    # 树节点的值互不相同，且在 1 到 10^5 之间。
class Solution:
    def balanceBST(self, root: TreeNode) -> TreeNode:
        res = []
        # 有序树转成有序数组
        def traversal(cur: TreeNode):
            if not cur: return
            traversal(cur.left)
            res.append(cur.val)
            traversal(cur.right)
        # 有序数组转成平衡二叉树
        def getTree(nums: List, left, right):
            if left > right: return 
            mid = left + (right -left) // 2
            root = TreeNode(nums[mid])
            root.left = getTree(nums, left, mid - 1)
            root.right = getTree(nums, mid + 1, right)
            return root
        traversal(root)
        return getTree(res, 0, len(res) - 1)


#20 100. 相同的树
    # 给定两个二叉树, 编写一个函数来检验它们是否相同。
    # 如果两个树在结构上相同, 并且节点具有相同的值, 则认为它们是相同的。
# 递归法
class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        if not p and not q: return True
        elif not p or not q: return False
        elif p.val != q.val: return False
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
# 迭代法
class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        if not p and not q: return True
        if not p or not q: return False
        que = collections.deque()
        que.append(p)
        que.append(q)
        while que:
            leftNode = que.popleft()
            rightNode = que.popleft()
            if not leftNode and not rightNode: continue 
            if not leftNode or not rightNode or leftNode.val != rightNode.val: return False 
            que.append(leftNode.left)
            que.append(rightNode.left)
            que.append(leftNode.right)
            que.append(rightNode.right)
        return True


#21 116. 填充每个节点的下一个右侧节点指针
    # 给定一个 完美二叉树 ，其所有叶子节点都在同一层，每个父节点都有两个子节点。二叉树定义如下：
    # struct Node {
    #   int val;
    #   Node *left;
    #   Node *right;
    #   Node *next;
    # }
    # 填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 NULL。
    # 初始状态下，所有 next 指针都被设置为 NULL。
    # 进阶：
    # 你只能使用常量级额外空间。
    # 使用递归解题也符合要求，本题中递归程序占用的栈空间不算做额外的空间复杂度。
# 递归法
class Solution:
    def connect(self, root: 'Node') -> 'Node':
        def traversal(cur: 'Node') -> 'Node':
            if not cur: return []
            if cur.left: cur.left.next = cur.right # 操作1
            if cur.right:
                if cur.next:
                    cur.right.next = cur.next.left # 操作2
                else:
                    cur.right.next = None
            traversal(cur.left) # 左
            traversal(cur.right) # 右
        traversal(root)
        return root
# 迭代法
class Solution:
    def connect(self, root: 'Node') -> 'Node':
        if not root: return 
        res = []
        queue = [root]
        while queue:
            size = len(queue)
            for i in range(size): # 开始每一层的遍历
                if i==0: 
                    nodePre = queue.pop(0) # 记录一层的头结点
                    node = nodePre
                else:
                    node = queue.pop(0)
                    nodePre.next = node # 本层前一个节点next指向本节点
                    nodePre = nodePre.next
                if node.left: queue.append(node.left)
                if node.right: queue.append(node.right)
            nodePre.next = None # 本层最后一个节点指向None
        return root


#22 52. N皇后II
    # 题目链接：https://leetcode.cn/problems/n-queens-ii/
    # n 皇后问题研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。
    # 上图为 8 皇后问题的一种解法。
    # 给定一个整数 n，返回 n 皇后不同的解决方案的数量。
    # 示例:
    # 输入: 4
    # 输出: 2
    # 解释: 4 皇后问题存在如下两个不同的解法。
    # 解法 1
    # [  [".Q..",   "...Q",   "Q...",   "..Q."],
    # 解法 2
    #  ["..Q.",   "Q...",   "...Q",   ".Q.."] ]



#23 649. Dota2 参议院
    # 力扣题目链接
    # Dota2 的世界里有两个阵营：Radiant(天辉)和 Dire(夜魇)
    # Dota2 参议院由来自两派的参议员组成。现在参议院希望对一个 Dota2 游戏里的改变作出决定。他们以一个基于轮为过程的投票进行。在每一轮中，每一位参议员都可以行使两项权利中的一项：
    # 禁止一名参议员的权利：参议员可以让另一位参议员在这一轮和随后的几轮中丧失所有的权利。
    # 宣布胜利：如果参议员发现有权利投票的参议员都是同一个阵营的，他可以宣布胜利并决定在游戏中的有关变化。
    # 给定一个字符串代表每个参议员的阵营。字母 “R” 和 “D” 分别代表了 Radiant（天辉）和 Dire（夜魇）。然后，如果有 n 个参议员，给定字符串的大小将是 n。
    # 以轮为基础的过程从给定顺序的第一个参议员开始到最后一个参议员结束。这一过程将持续到投票结束。所有失去权利的参议员将在过程中被跳过。
    # 假设每一位参议员都足够聪明，会为自己的政党做出最好的策略，你需要预测哪一方最终会宣布胜利并在 Dota2 游戏中决定改变。输出应该是 Radiant 或 Dire。
    # 示例 1：
    # 输入："RD"
    # 输出："Radiant"
    # 解释：第一个参议员来自 Radiant 阵营并且他可以使用第一项权利让第二个参议员失去权力，因此第二个参议员将被跳过因为他没有任何权利。然后在第二轮的时候，第一个参议员可以宣布胜利，因为他是唯一一个有投票权的人
    # 示例 2：
    # 输入："RDD"
    # 输出："Dire"
    # 解释： 第一轮中,第一个来自 Radiant 阵营的参议员可以使用第一项权利禁止第二个参议员的权利， 第二个来自 Dire 阵营的参议员会被跳过因为他的权利被禁止， 第三个来自 Dire 阵营的参议员可以使用他的第一项权利禁止第一个参议员的权利， 因此在第二轮只剩下第三个参议员拥有投票的权利,于是他可以宣布胜利。
class Solution:
    def predictPartyVictory(self, senate: str) -> str:
        # R = true表示本轮循环结束后，字符串里依然有R。D同理
        R , D = True, True

        # 当flag大于0时，R在D前出现，R可以消灭D。当flag小于0时，D在R前出现，D可以消灭R
        flag = 0

        senate = list(senate)
        while R and D: # 一旦R或者D为false，就结束循环，说明本轮结束后只剩下R或者D了
            R = False
            D = False
            for i in range(len(senate)) :
                if senate[i] == 'R' :
                    if flag < 0: senate[i] = '0' # 消灭R，R此时为false
                    else: R = True # 如果没被消灭，本轮循环结束有R
                    flag += 1
                if senate[i] == 'D':
                    if flag > 0: senate[i] = '0'
                    else: D = True
                    flag -= 1
        # 循环结束之后，R和D只能有一个为true
        return "Radiant" if R else "Dire"


#24 1221. 分割平衡字符串
    # 在一个 平衡字符串 中，'L' 和 'R' 字符的数量是相同的。
    # 给你一个平衡字符串 s，请你将它分割成尽可能多的平衡字符串。
    # 注意：分割得到的每个字符串都必须是平衡字符串。
    # 返回可以通过分割得到的平衡字符串的 最大数量 。
    # 示例 1：
    # 输入：s = "RLRRLLRLRL"
    # 输出：4
    # 解释：s 可以分割为 "RL"、"RRLL"、"RL"、"RL" ，每个子字符串中都包含相同数量的 'L' 和 'R' 。
    # 示例 2：
    # 输入：s = "RLLLLRRRLR"
    # 输出：3
    # 解释：s 可以分割为 "RL"、"LLLRRR"、"LR" ，每个子字符串中都包含相同数量的 'L' 和 'R' 。
    # 示例 3：
    # 输入：s = "LLLLRRRR"
    # 输出：1
    # 解释：s 只能保持原样 "LLLLRRRR".
    # 示例 4：
    # 输入：s = "RLRRRLLRLL"
    # 输出：2
    # 解释：s 可以分割为 "RL"、"RRRLLRLL" ，每个子字符串中都包含相同数量的 'L' 和 'R' 。
class Solution:
    def balancedStringSplit(self, s: str) -> int:
        diff = 0 #右左差值
        ans = 0
        for c in s:
            if c == "L":
                diff -= 1
            else:
                diff += 1
            if diff == 0:
                ans += 1
        return ans


#25 5.最长回文子串
    # 给你一个字符串 s, 找到 s 中最长的回文子串。
    # 示例 1：
    # 输入：s = "babad"
    # 输出："bab"
    # 解释："aba" 同样是符合题意的答案。
    # 示例 2：
    # 输入：s = "cbbd"
    # 输出："bb"
    # 示例 3：
    # 输入：s = "a"
    # 输出："a"
    # 示例 4：
    # 输入：s = "ac"
    # 输出："a"
class Solution:
    def longestPalindrome(self, s: str) -> str:
        dp = [[False] * len(s) for _ in range(len(s))]
        maxlenth = 0
        left = 0
        right = 0
        for i in range(len(s) - 1, -1, -1):
            for j in range(i, len(s)):
                if s[j] == s[i]:
                    if j - i <= 1 or dp[i + 1][j - 1]:
                        dp[i][j] = True
                if dp[i][j] and j - i + 1 > maxlenth:
                    maxlenth = j - i + 1
                    left = i
                    right = j
        return s[left:right + 1]
# 双指针
class Solution:
    def longestPalindrome(self, s: str) -> str:

        def find_point(i, j, s):
            while i >= 0 and j < len(s) and s[i] == s[j]:
                i -= 1
                j += 1
            return i + 1, j

        def compare(start, end, left, right):
            if right - left > end - start:
                return left, right
            else:
                return start, end

        start = 0
        end = 0
        for i in range(len(s)):
            left, right = find_point(i, i, s)
            start, end = compare(start, end, left, right)

            left, right = find_point(i, i + 1, s)
            start, end = compare(start, end, left, right)
        return s[start:end]


#26 132. 分割回文串 II
    # 给你一个字符串 s, 请你将 s 分割成一些子串, 使每个子串都是回文。
    # 返回符合要求的 最少分割次数
    # 示例 1：
    # 输入：s = "aab" 输出：1 解释：只需一次分割就可将 s 分割成 ["aa","b"] 这样两个回文子串。
    # 示例 2： 输入：s = "a" 输出：0
    # 示例 3： 输入：s = "ab" 输出：1
    # 提示：
    # 1 <= s.length <= 2000
    # s 仅由小写英文字母组成
class Solution:
    def minCut(self, s: str) -> int:

        isPalindromic=[[False]*len(s) for _ in range(len(s))]

        for i in range(len(s)-1,-1,-1):
            for j in range(i,len(s)):
                if s[i]!=s[j]:
                    isPalindromic[i][j] = False
                elif  j-i<=1 or isPalindromic[i+1][j-1]:
                    isPalindromic[i][j] = True

        # print(isPalindromic)
        dp=[sys.maxsize]*len(s)
        dp[0]=0

        for i in range(1,len(s)):
            if isPalindromic[0][i]:
                dp[i]=0
                continue
            for j in range(0,i):
                if isPalindromic[j+1][i]==True:
                    dp[i]=min(dp[i], dp[j]+1)
        return dp[-1]


#27 673.最长递增子序列的个数
    # 给定一个未排序的整数数组, 找到最长递增子序列的个数。
    # 示例 1:
    # 输入: [1,3,5,4,7]
    # 输出: 2
    # 解释: 有两个最长递增子序列，分别是 [1, 3, 4, 7] 和[1, 3, 5, 7]。
    # 示例 2:
    # 输入: [2,2,2,2,2]
    # 输出: 5
    # 解释: 最长递增子序列的长度是1，并且存在5个子序列的长度为1，因此输出5。
class Solution:
    def findNumberOfLIS(self, nums: List[int]) -> int:
        size = len(nums)
        if size<= 1: return size

        dp = [1 for i in range(size)]
        count = [1 for i in range(size)]

        maxCount = 0
        for i in range(1, size):
            for j in range(i):
                if nums[i] > nums[j]:
                    if dp[j] + 1 > dp[i] :
                        dp[i] = dp[j] + 1
                        count[i] = count[j]
                    elif dp[j] + 1 == dp[i] :
                        count[i] += count[j]
                if dp[i] > maxCount:
                    maxCount = dp[i];
        result = 0
        for i in range(size):
            if maxCount == dp[i]:
                result += count[i]
        return result;


#28 841.钥匙和房间
    # 有 N 个房间, 开始时你位于 0 号房间。每个房间有不同的号码：0, 1, 2, ..., N-1, 并且房间里可能有一些钥匙能使你进入下一个房间。
    # 在形式上，对于每个房间 i 都有一个钥匙列表 rooms[i]，每个钥匙 rooms[i][j] 由 [0,1，...，N-1] 中的一个整数表示，其中 N = rooms.length。 钥匙 rooms[i][j] = v 可以打开编号为 v 的房间。
    # 最初，除 0 号房间外的其余所有房间都被锁住。
    # 你可以自由地在房间之间来回走动。
    # 如果能进入每个房间返回 true，否则返回 false。
    # 示例 1：
    # 输入: [[1],[2],[3],[]]
    # 输出: true
    # 解释: 我们从 0 号房间开始，拿到钥匙 1。 之后我们去 1 号房间，拿到钥匙 2。 然后我们去 2 号房间，拿到钥匙 3。 最后我们去了 3 号房间。 由于我们能够进入每个房间，我们返回 true。
    # 示例 2：
    # 输入：[[1,3],[3,0,1],[2],[0]]
    # 输出：false
    # 解释：我们不能进入 2 号房间。
# 深度搜索优先
class Solution:
    def dfs(self, key: int, rooms: List[List[int]]  , visited : List[bool] ) :
        if visited[key] :
            return
        visited[key] = True
        keys = rooms[key]
        for i in range(len(keys)) :
            # 深度优先搜索遍历
            self.dfs(keys[i], rooms, visited)

    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        visited = [False for i in range(len(rooms))]

        self.dfs(0, rooms, visited)

        # 检查是否都访问到了
        for i in range(len(visited)):
            if not visited[i] :
                return False
        return True
# 广度搜索优先
class Solution:
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        visited = [False] * len(rooms)
        self.bfs(rooms, 0, visited)
        
        for room in visited:
            if room == False:
                return False
        
        return True
    
    def bfs(self, rooms, index, visited):
        q = collections.deque()
        q.append(index)

        visited[0] = True

        while len(q) != 0:
            index = q.popleft()
            for nextIndex in rooms[index]:
                if visited[nextIndex] == False:
                    q.append(nextIndex)
                    visited[nextIndex] = True


#29 127. 单词接龙
    # 字典 wordList 中从单词 beginWord 和 endWord 的 转换序列 是一个按下述规格形成的序列：
    # 序列中第一个单词是 beginWord 。
    # 序列中最后一个单词是 endWord 。
    # 每次转换只能改变一个字母。
    # 转换过程中的中间单词必须是字典 wordList 中的单词。
    # 给你两个单词 beginWord 和 endWord 和一个字典 wordList ，找到从 beginWord 到 endWord 的 最短转换序列 中的 单词数目 。如果不存在这样的转换序列，返回 0。
    # 示例 1：
    # 输入：beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
    # 输出：5
    # 解释：一个最短转换序列是 "hit" -> "hot" -> "dot" -> "dog" -> "cog", 返回它的长度 5。
    # 示例 2：
    # 输入：beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log"]
    # 输出：0
    # 解释：endWord "cog" 不在字典中，所以无法进行转换。
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        wordSet = set(wordList)
        if len(wordSet)== 0 or endWord not in wordSet:
            return 0
        mapping = {beginWord:1}
        queue = deque([beginWord]) 
        while queue:
            word = queue.popleft()
            path = mapping[word]
            for i in range(len(word)):
                word_list = list(word)
                for j in range(26):
                    word_list[i] = chr(ord('a')+j)
                    newWord = "".join(word_list)
                    if newWord == endWord:
                        return path+1
                    if newWord in wordSet and newWord not in mapping:
                        mapping[newWord] = path+1
                        queue.append(newWord)                      
        return 0


#30 684.冗余连接
    # 树可以看成是一个连通且 无环 的 无向 图。
    # 给定往一棵 n 个节点 (节点值 1～n) 的树中添加一条边后的图。添加的边的两个顶点包含在 1 到 n 中间，且这条附加的边不属于树中已存在的边。图的信息记录于长度为 n 的二维数组 edges ，edges[i] = [ai, bi] 表示图中在 ai 和 bi 之间存在一条边。
    # 请找出一条可以删去的边，删除后可使得剩余部分是一个有着 n 个节点的树。如果有多个答案，则返回数组 edges 中最后出现的边。
class Solution:
    def __init__(self):
        """
        初始化
        """
        self.n = 1005
        self.father = [i for i in range(self.n)]


    def find(self, u):
        """
        并查集里寻根的过程
        """
        if u == self.father[u]:
            return u
        self.father[u] = self.find(self.father[u])
        return self.father[u]

    def join(self, u, v):
        """
        将v->u 这条边加入并查集
        """
        u = self.find(u)
        v = self.find(v)
        if u == v : return
        self.father[v] = u
        pass


    def same(self, u, v ):
        """
        判断 u 和 v是否找到同一个根，本题用不上
        """
        u = self.find(u)
        v = self.find(v)
        return u == v

    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        for i in range(len(edges)):
            if self.same(edges[i][0], edges[i][1]) :
                return edges[i]
            else :
                self.join(edges[i][0], edges[i][1])
        return []
# Python简洁写法
class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        n = len(edges)
        p = [i for i in range(n+1)]
        def find(i):
            if p[i] != i:
                p[i] = find(p[i])
            return p[i]
        for u, v in edges:
            if p[find(u)] == find(v):
                return [u, v]
            p[find(u)] = find(v)


#31 685.冗余连接II
    # 在本问题中，有根树指满足以下条件的 有向 图。该树只有一个根节点，所有其他节点都是该根节点的后继。该树除了根节点之外的每一个节点都有且只有一个父节点，而根节点没有父节点。
    # 输入一个有向图，该图由一个有着 n 个节点（节点值不重复，从 1 到 n）的树及一条附加的有向边构成。附加的边包含在 1 到 n 中的两个不同顶点间，这条附加的边不属于树中已存在的边。
    # 结果图是一个以边组成的二维数组 edges 。 每个元素是一对 [ui, vi]，用以表示 有向 图中连接顶点 ui 和顶点 vi 的边，其中 ui 是 vi 的一个父节点。
    # 返回一条能删除的边，使得剩下的图是有 n 个节点的有根树。若有多个答案，返回最后出现在给定二维数组的答案。
class Solution:
    def __init__(self):
        self.n = 1010
        self.father = [i for i in range(self.n)]


    def find(self, u: int):
        """
        并查集里寻根的过程
        """
        if u == self.father[u]:
            return u
        self.father[u] = self.find(self.father[u])
        return self.father[u]

    def join(self, u: int, v: int):
        """
        将v->u 这条边加入并查集
        """
        u = self.find(u)
        v = self.find(v)
        if u == v : return
        self.father[v] = u
        pass


    def same(self, u: int, v: int ):
        """
        判断 u 和 v是否找到同一个根，本题用不上
        """
        u = self.find(u)
        v = self.find(v)
        return u == v

    def init_father(self):
        self.father = [i for i in range(self.n)]
        pass

    def getRemoveEdge(self, edges: List[List[int]]) -> List[int]:
        """
        在有向图里找到删除的那条边，使其变成树
        """

        self.init_father()
        for i in range(len(edges)):
            if self.same(edges[i][0], edges[i][1]): # 构成有向环了，就是要删除的边
                return edges[i]
            self.join(edges[i][0], edges[i][1]);
        return []

    def isTreeAfterRemoveEdge(self, edges: List[List[int]], deleteEdge: int) -> bool:
        """
        删一条边之后判断是不是树
        """

        self.init_father()
        for i in range(len(edges)):
            if i == deleteEdge: continue
            if self.same(edges[i][0], edges[i][1]): #  构成有向环了，一定不是树
                return False
            self.join(edges[i][0], edges[i][1]);
        return True

    def findRedundantDirectedConnection(self, edges: List[List[int]]) -> List[int]:
        inDegree = [0 for i in range(self.n)]

        for i in range(len(edges)):
            inDegree[ edges[i][1] ] += 1

        # 找入度为2的节点所对应的边，注意要倒序，因为优先返回最后出现在二维数组中的答案
        towDegree = []
        for i in range(len(edges))[::-1]:
            if inDegree[edges[i][1]] == 2 :
                towDegree.append(i)

        # 处理图中情况1 和 情况2
        # 如果有入度为2的节点，那么一定是两条边里删一个，看删哪个可以构成树
        if len(towDegree) > 0:
            if(self.isTreeAfterRemoveEdge(edges, towDegree[0])) :
                return edges[towDegree[0]]
            return edges[towDegree[1]]

        # 明确没有入度为2的情况，那么一定有有向环，找到构成环的边返回就可以了
        return self.getRemoveEdge(edges)


#32 657. 机器人能否返回原点
    # 在二维平面上, 有一个机器人从原点 (0, 0) 开始。给出它的移动顺序, 判断这个机器人在完成移动后是否在 (0, 0) 处结束。
    # 移动顺序由字符串表示。字符 move[i] 表示其第 i 次移动。机器人的有效动作有 R（右），L（左），U（上）和 D（下）。如果机器人在完成所有动作后返回原点，则返回 true。否则，返回 false。
    # 注意：机器人“面朝”的方向无关紧要。 “R” 将始终使机器人向右移动一次，“L” 将始终向左移动等。此外，假设每次移动机器人的移动幅度相同。
    # 示例 1:
    # 输入: "UD"
    # 输出: true
    # 解释：机器人向上移动一次，然后向下移动一次。所有动作都具有相同的幅度，因此它最终回到它开始的原点。因此，我们返回 true。
    # 示例 2:
    # 输入: "LL"
    # 输出: false
    # 解释：机器人向左移动两次。它最终位于原点的左侧，距原点有两次 “移动” 的距离。我们返回 false，因为它在移动结束时没有返回原点。
# 时间复杂度：O(n)
# 空间复杂度：O(1)
class Solution:
    def judgeCircle(self, moves: str) -> bool:
        x = 0 # 记录当前位置
        y = 0
        for i in range(len(moves)):
            if (moves[i] == 'U'):
                y += 1
            if (moves[i] == 'D'):
                y -= 1
            if (moves[i] == 'L'):
                x += 1
            if (moves[i] == 'R'):
                x -= 1
        return x == 0 and y == 0


#33 31.下一个排列
    # 实现获取 下一个排列 的函数, 算法需要将给定数字序列重新排列成字典序中下一个更大的排列。
    # 如果不存在下一个更大的排列, 则将数字重新排列成最小的排列（即升序排列）。
    # 必须 原地 修改, 只允许使用额外常数空间。
    # 示例 1：
    # 输入：nums = [1,2,3]
    # 输出：[1,3,2]
    # 示例 2：
    # 输入：nums = [3,2,1]
    # 输出：[1,2,3]
    # 示例 3：
    # 输入：nums = [1,1,5]
    # 输出：[1,5,1]
    # 示例 4：
    # 输入：nums = [1]
    # 输出：[1]
# 直接使用sorted()会开辟新的空间并返回一个新的list，故补充一个原地反转函数
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        length = len(nums)
        for i in range(length - 2, -1, -1): # 从倒数第二个开始
            if nums[i]>=nums[i+1]: continue # 剪枝去重
            for j in range(length - 1, i, -1):
                if nums[j] > nums[i]:
                    nums[j], nums[i] = nums[i], nums[j]
                    self.reverse(nums, i + 1, length - 1)
                    return  
        self.reverse(nums, 0, length - 1)
    
    def reverse(self, nums: List[int], left: int, right: int) -> None:
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1
"""
265 / 265 个通过测试用例
状态：通过
执行用时: 36 ms
内存消耗: 14.9 MB
"""


#34 463. 岛屿的周长
    # 给定一个 row x col 的二维网格地图 grid ，其中：grid[i][j] = 1 表示陆地， grid[i][j] = 0 表示水域。
    # 网格中的格子 水平和垂直 方向相连（对角线方向不相连）。整个网格被水完全包围，但其中恰好有一个岛屿（或者说，一个或多个表示陆地的格子相连组成的岛屿）。
    # 岛屿中没有“湖”（“湖” 指水域在岛屿内部且不和岛屿周围的水相连）。格子是边长为 1 的正方形。网格为长方形，且宽度和高度均不超过 100 。计算这个岛屿的周长。
    # 输入：grid = [[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]]
    # 输出：16
    # 解释：它的周长是上面图片中的 16 个黄色的边
    # 示例 2：
    # 输入：grid = [[1]]
    # 输出：4
    # 示例 3：
    # 输入：grid = [[1,0]]
    # 输出：4
    # 提示：
    # row == grid.length
    # col == grid[i].length
    # 1 <= row, col <= 100
    # grid[i][j] 为 0 或 1
# 扫描每个cell,如果当前位置为岛屿 grid[i][j] == 1， 从当前位置判断四边方向，如果边界或者是水域，证明有边界存在，res矩阵的对应cell加一。
class Solution:
    def islandPerimeter(self, grid: List[List[int]]) -> int:

        m = len(grid)
        n = len(grid[0])

        # 创建res二维素组记录答案
        res = [[0] * n for j in range(m)]

        for i in range(m):
            for j in range(len(grid[i])):
                # 如果当前位置为水域，不做修改或reset res[i][j] = 0
                if grid[i][j] == 0:
                    res[i][j] = 0
                # 如果当前位置为陆地，往四个方向判断，update res[i][j]
                elif grid[i][j] == 1:
                    if i == 0 or (i > 0 and grid[i-1][j] == 0):
                        res[i][j] += 1
                    if j == 0 or (j >0 and grid[i][j-1] == 0):
                        res[i][j] += 1
                    if i == m-1 or (i < m-1 and grid[i+1][j] == 0):
                        res[i][j] += 1
                    if j == n-1 or (j < n-1 and grid[i][j+1] == 0):
                        res[i][j] += 1

        # 最后求和res矩阵，这里其实不一定需要矩阵记录，可以设置一个variable res 记录边长，舍矩阵无非是更加形象而已
        ans = sum([sum(row) for row in res])

        return ans


#35 1356. 根据数字二进制下 1 的数目排序
    # 题目链接：https://leetcode.cn/problems/sort-integers-by-the-number-of-1-bits/
    # 给你一个整数数组 arr 。请你将数组中的元素按照其二进制表示中数字 1 的数目升序排序。
    # 如果存在多个数字二进制中 1 的数目相同，则必须将它们按照数值大小升序排列。
    # 请你返回排序后的数组。
    # 示例 1：
    # 输入：arr = [0,1,2,3,4,5,6,7,8]
    # 输出：[0,1,2,4,8,3,5,6,7]
    # 解释：[0] 是唯一一个有 0 个 1 的数。 [1,2,4,8] 都有 1 个 1 。 [3,5,6] 有 2 个 1 。 [7] 有 3 个 1 。按照 1 的个数排序得到的结果数组为 [0,1,2,4,8,3,5,6,7]
    # 示例 2：
    # 输入：arr = [1024,512,256,128,64,32,16,8,4,2,1]
    # 输出：[1,2,4,8,16,32,64,128,256,512,1024]
    # 解释：数组中所有整数二进制下都只有 1 个 1 ，所以你需要按照数值大小将它们排序。
    # 示例 3：
    # 输入：arr = [10000,10000]
    # 输出：[10000,10000]
    # 示例 4：
    # 输入：arr = [2,3,5,7,11,13,17,19]
    # 输出：[2,3,5,17,7,11,13,19]
    # 示例 5：
    # 输入：arr = [10,100,1000,10000]
    # 输出：[10,100,10000,1000]
class Solution:
    def sortByBits(self, arr: List[int]) -> List[int]:
        arr.sort(key=lambda num: (self.count_bits(num), num))
        return arr

    def count_bits(self, num: int) -> int:
        count = 0
        while num:
            num &= num - 1
            count += 1
        return count
