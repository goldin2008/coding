"""

"""
#1 Easy 27. 移除元素
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


#2 Easy 344.反转字符串
    # 编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 char[] 的形式给出。
    # 不要给另外的数组分配额外的空间，你必须原地修改输入数组、使用 O(1) 的额外空间解决这一问题。
    # 你可以假设数组中的所有字符都是 ASCII 码表中的可打印字符。
    # 示例 1：
    # 输入：["h","e","l","l","o"]
    # 输出：["o","l","l","e","h"]
    # 示例 2：
    # 输入：["H","a","n","n","a","h"]
    # 输出：["h","a","n","n","a","H"]
# （版本一） 双指针
class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        left, right = 0, len(s) - 1
        
        # 该方法已经不需要判断奇偶数，经测试后时间空间复杂度比用 for i in range(len(s)//2)更低
        # 因为while每次循环需要进行条件判断，而range函数不需要，直接生成数字，因此时间复杂度更低。推荐使用range
        while left < right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1
# （版本二） 使用栈
# class Solution:
#     def reverseString(self, s: List[str]) -> None:
#         """
#         Do not return anything, modify s in-place instead.
#         """
#         stack = []
#         for char in s:
#             stack.append(char)
#         for i in range(len(s)):
#             s[i] = stack.pop()
# # （版本三） 使用range
# class Solution:
#     def reverseString(self, s: List[str]) -> None:
#         """
#         Do not return anything, modify s in-place instead.
#         """
#         n = len(s)
#         for i in range(n // 2):
#             s[i], s[n - i - 1] = s[n - i - 1], s[i]
# # （版本四） 使用reversed
# class Solution:
#     def reverseString(self, s: List[str]) -> None:
#         """
#         Do not return anything, modify s in-place instead.
#         """
#         s[:] = reversed(s)
# # （版本五） 使用切片
# class Solution:
#     def reverseString(self, s: List[str]) -> None:
#         """
#         Do not return anything, modify s in-place instead.
#         """
#         s[:] = s[::-1]
# # （版本六） 使用列表推导
# class Solution:
#     def reverseString(self, s: List[str]) -> None:
#         """
#         Do not return anything, modify s in-place instead.
#         """
#         s[:] = [s[i] for i in range(len(s) - 1, -1, -1)]
# # （版本七） 使用reverse()
# class Solution:
#     def reverseString(self, s: List[str]) -> None:
#         """
#         Do not return anything, modify s in-place instead.
#         """
#         # 原地反转,无返回值
#         s.reverse()


#3 Easy 替换数字
    # 给定一个字符串 s，它包含小写字母和数字字符，请编写一个函数，将字符串中的字母字符保持不变，而将每个数字字符替换为number。
    # 例如，对于输入字符串 "a1b2c3"，函数应该将其转换为 "anumberbnumbercnumber"。
    # 对于输入字符串 "a5b"，函数应该将其转换为 "anumberb"
    # 输入：一个字符串 s,s 仅包含小写字母和数字字符。
    # 输出：打印一个新的字符串，其中每个数字字符都被替换为了number
    # 样例输入：a1b2c3
    # 样例输出：anumberbnumbercnumber
class Solution:
    def change(self, s):
        lst = list(s) # Python里面的string也是不可改的，所以也是需要额外空间的。空间复杂度：O(n)。
        for i in range(len(lst)):
            if lst[i].isdigit():
                lst[i] = "number"
        return ''.join(lst)


#4 Medium 151.翻转字符串里的单词
    # 给定一个字符串，逐个翻转字符串中的每个单词。
    # 示例 1：
    # 输入: "the sky is blue"
    # 输出: "blue is sky the"
    # 示例 2：
    # 输入: "  hello world!  "
    # 输出: "world! hello"
    # 解释: 输入字符串可以在前面或者后面包含多余的空格，但是反转后的字符不能包括。
    # 示例 3：
    # 输入: "a good   example"
    # 输出: "example good a"
    # 解释: 如果两个单词间有多余的空格，将反转后单词间的空格减少到只含一个。
# （版本一）先删除空白，然后整个反转，最后单词反转。 因为字符串是不可变类型，所以反转单词的时候，需要将其转换成列表，然后通过join函数再将其转换成列表，所以空间复杂度不是O(1)
class Solution:
    def reverseWords(self, s: str) -> str:
        # 删除前后空白
        s = s.strip()
        # 反转整个字符串
        s = s[::-1]
        # 将字符串拆分为单词，并反转每个单词
        s = ' '.join(word[::-1] for word in s.split())
        return s
# （版本二）使用双指针
class Solution:
    def reverseWords(self, s: str) -> str:
        # 将字符串拆分为单词，即转换成列表类型
        words = s.split()

        # 反转单词
        left, right = 0, len(words) - 1
        while left < right:
            words[left], words[right] = words[right], words[left]
            left += 1
            right -= 1

        # 将列表转换成字符串
        return " ".join(words)
# （版本三）不使用build-in function
class Solution:
        #1.去除多余的空格
        def trim_spaces(self, s):     
            n = len(s)
            left = 0
            right = n-1
        
            while left <= right and s[left] == ' ':    #去除开头的空格
                left += 1
            while left <= right and s[right] == ' ':   #去除结尾的空格
                right = right-1
            tmp = []
            while left <= right:                      #去除单词中间多余的空格
                if s[left] != ' ':
                    tmp.append(s[left])
                elif tmp[-1] != ' ':                 #当前位置是空格，但是相邻的上一个位置不是空格，则该空格是合理的
                    tmp.append(s[left])
                left += 1
            return tmp
        #2.翻转字符数组
        def reverse_string(self, nums, left, right):
            while left < right:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1
            return None
        #3.翻转每个单词
        def reverse_each_word(self, nums):
            start = 0
            end = 0
            n = len(nums)
            while start < n:
                while end < n and nums[end] != ' ':
                    end += 1
                self.reverse_string(nums, start, end-1)
                end += 1
                start = end
            return None
        #4.翻转字符串里的单词
        def reverseWords(self, s):                #测试用例："the sky is blue"
            l = self.trim_spaces(s)               #输出：['t', 'h', 'e', ' ', 's', 'k', 'y', ' ', 'i', 's', ' ', 'b', 'l', 'u', 'e'
            self.reverse_string(l,  0, len(l)-1)  #输出：['e', 'u', 'l', 'b', ' ', 's', 'i', ' ', 'y', 'k', 's', ' ', 'e', 'h', 't']
            self.reverse_each_word(l)             #输出：['b', 'l', 'u', 'e', ' ', 'i', 's', ' ', 's', 'k', 'y', ' ', 't', 'h', 'e']
            return ''.join(l)                    #输出：blue is sky the


#5 Easy 206.反转链表
    # 题意：反转一个单链表。
    # 示例: 输入: 1->2->3->4->5->NULL 输出: 5->4->3->2->1->NULL
# （版本一）双指针法
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        pre = None
        cur = head   
        
        while cur:
            temp = cur.next # 保存一下 cur的下一个节点，因为接下来要改变cur->next
            cur.next = pre #反转
            #更新pre、cur指针
            pre = cur
            cur = temp
        return pre
# （版本二）递归法
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# class Solution:
#     def reverseList(self, head: ListNode) -> ListNode:
#         return self.reverse(head, None)
#     def reverse(self, cur: ListNode, pre: ListNode) -> ListNode:
#         if cur == None:
#             return pre
#         temp = cur.next
#         cur.next = pre
#         return self.reverse(temp, cur)


#6 Medium 19.删除链表的倒数第N个节点
    # 给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。
    # 进阶：你能尝试使用一趟扫描实现吗？
    # 示例 1：
    # 输入：head = [1,2,3,4,5], n = 2 输出：[1,2,3,5] 示例 2：
    # 输入：head = [1], n = 1 输出：[] 示例 3：
    # 输入：head = [1,2], n = 1 输出：[1]
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        # 创建一个虚拟节点，并将其下一个指针设置为链表的头部
        dummy_head = ListNode(0, head)
        
        # 创建两个指针，慢指针和快指针，并将它们初始化为虚拟节点
        slow = fast = dummy_head
        
        # 快指针比慢指针快 n+1 步
        for i in range(n+1):
            fast = fast.next
        
        # 移动两个指针，直到快速指针到达链表的末尾
        while fast:
            slow = slow.next
            fast = fast.next
        
        # 通过更新第 (n-1) 个节点的 next 指针删除第 n 个节点
        slow.next = slow.next.next
        
        return dummy_head.next


#7 Easy Medium 160.链表相交
    # 给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。如果两个链表没有交点，返回 null 。
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


#8 Medium 142.环形链表II
    # 题意： 给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。
    # 为了表示给定链表中的环，使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。
    # 说明：不允许修改给定的链表。
# （版本一）快慢指针法
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        slow = head
        fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            # If there is a cycle, the slow and fast pointers will eventually meet
            if slow == fast:
                # Move one of the pointers back to the start of the list
                slow = head
                while slow != fast:
                    slow = slow.next
                    fast = fast.next
                return slow
        # If there is no cycle, return None
        return None
# （版本二）集合法
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        visited = set()
        
        while head:
            if head in visited:
                return head
            visited.add(head)
            head = head.next
        
        return None


#X9 Medium 15. 三数之和 3Sum
    # 给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有满足条件且不重复的三元组。
    # 注意： 答案中不可以包含重复的三元组。
    # 示例：
    # 给定数组 nums = [-1, 0, 1, 2, -1, -4]，
    # 满足要求的三元组集合为： [ [-1, 0, 1], [-1, -1, 2] ]
# ***（版本一） 双指针
# 先定一个数，然后用双指针法找另外两个数
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        result = []
        nums.sort()
        n = len(nums)

        for i in range(n):
            # 如果第一个元素已经大于0，不需要进一步检查
            if nums[i] > 0:
                # return result
                break
            # 跳过相同的元素以避免重复
            if i > 0 and nums[i] == nums[i - 1]:
                continue
                
            left = i + 1
            right = n - 1
            
            while left < right:
                sum_ = nums[i] + nums[left] + nums[right]
                if sum_ < 0:
                    left += 1
                elif sum_ > 0:
                    right -= 1
                else:
                    result.append([nums[i], nums[left], nums[right]])
                    # 跳过相同的元素以避免重复
                    left += 1
                    while right > left and nums[left] == nums[left - 1]:
                        left += 1
        return result
# （版本二） 使用字典
# class Solution:
#     def threeSum(self, nums: List[int]) -> List[List[int]]:
#         result = []
#         nums.sort()
#         # 找出a + b + c = 0
#         # a = nums[i], b = nums[j], c = -(a + b)
#         for i in range(len(nums)):
#             # 排序之后如果第一个元素已经大于零，那么不可能凑成三元组
#             if nums[i] > 0:
#                 break
#             if i > 0 and nums[i] == nums[i - 1]: #三元组元素a去重
#                 continue
#             d = {}
#             for j in range(i + 1, len(nums)):
#                 if j > i + 2 and nums[j] == nums[j-1] == nums[j-2]: # 三元组元素b去重
#                     continue
#                 c = 0 - (nums[i] + nums[j])
#                 if c in d:
#                     result.append([nums[i], nums[j], c])
#                     d.pop(c) # 三元组元素c去重
#                 else:
#                     d[nums[j]] = j
#         return result

# Triplet Sum
# Given an array of integers, return all triplets [a, b, c] such that a + b + c = 0 . 
# The solution must not contain duplicate triplets (e.g., [1, 2, 3] and [2, 3, 1] are considered duplicates). 
# If no such triplets are found, return an empty array.
# Each triplet can be arranged in any order, and the output can be returned in any order.
# Example:
# Input: nums = [0, -1, 2, -3, 1]
# Output: [[-3, 1, 2], [-1, 0, 1]]
def triplet_sum_brute_force(nums: List[int]) -> List[List[int]]:
    n = len(nums)
    # Use a hash set to ensure we don't add duplicate triplets.
    triplets = set()
    # Iterate through the indexes of all triplets.
    for i in range(n):
        for j in range(i + 1, n):          
            for k in range(j + 1, n):
                if nums[i] + nums[j] + nums[k] == 0:
                    # Sort the triplet before including it in the 
                    # hash set.
                    triplet = tuple(sorted([nums[i], nums[j], nums[k]]))
                    triplets.add(triplet)
    return [list(triplet) for triplet in triplets]

# *** Pair Sum - Sorted *** BEST SOLUTION
def triplet_sum(nums: List[int]) -> List[List[int]]:
    triplets = []
    nums.sort()
    for i in range(len(nums)):
        # Optimization: triplets consisting of only positive numbers 
        # will never sum to 0.
        
        if nums[i] > 0:
            break
        # To avoid duplicate triplets, skip 'a' if it's the same as 
        # the previous number.
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        # Find all pairs that sum to a target of '-a' (-nums[i]).
        pairs = pair_sum_sorted_all_pairs(nums, i + 1, -nums[i])
        for pair in pairs:
            triplets.append([nums[i]] + pair)
    return triplets

def pair_sum_sorted_all_pairs(nums: List[int], start: int, target: int) -> List[int]:
    pairs = []
    left, right = start, len(nums) - 1
    while left < right:
        sum = nums[left] + nums[right]
        if sum < target:
            left += 1
        elif sum > target:
            right -= 1
        else:
            pairs.append([nums[left], nums[right]])
            left += 1
            # To avoid duplicate '[b, c]' pairs, skip 'b' if it's the 
            # same as the previous number.
            while left < right and nums[left] == nums[left - 1]:
                left += 1
    return pairs

#10 Medium 第18题. 四数之和 4Sum
    # 题意：给定一个包含 n 个整数的数组 nums 和一个目标值 target，判断 nums 中是否存在四个元素 a，b，c 和 d ，使得 a + b + c + d 的值与 target 相等？找出所有满足条件且不重复的四元组。
    # 注意：
    # 答案中不可以包含重复的四元组。
    # 示例： 给定数组 nums = [1, 0, -1, 0, -2, 2]，和 target = 0。 满足要求的四元组集合为： [ [-1, 0, 0, 1], [-2, -1, 1, 2], [-2, 0, 0, 2] ]
# (版本一) 双指针
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()
        n = len(nums)
        result = []
        for i in range(n):
            # if nums[i] > target and nums[i] > 0 and target > 0:# 剪枝（可省）
            #     break
            if i > 0 and nums[i] == nums[i-1]:# 去重
                continue
            for j in range(i+1, n):
                # if nums[i] + nums[j] > target and target > 0: #剪枝（可省）
                #     break
                if j > i+1 and nums[j] == nums[j-1]: # 去重
                    continue
                left = j + 1
                right = n - 1
                while left < right:
                    sum_ = nums[i] + nums[j] + nums[left] + nums[right]
                    if sum_ < target:
                        left += 1
                    elif sum_ > target:
                        right -= 1
                    else:
                        result.append([nums[i], nums[j], nums[left], nums[right]])
                        left += 1
                        while left < right and nums[left] == nums[left - 1]:
                            left += 1
        return result
# (版本二) 使用字典
# class Solution(object):
#     def fourSum(self, nums, target):
#         """
#         :type nums: List[int]
#         :type target: int
#         :rtype: List[List[int]]
#         """
#         # 创建一个字典来存储输入列表中每个数字的频率
#         freq = {}
#         for num in nums:
#             freq[num] = freq.get(num, 0) + 1
        
#         # 创建一个集合来存储最终答案，并遍历4个数字的所有唯一组合
#         ans = set()
#         for i in range(len(nums)):
#             for j in range(i + 1, len(nums)):
#                 for k in range(j + 1, len(nums)):
#                     val = target - (nums[i] + nums[j] + nums[k])
#                     if val in freq:
#                         # 确保没有重复
#                         count = (nums[i] == val) + (nums[j] == val) + (nums[k] == val)
#                         if freq[val] > count:
#                             ans.add(tuple(sorted([nums[i], nums[j], nums[k], val])))
        
#         return [list(x) for x in ans]


#X11 Easy 2Sum Pair Sum - Sorted
# Given an array of integers sorted in ascending order and a target value, 
# return the indexes of any pair of numbers in the array that sum to the target. 
# The order of the indexes in the result doesn't matter. If no pair is found, return an empty array.
# Example 1:
# Input: nums = [-5, -2, 3, 4, 6], target = 7
# Output: [2, 3]
# Explanation: nums[2] + nums[3] = 3 + 4 = 7
# Example 2:
# Input: nums = [1, 1, 1], target = 2
# Output: [0, 1]
# Explanation: other valid outputs could be [1, 0], [0, 2], [2, 0], [1, 2] or [2, 1].
def pair_sum_sorted_brute_force(nums: List[int], target: int) -> List[int]:
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []

def pair_sum_sorted(nums: List[int], target: int) -> List[int]:
    left, right = 0, len(nums) - 1
    while left < right:
        sum = nums[left] + nums[right]
        # If the sum is smaller, increment the left pointer, aiming
        # to increase the sum toward the target value.
        if sum < target:
            left += 1
        # If the sum is larger, decrement the right pointer, aiming
        # to decrease the sum toward the target value.
        elif sum > target:
            right -= 1
        # If the target pair is found, return its indexes.
        else:   
            return [left, right]
    return []

#X12 Easy Is Palindrome Valid
# A palindrome is a sequence of characters that reads the same forward and backward.
# Given a string, determine if it's a palindrome after removing all non-alphanumeric characters. 
# A character is alphanumeric if it's either a letter or a number.
# Example 1:
# Input: s = 'a dog! a panic in a pagoda.'
# Output: True
def is_palindrome_valid(s: str) -> bool:    
    left, right = 0, len(s) - 1
    while left < right:   
        # Skip non-alphanumeric characters from the left.     
        while left < right and not s[left].isalnum():
            left += 1
        # Skip non-alphanumeric characters from the right.
        while left < right and not s[right].isalnum():
            right -= 1
        # If the characters at the left and right pointers don't
        # match, the string is not a palindrome.
        if s[left] != s[right]:
            return False 
        left += 1
        right -= 1
    return True

#X13 Medium Largest Container
# You are given an array of numbers, each representing the height of a vertical 
# line on a graph. A container can be formed with any pair of these lines, 
# along with the x-axis of the graph. Return the amount of water which the largest container can hold.
# Input: heights = [2, 7, 8, 3, 7, 6]
# Output: 24
def largest_container_brute_force(heights: List[int]) -> int:
    n = len(heights)
    max_water = 0
    # Find the maximum amount of water stored between all pairs of
    # lines.
    for i in range(n):
        for j in range(i + 1, n):
            water = min(heights[i], heights[j]) * (j - i)
            max_water = max(max_water, water)
    return max_water

def largest_container(heights: List[int]) -> int:
    max_water = 0
    left, right = 0, len(heights) - 1
    while (left < right):
        # Calculate the water contained between the current pair of 
        # lines.
        water = min(heights[left], heights[right]) * (right - left)
        max_water = max(max_water, water)
        # Move the pointers inward, always moving the pointer at the 
        # shorter line. If both lines have the same height, move both 
        # pointers inward.
        if (heights[left] < heights[right]):
            left += 1
        elif (heights[left] > heights[right]):
            right -= 1
        else:
            left += 1
            right -= 1
    return max_water

#X14 Easy Shift Zeros to the End
# Given an array of integers, modify the array in place to move all zeros to the end 
# while maintaining the relative order of non-zero elements.
# Example:
# Input: nums = [0, 1, 0, 3, 2]
# Output: [1, 3, 2, 0, 0]
def shift_zeros_to_the_end_naive(nums: List[int]) -> None:
    temp = [0] * len(nums)
    i = 0
    # Add all non-zero elements to the left of 'temp'.
    for num in nums:
        if num != 0:
            temp[i] = num
            i += 1
    # Set 'nums' to 'temp'.
    for j in range(len(nums)):
        nums[j] = temp[j]

def shift_zeros_to_the_end(nums: List[int]) -> None:
    # The 'left' pointer is used to position non-zero elements.
    left = 0
    # Iterate through the array using a 'right' pointer to locate non-zero 
    # elements.
    for right in range(len(nums)):
        if nums[right] != 0:
            nums[left], nums[right] = nums[right], nums[left]
            # Increment 'left' since it now points to a position already occupied 
            # by a non-zero element.
            left += 1

#X15 Medium Next Lexicographical Sequence
# Given a string of lowercase English letters, rearrange the characters to form a 
# new string representing the next immediate sequence in lexicographical (alphabetical) 
# order. If the given string is already last in lexicographical order among all 
# possible arrangements, return the arrangement that's first in lexicographical order.
# Example 1:
# Input: s = 'abcd'
# Output: 'abdc'
# Explanation: "abdc" is the next sequence in lexicographical order after rearranging "abcd".
# Example 2:
# Input: s = 'dcba'
# Output: 'abcd'
# Explanation: Since "dcba" is the last sequence in lexicographical order, we return the first sequence: "abcd".
def next_lexicographical_sequence(s: str) -> str:
    letters = list(s)
    # Locate the pivot, which is the first character from the right that breaks 
    # non-increasing order. Start searching from the second-to-last position.
    pivot = len(letters) - 2
    while pivot >= 0 and letters[pivot] >= letters[pivot + 1]:
        pivot -= 1
    # If pivot is not found, the string is already in its largest permutation. In
    # this case, reverse the string to obtain the smallest permutation.
    if pivot == -1:
        return ''.join(reversed(letters))
    # Find the rightmost successor to the pivot.
    rightmost_successor = len(letters) - 1
    while letters[rightmost_successor] <= letters[pivot]:
        rightmost_successor -= 1
    # Swap the rightmost successor with the pivot to increase the lexicographical
    # order of the suffix.
    letters[pivot], letters[rightmost_successor] = (letters[rightmost_successor], letters[pivot])
    # Reverse the suffix after the pivot to minimize its permutation.
    letters[pivot + 1:] = reversed(letters[pivot + 1:])
    return ''.join(letters)


# Fast And Slow Pointers
#X16 Easy 141. Easy Linked List Loop
# Given a singly linked list, determine if it contains a cycle. 
# A cycle occurs if a node's next pointer references an earlier node in the linked list, causing a loop.
from ds import ListNode
"""
Definition of ListNode:
class ListNode:
    def __init__(self, val=None, next=None):
        self.val = val
        self.next = next
"""
def linked_list_loop_naive(head: ListNode) -> bool:
    visited = set()  
    curr = head
    while curr:  
        # Cycle detected if the current node has already been visited.
        if curr in visited:
            return True  
        visited.add(curr)  
        curr = curr.next
    return False  

def linked_list_loop(head: ListNode) -> bool:
    slow = fast = head
    # Check both 'fast' and 'fast.next' to avoid null pointer
    # exceptions when we perform 'fast.next' and 'fast.next.next'.
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if fast == slow:
            return True
    return False

#X17 Easy Linked List Midpoint
# Given a singly linked list, find and return its middle node. If there are two middle nodes, 
# return the second one.
def linked_list_midpoint(head: ListNode) -> ListNode:
    slow = fast = head
    # When the fast pointer reaches the end of the list, the slow
    # pointer will be at the midpoint of the linked list.
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow

#X18 Medium Happy Number
# In number theory, a happy number is defined as a number that, when repeatedly subjected to the 
# process of squaring its digits and summing those squares, eventually leads to 1. 
# An unhappy number will never reach 1 during this process, and will get stuck in an infinite loop.
# Given an integer, determine if it's a happy number.
# Example:
# Input: n = 23
# Output: True
def happy_number(n: int) -> bool:
    slow = fast = n
    while True:
        slow = get_next_num(slow)
        fast = get_next_num(get_next_num(fast))
        if fast == 1:
            return True
        # If the fast and slow pointers meet, a cycle is detected. 
        # Hence, 'n' is not a happy number.
        elif fast == slow:
            return False

def get_next_num(x: int) -> int:
    next_num = 0
    while x > 0:
        # Extract the last digit of 'x'.
        digit = x % 10
        # Truncate (remove) the last digit from 'x' using floor 
        # division.
        x //= 10
        # Add the square of the extracted digit to the sum.
        next_num += digit ** 2
    return next_num

class Solution:
    def isHappy(self, n: int) -> bool:
        def get_next(n):
            total_sum = 0
            while n > 0:
                n, digit = divmod(n, 10)
                total_sum += digit ** 2
            return total_sum

        seen = set()
        while n != 1 and n not in seen:
            seen.add(n)
            n = get_next(n)

        return n == 1

class Solution:
    def isHappy(self, n: int) -> bool:
        def get_next(x):
            res = 0
            while x > 0:
                res += (x % 10) ** 2
                x = x // 10
            return res
        
        seen = set()
        while True:
            n = get_next(n)
            if n in seen:
                return False
            if n == 1:
                return True
            seen.add(n)