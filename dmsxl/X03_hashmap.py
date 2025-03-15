"""
双指针法将时间复杂度: O(n^2)的解法优化为 O(n)的解法。也就是降一个数量级，题目如下：
27.移除元素
15.三数之和
18.四数之和

链表相关双指针题目：
206.反转链表
19.删除链表的倒数第N个节点
面试题 02.07. 链表相交
142题.环形链表II
"""
#1 Easy 242.有效的字母异位词
    # 给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的字母异位词。
    # 示例 1: 输入: s = "anagram", t = "nagaram" 输出: true
    # 示例 2: 输入: s = "rat", t = "car" 输出: false
    # 说明: 你可以假设字符串只包含小写字母。
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        record = [0] * 26
        for i in s:
            #并不需要记住字符a的ASCII，只要求出一个相对数值就可以了
            record[ord(i) - ord("a")] += 1
        for i in t:
            record[ord(i) - ord("a")] -= 1
        for i in range(26):
            if record[i] != 0:
                #record数组如果有的元素不为零0，说明字符串s和t 一定是谁多了字符或者谁少了字符。
                return False
        return True
# Python写法二（没有使用数组作为哈希表，只是介绍defaultdict这样一种解题思路）：
# class Solution:
#     def isAnagram(self, s: str, t: str) -> bool:
#         from collections import defaultdict
        
#         s_dict = defaultdict(int)
#         t_dict = defaultdict(int)
#         for x in s:
#             s_dict[x] += 1
        
#         for x in t:
#             t_dict[x] += 1
#         return s_dict == t_dict
# # Python写法三(没有使用数组作为哈希表，只是介绍Counter这种更方便的解题思路)：
# class Solution(object):
#     def isAnagram(self, s: str, t: str) -> bool:
#         from collections import Counter
#         a_count = Counter(s)
#         b_count = Counter(t)
#         return a_count == b_count


#2 Easy 349. 两个数组的交集
    # 题意：给定两个数组，编写一个函数来计算它们的交集。
# （版本一） 使用字典和集合
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
    # 使用哈希表存储一个数组中的所有元素
        table = {}
        for num in nums1:
            table[num] = table.get(num, 0) + 1
        
        # 使用集合存储结果
        res = set()
        for num in nums2:
            if num in table:
                res.add(num)
                del table[num]
        
        return list(res)
# （版本二） 使用数组
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        count1 = [0]*1001
        count2 = [0]*1001
        result = []
        for i in range(len(nums1)):
            count1[nums1[i]]+=1
        for j in range(len(nums2)):
            count2[nums2[j]]+=1
        for k in range(1001):
            if count1[k]*count2[k]>0:
                result.append(k)
        return result
# （版本三） 使用集合
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        return list(set(nums1) & set(nums2))
# ***
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        val_dict = {}
        ans = []
        for num in nums1:
            val_dict[num] = 1

        for num in nums2:
            if num in val_dict.keys() and val_dict[num] == 1:
                ans.append(num)
                val_dict[num] = 0
        return ans

#3 Easy 第202题. 快乐数
    # 编写一个算法来判断一个数 n 是不是快乐数。
    # 「快乐数」定义为：对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和，然后重复这个过程直到这个数变为 1，也可能是 无限循环 但始终变不到 1。如果 可以变为  1，那么这个数就是快乐数。
    # 如果 n 是快乐数就返回 True ；不是，则返回 False 。
    # 示例：
    # 输入：19
    # 输出：true
    # 解释：
    # 1^2 + 9^2 = 82
    # 8^2 + 2^2 = 68
    # 6^2 + 8^2 = 100
    # 1^2 + 0^2 + 0^2 = 1
# (版本一)使用集合
class Solution:
    def isHappy(self, n: int) -> bool:        
        record = set()

        while True:
            n = self.get_sum(n)
            if n == 1:
                return True
            # 如果中间结果重复出现，说明陷入死循环了，该数不是快乐数
            if n in record:
                return False
            else:
                record.add(n)

    def get_sum(self,n: int) -> int: 
        new_num = 0
        while n:
            n, r = divmod(n, 10)
            new_num += r ** 2
        return new_num
# (版本二)使用集合
class Solution:
   def isHappy(self, n: int) -> bool:
       record = set()
       while n not in record:
           record.add(n)
           new_num = 0
           n_str = str(n)
           for i in n_str:
               new_num+=int(i)**2
           if new_num==1: return True
           else: n = new_num
       return False
# (版本三)使用数组
class Solution:
   def isHappy(self, n: int) -> bool:
       record = []
       while n not in record:
           record.append(n)
           new_num = 0
           n_str = str(n)
           for i in n_str:
               new_num+=int(i)**2
           if new_num==1: return True
           else: n = new_num
       return False
# (版本四)使用快慢指针
class Solution:
   def isHappy(self, n: int) -> bool:        
       slow = n
       fast = n
       while self.get_sum(fast) != 1 and self.get_sum(self.get_sum(fast)):
           slow = self.get_sum(slow)
           fast = self.get_sum(self.get_sum(fast))
           if slow == fast:
               return False
       return True
   def get_sum(self,n: int) -> int: 
       new_num = 0
       while n:
           n, r = divmod(n, 10)
           new_num += r ** 2
       return new_num
# (版本五)使用集合+精简
class Solution:
   def isHappy(self, n: int) -> bool:
       seen = set()
       while n != 1:
           n = sum(int(i) ** 2 for i in str(n))
           if n in seen:
               return False
           seen.add(n)
       return True
# (版本六)使用数组+精简
class Solution:
   def isHappy(self, n: int) -> bool:
       seen = []
       while n != 1:
           n = sum(int(i) ** 2 for i in str(n))
           if n in seen:
               return False
           seen.append(n)
       return True


#4 Easy 1.两数之和 2Sum
    # 给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。
    # 你可以假设每种输入只会对应一个答案。但是，数组中同一个元素不能使用两遍。
    # 示例:
    # 给定 nums = [2, 7, 11, 15], target = 9
    # 因为 nums[0] + nums[1] = 2 + 7 = 9
    # 所以返回 [0, 1]
# ***（版本一） 使用字典
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        records = dict()

        for index, value in enumerate(nums):  
            if target - value in records:   # 遍历当前元素，并在map中寻找是否有匹配的key
                return [records[target- value], index]
            records[value] = index    # 如果没找到匹配对，就把访问过的元素和下标加入到map中
        return []
# （版本二）使用集合
# class Solution:
#     def twoSum(self, nums: List[int], target: int) -> List[int]:
#         #创建一个集合来存储我们目前看到的数字
#         seen = set()             
#         for i, num in enumerate(nums):
#             complement = target - num
#             if complement in seen:
#                 return [nums.index(complement), i]
#             seen.add(num)
# # （版本三）使用双指针
# class Solution:
#     def twoSum(self, nums: List[int], target: int) -> List[int]:
#         # 对输入列表进行排序
#         nums_sorted = sorted(nums)
        
#         # 使用双指针
#         left = 0
#         right = len(nums_sorted) - 1
#         while left < right:
#             current_sum = nums_sorted[left] + nums_sorted[right]
#             if current_sum == target:
#                 # 如果和等于目标数，则返回两个数的下标
#                 left_index = nums.index(nums_sorted[left])
#                 right_index = nums.index(nums_sorted[right])
#                 if left_index == right_index:
#                     right_index = nums[left_index+1:].index(nums_sorted[right]) + left_index + 1
#                 return [left_index, right_index]
#             elif current_sum < target:
#                 # 如果总和小于目标，则将左侧指针向右移动
#                 left += 1
#             else:
#                 # 如果总和大于目标值，则将右指针向左移动
#                 right -= 1
# # （版本四）暴力法
# class Solution:
#     def twoSum(self, nums: List[int], target: int) -> List[int]:
#         for i in range(len(nums)):
#             for j in range(i+1, len(nums)):
#                 if nums[i] + nums[j] == target:
#                     return [i,j]


#5 Medium 第454题.四数相加II
    # 给定四个包含整数的数组列表 A , B , C , D ,计算有多少个元组 (i, j, k, l) ，使得 A[i] + B[j] + C[k] + D[l] = 0。
    # 为了使问题简单化，所有的 A, B, C, D 具有相同的长度 N，且 0 ≤ N ≤ 500 。所有整数的范围在 -2^28 到 2^28 - 1 之间，最终结果不会超过 2^31 - 1 。
    # 例如:
    # 输入:
    # A = [ 1, 2]
    # B = [-2,-1]
    # C = [-1, 2]
    # D = [ 0, 2]
    # 输出:
    # 2
    # 解释:
    # 两个元组如下:
    # (0, 0, 0, 1) -> A[0] + B[0] + C[0] + D[1] = 1 + (-2) + (-1) + 2 = 0
    # (1, 1, 0, 0) -> A[1] + B[1] + C[0] + D[0] = 2 + (-1) + (-1) + 0 = 0
# （版本一） 使用字典
class Solution(object):
    def fourSumCount(self, nums1, nums2, nums3, nums4):
        # 使用字典存储nums1和nums2中的元素及其和
        hashmap = dict()
        for n1 in nums1:
            for n2 in nums2:
                if n1 + n2 in hashmap:
                    hashmap[n1+n2] += 1
                else:
                    hashmap[n1+n2] = 1
        
        # 如果 -(n1+n2) 存在于nums3和nums4, 存入结果
        count = 0
        for n3 in nums3:
            for n4 in nums4:
                key = - n3 - n4
                if key in hashmap:
                    count += hashmap[key]
        return count
# （版本二） 使用字典
class Solution(object):
    def fourSumCount(self, nums1, nums2, nums3, nums4):
        # 使用字典存储nums1和nums2中的元素及其和
        hashmap = dict()
        for n1 in nums1:
            for n2 in nums2:
                hashmap[n1+n2] = hashmap.get(n1+n2, 0) + 1
        
        # 如果 -(n1+n2) 存在于nums3和nums4, 存入结果
        count = 0
        for n3 in nums3:
            for n4 in nums4:
                key = - n3 - n4
                if key in hashmap:
                    count += hashmap[key]
        return count
# （版本三）使用 defaultdict
from collections import defaultdict 
class Solution:
    def fourSumCount(self, nums1: list, nums2: list, nums3: list, nums4: list) -> int:
        rec, cnt = defaultdict(lambda : 0), 0
        for i in nums1:
            for j in nums2:
                rec[i+j] += 1
        for i in nums3:
            for j in nums4:
                cnt += rec.get(-(i+j), 0) 
        return cnt

class Solution:
    def fourSumCount(self, nums1: List[int], nums2: List[int], nums3: List[int], nums4: List[int]) -> int:
        hashmap = {}
        for num1 in nums1:
            for num2 in nums2:
                sum_ = num1 + num2
                hashmap[sum_] = hashmap.get(sum_, 0) + 1
        
        count = 0
        for num3 in nums3:
            for num4 in nums4:
                sum_com = -(num3 + num4)
                count += hashmap.get(sum_com, 0)
        return count

#6 Easy 383.赎金信
    # 给定一个赎金信 (ransom) 字符串和一个杂志(magazine)字符串，判断第一个字符串 ransom 能不能由第二个字符串 magazines 里面的字符构成。如果可以构成，返回 true ；否则返回 false。
    # (题目说明：为了不暴露赎金信字迹，要从杂志上搜索各个需要的字母，组成单词来表达意思。杂志字符串中的每个字符只能在赎金信字符串中使用一次。)
    # 注意：
    # 你可以假设两个字符串均只含有小写字母。
    # canConstruct("a", "b") -> false
    # canConstruct("aa", "ab") -> false
    # canConstruct("aa", "aab") -> true
class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        record = [0] * 26
        for r in magazine:
            record[ord(r)-ord('a')] += 1

        for m in ransomNote:
            if record[ord(m)-ord('a')] < 1:
                return False
            record[ord(m)-ord('a')] -= 1
        
        return True
# ***（版本一）使用数组
class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        ransom_count = [0] * 26
        magazine_count = [0] * 26
        for c in ransomNote:
            ransom_count[ord(c) - ord('a')] += 1
        for c in magazine:
            magazine_count[ord(c) - ord('a')] += 1
        return all(ransom_count[i] <= magazine_count[i] for i in range(26))
# （版本二）使用defaultdict
# from collections import defaultdict

# class Solution:
#     def canConstruct(self, ransomNote: str, magazine: str) -> bool:

#         hashmap = defaultdict(int)

#         for x in magazine:
#             hashmap[x] += 1

#         for x in ransomNote:
#             value = hashmap.get(x)
#             if not value or not value:
#                 return False
#             else:
#                 hashmap[x] -= 1

#         return True
# （版本三）使用字典
class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        counts = {}
        for c in magazine:
            counts[c] = counts.get(c, 0) + 1
        for c in ransomNote:
            if c not in counts or counts[c] == 0:
                return False
            counts[c] -= 1
        return True
# （版本四）使用Counter
# from collections import Counter

# class Solution:
#     def canConstruct(self, ransomNote: str, magazine: str) -> bool:
#         return not Counter(ransomNote) - Counter(magazine)
# # （版本五）使用count
# class Solution:
#     def canConstruct(self, ransomNote: str, magazine: str) -> bool:
#         return all(ransomNote.count(c) <= magazine.count(c) for c in set(ransomNote))
# # (版本六）使用count(简单易懂)
# class Solution:
#     def canConstruct(self, ransomNote: str, magazine: str) -> bool:
#         for char in ransomNote:
#             if char in magazine and ransomNote.count(char) <= magazine.count(char):
#                 continue
#             else:
#                 return False
#         return True


#7 Medium 第15题. 三数之和 3Sum
    # 给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有满足条件且不重复的三元组。
    # 注意： 答案中不可以包含重复的三元组。
    # 示例：
    # 给定数组 nums = [-1, 0, 1, 2, -1, -4]，
    # 满足要求的三元组集合为： [ [-1, 0, 1], [-1, -1, 2] ]
# 两数之和 就不能使用双指针法，因为1.两数之和 要求返回的是索引下标， 而双指针法一定要排序，一旦排序之后原数组的索引就被改变了。
# 如果1.两数之和 要求返回的是数值的话，就可以使用双指针法了。

# 说到去重，其实主要考虑三个数的去重。 a, b ,c, 对应的就是 nums[i]，nums[left]，nums[right]
# a 如果重复了怎么办，a是nums里遍历的元素，那么应该直接跳过去。
# ***（版本一） 双指针
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        result = []
        nums.sort()
        
        for i in range(len(nums)):
            # 如果第一个元素已经大于0，不需要进一步检查
            if nums[i] > 0:
                return result
            
            # a去重, 跳过相同的元素以避免重复
            if i > 0 and nums[i] == nums[i - 1]:
                continue
                
            left = i + 1
            right = len(nums) - 1
            
            while right > left:
                sum_ = nums[i] + nums[left] + nums[right]
                
                if sum_ < 0:
                    left += 1
                elif sum_ > 0:
                    right -= 1
                else:
                    result.append([nums[i], nums[left], nums[right]])
                    # b,c去重, 跳过相同的元素以避免重复
                    # while right > left and nums[right] == nums[right - 1]:
                    #     right -= 1
                    left += 1
                    while right > left and nums[left] == nums[left - 1]:
                        left += 1
                    # right -= 1
                    # left += 1
                    
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


#8 Medium 第18题. 四数之和 4Sum
    # 题意：给定一个包含 n 个整数的数组 nums 和一个目标值 target，判断 nums 中是否存在四个元素 a，b，c 和 d ，使得 a + b + c + d 的值与 target 相等？找出所有满足条件且不重复的四元组。
    # 注意：
    # 答案中不可以包含重复的四元组。
    # 示例： 给定数组 nums = [1, 0, -1, 0, -2, 2]，和 target = 0。 满足要求的四元组集合为： [ [-1, 0, 0, 1], [-2, -1, 1, 2], [-2, 0, 0, 2] ]
# 不要判断nums[k] > target 就返回了，三数之和 可以通过 nums[i] > 0 就返回了，因为 0 已经是确定的数了，四数之和这道题目 target是任意值。
# 比如：数组是[-4, -3, -2, -1]，target是-10，不能因为-4 > -10而跳过。
# 但是我们依旧可以去做剪枝，逻辑变成nums[i] > target && (nums[i] >=0 || target >= 0)就可以了。
    
# 15.三数之和 的双指针解法是一层for循环num[i]为确定值，然后循环内有left和right下标作为双指针，找到nums[i] + nums[left] + nums[right] == 0。
# 四数之和的双指针解法是两层for循环nums[k] + nums[i]为确定值，依然是循环内有left和right下标作为双指针，
# 找出nums[k] + nums[i] + nums[left] + nums[right] == target的情况，三数之和的时间复杂度是O(n^2)，四数之和的时间复杂度是O(n^3) 。
# 那么一样的道理，五数之和、六数之和等等都采用这种解法。
# 对于15.三数之和 双指针法就是将原本暴力O(n^3)的解法，降为O(n^2)的解法，四数之和的双指针解法就是将原本暴力O(n^4)的解法，降为O(n^3)的解法。
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

                left, right = j+1, n-1
                while left < right:
                    s = nums[i] + nums[j] + nums[left] + nums[right]
                    if s == target:
                        result.append([nums[i], nums[j], nums[left], nums[right]])
                        left += 1
                        while left < right and nums[left] == nums[left-1]:
                            left += 1
                        # while left < right and nums[right] == nums[right-1]:
                        #     right -= 1
                        # right -= 1
                    elif s < target:
                        left += 1
                    else:
                        right -= 1
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

#X9 Easy 1.Pair Sum - Unsorted 2Sum
# Given an array of integers, return the indexes of any two numbers that add up to a target. 
# The order of the indexes in the result doesn't matter. If no pair is found, return an empty array.
# Example:
# Input: nums = [-1, 3, 4, 2], target = 3
# Output: [0, 2]
# Explanation: nums[0] + nums[2] = -1 + 4 = 3
def pair_sum_unsorted(nums: List[int], target: int) -> List[int]:
    hashmap = {}   
    for i, x in enumerate(nums):
        if target - x in hashmap:
            return [hashmap[target - x], i]
        hashmap[x] = i
    return []

def pair_sum_unsorted_two_pass(nums: List[int], target: int) -> List[int]:
    num_map = {}
    # First pass: Populate the hash map with each number and its 
    # index.
    for i, num in enumerate(nums):
        num_map[num] = i
    # Second pass: Check for each number's complement in the hash map.
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_map and num_map[complement] != i:
            return [i, num_map[complement]]
    return []

#X10 Medium 36.Verify Sudoku Board
# Given a partially completed 9×9 Sudoku board, determine if the current state of the board adheres to the rules of the game:
# Each row and column must contain unique numbers between 1 and 9, or be empty (represented as 0).
# Each of the nine 3×3 subgrids that compose the grid must contain unique numbers between 1 and 9, or be empty.
# Note: You are asked to determine whether the current state of the board is valid given these rules, not whether 
# the board is solvable.
def verify_sudoku_board(board: List[List[int]]) -> bool:
    # Create hash sets for each row, column, and subgrid to keep 
    # track of numbers previously seen on any given row, column, or 
    # subgrid.
    row_sets = [set() for _ in range(9)]
    column_sets = [set() for _ in range(9)]
    subgrid_sets = [[set() for _ in range(3)] for _ in range(3)]
    for r in range(9):
        for c in range(9):
            num = board[r][c]
            if num == 0:
                continue
            # Check if 'num' has been seen in the current row, 
            # column, or subgrid.
            if num in row_sets[r]:
                return False
            if num in column_sets[c]:
                return False
            if num in subgrid_sets[r // 3][c // 3]:
                return False
            # If we passed the above checks, mark this value as seen 
            # by adding it to its corresponding hash sets.
            row_sets[r].add(num)
            column_sets[c].add(num)
            subgrid_sets[r // 3][c // 3].add(num)
    return True

#X11 Medium 7.Zero Striping
# For each zero in an m x n matrix, set its entire row and column to zero in place.
def zero_striping(matrix: List[List[int]]) -> None:
    if not matrix or not matrix[0]:
        return
    m, n = len(matrix), len(matrix[0])
    # Check if the first row initially contains a zero.
    first_row_has_zero = False
    for c in range(n):
        if matrix[0][c] == 0:
            first_row_has_zero = True
            break
    # Check if the first column initially contains a zero.
    first_col_has_zero = False
    for r in range(m):
        if matrix[r][0] == 0:
            first_col_has_zero = True
            break
    # Use the first row and column as markers. If an element in the 
    # submatrix is zero, mark its corresponding row and column in the 
    # first row and column as 0.
    for r in range(1, m):
        for c in range(1, n):
            if matrix[r][c] == 0:
                matrix[0][c] = 0
                matrix[r][0] = 0
    # Update the submatrix using the markers in the first row and 
    # column.
    for r in range(1, m):
        for c in range(1, n):
            if matrix[0][c] == 0 or matrix[r][0] == 0:
                matrix[r][c] = 0
    # If the first row had a zero initially, set all elements in the 
    # first row to zero.
    if first_row_has_zero:
        for c in range(n):
            matrix[0][c] = 0
    # If the first column had a zero initially, set all elements in 
    # the first column to zero.
    if first_col_has_zero:
        for r in range(m):
            matrix[r][0] = 0

def zero_striping_hash_sets(matrix: List[List[int]]) -> None:     
    if not matrix or not matrix[0]:
        return
    m, n = len(matrix), len(matrix[0])
    zero_rows, zero_cols = set(), set()
    # Pass 1: Traverse through the matrix to identify the rows and
    # columns containing zeros and store their indexes in the 
    # appropriate hash sets.
    for r in range(m):
        for c in range(n):
            if matrix[r][c] == 0:
                zero_rows.add(r)
                zero_cols.add(c)
    # Pass 2: Set any cell in the matrix to zero if its row index is 
    # in 'zero_rows' or its column index is in 'zero_cols'.
    for r in range(m):
        for c in range(n):
            if r in zero_rows or c in zero_cols:
                matrix[r][c] = 0

#X12 Medium 128.Longest Chain of Consecutive Numbers
# Find the longest chain of consecutive numbers in an*array. 
# Two numbers are consecutive if they have a difference of 1.
# Example:
# Input: nums = [1, 6, 2, 5, 8, 7, 10, 3]
# Output: 4
# Explanation: The longest chain of consecutive numbers is 5, 6, 7, 8.
def longest_chain_of_consecutive_numbers_brute_force(nums: List[int]) -> int:
    if not nums:
        return 0
    longest_chain = 0
    # Look for chains of consecutive numbers that start from each number.
    for num in nums:
        current_num = num
        current_chain = 1
        # Continue to find the next consecutive numbers in the chain.
        while (current_num + 1) in nums:
            current_num += 1
            current_chain += 1
        longest_chain = max(longest_chain, current_chain)
    return longest_chain

def longest_chain_of_consecutive_numbers(nums: List[int]) -> int:
    if not nums:
        return 0
    num_set = set(nums)
    longest_chain = 0
    for num in num_set:
        # If the current number is the smallest number in its chain, search for
        # the length of its chain.
        if num - 1 not in num_set:
            current_num = num
            current_chain = 1
            # Continue to find the next consecutive numbers in the chain.
            while current_num + 1 in num_set:
                current_num += 1
                current_chain += 1
            longest_chain = max(longest_chain, current_chain)
    return longest_chain

class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        if not nums:
            return 0

        nums.sort()
        max_len = 1
        count = 1
        for i in range(len(nums)):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            if i > 0 and nums[i] == nums[i-1] + 1:
                count += 1
                print(count)
            else:
                max_len = max(max_len, count)
                count = 1
        
        return max(max_len, count)
            
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        if not nums:
            return 0

        nums.sort()

        longest_streak = 1
        current_streak = 1

        for i in range(1, len(nums)):
            if nums[i] != nums[i - 1]:
                if nums[i] == nums[i - 1] + 1:
                    current_streak += 1
                else:
                    longest_streak = max(longest_streak, current_streak)
                    current_streak = 1

        return max(longest_streak, current_streak)

#X13 Medium Geometric Sequence Triplets
# A geometric sequence triplet is a sequence of three numbers where each successive number 
# is obtained by multiplying the preceding number by a constant called the common ratio.
# Let's examine three triplets to understand how this works:
# (1, 2, 4): This is a geometric sequence with a ratio of 2 (i.e., [1, 1⋅2 = 2, 2⋅2 = 4]).
# (5, 15, 45): This is a geometric sequence with a ratio of 3 (i.e., [5, 5⋅3 = 15, 15⋅3 = 45]).
# (2, 3, 4): Not a geometric sequence.
# Given an array of integers and a common ratio r, find all triplets of indexes (i, j, k) 
# that follow a geometric sequence for i < j < k. It's possible to encounter duplicate triplets in the array.
# Example:
# Input: nums = [2, 1, 2, 4, 8, 8], r = 2
# Output: 5
from collections import defaultdict
from typing import List


def geometric_sequence_triplets(nums: List[int], r: int) -> int:
    # Use 'defaultdict' to ensure the default value of 0 is returned when 
    # accessing a key that doesn’t exist in the hash map. This effectively sets 
    # the default frequency of all elements to 0.
    left_map = defaultdict(int)
    right_map = defaultdict(int)
    count = 0
    # Populate 'right_map' with the frequency of each element in the array.
    for x in nums:
        right_map[x] += 1
    # Search for geometric triplets that have x as the center.
    for x in nums:
        # Decrement the frequency of x in 'right_map' since x is now being
        # processed and is no longer to the right.
        right_map[x] -= 1
        if x % r == 0:
            count += left_map[x // r] * right_map[x * r]
        # Increment the frequency of x in 'left_map' since it'll be a part of the
        # left side of the array once we iterate to the next value of x.
        left_map[x] += 1
    return count