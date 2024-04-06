"""

"""
#1 242.有效的字母异位词
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


#2 349. 两个数组的交集
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


#3 第202题. 快乐数
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


#4 1. 两数之和
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


#5 第454题.四数相加II
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


#6 383. 赎金信
    # 给定一个赎金信 (ransom) 字符串和一个杂志(magazine)字符串，判断第一个字符串 ransom 能不能由第二个字符串 magazines 里面的字符构成。如果可以构成，返回 true ；否则返回 false。
    # (题目说明：为了不暴露赎金信字迹，要从杂志上搜索各个需要的字母，组成单词来表达意思。杂志字符串中的每个字符只能在赎金信字符串中使用一次。)
    # 注意：
    # 你可以假设两个字符串均只含有小写字母。
    # canConstruct("a", "b") -> false
    # canConstruct("aa", "ab") -> false
    # canConstruct("aa", "aab") -> true
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


#7 第15题. 三数之和
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
                    while right > left and nums[right] == nums[right - 1]:
                        right -= 1
                    while right > left and nums[left] == nums[left + 1]:
                        left += 1
                        
                    right -= 1
                    left += 1
                    
        return result
# （版本二） 使用字典
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        result = []
        nums.sort()
        # 找出a + b + c = 0
        # a = nums[i], b = nums[j], c = -(a + b)
        for i in range(len(nums)):
            # 排序之后如果第一个元素已经大于零，那么不可能凑成三元组
            if nums[i] > 0:
                break
            if i > 0 and nums[i] == nums[i - 1]: #三元组元素a去重
                continue
            d = {}
            for j in range(i + 1, len(nums)):
                if j > i + 2 and nums[j] == nums[j-1] == nums[j-2]: # 三元组元素b去重
                    continue
                c = 0 - (nums[i] + nums[j])
                if c in d:
                    result.append([nums[i], nums[j], c])
                    d.pop(c) # 三元组元素c去重
                else:
                    d[nums[j]] = j
        return result


#8 第18题. 四数之和
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
            if nums[i] > target and nums[i] > 0 and target > 0:# 剪枝（可省）
                break
            if i > 0 and nums[i] == nums[i-1]:# 去重
                continue
            for j in range(i+1, n):
                if nums[i] + nums[j] > target and target > 0: #剪枝（可省）
                    break
                if j > i+1 and nums[j] == nums[j-1]: # 去重
                    continue
                left, right = j+1, n-1
                while left < right:
                    s = nums[i] + nums[j] + nums[left] + nums[right]
                    if s == target:
                        result.append([nums[i], nums[j], nums[left], nums[right]])
                        while left < right and nums[left] == nums[left+1]:
                            left += 1
                        while left < right and nums[right] == nums[right-1]:
                            right -= 1
                        left += 1
                        right -= 1
                    elif s < target:
                        left += 1
                    else:
                        right -= 1
        return result
# (版本二) 使用字典
class Solution(object):
    def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        # 创建一个字典来存储输入列表中每个数字的频率
        freq = {}
        for num in nums:
            freq[num] = freq.get(num, 0) + 1
        
        # 创建一个集合来存储最终答案，并遍历4个数字的所有唯一组合
        ans = set()
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                for k in range(j + 1, len(nums)):
                    val = target - (nums[i] + nums[j] + nums[k])
                    if val in freq:
                        # 确保没有重复
                        count = (nums[i] == val) + (nums[j] == val) + (nums[k] == val)
                        if freq[val] > count:
                            ans.add(tuple(sorted([nums[i], nums[j], nums[k], val])))
        
        return [list(x) for x in ans]
