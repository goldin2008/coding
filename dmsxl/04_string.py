"""
python的string为不可变, 需要开辟同样大小的list空间来修改
res = list(s)
return ''.join(res)

344.反转字符串:
第一次讲到反转一个字符串应该怎么做, 使用了双指针法。
541. 反转字符串II:
这里开始给反转加上了一些条件, 当需要固定规律一段一段去处理字符串的时候, 要想想在for循环的表达式上做做文章。
151.翻转字符串里的单词:
要对一句话里的单词顺序进行反转, 发现先整体反转再局部反转 是一个很妙的思路。
左旋转字符串:
是先局部反转再 整体反转, 与151.翻转字符串里的单词类似, 但是也是一种新的思路。
"""
class Solution:
    #1.去除多余的空格
    def trim_spaces(self, s):     
        n=len(s)
        left=0
        right=n-1

        while left<=right and s[left]==' ':    #去除开头的空格
            left+=1
        while left<=right and s[right]==' ':   #去除结尾的空格
            right=right-1
        tmp=[]
        while left<=right:    #去除单词中间多余的空格
            if s[left]!=' ':
                tmp.append(s[left])
            elif tmp[-1]!=' ':   #当前位置是空格, 但是相邻的上一个位置不是空格, 则该空格是合理的
                tmp.append(s[left])
            left+=1
        return tmp
    #2.翻转字符数组
    def reverse_string(self, nums, left, right):
        while left<right:
            nums[left], nums[right]=nums[right],nums[left]
            left+=1
            right-=1
        return None
    #3.翻转每个单词
    def reverse_each_word(self, nums):
        start=0
        end=0
        n=len(nums)
        while start<n:
            while end<n and nums[end]!=' ':
                end+=1
            self.reverse_string(nums,start,end-1)
            start=end+1
            end+=1
        return None

    #4.翻转字符串里的单词
    def reverseWords(self, s): #测试用例："the sky is blue"
        l = self.trim_spaces(s)   #输出：['t', 'h', 'e', ' ', 's', 'k', 'y', ' ', 'i', 's', ' ', 'b', 'l', 'u', 'e'
        self.reverse_string(l, 0, len(l)-1)   #输出：['e', 'u', 'l', 'b', ' ', 's', 'i', ' ', 'y', 'k', 's', ' ', 'e', 'h', 't']
        self.reverse_each_word(l)               #输出：['b', 'l', 'u', 'e', ' ', 'i', 's', ' ', 's', 'k', 'y', ' ', 't', 'h', 'e']
        return ''.join(l)         #输出：blue is sky the

#1 344.反转字符串
    # 编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 char[] 的形式给出。
    # 不要给另外的数组分配额外的空间，你必须原地修改输入数组、使用 O(1) 的额外空间解决这一问题。
    # 你可以假设数组中的所有字符都是 ASCII 码表中的可打印字符。
    # 示例 1：
    # 输入：["h","e","l","l","o"]
    # 输出：["o","l","l","e","h"]
    # 示例 2：
    # 输入：["H","a","n","n","a","h"]
    # 输出：["h","a","n","n","a","H"]
# ***（版本一） 双指针
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


#2 541. 反转字符串II
    # 给定一个字符串 s 和一个整数 k，从字符串开头算起, 每计数至 2k 个字符，就反转这 2k 个字符中的前 k 个字符。
    # 如果剩余字符少于 k 个，则将剩余字符全部反转。
    # 如果剩余字符小于 2k 但大于或等于 k 个，则反转前 k 个字符，其余字符保持原样。
    # 示例:
    # 输入: s = "abcdefg", k = 2
    # 输出: "bacdfeg"
class Solution:
    def reverseStr(self, s: str, k: int) -> str:
        """
        1. 使用range(start, end, step)来确定需要调换的初始位置
        2. 对于字符串s = 'abc', 如果使用s[0:999] ===> 'abc'。字符串末尾如果超过最大长度，则会返回至字符串最后一个值，这个特性可以避免一些边界条件的处理。
        3. 用切片整体替换，而不是一个个替换.
        """
        def reverse_substring(text):
            left, right = 0, len(text) - 1
            while left < right:
                text[left], text[right] = text[right], text[left]
                left += 1
                right -= 1
            return text
        
        res = list(s)

        for cur in range(0, len(s), 2 * k):
            res[cur: cur + k] = reverse_substring(res[cur: cur + k])
        
        return ''.join(res)
# class Solution:
#     def reverseStr(self, s: str, k: int) -> str:
#         # Two pointers. Another is inside the loop.
#         p = 0
#         while p < len(s):
#             p2 = p + k
#             # Written in this could be more pythonic.
#             s = s[:p] + s[p: p2][::-1] + s[p2:]
#             p = p + 2 * k
#         return s


#3 替换数字
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


#4 151.翻转字符串里的单词
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


#5 右旋字符串
    # 字符串的右旋转操作是把字符串尾部的若干个字符转移到字符串的前面。给定一个字符串 s 和一个正整数 k，请编写一个函数，将字符串中的后面 k 个字符移到字符串的前面，实现字符串的右旋转操作。
    # 例如，对于输入字符串 "abcdefg" 和整数 2，函数应该将其转换为 "fgabcde"。
    # 输入：输入共包含两行，第一行为一个正整数 k，代表右旋转的位数。第二行为字符串 s，代表需要旋转的字符串。
    # 输出：输出共一行，为进行了右旋转操作后的字符串。
    # 样例输入：
    # 2
    # abcdefg 
    # 样例输出：
    # fgabcde
#获取输入的数字k和字符串
k = int(input())
s = input()
#通过切片反转第一段和第二段字符串
#注意：python中字符串是不可变的，所以也需要额外空间
s = s[len(s)-k:] + s[:len(s)-k]
print(s)


#6 ??? 28. 实现 strStr()
    # 实现 strStr() 函数。
    # 给定一个 haystack 字符串和一个 needle 字符串，在 haystack 字符串中找出 needle 字符串出现的第一个位置 (从0开始)。如果不存在，则返回  -1。
    # 示例 1: 输入: haystack = "hello", needle = "ll" 输出: 2
    # 示例 2: 输入: haystack = "aaaaa", needle = "bba" 输出: -1
    # 说明: 当 needle 是空字符串时，我们应当返回什么值呢？这是一个在面试中很好的问题。 对于本题而言，当 needle 是空字符串时我们应当返回 0 。这与C语言的 strstr() 以及 Java的 indexOf() 定义相符。
# KMP的经典思想就是:当出现字符串不匹配时，可以记录一部分之前已经匹配的文本内容，利用这些信息避免从头再去做匹配。
# （版本一）前缀表（减一）
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        if not needle:
            return 0
        next = [0] * len(needle)
        self.getNext(next, needle)
        j = -1
        for i in range(len(haystack)):
            while j >= 0 and haystack[i] != needle[j+1]:
                j = next[j]
            if haystack[i] == needle[j+1]:
                j += 1
            if j == len(needle) - 1:
                return i - len(needle) + 1
        return -1

    def getNext(self, next, s):
        j = -1
        next[0] = j
        for i in range(1, len(s)):
            while j >= 0 and s[i] != s[j+1]:
                j = next[j]
            if s[i] == s[j+1]:
                j += 1
            next[i] = j
# （版本二）前缀表（不减一）
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        if len(needle) == 0:
            return 0
        next = [0] * len(needle)
        self.getNext(next, needle)
        j = 0
        for i in range(len(haystack)):
            while j > 0 and haystack[i] != needle[j]:
                j = next[j - 1]
            if haystack[i] == needle[j]:
                j += 1
            if j == len(needle):
                return i - len(needle) + 1
        return -1

    def getNext(self, next: List[int], s: str) -> None:
        j = 0
        next[0] = 0
        for i in range(1, len(s)):
            while j > 0 and s[i] != s[j]:
                j = next[j - 1]
            if s[i] == s[j]:
                j += 1
            next[i] = j
# （版本三）暴力法
# class Solution(object):
#     def strStr(self, haystack, needle):
#         """
#         :type haystack: str
#         :type needle: str
#         :rtype: int
#         """
#         m, n = len(haystack), len(needle)
#         for i in range(m):
#             if haystack[i:i+n] == needle:
#                 return i
#         return -1    
# # （版本四）使用 index
# class Solution:
#     def strStr(self, haystack: str, needle: str) -> int:
#         try:
#             return haystack.index(needle)
#         except ValueError:
#             return -1
# # （版本五）使用 find
# class Solution:
#     def strStr(self, haystack: str, needle: str) -> int:
#         return haystack.find(needle)


#7 459.重复的子字符串
    # 给定一个非空的字符串，判断它是否可以由它的一个子串重复多次构成。给定的字符串只含有小写英文字母，并且长度不超过10000。
    # 示例 1:
    # 输入: "abab"
    # 输出: True
    # 解释: 可由子字符串 "ab" 重复两次构成。
    # 示例 2:
    # 输入: "aba"
    # 输出: False
    # 示例 3:
    # 输入: "abcabcabcabc"
    # 输出: True
    # 解释: 可由子字符串 "abc" 重复四次构成。 (或者子字符串 "abcabc" 重复两次构成。)
# （版本一） 前缀表 减一
class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:  
        if len(s) == 0:
            return False
        nxt = [0] * len(s)
        self.getNext(nxt, s)
        if nxt[-1] != -1 and len(s) % (len(s) - (nxt[-1] + 1)) == 0:
            return True
        return False
    
    def getNext(self, nxt, s):
        nxt[0] = -1
        j = -1
        for i in range(1, len(s)):
            while j >= 0 and s[i] != s[j+1]:
                j = nxt[j]
            if s[i] == s[j+1]:
                j += 1
            nxt[i] = j
        return nxt
# （版本二） 前缀表 不减一
class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:  
        if len(s) == 0:
            return False
        nxt = [0] * len(s)
        self.getNext(nxt, s)
        if nxt[-1] != 0 and len(s) % (len(s) - nxt[-1]) == 0:
            return True
        return False
    
    def getNext(self, nxt, s):
        nxt[0] = 0
        j = 0
        for i in range(1, len(s)):
            while j > 0 and s[i] != s[j]:
                j = nxt[j - 1]
            if s[i] == s[j]:
                j += 1
            nxt[i] = j
        return nxt
# （版本三） 使用 find
class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        n = len(s)
        if n <= 1:
            return False
        ss = s[1:] + s[:-1] 
        print(ss.find(s))              
        return ss.find(s) != -1
# （版本四） 暴力法
class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        n = len(s)
        if n <= 1:
            return False
        
        substr = ""
        for i in range(1, n//2 + 1):
            if n % i == 0:
                substr = s[:i]
                if substr * (n//i) == s:
                    return True
                
        return False
