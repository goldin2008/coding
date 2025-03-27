"""
DEShaw

4, 22, 79, 85, 124, 142, 240, 239, 430, 797, 815, 1059, 1387, 

https://math.stackexchange.com/questions/2991347/catalan-numbers-sequence-of-balanced-parentheses

https://www.geeksforgeeks.org/maximum-number-of-overlapping-intervals/

https://medium.com/@Todd_Rizley/technical-interview-prep-1-collatz-25debc24a972

"""
#1 Count Integer Partitions
# Given a positive integer n, find out how many ways of writing n as a sum of positive integers. 
# Two sums that differ only in the order of their summands are considered the same partition.
    # Example:
    # Input: 5
    # Output: 6
    # Explanation:
    # 1. 1 + 1 + 1 + 1 + 1
    # 2. 1 + 1 + 1 + 2
    # 3. 1 + 1 + 3
    # 4. 1 + 4
    # 5. 1 + 2 + 2
    # 6. 2 + 3
# Use a 1D array dp where dp[i] represents the number of ways to partition the integer i using any combination of positive integers.
# For each integer k from 1 to n, update the DP table to account for partitions that include k. 
# This is done by iterating through the integers from k to n and updating dp[i] with the sum of dp[i - k].
def count_partitions(n):
    if n == 0:
        return 0  # According to problem constraints, n is positive, but handle n=0 if required

    dp = [0] * (n + 1)
    dp[0] = 1  # Base case: one way to partition 0
    
    for k in range(1, n + 1):
        for i in range(k, n + 1):
            dp[i] += dp[i - k]
    
    return dp[n] - 1  # Subtract 1 to exclude the single-term partition

# Example usage:
n = 5
print(f"The number of partitions of {n} is: {count_partitions(n)}")

#2 Given an array of strings words, find the number of pairs where either the strings are equal 
# or one string ends with another. In other words, find the number of such pairs i, j j ≤ i < j< words. 
# length ) that words[i] is a suffix of words[j], or words[j] is a suffix of words[i].
    # For words = ["back", "backdoor", "gammon", "backgammon"
    # "comeback", "come", "door"], the output should be
    # solution (words) = 3 .
    # The relevant pairs are:
    # 1. words[0] = "back" and words[4] = "comeback"
    # 2. words[1] = "backdoor" and words[6] = "door"
    # 3. words[2] = "gammon" and words[3] = "backgammon"
    # For words = ["cba", "a", "a", "b", "ba", "ca"], the output should be solution(words) = 8.
    # The relevant pairs are:
    # 1. words[0] = "cba" and words[1] = "a"
    # 2. words[0] = "cba" and words[2] = "a"
    # 3. words[0] = "cba" and words[4] = "ba"
    # 4. words[1] = "a" and words[2] = "a"
    # 5. words[1] = "a" and words[4] = "ba"
    # 6. words[1] = "a" and words[5] = "ca"
    # 7. words[2] = "a" and words[4] = "ba"
    # 8. words[2] = "a" and words[5] = "ca"

from collections import defaultdict

def countSuffixPairs(words):
    from collections import defaultdict

    suffix_count = defaultdict(int)
    count = 0

    freq = defaultdict(int)
    duplicates = 0
    # seen = set()
    # First pass to count duplicates
    for word in words:
        duplicates += freq[word]
        freq[word] += 1

    # First pass: populate the suffix_count with all possible suffixes
    for word in words:
        # Generate all suffixes for the current word
        for i in range(len(word)):
            suffix = word[i:]
            suffix_count[suffix] += 1

    # print(suffix_count)
    # Second pass: for each word, count how many times it appears as a suffix in other words
    for word in words:
        # The number of times 'word' appears as a suffix in other words
        # We subtract 1 to exclude the current word itself
        # print(word, suffix_count[word])
        count += suffix_count[word] - 1

    # Since each pair is counted twice (i,j and j,i), we divide by 2
    return count - duplicates

# Test the function
words = ["cba", "a", "a", "b", "ba", "ca"]
print(countSuffixPairs(words))  # Expected output: 8
words = ["back", "backdoor", "gammon", "backgammon", "comeback", "come", "door"]
print(countSuffixPairs(words))  # Expected output: 3
words = ["cba", "a", "a", "a", "a"]
print(countSuffixPairs(words))  # Expected output: 2

#3 There are some lamps placed on a coordinate line. Each of these lamps illuminates some 
# space around it within a given radius. You are given the coordinates of the lamps on the line, 
# and the effective radius of each of the lamps' light. In other words, you are given a 
# twb-dimensional array lamps, where Tamps [i] contains information about the ith lamp. 
# Tamps [i] [0] is an integer representing the lamp's coordinate, and lamps [i][1] is a 
# positive integer representing the effective radius of the ith lamp. That means that the 
# ith lamp illuminates everything in a range from Tamps [i][0] - lamps [i][1] to 
# lamps [i][0] + lamps [i][1] inclusive. Your task is to find the coordinate of the point 
# that is illuminated by the highest number of lamps. In case of a tie, return the point 
# among them with the minimal possible coordinate. Example • For lamps = [ [-2, 3], [2, 3], [2, 1] ], 
# the output should be solution(lamps) = 1
def solution(lamps):
    points = []
    for x, r in lamps:
        start = x - r
        end = x + r
        points.append((start, 'start'))
        points.append((end, 'end'))
    
    # Sort points:
    # For the same coordinate, process 'start' points before 'end' to ensure correct counting
    # Primary Sort Key (e[0]):
    # The first element of each tuple (e[0]) is used as the primary sorting criterion.
    # This means all points will first be sorted based on their first element in ascending order.
    # Secondary Sort Key (e[1] == 'end'):
    # The second element (e[1]) is checked to see if it equals the string 'end'.
    # This comparison returns a boolean (True or False).
    # In Python, False (0) is considered "smaller" than True (1) when sorting.
    # Therefore, points where e[1] is not 'end' will appear before points where e[1] is 'end' when their e[0] values are equal.
    points.sort(key=lambda e: (e[0], e[1] == 'end'))
    print(points)
    
    active_intervals = 0
    max_overlaps = 0
    best_point = None

    for point, point_type in points:
        if point_type == 'start':
            active_intervals += 1
        else:
            active_intervals -= 1

        if active_intervals > max_overlaps:
            max_overlaps = active_intervals
            best_point = point
    
    return best_point

lamps = [ [-2, 3], [2, 3], [2, 1] ]
print(solution(lamps))  # Expected output: 1

#4 You are given an array of integers numbers. Your task is to count the number of 
# distinct pairs (i,j) suchthat 0 <= i < j < numbers.length, numbers[i] and numbers[j] 
# have the same number of digits, and only one of the digits differ between numbers[1] 
# and numbers[j] . 
    # Example For numbers = [1, 151, 241, 1, 9, 22, 351] , the output should 
    # be solution (numbers) - 3. • numbers[0] = 1 differs from numbers[4] = 9 on the one and 
    # only digit in both numbers. • numbers[1] = 151 differs from numbers[6] = 351 on the 
    # first digit. • numbers[3] = 1 differs from numbers[4] = 9 on the one and only digit 
    # in both numbers. Note that numbers (01 = 1 and numbers[3] = 1 do not differ from each 
    # other at all and thus do not count as a valid pair.
    # Input/Output • [execution time limit] 4 seconds (py3) • [input] array.integer numbers 
    # An array of positive integers. Guaranteed constraints: 1 ≤ numbers. 
    # length ≤ 104 1 ≤ numbers (1] $ 10° • [output] integer The count of pairs from numbers 
    # which have the same number of digits but differ on exactly one of the digits.
def solution(numbers):
    from collections import defaultdict

    # Group numbers by their digit lengths
    digit_groups = defaultdict(list)
    for num in numbers:
        digit_length = len(str(num))
        digit_groups[digit_length].append(num)

    print(digit_groups)
    
    count = 0

    # Iterate over each group of numbers with the same digit length
    for group in digit_groups.values():
        n = len(group)
        # Compare each number with every other number in the same group
        for i in range(n):
            for j in range(i + 1, n):
                num1 = str(group[i])
                num2 = str(group[j])
                diff = 0
                # Compare each digit
                for d1, d2 in zip(num1, num2):
                    if d1 != d2:
                        diff += 1
                        if diff > 1:
                            break
                if diff == 1:
                    count += 1
    return count

# Example usage:
numbers = [1, 151, 241, 1, 9, 22, 351]
print(solution(numbers))  # Expected output: 3

#5 Imagine a long row of trees is being planted just in front of your house, and you are 
# monitoring the density of the trees. This row of trees is represented by a number line, 
# where a tree can be planted at each integer point on the line. You start monitoring 
# right before the first tree is planted, so you can assume there are no trees on the 
# line at the beginning. You are given queries - an array of integers representing the 
# locations of planted trees, in the same order as they are planted. After each tree is 
# planted, your task is to find the longest segment of consecutive trees along the line. 
# Return an array of two-element arrays representing the longest segment of consecutive 
# trees along the line after each tree is planted. Specifically, the ith element of this 
# array should be a two-element array [start, end] representing the longest segment after 
# the queries[i] tree is planted. If there are multiple segments that could qualify as the 
# longest, return any one of them. NOTE: It is guaranteed that elements in queries are 
# pairwise unique, so trees will not be planted at the same location twice.
    # Example: For queries = [2, 1, 3], the output should be solution (queries) = [ [2, 2], [1, 2], [1, 3] ]
import bisect

def solution(queries):
    trees = set()
    longest_segments = []

    for q in queries:
        trees.add(q)
        sorted_trees = sorted(trees)

        # Finding the longest consecutive segment
        max_len = 0
        longest_segment = [sorted_trees[0], sorted_trees[0]]

        start = sorted_trees[0]
        for i in range(1, len(sorted_trees)):
            if sorted_trees[i] != sorted_trees[i - 1] + 1:
                # New segment found
                segment_length = sorted_trees[i - 1] - start + 1
                if segment_length > max_len:
                    max_len = segment_length
                    longest_segment = [start, sorted_trees[i - 1]]
                start = sorted_trees[i]
        
        # Final check for the last segment
        segment_length = sorted_trees[-1] - start + 1
        if segment_length > max_len:
            longest_segment = [start, sorted_trees[-1]]

        longest_segments.append(longest_segment)

    return longest_segments

# Example usage:
queries = [2, 1, 3]
print(solution(queries))  # Expected output: [[2, 2], [1, 2], [1, 3]]

#6 Analysts in your company have been researching a new format for analyzing stock 
# performance data - a golden rectangle. This can be applied to stock data represented 
# by a matrix prices, with rows representing days and columns representing different times 
# within each day. Specifically, prices (day] (time] contains the price of the stock at 
# the specific day and time. A rectangle (submatrix) within such prices is considered to 
# be golden if it has a width of at least two cells and values across all rows are strictly 
# increasing from left to right, suggesting that prices within each day are consistently 
# increasing. The size of the golden rectangle is the number of cells within it. Given a 
# matrix of stock data prices, your task is to find the size of the largest golden rectangle 
# within this matrix. 
    # Example For prices = [ [1,2,5,2,1,9], [3,4,4,4,5,9], [4,3,4,4,7,8], [1,2,3,2,4,3], [5,6,4,7,8,9] ] 
    # output should be solution (prices) = 8
    # the items satisfying the golden rule in the matrix is presented by ., is 
    # [ [1,2,5,2,1,9], [3,4,4, ., .,9], [4,3,4, ., .,8], [1,2,3, ., .,3], [5,6,4, ., .,9] ], 
    # which means we want to find in the same time interval in continuous days, at least two continuous 
    # time slot the prices are consistently increasing. And also in continuous days. Want to find this 
    # maximum number which is the product of number of continuous increasing prices and the number of continuous days.
# stack: An empty list that will be used to store indices of the bars in the histogram. 
# This stack helps in tracking the bars that are currently being considered for forming rectangles.

# width = i if not stack else i - stack[-1] - 1: Calculates the width of the rectangle. 
# If the stack is empty after popping, it means the rectangle extends from the start of 
# the histogram to the current index i, so the width is i. Otherwise, the width is the 
# difference between the current index i and the index now at the top of the stack (stack[-1]), 
# minus one. This calculation determines the number of bars that can form a rectangle with the height height.

# heights.append(0): A sentinel value (0) is appended to the heights list. This ensures that all bars in the 
# histogram are processed by the time the loop completes, as it provides a guaranteed condition to finalize 
# calculations for any remaining bars in the stack.
def largest_golden_rectangle(prices):
    if not prices or not prices[0]:
        return 0

    rows, cols = len(prices), len(prices[0])
    dp = [[1] * cols for _ in range(rows)]

    # Build the DP table
    # The dp table (dynamic programming table) is initialized with the same dimensions as prices, 
    # filled with zeros. This table will be used to store the length of the longest increasing sequences ending at each cell.
    for r in range(rows):
        for c in range(cols):
            if prices[r][c] > prices[r][c - 1]:
                dp[r][c] = dp[r][c - 1] + 1

    # Function to calculate the largest rectangle area in a histogram
    def largest_rectangle_area(heights):
        stack = []
        max_area = 0
        heights.append(0)  # Sentinel value to ensure the last element is processed

        for i, h in enumerate(heights):
            while stack and heights[stack[-1]] >= h:
                height = heights[stack.pop()]
                width = i if not stack else i - stack[-1] - 1
                if width >= 2:  # Width must be at least 2
                    max_area = max(max_area, width * height)
            stack.append(i)

        heights.pop()  # Remove the sentinel value
        return max_area

    max_golden_area = 0

    # Calculate the maximum golden rectangle area
    for c in range(cols):
        heights = [dp[r][c] for r in range(rows)]
        max_golden_area = max(max_golden_area, largest_rectangle_area(heights))

    print(dp)
    
    return max_golden_area

# Example usage:
prices = [
    [1, 2, 5, 2, 1, 9],
    [3, 4, 4, 4, 5, 9],
    [4, 3, 4, 4, 7, 8],
    [1, 2, 3, 2, 4, 3],
    [5, 6, 4, 7, 8, 9]
]

print(largest_golden_rectangle(prices))  # Output: 8

#7 Given an empty array that should contain integers numbers, your task is to process 
# a list of queries on it. Specifically, there are two types of queries: • "+x" - add 
# integer * to numbers. numbers may contain multiple instances of the same integer. • 
# "-*" - remove a single instance of integer x from numbers After processing each query 
# record the number of pairs in numbers with a difference equal to a given diff. The final 
# output should be an array of such values for all queries . main.pys 1 3 def Saved solut 
# diff Notes: • All numbers in queries are guaranteed to be in the range of 1-10°, 10°1 . 
# It is also guaranteed that for every - query, the specified number * exists in numbers. 
# • It is guaranteed that the answer for each query fits into a 32-bit integer type. 
    # Example • For queries = ['+4", '+5", "+2". "-4"] and diff = 1, the output should be solution 
    # (queries, diff) = 10, 1. 1, 01 • • First, process queries(0] = "+4" andadd to numbers, 
    # resulting in numbers = [4] . There are no pairs with dift = 1. so append 0 to the output. 
    # • Next, process queries(1] - "+5" and add 5 to numbers, resulting in numbers - [4, 51 - 
    # The numbers 4 and 5 have difference diff - 1, so append 1 to the output. • Process 
    # queries(2] - "+2" andadd 2 to numbers, resulting in numbers = 14, 5, 21. The number of 
    # pairs with difference diff = 1 remains the same, so append 1 to the output. • Process 
    # queries (31 - and and remove an instance of number 4, resulting in numbers = [5, 2]. 
    # There are no pairs with difference diff = 1, so append 0 to the output. The final output is [0, 1, 1, 0]
from collections import defaultdict

def solution(queries, diff):
    freq = defaultdict(int)
    output = []
    count_pairs = 0
    
    for query in queries:
        if query.startswith('+'):
            x = int(query[1:])
            # Before adding x, check how many existing numbers can form pairs with x
            count_pairs += freq.get(x - diff, 0)
            count_pairs += freq.get(x + diff, 0)
            freq[x] += 1
        elif query.startswith('-'):
            x = int(query[1:])
            # Before removing x, decrease the count of pairs involving x
            freq[x] -= 1
            count_pairs -= freq.get(x - diff, 0)
            count_pairs -= freq.get(x + diff, 0)
        output.append(count_pairs)
    
    return output

# Example usage:
queries = ['+4', '+5', '+2', '-4']
diff = 1
print(solution(queries, diff))  # Expected output: [0, 1, 1, 0]

#8 You are given a positive binary number represented as a string of 1s and os, 
# and an array of requests of the following two types: • requests [1] = "+": 
# increment the current value of number • requests [i] = "?": find the amount 
# of 1s in the binary representation of the current value of number Your task is 
# to return an array containing the answers for the requests of the second type 
# in the order they are presented in the requests array. 
    # Example For number = "1101" 
    # and requests - ["2", "+", "?", "+", "?", "+*, *?"], the output should be solution 
    # (number, requests) = [3, 3, 4, 1] • requests [0] = "2": the current value of number 
    # is 11012, which contains 3 ones; • requests [1] = +: the current value of number is 
    # 11012 = 1310, which after incrementing becomes 11102 = 1410 : • requests [2] = "?" : t
    # he current value of number is 11102 which contains 3 ones: • requests [3] - +: the 
    # current value of number is 11102 - 1410, which after incrementing becomes 11112 = 1510 : 
    # • requests [4] = "2": the current value of number iS 11112, which contains 4 ones; 
    # • requests [5] = +: the current value of number is 11112 = 1510. which after incrementing 
    # becomes 100002 = 1610 • requests (6] = 2: the current value of number is 100002 which 
    # contains a single 1 The answers for the requests of the second type are 3. 3, 4, and 1 
    # respectively, so the final result is [3, 3, 4, 1]
def solution(number, requests):
    # current = int(number, 2)
    current = binary_to_decimal(number)

    result = []
    for req in requests:
        if req == "+":
            current += 1
        elif req == "?":
            # count = 0
            # n = current
            # while n:
            #     n &= n - 1
            #     count += 1
            # result.append(count)
            result.append(count_set_bits(current))
    return result

def binary_to_decimal(binary_str):
    decimal = 0
    for bit in binary_str:
        decimal = decimal * 2 + (1 if bit == '1' else 0)
    return decimal

# Use bit manipulation to count the number of 1s in the binary representation of the current integer. 
# This can be done efficiently using Brian Kernighan's algorithm, which repeatedly flips the least 
# significant 1 bit to 0 and counts the number of flips until the number becomes 0.
def count_set_bits(n):
    count = 0
    while n:
        n &= n - 1  # Clear the least significant set bit
        count += 1
    return count

# Example usage:
number = "1101"
requests = ["?", "+", "?", "+", "?", "+", "?"]
print(solution(number, requests))  # Expected output: [3, 3, 4, 1]


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


# (Medium) 322.零钱兑换
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
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0

        for coin in coins:
            for i in range(coin, amount + 1): # 进行优化,从能装得下的背包开始计算,则不需要进行比较
                # 更新凑成金额 i 所需的最少硬币数量
                dp[i] = min(dp[i], dp[i - coin] + 1)

        return dp[amount] if dp[amount] != float('inf') else -1