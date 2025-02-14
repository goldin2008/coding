"""
Bit Manipulation, Math
"""
#1 191. Number of 1 Bits
    # Given a positive integer n, write a function that returns the number of set bits
    #  in its binary representation (also known as the Hamming weight).
class Solution:
    def hammingWeight(self, n: int) -> int:
        count = 0
        while (n > 0):
            count += n & 1
            n >>= 1
        return count
# Hamming Weights of Integers
    # The Hamming weight of a number is the number of set bits (1-bits) in its binary 
    # representation. Given a positive integer n, return an array where the ith element
    # is the Hamming weight of integer i for all integers from 0 to n.
    # Example:
    # Input: n = 7
    # Output: [0, 1, 1, 2, 1, 2, 2, 3]
# Bitwise
from typing import List
def hamming_weights_of_integers(n: int) -> List[int]:
    # Write your code here
    return [count_set_bits(x) for x in range(n+1)]

def count_set_bits(n):
    count = 0
    while (n > 0):
        count += n & 1
        n >>= 1
    return count
# DP
def hamming_weights_of_integers(n: int) -> List[int]:
    # Write your code here
    dp = [0] * (n+1)
    for x in range(1, n+1):
        dp[x] = dp[x >> 1] + (x & 1)
    return dp

# Lonely Integer
    # Given an integer array where each number occurs twice except for one of them, find the unique number.
    # Example:
    # Input: nums = [1, 3, 3, 2, 1]
    # Output: 2
from typing import List

def lonely_integer(nums: List[int]) -> int:
    # Write your code here
    res = 0
    for num in nums:
        res ^= num
    return res

# Swap Odd and Even Bits
    # Given an unsigned 32-bit integer n, return an integer where all of 
    # n's even bits are swapped with their adjacent odd bits.
def swap_odd_and_even_bits(n: int) -> int:
    # Write your code here
    even_mask = 0x55555555
    odd_mask = 0xAAAAAAAA
    even_bits = n & even_mask
    odd_bits = n & odd_mask
    return (even_bits << 1) | (odd_bits >> 1)

