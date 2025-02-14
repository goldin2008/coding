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

