"""
Bit Manipulation, Math, Tries, Heaps

Basics of Bit Manipulation

https://www.hackerearth.com/practice/basic-programming/bit-manipulation/basics-of-bit-manipulation/tutorial/

by applying (id+1)^1-1, you toggle the LSB of the input number, which changes odd numbers to even and even numbers to odd. 
This operation works because it takes advantage of the properties of bitwise XOR and subtraction to toggle and reset the LSB, respectively.
"""
#X1 (Easy) 191.Number of 1 Bits
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


#X2 (Easy) Lonely Integer
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


#X3 (Medium) Swap Odd and Even Bits
    # Given an unsigned 32-bit integer n, return an integer where all of 
    # n's even bits are swapped with their adjacent odd bits.
def swap_odd_and_even_bits(n: int) -> int:
    # Write your code here
    even_mask = 0x55555555
    odd_mask = 0xAAAAAAAA
    even_bits = n & even_mask
    odd_bits = n & odd_mask
    return (even_bits << 1) | (odd_bits >> 1)


#X4 To find the Greatest Common Divisor (GCD) of two integers, we can use the Euclidean Algorithm
    # Euclidean Algorithm
    # The algorithm is based on the principle that the GCD of two numbers also divides their difference. Here's how it works:
    # Divide: Divide the larger number by the smaller number.
    # Remainder: Take the remainder of the division.
    # Repeat: Replace the larger number with the smaller number and the smaller number with the remainder.
    # Stop: When the remainder is 0, the smaller number at that step is the GCD.
# Explanation of the Code
# The function gcd(a, b) takes two integers as input.
# It uses a while loop to repeatedly apply the Euclidean Algorithm until b becomes 0.
# In each iteration:
# a is replaced with b.
# b is replaced with the remainder of a % b.
# When b becomes 0, the loop ends, and a is returned as the GCD.
def gcd(a, b):
    while b != 0:
        a, b = b, a % b  # Replace a with b, and b with the remainder of a divided by b
    return a
# Example usage
print(gcd(56, 98))  # Output: 14


# Trie
#X5 (Medium) Design a Trie
    # Design and implement a trie data structure that supports the following operations:
    # insert(word: str) -> None: Inserts a word into the trie.
    # search(word: str) -> bool: Returns true if a word exists in the trie, and false if not.
    # has_prefix(prefix: str) -> bool: Returns true if the trie contains a word with the given prefix, and false if not.
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for c in word:
            # For each character in the word, if it's not a child of
            # the current node, create a new TrieNode for that
            # character.
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
            # Mark the last node as the end of a word.
        node.is_word = True

    def search(self, word: str) -> bool:
        node = self.root
        for c in word:
            # For each character in the word, if it's not a child of
            # the current node, the word doesn't exist in the Trie.
            if c not in node.children:
                return False
            node = node.children[c]
            # Return whether the current node is marked as the end of the
            # word.
        return node.is_word
        
    def has_prefix(self, prefix: str) -> bool:
        node = self.root
        for c in prefix:
            if c not in node.children:
                return False
            node = node.children[c]
        # Once we've traversed the nodes corresponding to each
        # character in the prefix, return True.
        return True


#X6 (Medium) Insert and Search Words with Wildcards
    # Design and implement a data structure that supports the following operations:
    # insert(word: str) -> None: Inserts a word into the data structure.
    # search(word: str) -> bool: Returns true if a word exists in the data 
    # structure and false if not. The word may contain wildcards ('.') that can represent any letter.
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False

class InsertAndSearchWordsWithWildcards:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.is_word = True

    def search(self, word: str) -> bool:
        # Start searching from the root of the trie.
        return self.search_helper(0, word, self.root)

    def search_helper(self, word_index: int, word: str, node: TrieNode) -> bool:
        for i in range(word_index, len(word)):
            c = word[i]
            # If a wildcard character is encountered, recursively
            # search for the rest of the word from each child node.
            if c == '.':
                for child in node.children.values():
                    # If a match is found, return true.
                    if self.search_helper(i + 1, word, child):
                        return True
                return False
            elif c in node.children:
                node = node.children[c]
            else:
                return False
            # After processing the last character, return true if we've
            # reached the end of a word.
        return node.is_word


#X7 (Hard) Find All Words on a Board
    # Given a 2D board of characters and an array of words, find all the words 
    # in the array that can be formed by tracing a path through adjacent cells 
    # in the board. Adjacent cells are those which horizontally or vertically 
    # neighbor each other. We can't use the same cell more than once for a single word.
from typing import List

class TrieNode:
    def __init__(self):
        self.children = {}
        self.word = None

def find_all_words_on_a_board(board: List[List[str]], words: List[str]) -> List[str]:
    root = TrieNode()
    # Insert every word into the trie.
    for word in words:
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.word = word
    res = []
    # Start a DFS call from each cell of the board that contains a
    # child of the root node, which represents the first letter of a
    # word in the trie.
    for r in range(len(board)):
        for c in range(len(board[0])):
            if board[r][c] in root.children:
                dfs(board, r, c, root.children[board[r][c]], res)
    return res

def dfs(board: List[List[str]], r: int, c: int, node: TrieNode, res: List[str]) -> None:
    # If the current node represents the end of a word, add the word to
    # the result.
    if node.word:
        res.append(node.word)
        # Ensure the current word is only added once.
        node.word = None
    temp = board[r][c]
    # Mark the current cell as visited.
    board[r][c] = '#'
    # Explore all adjacent cells that correspond with a child of the
    # current TrieNode.
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for d in dirs:
        next_r, next_c = r + d[0], c + d[1]
        if (is_within_bounds(next_r, next_c, board) and board[next_r][next_c] in node.children):
            dfs(board, next_r, next_c, node.children[board[next_r][next_c]], res)
    # Backtrack by reverting the cell back to its original character.
    board[r][c] = temp

def is_within_bounds(r: int, c: int, board: List[str]) -> bool:
    return 0 <= r < len(board) and 0 <= c < len(board[0])