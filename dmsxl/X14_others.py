"""
Bit Manipulation, Math, Tries, Heaps

Basics of Bit Manipulation

https://www.hackerearth.com/practice/basic-programming/bit-manipulation/basics-of-bit-manipulation/tutorial/

by applying (id+1)^1-1, you toggle the LSB of the input number, which changes odd numbers to even and even numbers to odd. 
This operation works because it takes advantage of the properties of bitwise XOR and subtraction to toggle and reset the LSB, respectively.
"""
#X1 191. Number of 1 Bits
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

#X2 Lonely Integer
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

#X3 Swap Odd and Even Bits
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
#X5 Design a Trie
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

#X6 Insert and Search Words with Wildcards
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
    
#X7 Find All Words on a Board
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



# Heaps
#X8 Medium K Most Frequent Strings
# Find the k most frequently occurring strings in an array, and return 
# them sorted by frequency in descending order. If two strings have the 
# same frequency, sort them in lexicographical order.
# Example:
# Input: strs = ['go', 'coding', 'byte', 'byte', 'go', 'interview', 'go'], k = 2
# Output: ['go', 'byte']
# Explanation: The strings "go" and "byte" appear the most frequently, 
# with frequencies of 3 and 2, respectively.
# Constraints:
# k ≤ n, where n denotes the length of the array.
from collections import Counter
import heapq
from typing import List


class Pair:
    def __init__(self, str, freq):
        self.str = str
        self.freq = freq

    # Define a custom comparator.
    def __lt__(self, other):
        # Prioritize lexicographical order for strings with equal
        # frequencies.
        if self.freq == other.freq:
            return self.str < other.str
        # Otherwise, prioritize strings with higher frequencies.
        return self.freq > other.freq
   
def k_most_frequent_strings_max_heap(strs: List[str], k: int) -> List[str]:
    # We use 'Counter' to create a hash map that counts the frequency 
    # of each string.
    freqs = Counter(strs)
    # Create the max heap by performing heapify on all string-frequency 
    # pairs.
    max_heap = [Pair(str, freq) for str, freq in freqs.items()]
    heapq.heapify(max_heap)
    # Pop the most frequent string off the heap 'k' times and return 
    # these 'k' most frequent strings.
    return [heapq.heappop(max_heap).str for _ in range(k)]


from collections import Counter
import heapq
from typing import List


class Pair:
    def __init__(self, str, freq):
        self.str = str
        self.freq = freq
    # Since this is a min-heap comparator, we can use the same 
    # comparator as the one used in the max-heap, but reversing the 
    # inequality signs to invert the priority.
    def __lt__(self, other):
        if self.freq == other.freq:
            return self.str > other.str
        return self.freq < other.freq
   
def k_most_frequent_strings_min_heap(strs: List[str], k: int) -> List[str]:
    freqs = Counter(strs)
    min_heap = []
    for str, freq in freqs.items():
        heapq.heappush(min_heap, Pair(str, freq))
        # If heap size exceeds 'k', pop the lowest frequency string to 
        # ensure the heap only contains the 'k' most frequent words so 
        # far.
        if len(min_heap) > k:
            heapq.heappop(min_heap)
    # Return the 'k' most frequent strings by popping the remaining 'k' 
    # strings from the heap. Since we're using a min-heap, we need to 
    # reverse the result after popping the elements to ensure the most 
    # frequent strings are listed first.
    res = [heapq.heappop(min_heap).str for _ in range(k)]
    res.reverse()
    return res


#X9 Medium Combine Sorted Linked Lists
# Given k singly linked lists, each sorted in ascending order, 
# combine them into one sorted linked list.
import heapq
from ds import ListNode
from typing import List
"""
Definition of ListNode:
class ListNode:
    def __init__(self, val=None, next=None):
        self.val = val
        self.next = next
"""

def combine_sorted_linked_lists(lists: List[ListNode]) -> ListNode:
    # Define a custom comparator for 'ListNode', enabling the min-heap 
    # to prioritize nodes with smaller values.
    ListNode.__lt__ = lambda self, other: self.val < other.val
    heap = []
    # Push the head of each linked list into the heap.
    for head in lists:
        if head:
            heapq.heappush(heap, head)
    # Set a dummy node to point to the head of the output linked list.
    dummy = ListNode(-1)
    # Create a pointer to iterate through the combined linked list as 
    # we add nodes to it.
    curr = dummy
    while heap:
        # Pop the node with the smallest value from the heap and add it 
        # to the output linked list.
        smallest_node = heapq.heappop(heap)
        curr.next = smallest_node
        curr = curr.next
        # Push the popped node's subsequent node to the heap.
        if smallest_node.next:
            heapq.heappush(heap, smallest_node.next)
    return dummy.next


#X10 Hard Median of an Integer Stream
# Design a data structure that supports adding integers from a data stream 
# and retrieving the median of all elements received at any point.
# add(num: int) -> None: adds an integer to the data structure.
# get_median() -> float: returns the median of all integers so far.
# Example:
# Input: [add(3), add(6), get_median(), add(1), get_median()]
# Output: [4.5, 3.0]
# Explanation:
# add(3)        # data structure contains [3] when sorted
# add(6)        # data structure contains [3, 6] when sorted
# get_median()  # median is (3 + 6) / 2 = 4.5
# add(1)        # data structure contains [1, 3, 6] when sorted
# get_median()  # median is 3.0
# Constraints:
# At least one value will have been added before get_median is called.
import heapq

class MedianOfAnIntegerStream:
    def __init__(self):
        self.left_half = []  # Max-heap.
        self.right_half = []  # Min-heap.

    def add(self, num: int) -> None:
        # If 'num' is less than or equal to the max of 'left_half', it 
        # belongs to the left half.
        if not self.left_half or num <= -self.left_half[0]:
            heapq.heappush(self.left_half, -num)
            # Rebalance the heaps if the size of the 'left_half' 
            # exceeds the size of the 'right_half' by more than one.
            if len(self.left_half) > len(self.right_half) + 1:
                heapq.heappush(self.right_half, -heapq.heappop(self.left_half))
        # Otherwise, it belongs to the right half.
        else:
            heapq.heappush(self.right_half, num)
            # Rebalance the heaps if 'right_half' is larger than 
            # 'left_half'.
            if len(self.left_half) < len(self.right_half):
                heapq.heappush(self.left_half, -heapq.heappop(self.right_half))

    def get_median(self) -> float:
        if len(self.left_half) == len(self.right_half):
            return (-self.left_half[0] + self.right_half[0]) / 2.0
        return -self.left_half[0]


#X11 Medium Sort a K-Sorted Array
# Given an integer array where each element is at most k positions away 
# from its sorted position, sort the array in a non-decreasing order.
# Example:
# Input: nums = [5, 1, 9, 4, 7, 10], k = 2
# Output: [1, 4, 5, 7, 9, 10]
import heapq
from typing import List


def sort_a_k_sorted_array(nums: List[int], k: int) -> List[int]:
    # Populate a min-heap with the first k + 1 values in 'nums'.
    min_heap = nums[:k+1]
    heapq.heapify(min_heap)
    # Replace elements in the array with the minimum from the heap at each 
    # iteration.
    insert_index = 0
    for i in range(k + 1, len(nums)):
        nums[insert_index] = heapq.heappop(min_heap)
        insert_index += 1
        heapq.heappush(min_heap, nums[i])
    # Pop the remaining elements from the heap to finish sorting the array.
    while min_heap:
        nums[insert_index] = heapq.heappop(min_heap)
        insert_index += 1
    return nums