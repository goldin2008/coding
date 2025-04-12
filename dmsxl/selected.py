"""

"""
#1 (Medium) 1387. Sort Integers by The Power Value
    # The power of an integer x is defined as the number of steps needed to transform x into 1 using the following steps:
    # if x is even then x = x / 2
    # if x is odd then x = 3 * x + 1
    # For example, the power of x = 3 is 7 because 3 needs 7 steps to become 1 (3 --> 10 --> 5 --> 16 --> 8 --> 4 --> 2 --> 1).
    # Given three integers lo, hi and k. The task is to sort all integers in the interval [lo, hi] by the power value in ascending order, if two or more integers have the same power value sort them by ascending order.
    # Return the kth integer in the range [lo, hi] sorted by the power value.
    # Notice that for any integer x (lo <= x <= hi) it is guaranteed that x will transform into 1 using these steps and that the power of x is will fit in a 32-bit signed integer.
# Example 1:
# Input: lo = 12, hi = 15, k = 2
# Output: 13
# Explanation: The power of 12 is 9 (12 --> 6 --> 3 --> 10 --> 5 --> 16 --> 8 --> 4 --> 2 --> 1)
# The power of 13 is 9
# The power of 14 is 17
# The power of 15 is 17
# The interval sorted by the power value [12,13,14,15]. For k = 2 answer is the second element which is 13.
# Notice that 12 and 13 have the same power value and we sorted them in ascending order. Same for 14 and 15.
# Example 2:
# Input: lo = 7, hi = 11, k = 4
# Output: 7
# Explanation: The power array corresponding to the interval [7, 8, 9, 10, 11] is [16, 3, 19, 6, 14].
# The interval sorted by power is [8, 10, 11, 7, 9].
# The fourth number in the sorted array is 7.
def getKth(lo: int, hi: int, k: int) -> int:
    memo = {}
    
    def get_power(x):
        if x == 1:
            return 0
        if x in memo:
            return memo[x]
        if x % 2 == 0:
            memo[x] = 1 + get_power(x // 2)
        else:
            memo[x] = 1 + get_power(3 * x + 1)
        return memo[x]  # Ensure this is outside the if-else blocks
    
    numbers = []
    for num in range(lo, hi + 1):
        power = get_power(num)
        numbers.append((power, num))
    
    numbers.sort()
    return numbers[k-1][1]


# X07_binary_tree.py
#19 (Easy) 111.二叉树的最小深度
    # 给定一个二叉树, 找出其最小深度。
    # 最小深度是从根节点到最近叶子节点的最短路径上的节点数量。
# 需要注意的是，只有当左右孩子都为空的时候，才说明遍历到最低点了。如果其中一个孩子不为空则不是最低点
# 递归法（版本一）后序遍历 postorder ???
class Solution:
    def minDepth(self, root):
        return self.getDepth(root)

    def getDepth(self, node):
        if node is None:
            return 0
        leftDepth = self.getDepth(node.left)  # 左
        rightDepth = self.getDepth(node.right)  # 右
        
        # 当一个左子树为空，右不为空，这时并不是最低点
        if node.left is None and node.right is not None:
            return 1 + rightDepth
        
        # 当一个右子树为空，左不为空，这时并不是最低点
        if node.left is not None and node.right is None:
            return 1 + leftDepth
        
        # The current node's depth is calculated after its children's depths are computed
        result = 1 + min(leftDepth, rightDepth) # 中 
        return result