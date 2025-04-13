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

#21 (Easy) 110.平衡二叉树
    # 给定一个二叉树, 判断它是否是高度平衡的二叉树。
    # 本题中，一棵高度平衡二叉树定义为：一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过1。
    # 示例 1:
    # 给定二叉树 [3,9,20,null,null,15,7]
    # 返回 true 。
    # 示例 2:
    # 给定二叉树 [1,2,2,3,3,null,null,4,4]
    # 返回 false 。
"""
求深度可以从上到下去查 所以需要**前序遍历 Preorder**（中左右），而高度只能从下到上去查，所以只能**后序遍历 Postorder**（左右中）
都知道回溯法其实就是递归，但是很少人用迭代的方式去实现回溯算法！
因为对于回溯算法已经是非常复杂的递归了，如果再用迭代的话，就是自己给自己找麻烦，效率也并不一定高。
讲了这么多二叉树题目的迭代法，有的同学会疑惑，迭代法中究竟什么时候用队列，什么时候用栈？
如果是模拟前中后序遍历就用栈，如果是适合层序遍历就用队列，当然还是其他情况，那么就是 先用队列试试行不行，不行就用栈。
"""
# 递归法
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        if self.get_height(root) != -1:
            return True
        else:
            return False

    def get_height(self, root: TreeNode) -> int:
        # Base Case
        if not root:
            return 0
        # 左
        if (left_height := self.get_height(root.left)) == -1:
            return -1
        # 右
        if (right_height := self.get_height(root.right)) == -1:
            return -1
        # 中
        if abs(left_height - right_height) > 1:
            return -1
        else:
            return 1 + max(left_height, right_height)
# *** 递归法精简版, 后序遍历
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        return self.get_hight(root) != -1
    def get_hight(self, node):
        if not node:
            return 0
        left = self.get_hight(node.left)
        right = self.get_hight(node.right)
        if left == -1 or right == -1 or abs(left - right) > 1:
            return -1
        return max(left, right) + 1

"""
1. For Boolean Checks (Early Return):
def traversal(node, count):
    if not node.left and not node.right:
        return count == 0  # True if match, False otherwise
    
    if node.left and traversal(node.left, count - node.left.val):
        return True
    if node.right and traversal(node.right, count - node.right.val):
        return True
    
    return False  # No path found


2. For Collecting Paths (No Early Return):
def traversal(node, count, path, result):
    if not node.left and not node.right:
        if count == 0:
            result.append(path.copy())
        return  # No return value needed
    
    if node.left:
        path.append(node.left.val)
        traversal(node.left, count - node.left.val, path, result)
        path.pop()  # Backtrack
    
    if node.right:
        path.append(node.right.val)
        traversal(node.right, count - node.right.val, path, result)
        path.pop()  # Backtrack
"""
#25 (Easy) 112.路径总和
    # 给定一个二叉树和一个目标和, 判断该树中是否存在根节点到叶子节点的路径, 这条路径上所有节点值相加等于目标和。
    # 说明: 叶子节点是指没有子节点的节点。
    # 示例: 给定如下二叉树，以及目标和 sum = 22，
    # 返回 true, 因为存在目标和为 22 的根节点到叶子节点的路径 5->4->11->2。
# 统一写法112和113
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:

        if not root:
            return False
        
        path = [root.val]
        res = []
        def traversal(node, path):
            if not node.left and not node.right:
                if sum(path) == targetSum:
                    res.append(path[:])
                return

            if node.left:
                path.append(node.left.val)
                traversal(node.left, path)
                path.pop()
            if node.right:
                path.append(node.right.val)
                traversal(node.right, path)
                path.pop()

        traversal(root, path)
        return bool(res)

# *** (版本一) 递归
class Solution:
    def hasPathSum(self, root: TreeNode, sum: int) -> bool:
        if root is None:
            return False
        
        return self.traversal(root, sum - root.val)
    
    def traversal(self, cur: TreeNode, count: int) -> bool:
        if not cur.left and not cur.right and count == 0: # 遇到叶子节点，并且计数为0
            return True
        if not cur.left and not cur.right: # 遇到叶子节点直接返回
            return False
        
        if cur.left: # 左
            # count -= cur.left.val
            # if self.traversal(cur.left, count): # 递归，处理节点
            #     return True
            # count += cur.left.val # 回溯，撤销处理结果
            if self.traversal(cur.left, count-cur.left.val): # 递归，处理节点
                return True
 
        if cur.right: # 右
            # count -= cur.right.val
            # if self.traversal(cur.right, count): # 递归，处理节点
            #     return True
            # count += cur.right.val # 回溯，撤销处理结果
            if self.traversal(cur.right, count-cur.right.val): # 递归，处理节点
                return True            

        return False
# (版本二) 递归 + 精简
# class Solution:
#     def hasPathSum(self, root: TreeNode, sum: int) -> bool:
#         if not root:
#             return False
#         if not root.left and not root.right and sum == root.val:
#             return True
#         return self.hasPathSum(root.left, sum - root.val) or self.hasPathSum(root.right, sum - root.val)
# *** (版本三) 迭代
class Solution:
    def hasPathSum(self, root: TreeNode, sum: int) -> bool:
        if not root:
            return False
        # 此时栈里要放的是pair<节点指针，路径数值>
        st = [(root, root.val)]
        while st:
            node, path_sum = st.pop()
            # 如果该节点是叶子节点了，同时该节点的路径数值等于sum，那么就返回true
            if not node.left and not node.right and path_sum == sum:
                return True
            # 左节点，压进去一个节点的时候，将该节点的路径数值也记录下来
            if node.left:
                st.append((node.left, path_sum + node.left.val))
            # 右节点，压进去一个节点的时候，将该节点的路径数值也记录下来
            if node.right:
                st.append((node.right, path_sum + node.right.val))
        return False


#26 (Medium) 113.路径总和-ii
    # 给定一个二叉树和一个目标和，找到所有从根节点到叶子节点路径总和等于给定目标和的路径。
    # 说明: 叶子节点是指没有子节点的节点。
    # 示例: 给定如下二叉树，以及目标和 sum = 22，
# 统一写法112和113
# separate functions
"""
Potential Issues (Not Bugs, But Optimizations)
Memory Usage:
    path + [node.val] creates a new list at every step, which uses more memory than append/pop backtracking.
    For large trees, this could lead to higher memory consumption.

Redundant Copies:
    path[:] creates another copy when appending to res, which is slightly redundant since path + [node.val] already created a new list.

path.copy(): Explicitly calls the copy() method of the list (introduced in Python 3.3).
path[:]: Uses slice notation to create a copy (works in all Python versions).
"""
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetsum: int) -> List[List[int]]:

        if not root:
            return []
        
        path = [root.val]
        res = []

        self.traversal(root, targetsum-root.val, path, res)
        return res

    def traversal(self, node, count, path, res):
        if not node.left and not node.right:
            if count == 0:
                # res.append(path[:])
                res.append(path.copy())
            return

        if node.left:
            path.append(node.left.val)
            self.traversal(node.left, count-node.left.val, path, res)
            path.pop()
            # self.traversal(node.left, count-node.left.val, path+[node.left.val], res)
        if node.right:
            path.append(node.right.val)
            self.traversal(node.right, count-node.right.val, path, res)
            path.pop()
            # self.traversal(node.right, count-node.right.val, path+[node.right.val], res)
# nested function with targetsum, path and undo
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetsum: int) -> List[List[int]]:

        if not root:
            return []
        
        path = [root.val]
        res = []
        def traversal(node, count, path):
            # print(path)
            if not node.left and not node.right:
                if count == 0:
                    res.append(path[:])
                return

            if node.left:
                # path.append(node.left.val)
                # path.append(node.left.val) modifies path in-place and returns None (since list.append() is a void method).
                # traversal(node.left, count-node.left.val, path.append(node.left.val))
                traversal(node.left, count-node.left.val, path+[node.left.val])
                # path.pop()
                # Your commented-out path.pop() suggests you tried backtracking, but the current code doesn’t undo append operations.
            if node.right:
                traversal(node.right, count-node.right.val, path+[node.right.val])

        traversal(root, targetsum-root.val, path)
        return res
# with targetSum
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetsum: int) -> List[List[int]]:

        if not root:
            return []
        
        path = [root.val]
        res = []
        def traversal(node, count):
            if not node.left and not node.right:
                if count == 0:
                    res.append(path[:])
                return

            if node.left:
                path.append(node.left.val)
                traversal(node.left, count-node.left.val)
                path.pop()
            if node.right:
                path.append(node.right.val)
                traversal(node.right, count-node.right.val)
                path.pop()

        traversal(root, targetsum-root.val)
        return res
# func with nothing
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetsum: int) -> List[List[int]]:

        if not root:
            return []
        
        path = [root.val]
        res = []
        def traversal(node):
            if not node.left and not node.right:
                if sum(path) == targetsum:
                    res.append(path[:])
                return

            if node.left:
                path.append(node.left.val)
                traversal(node.left)
                path.pop()
            if node.right:
                path.append(node.right.val)
                traversal(node.right)
                path.pop()

        traversal(root)
        return res
# func with path
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetsum: int) -> List[List[int]]:

        if not root:
            return []
        
        path = [root.val]
        res = []
        def traversal(node, path):
            if not node.left and not node.right:
                if sum(path) == targetsum:
                    res.append(path[:])
                return

            if node.left:
                path.append(node.left.val)
                traversal(node.left, path)
                path.pop()
            if node.right:
                path.append(node.right.val)
                traversal(node.right, path)
                path.pop()

        traversal(root, path)
        return res

# *** (版本一) 递归
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetsum: int) -> List[List[int]]:
        def traversal(cur_node, remain): 
            if not cur_node.left and not cur_node.right:
                if remain == 0: 
                    result.append(path[:])
                return

            if cur_node.left: 
                path.append(cur_node.left.val)
                traversal(cur_node.left, remain-cur_node.left.val)
                path.pop()

            if cur_node.right: 
                path.append(cur_node.right.val)
                traversal(cur_node.right, remain-cur_node.right.val)
                path.pop()

        result, path = [], []
        if not root: 
            return []
        path.append(root.val)
        traversal(root, targetsum - root.val)
        return result
    
class Solution:
    def __init__(self):
        self.result = []
        self.path = []
    def pathSum(self, root: TreeNode, sum: int) -> List[List[int]]:
        self.result.clear()
        self.path.clear()
        if not root:
            return self.result
        self.path.append(root.val) # 把根节点放进路径
        self.traversal(root, sum - root.val)
        return self.result
    def traversal(self, cur, count):
        if not cur.left and not cur.right and count == 0: # 遇到了叶子节点且找到了和为sum的路径
            # update final result
            self.result.append(self.path[:])
            return

        if not cur.left and not cur.right: # 遇到叶子节点而没有找到合适的边，直接返回
            return

        if cur.left: # 左 （空节点不遍历）
            # update path
            self.path.append(cur.left.val)
            # count -= cur.left.val
            # self.traversal(cur.left, count) # 递归
            # count += cur.left.val # 回溯
            self.traversal(cur.left, count-cur.left.val)
            self.path.pop() # 回溯

        if cur.right: #  右 （空节点不遍历）
            # update path
            self.path.append(cur.right.val) 
            # count -= cur.right.val
            # self.traversal(cur.right, count) # 递归
            # count += cur.right.val # 回溯
            self.traversal(cur.right, count-cur.right.val)
            self.path.pop() # 回溯

        return
# (版本二) 递归 + 精简
# class Solution:
#     def pathSum(self, root: TreeNode, targetSum: int) -> List[List[int]]:
        
#         result = []
#         self.traversal(root, targetSum, [], result)
#         return result
#     def traversal(self,node, count, path, result):
#             if not node:
#                 return
#             path.append(node.val)
#             count -= node.val
#             if not node.left and not node.right and count == 0:
#                 result.append(list(path))
#             self.traversal(node.left, count, path, result)
#             self.traversal(node.right, count, path, result)
#             path.pop()
# *** (版本三) 迭代
class Solution:
    def pathSum(self, root: TreeNode, targetSum: int) -> List[List[int]]:
        if not root:
            return []
        stack = [(root, [root.val])]
        res = []
        while stack:
            node, path = stack.pop()
            if not node.left and not node.right and sum(path) == targetSum:
                res.append(path)
            if node.left:
                stack.append((node.left, path + [node.left.val]))
            if node.right:
                stack.append((node.right, path + [node.right.val]))
        return res
