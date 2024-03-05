"""
===================================================================================================
NOTE:
递归(深度和广度)只需要function call function, 用一个result储存结果;

DFS 深度(迭代)需要额外用stack来存遍历的node, 用while循环来遍历stack里面的所有node。数据结构栈stack用list [], st = []
可用st.pop()获得item (last one in the stack)
深度遍历先遍历子node, 然后再处理node. 用同一写法的话,都是先遍历(包括用None来先标记处理node)在处理."所以是preorder的话, 先处理node再添加子node; 如果不是preorder, 需要用None标记待处理node"
重点:要区分先处理还是后处理node

BFS 广度(迭代)需要额外用queue(FIFO)来存遍历的node, 用while循环来遍历queue里面的所有node。数据结构队列queue用deque([]), q=collections.deque()
可以用q.popleft()获得item (first one in the queue)
广度遍历level order先处理node, 然后再遍历添加子node
广度(递归和迭代)都是先处理node, 再遍历添加子node
===================================================================================================

满二叉树: 如果一棵二叉树只有度为0的结点和度为2的结点, 并且度为0的结点在同一层上, 则这棵二叉树为满二叉树。
这棵二叉树为满二叉树, 也可以说深度为k, 有2^k-1个节点的二叉树。

完全二叉树的定义如下: 在完全二叉树中, 除了最底层节点可能没填满外, 其余每层节点数都达到最大值, 
并且最下面一层的节点都集中在该层最左边的若干位置。若最底层为第 h 层, 则该层包含 1~ 2^(h-1)  个节点。
优先级队列其实是一个堆, 堆就是一棵完全二叉树, 同时保证父子节点的顺序关系。

二叉搜索树是有数值的了, 二叉搜索树是一个有序树。
1. 若它的左子树不空, 则左子树上所有结点的值均小于它的根结点的值;
2. 若它的右子树不空, 则右子树上所有结点的值均大于它的根结点的值;
3. 它的左、右子树也分别为二叉排序树

平衡二叉搜索树: 又被称为AVL(Adelson-Velsky and Landis)树, 
且具有以下性质: 它是一棵空树或它的左右两个子树的高度差的绝对值不超过1, 并且左右两个子树都是一棵平衡二叉树。

链式存储方式就用指针,  顺序存储的方式就是用数组

二叉树主要有两种遍历方式:
1. 深度优先遍历: 先往深走, 遇到叶子节点再往回走。
2. 广度优先遍历: 一层一层的去遍历。
深度优先遍历
1. 前序遍历(递归法, 迭代法)
2. 中序遍历(递归法, 迭代法)
3. 后序遍历(递归法, 迭代法)
广度优先遍历
1. 层次遍历(迭代法)

在深度优先遍历中：有三个顺序，前中后序遍历， 有同学总分不清这三个顺序，经常搞混，我这里教大家一个技巧。
这里前中后，其实指的就是中间节点的遍历顺序，只要大家记住 前中后序指的就是中间节点的位置就可以了。
看如下中间节点的顺序，就可以发现，中间节点的顺序就是所谓的遍历方式
前序遍历：中左右
中序遍历：左中右
后序遍历：左右中

二叉树中深度优先和广度优先遍历实现方式, 我们做二叉树相关题目, 经常会使用递归的方式来实现深度优先遍历, 也就是实现前中后序遍历, 使用递归是比较方便的。
栈其实就是递归的一种实现结构, 也就说前中后序遍历的逻辑其实都是可以借助栈使用递归的方式来实现的。
广度优先遍历的实现一般使用队列来实现, 这也是队列先进先出的特点所决定的, 因为需要先进先出的结构, 才能一层一层的来遍历二叉树。

有同学会把红黑树和二叉平衡搜索树弄分开了, 其实红黑树就是一种二叉平衡搜索树, 这两个树不是独立的, 所以C++中map、multimap、set、multiset的底层实现机制是二叉平衡搜索树, 再具体一点是红黑树。

morris遍历是二叉树遍历算法的超强进阶算法, morris遍历可以将非递归遍历中的空间复杂度降为O(1), 感兴趣大家就去查一查学习学习, 比较小众, 面试几乎不会考。我其实也没有研究过, 就不做过多介绍了。
"""
# 二叉树节点定义
class TreeNode:
    def __init__(self, val, left = None, right = None):
        self.val = val
        self.left = left
        self.right = right

"""
DFS 深度优先(前中后序遍历) 递归
"""
# 前序遍历-递归-LC144_二叉树的前序遍历
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# class Solution:
#     def preorderTraversal(self, root: TreeNode) -> List[int]:
#         # 保存结果
#         result = []
#         def traversal(root: TreeNode):
#             if root == None:
#                 return
#             result.append(root.val) # 前序
#             traversal(root.left)    # 左
#             traversal(root.right)   # 右
#         traversal(root)
#         return result
# 这第二种方法更好,因为func call func, 只用一个func就够了,不需要再早一点func了
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        if not root:
            return []

        left = self.preorderTraversal(root.left)
        right = self.preorderTraversal(root.right)

        return  [root.val] + left +  right


# # 中序遍历-递归-LC94_二叉树的中序遍历
# class Solution:
#     def inorderTraversal(self, root: TreeNode) -> List[int]:
#         result = []
#         def traversal(root: TreeNode):
#             if root == None:
#                 return
#             traversal(root.left)    # 左
#             result.append(root.val) # 中序
#             traversal(root.right)   # 右
#         traversal(root)
#         return result
# 中序遍历-递归-LC94_二叉树的中序遍历
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        if root is None:
            return []

        left = self.inorderTraversal(root.left)
        right = self.inorderTraversal(root.right)

        return left + [root.val] + right


# # 后序遍历-递归-LC145_二叉树的后序遍历
# class Solution:
#     def postorderTraversal(self, root: TreeNode) -> List[int]:
#         result = []
#         def traversal(root: TreeNode):
#             if root == None:
#                 return
#             traversal(root.left)    # 左
#             traversal(root.right)   # 右
#             result.append(root.val) # 后序
#         traversal(root)
#         return result
# 后序遍历-递归-LC145_二叉树的后序遍历
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        if not root:
            return []

        left = self.postorderTraversal(root.left)
        right = self.postorderTraversal(root.right)

        return left + right + [root.val]


"""
DFS 深度优先(前中后序遍历) 迭代
数据结构: 栈 stack, [] in python
注意放入stack的顺序,如是前中后,放入的应该是后中前,因为先遍历再处理.因为用的是stack,所以遍历是处理的反顺序.
"""
# 迭代统一写法
# 迭代法前序遍历：
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        result = []
        st= []
        if root:
            st.append(root)
        while st:
            node = st.pop()
            if node != None:
                if node.right: #右
                    st.append(node.right)
                if node.left: #左
                    st.append(node.left)
                st.append(node) #中
                st.append(None)
            else:
                node = st.pop()
                result.append(node.val)
        return result
# 迭代法中序遍历：
    # 深度遍历先遍历子node, 然后再处理node
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        result = []
        st = []
        if root:
            st.append(root)
        while st:
            node = st.pop()
            if node != None:
                # 深度遍历先遍历子node, 然后再处理node
                if node.right: #添加右节点(空节点不入栈)
                    st.append(node.right)
                
                st.append(node) #添加中节点
                st.append(None) #中节点访问过, 但是还没有处理, 加入空节点做为标记。
                
                if node.left: #添加左节点(空节点不入栈)
                    st.append(node.left)
            else: #只有遇到空节点的时候, 才将下一个节点放进结果集
                # 深度遍历先遍历子node, 然后再处理node
                node = st.pop() #重新取出栈中元素
                result.append(node.val) #加入到结果集
        return result
# 迭代法后序遍历：
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        result = []
        st = []
        if root:
            st.append(root)
        while st:
            node = st.pop()
            if node != None:
                st.append(node) #中
                st.append(None)
                
                if node.right: #右
                    st.append(node.right)
                if node.left: #左
                    st.append(node.left)
            else:
                node = st.pop()
                result.append(node.val)
        return result


"""
BFS 广度优先(层序遍历) 递归
"""
# 102.二叉树的层序遍历
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        res = []
        def helper(root, depth):
            if not root: return []
            if len(res) == depth: res.append([]) # start the current depth
            res[depth].append(root.val) # fulfil the current depth
            if  root.left: helper(root.left, depth + 1) # process child nodes for the next depth
            if  root.right: helper(root.right, depth + 1)
        helper(root, 0)
        return res
# 递归法
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        levels = []
        self.helper(root, 0, levels)
        return levels
    
    def helper(self, node, level, levels):
        if not node:
            return
        if len(levels) == level:
            levels.append([])
        levels[level].append(node.val)
        self.helper(node.left, level + 1, levels)
        self.helper(node.right, level + 1, levels)


"""
BFS 广度优先(层序遍历) 迭代
数据结构: 队列 queue, deque() in python
"""
# 利用迭代法
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    """二叉树层序遍历迭代解法"""
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        
        result = []
        que = collections.deque([root])

        while que:
            # que里面现有的所有node都是一层要遍历的node
            level = []
            # 遍历一层的所有node
            for _ in range(len(que)):
                # 处理node
                cur = que.popleft()
                level.append(cur.val)
                # 添加左右子node
                if cur.left:
                    que.append(cur.left)
                if cur.right:
                    que.append(cur.right)
            result.append(level)

        return result


"""
二叉搜索树中的搜索
"""
# 700.二叉搜索树中的搜索
# 递归法 (方法一)
class Solution:
    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        # 为什么要有返回值: 
        #   因为搜索到目标节点就要立即return, 
        #   这样才是找到节点就返回(搜索某一条边), 如果不加return, 就是遍历整棵树了。
        if not root or root.val == val: 
            return root

        if root.val > val: 
            return self.searchBST(root.left, val)

        if root.val < val: 
            return self.searchBST(root.right, val)

# 迭代法 (方法二)
class Solution:
    def searchBST(self, root: TreeNode, val: int) -> TreeNode:
        while root:
            if val < root.val: root = root.left
            elif val > root.val: root = root.right
            else: return root
        return None



#(1-10) 107.二叉树的层次遍历 II
class Solution:
    """二叉树层序遍历II迭代解法"""

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrderBottom(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        queue = collections.deque([root])
        result = []
        while queue:
            level = []
            for _ in range(len(queue)):
                cur = queue.popleft()
                level.append(cur.val)
                if cur.left:
                    queue.append(cur.left)
                if cur.right:
                    queue.append(cur.right)
            result.append(level)
        return result[::-1]


# 199.二叉树的右视图
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rightSideView(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        
        queue = collections.deque([root])
        right_view = []
        
        while queue:
            level_size = len(queue)
            
            for i in range(level_size):
                node = queue.popleft()
                
                if i == level_size - 1:
                    right_view.append(node.val)
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        
        return right_view


# 637.二叉树的层平均值
class Solution:
    """二叉树层平均值迭代解法"""

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def averageOfLevels(self, root: TreeNode) -> List[float]:
        if not root:
            return []

        queue = collections.deque([root])
        averages = []
        
        while queue:
            size = len(queue)
            level_sum = 0
            for i in range(size):
                node = queue.popleft()                
                level_sum += node.val
                    
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            averages.append(level_sum / size)
        
        return averages


# 429.N叉树的层序遍历
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""

class Solution:
    def levelOrder(self, root: 'Node') -> List[List[int]]:
        if not root:
            return []

        result = []
        queue = collections.deque([root])

        while queue:
            level_size = len(queue)
            level = []

            for _ in range(level_size):
                node = queue.popleft()
                level.append(node.val)

                for child in node.children:
                    queue.append(child)

            result.append(level)

        return result


# 515.在每个树行中找最大值
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def largestValues(self, root: TreeNode) -> List[int]:
        if not root:
            return []

        result = []
        queue = collections.deque([root])

        while queue:
            level_size = len(queue)
            max_val = float('-inf')

            for _ in range(level_size):
                node = queue.popleft()
                max_val = max(max_val, node.val)

                if node.left:
                    queue.append(node.left)

                if node.right:
                    queue.append(node.right)

            result.append(max_val)

        return result


# 116.填充每个节点的下一个右侧节点指针
"""
# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""
class Solution:
    def connect(self, root: 'Node') -> 'Node':
        if not root:
            return root
        
        queue = collections.deque([root])
        
        while queue:
            level_size = len(queue)
            prev = None
            
            for i in range(level_size):
                node = queue.popleft()
                
                if prev:
                    prev.next = node
                
                prev = node
                
                if node.left:
                    queue.append(node.left)
                
                if node.right:
                    queue.append(node.right)
        
        return root


# 117.填充每个节点的下一个右侧节点指针II
# 层序遍历解法
"""
# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""
class Solution:
    def connect(self, root: 'Node') -> 'Node':
        if not root:
            return root
        
        queue = collections.deque([root])
        
        while queue:
            level_size = len(queue)
            prev = None
            
            for i in range(level_size):
                node = queue.popleft()
                
                if prev:
                    prev.next = node
                
                prev = node
                
                if node.left:
                    queue.append(node.left)
                
                if node.right:
                    queue.append(node.right)
        
        return root


# 104.二叉树的最大深度
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        depth = 0
        queue = collections.deque([root])
        
        while queue:
            depth += 1
            for _ in range(len(queue)):
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        
        return depth


# 111.二叉树的最小深度
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def minDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        depth = 0
        queue = collections.deque([root])
        
        while queue:
            depth += 1 
            for _ in range(len(queue)):
                node = queue.popleft()
                
                if not node.left and not node.right:
                    return depth
            
                if node.left:
                    queue.append(node.left)
                    
                if node.right:
                    queue.append(node.right)

        return depth


#1 226.翻转二叉树
## 递归法：前序遍历
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return None
        root.left, root.right = root.right, root.left #中
        self.invertTree(root.left) #左
        self.invertTree(root.right) #右
        return root
# 迭代法：深度优先遍历(前序遍历)
    # 注意中左右的遍历顺序,一般用stack应该用右左中的反顺序
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return root
        st = []
        st.append(root)
        while st:
            node = st.pop()
            node.left, node.right = node.right, node.left #中
            if node.left:
                st.append(node.left) #左
            if node.right:
                st.append(node.right) #右
        return root


# 递归法：中序遍历
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return None
        self.invertTree(root.left)
        root.left, root.right = root.right, root.left
        self.invertTree(root.left)
        return root
# 迭代法：中序遍历
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return None      
        stack = [root]        
        while stack:
            node = stack.pop()                   
            if node.left:
                stack.append(node.left)
            node.left, node.right = node.right, node.left               
            if node.right:
                stack.append(node.right)       
        return root


# 递归法：后序遍历
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return None
        self.invertTree(root.left)
        self.invertTree(root.right)
        root.left, root.right = root.right, root.left
        return root
# 迭代法：后序遍历
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return None      
        stack = [root]        
        while stack:
            node = stack.pop()                   
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)  
            node.left, node.right = node.right, node.left
        return root


# 迭代法：广度优先遍历(层序遍历)
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if not root: 
            return None

        queue = collections.deque([root])
        while queue:
            size = len(queue)
            for i in range(size):
                node = queue.popleft()
                node.left, node.right = node.right, node.left #节点处理
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return root


#2 101. 对称二叉树
# 给定一个二叉树, 检查它是否是镜像对称的。
# 递归法
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        if not root:
            return True
        return self.compare(root.left, root.right)
        
    def compare(self, left, right):
        #首先排除空节点的情况
        if left == None and right != None: return False
        elif left != None and right == None: return False
        elif left == None and right == None: return True
        #排除了空节点, 再排除数值不相同的情况
        elif left.val != right.val: return False
        
        #此时就是：左右节点都不为空, 且数值相同的情况
        #此时才做递归, 做下一层的判断
        outside = self.compare(left.left, right.right) #左子树：左、 右子树：右
        inside = self.compare(left.right, right.left) #左子树：右、 右子树：左
        isSame = outside and inside #左子树：中、 右子树：中 (逻辑处理)
        return isSame
# 迭代法: 使用队列
import collections
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        if not root:
            return True
        queue = collections.deque()
        queue.append(root.left) #将左子树头结点加入队列
        queue.append(root.right) #将右子树头结点加入队列
        while queue: #接下来就要判断这这两个树是否相互翻转
            leftNode = queue.popleft()
            rightNode = queue.popleft()
            if not leftNode and not rightNode: #左节点为空、右节点为空，此时说明是对称的
                continue
            
            #左右一个节点不为空，或者都不为空但数值不相同，返回false
            if not leftNode or not rightNode or leftNode.val != rightNode.val:
                return False
            queue.append(leftNode.left) #加入左节点左孩子
            queue.append(rightNode.right) #加入右节点右孩子
            queue.append(leftNode.right) #加入左节点右孩子
            queue.append(rightNode.left) #加入右节点左孩子
        return True
# 迭代法：使用栈
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        if not root:
            return True
        st = [] #这里改成了栈
        st.append(root.left)
        st.append(root.right)
        while st:
            rightNode = st.pop()
            leftNode = st.pop()
            if not leftNode and not rightNode:
                continue
            if not leftNode or not rightNode or leftNode.val != rightNode.val:
                return False
            st.append(leftNode.left)
            st.append(rightNode.right)
            st.append(leftNode.right)
            st.append(rightNode.left)
        return True
# 层次遍历
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        if not root:
            return True
        
        queue = collections.deque([root.left, root.right])
        
        while queue:
            level_size = len(queue)
            
            if level_size % 2 != 0:
                return False
            
            level_vals = []
            for i in range(level_size):
                node = queue.popleft()
                if node:
                    level_vals.append(node.val)
                    queue.append(node.left)
                    queue.append(node.right)
                else:
                    level_vals.append(None)
                    
            if level_vals != level_vals[::-1]:
                return False
            
        return True


#3 104.二叉树的最大深度
# 给定一个二叉树, 找出其最大深度。
# 二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。
# 递归法
class solution:
    def maxdepth(self, root: treenode) -> int:
        return self.getdepth(root)
        
    def getdepth(self, node):
        if not node:
            return 0
        leftheight = self.getdepth(node.left) #左
        rightheight = self.getdepth(node.right) #右
        height = 1 + max(leftheight, rightheight) #中
        return height
# 递归法：精简代码
class solution:
    def maxdepth(self, root: treenode) -> int:
        if not root:
            return 0
        return 1 + max(self.maxdepth(root.left), self.maxdepth(root.right))
# 层序遍历迭代法
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        depth = 0
        queue = collections.deque([root])
        
        while queue:
            depth += 1
            for _ in range(len(queue)):
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        
        return depth


#4 559.n叉树的最大深度
# 递归法
class Solution:
    def maxDepth(self, root: 'Node') -> int:
        if not root:
            return 0
        
        max_depth = 1
        
        for child in root.children:
            max_depth = max(max_depth, self.maxDepth(child) + 1)
        
        return max_depth
# 迭代法
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        depth = 0
        queue = collections.deque([root])
        
        while queue:
            depth += 1
            for _ in range(len(queue)):
                node = queue.popleft()
                for child in node.children:
                    queue.append(child)
        
        return depth
# 使用栈
class Solution:
    def maxDepth(self, root: 'Node') -> int:
        if not root:
            return 0
        
        max_depth = 0
        
        stack = [(root, 1)]
        
        while stack:
            node, depth = stack.pop()
            max_depth = max(max_depth, depth)
            for child in node.children:
                stack.append((child, depth + 1))
        
        return max_depth
# class solution:
#     def maxdepth(self, root: treenode) -> int:
#         return self.getdepth(root)
        
#     def getdepth(self, node):
#         if not node:
#             return 0
#         leftdepth = self.getdepth(node.left) #左
#         rightdepth = self.getdepth(node.right) #右
#         depth = 1 + max(leftdepth, rightdepth) #中
#         return depth

# class solution:
#     def maxdepth(self, root: 'node') -> int:
#         if not root:
#             return 0
#         depth = 0
#         for i in range(len(root.children)):
#             depth = max(depth, self.maxdepth(root.children[i]))
#         return depth + 1


#5 111.二叉树的最小深度
# 给定一个二叉树, 找出其最小深度。
# 最小深度是从根节点到最近叶子节点的最短路径上的节点数量。
# 递归法（版本一）
class Solution:
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
        
        result = 1 + min(leftDepth, rightDepth)
        return result

    def minDepth(self, root):
        return self.getDepth(root)
# 递归法（版本二）
class Solution:
    def minDepth(self, root):
        if root is None:
            return 0
        if root.left is None and root.right is not None:
            return 1 + self.minDepth(root.right)
        if root.left is not None and root.right is None:
            return 1 + self.minDepth(root.left)
        return 1 + min(self.minDepth(root.left), self.minDepth(root.right))
# 递归法（版本三）前序
class Solution:
    def __init__(self):
        self.result = float('inf')

    def getDepth(self, node, depth):
        if node is None:
            return
        if node.left is None and node.right is None:
            self.result = min(self.result, depth)
        if node.left:
            self.getDepth(node.left, depth + 1)
        if node.right:
            self.getDepth(node.right, depth + 1)

    def minDepth(self, root):
        if root is None:
            return 0
        self.getDepth(root, 1)
        return self.result
# 迭代法
class Solution:
    def minDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        depth = 0
        queue = collections.deque([root])
        
        while queue:
            depth += 1 
            for _ in range(len(queue)):
                node = queue.popleft()
                
                if not node.left and not node.right:
                    return depth
            
                if node.left:
                    queue.append(node.left)
                    
                if node.right:
                    queue.append(node.right)

        return depth
# class Solution:
#     def minDepth(self, root: TreeNode) -> int:
#         if not root:
#             return 0
#         if not root.left and not root.right:
#             return 1

#         min_depth = 10**9
#         if root.left:
#             min_depth = min(self.minDepth(root.left), min_depth) # 获得左子树的最小高度
#         if root.right:
#             min_depth = min(self.minDepth(root.right), min_depth) # 获得右子树的最小高度
#         return min_depth + 1


#6 ??? 222.完全二叉树的节点个数
# 递归法
class Solution:
    def countNodes(self, root: TreeNode) -> int:
        return self.getNodesNum(root)
        
    def getNodesNum(self, cur):
        if not cur:
            return 0
        leftNum = self.getNodesNum(cur.left) #左
        rightNum = self.getNodesNum(cur.right) #右
        treeNum = leftNum + rightNum + 1 #中
        return treeNum
# 递归法：精简版
class Solution:
    def countNodes(self, root: TreeNode) -> int:
        if not root:
            return 0
        return 1 + self.countNodes(root.left) + self.countNodes(root.right)
# 迭代法
import collections
class Solution:
    def countNodes(self, root: TreeNode) -> int:
        queue = collections.deque()
        if root:
            queue.append(root)
        result = 0
        while queue:
            size = len(queue)
            for i in range(size):
                node = queue.popleft()
                result += 1 #记录节点数量
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return result
# 完全二叉树
class Solution:
    def countNodes(self, root: TreeNode) -> int:
        if not root:
            return 0
        left = root.left
        right = root.right
        leftDepth = 0 #这里初始为0是有目的的，为了下面求指数方便
        rightDepth = 0
        while left: #求左子树深度
            left = left.left
            leftDepth += 1
        while right: #求右子树深度
            right = right.right
            rightDepth += 1
        if leftDepth == rightDepth:
            return (2 << leftDepth) - 1 #注意(2<<1) 相当于2^2，所以leftDepth初始为0
        return self.countNodes(root.left) + self.countNodes(root.right) + 1
# 完全二叉树写法2
class Solution: # 利用完全二叉树特性
    def countNodes(self, root: TreeNode) -> int:
        if not root: return 0
        count = 1
        left = root.left; right = root.right
        while left and right:
            count+=1
            left = left.left; right = right.right
        if not left and not right: # 如果同时到底说明是满二叉树，反之则不是
            return 2**count-1
        return 1+self.countNodes(root.left)+self.countNodes(root.right)  
# 完全二叉树写法3
class Solution: # 利用完全二叉树特性
    def countNodes(self, root: TreeNode) -> int:
        if not root: return 0
        count = 0
        left = root.left; right = root.right
        while left and right:
            count+=1
            left = left.left; right = right.right
        if not left and not right: # 如果同时到底说明是满二叉树，反之则不是
            return (2<<count)-1
        return 1+self.countNodes(root.left)+self.countNodes(root.right) 


# 110.平衡二叉树
# 给定一个二叉树, 判断它是否是高度平衡的二叉树。
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
# 递归法精简版
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
# 迭代法
class Solution:
    def getDepth(self, cur):
        st = []
        if cur is not None:
            st.append(cur)
        depth = 0
        result = 0
        while st:
            node = st[-1]
            if node is not None:
                st.pop()
                st.append(node)                           # 中
                st.append(None)
                depth += 1
                if node.right:
                    st.append(node.right)                 # 右
                if node.left:
                    st.append(node.left)                   # 左

            else:               
                node = st.pop()
                st.pop()
                depth -= 1
            result = max(result, depth)
        return result

    def isBalanced(self, root):
        st = []
        if root is None:
            return True
        st.append(root)
        while st:
            node = st.pop()                                 # 中
            if abs(self.getDepth(node.left) - self.getDepth(node.right)) > 1:
                return False
            if node.right:
                st.append(node.right)                       # 右（空节点不入栈）
            if node.left:
                st.append(node.left)                         # 左（空节点不入栈）
        return True
# 迭代法精简版
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        if not root:
            return True

        height_map = {}
        stack = [root]
        while stack:
            node = stack.pop()
            if node:
                stack.append(node)
                stack.append(None)
                if node.left: stack.append(node.left)
                if node.right: stack.append(node.right)
            else:
                real_node = stack.pop()
                left, right = height_map.get(real_node.left, 0), height_map.get(real_node.right, 0)
                if abs(left - right) > 1:
                    return False
                height_map[real_node] = 1 + max(left, right)
        return True


# 257. 二叉树的所有路径
# 给定一个二叉树, 返回所有从根节点到叶子节点的路径。
# 递归法+回溯
# Definition for a binary tree node.
class Solution:
    def traversal(self, cur, path, result):
        path.append(cur.val)  # 中
        if not cur.left and not cur.right:  # 到达叶子节点
            sPath = '->'.join(map(str, path))
            result.append(sPath)
            return
        if cur.left:  # 左
            self.traversal(cur.left, path, result)
            path.pop()  # 回溯
        if cur.right:  # 右
            self.traversal(cur.right, path, result)
            path.pop()  # 回溯

    def binaryTreePaths(self, root):
        result = []
        path = []
        if not root:
            return result
        self.traversal(root, path, result)
        return result
# 递归法+隐形回溯（版本一）
from typing import List, Optional

class Solution:
    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
        if not root:
            return []
        result = []
        self.traversal(root, [], result)
        return result
    
    def traversal(self, cur: TreeNode, path: List[int], result: List[str]) -> None:
        if not cur:
            return
        path.append(cur.val)
        if not cur.left and not cur.right:
            result.append('->'.join(map(str, path)))
        if cur.left:
            self.traversal(cur.left, path[:], result)
        if cur.right:
            self.traversal(cur.right, path[:], result)
# 递归法+隐形回溯（版本二）
class Solution:
    def binaryTreePaths(self, root: TreeNode) -> List[str]:
        path = ''
        result = []
        if not root: return result
        self.traversal(root, path, result)
        return result
    
    def traversal(self, cur: TreeNode, path: str, result: List[str]) -> None:
        path += str(cur.val)
        # 若当前节点为leave，直接输出
        if not cur.left and not cur.right:
            result.append(path)

        if cur.left:
            # + '->' 是隐藏回溯
            self.traversal(cur.left, path + '->', result)
        
        if cur.right:
            self.traversal(cur.right, path + '->', result)
# 迭代法
class Solution:

    def binaryTreePaths(self, root: TreeNode) -> List[str]:
        # 题目中节点数至少为1
        stack, path_st, result = [root], [str(root.val)], []

        while stack:
            cur = stack.pop()
            path = path_st.pop()
            # 如果当前节点为叶子节点，添加路径到结果中
            if not (cur.left or cur.right):
                result.append(path)
            if cur.right:
                stack.append(cur.right)
                path_st.append(path + '->' + str(cur.right.val))
            if cur.left:
                stack.append(cur.left)
                path_st.append(path + '->' + str(cur.left.val))

        return result


# 404.左叶子之和
# 计算给定二叉树的所有左叶子之和。
# 递归
class Solution:
    def sumOfLeftLeaves(self, root):
        if root is None:
            return 0
        if root.left is None and root.right is None:
            return 0
        
        leftValue = self.sumOfLeftLeaves(root.left)  # 左
        if root.left and not root.left.left and not root.left.right:  # 左子树是左叶子的情况
            leftValue = root.left.val
            
        rightValue = self.sumOfLeftLeaves(root.right)  # 右

        sum_val = leftValue + rightValue  # 中
        return sum_val
# 递归精简版
class Solution:
    def sumOfLeftLeaves(self, root):
        if root is None:
            return 0
        leftValue = 0
        if root.left is not None and root.left.left is None and root.left.right is None:
            leftValue = root.left.val
        return leftValue + self.sumOfLeftLeaves(root.left) + self.sumOfLeftLeaves(root.right)
# 迭代法
class Solution:
    def sumOfLeftLeaves(self, root):
        if root is None:
            return 0
        st = [root]
        result = 0
        while st:
            node = st.pop()
            if node.left and node.left.left is None and node.left.right is None:
                result += node.left.val
            if node.right:
                st.append(node.right)
            if node.left:
                st.append(node.left)
        return result


# 513.找树左下角的值
# 给定一个二叉树, 在树的最后一行找到最左边的值。
# （版本一）递归法 + 回溯
class Solution:
    def findBottomLeftValue(self, root: TreeNode) -> int:
        self.max_depth = float('-inf')
        self.result = None
        self.traversal(root, 0)
        return self.result
    
    def traversal(self, node, depth):
        if not node.left and not node.right:
            if depth > self.max_depth:
                self.max_depth = depth
                self.result = node.val
            return
        
        if node.left:
            depth += 1
            self.traversal(node.left, depth)
            depth -= 1
        if node.right:
            depth += 1
            self.traversal(node.right, depth)
            depth -= 1
# （版本二）递归法+精简
class Solution:
    def findBottomLeftValue(self, root: TreeNode) -> int:
        self.max_depth = float('-inf')
        self.result = None
        self.traversal(root, 0)
        return self.result
    
    def traversal(self, node, depth):
        if not node.left and not node.right:
            if depth > self.max_depth:
                self.max_depth = depth
                self.result = node.val
            return
        
        if node.left:
            self.traversal(node.left, depth+1)
        if node.right:
            self.traversal(node.right, depth+1)
# (版本三） 迭代法
from collections import deque
class Solution:
    def findBottomLeftValue(self, root):
        if root is None:
            return 0
        queue = deque()
        queue.append(root)
        result = 0
        while queue:
            size = len(queue)
            for i in range(size):
                node = queue.popleft()
                if i == 0:
                    result = node.val
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return result


# 112. 路径总和
# 给定一个二叉树和一个目标和, 判断该树中是否存在根节点到叶子节点的路径, 这条路径上所有节点值相加等于目标和。
# (版本一) 递归
class Solution:
    def traversal(self, cur: TreeNode, count: int) -> bool:
        if not cur.left and not cur.right and count == 0: # 遇到叶子节点，并且计数为0
            return True
        if not cur.left and not cur.right: # 遇到叶子节点直接返回
            return False
        
        if cur.left: # 左
            count -= cur.left.val
            if self.traversal(cur.left, count): # 递归，处理节点
                return True
            count += cur.left.val # 回溯，撤销处理结果
            
        if cur.right: # 右
            count -= cur.right.val
            if self.traversal(cur.right, count): # 递归，处理节点
                return True
            count += cur.right.val # 回溯，撤销处理结果
            
        return False
    
    def hasPathSum(self, root: TreeNode, sum: int) -> bool:
        if root is None:
            return False
        return self.traversal(root, sum - root.val)  
# (版本二) 递归 + 精简
class Solution:
    def hasPathSum(self, root: TreeNode, sum: int) -> bool:
        if not root:
            return False
        if not root.left and not root.right and sum == root.val:
            return True
        return self.hasPathSum(root.left, sum - root.val) or self.hasPathSum(root.right, sum - root.val)
# (版本三) 迭代
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
            # 右节点，压进去一个节点的时候，将该节点的路径数值也记录下来
            if node.right:
                st.append((node.right, path_sum + node.right.val))
            # 左节点，压进去一个节点的时候，将该节点的路径数值也记录下来
            if node.left:
                st.append((node.left, path_sum + node.left.val))
        return False


# 0113.路径总和-ii
# (版本一) 递归
class Solution:
    def __init__(self):
        self.result = []
        self.path = []

    def traversal(self, cur, count):
        if not cur.left and not cur.right and count == 0: # 遇到了叶子节点且找到了和为sum的路径
            self.result.append(self.path[:])
            return

        if not cur.left and not cur.right: # 遇到叶子节点而没有找到合适的边，直接返回
            return

        if cur.left: # 左 （空节点不遍历）
            self.path.append(cur.left.val)
            count -= cur.left.val
            self.traversal(cur.left, count) # 递归
            count += cur.left.val # 回溯
            self.path.pop() # 回溯

        if cur.right: #  右 （空节点不遍历）
            self.path.append(cur.right.val) 
            count -= cur.right.val
            self.traversal(cur.right, count) # 递归
            count += cur.right.val # 回溯
            self.path.pop() # 回溯

        return

    def pathSum(self, root: TreeNode, sum: int) -> List[List[int]]:
        self.result.clear()
        self.path.clear()
        if not root:
            return self.result
        self.path.append(root.val) # 把根节点放进路径
        self.traversal(root, sum - root.val)
        return self.result 
# (版本二) 递归 + 精简
class Solution:
    def pathSum(self, root: TreeNode, targetSum: int) -> List[List[int]]:
        
        result = []
        self.traversal(root, targetSum, [], result)
        return result
    def traversal(self,node, count, path, result):
            if not node:
                return
            path.append(node.val)
            count -= node.val
            if not node.left and not node.right and count == 0:
                result.append(list(path))
            self.traversal(node.left, count, path, result)
            self.traversal(node.right, count, path, result)
            path.pop()
# (版本三) 迭代
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
            if node.right:
                stack.append((node.right, path + [node.right.val]))
            if node.left:
                stack.append((node.left, path + [node.left.val]))
        return res


# 105.从前序与中序遍历序列构造二叉树
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        # 第一步: 特殊情况讨论: 树为空. 或者说是递归终止条件
        if not preorder:
            return None

        # 第二步: 前序遍历的第一个就是当前的中间节点.
        root_val = preorder[0]
        root = TreeNode(root_val)

        # 第三步: 找切割点.
        separator_idx = inorder.index(root_val)

        # 第四步: 切割inorder数组. 得到inorder数组的左,右半边.
        inorder_left = inorder[:separator_idx]
        inorder_right = inorder[separator_idx + 1:]

        # 第五步: 切割preorder数组. 得到preorder数组的左,右半边.
        # ⭐️ 重点1: 中序数组大小一定跟前序数组大小是相同的.
        preorder_left = preorder[1:1 + len(inorder_left)]
        preorder_right = preorder[1 + len(inorder_left):]

        # 第六步: 递归
        root.left = self.buildTree(preorder_left, inorder_left)
        root.right = self.buildTree(preorder_right, inorder_right)
        # 第七步: 返回答案
        return root


# 106.从中序与后序遍历序列构造二叉树
# 根据一棵树的中序遍历与后序遍历构造二叉树。
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> TreeNode:
        # 第一步: 特殊情况讨论: 树为空. (递归终止条件)
        if not postorder:
            return None

        # 第二步: 后序遍历的最后一个就是当前的中间节点.
        root_val = postorder[-1]
        root = TreeNode(root_val)

        # 第三步: 找切割点.
        separator_idx = inorder.index(root_val)

        # 第四步: 切割inorder数组. 得到inorder数组的左,右半边.
        inorder_left = inorder[:separator_idx]
        inorder_right = inorder[separator_idx + 1:]

        # 第五步: 切割postorder数组. 得到postorder数组的左,右半边.
        # ⭐️ 重点1: 中序数组大小一定跟后序数组大小是相同的.
        postorder_left = postorder[:len(inorder_left)]
        postorder_right = postorder[len(inorder_left): len(postorder) - 1]

        # 第六步: 递归
        root.left = self.buildTree(inorder_left, postorder_left)
        root.right = self.buildTree(inorder_right, postorder_right)
         # 第七步: 返回答案
        return root


# 654.最大二叉树
# 给定一个不含重复元素的整数数组。一个以此数组构建的最大二叉树定义如下：
# 二叉树的根是数组中的最大元素。
# 左子树是通过数组中最大值左边部分构造出的最大二叉树。
# 右子树是通过数组中最大值右边部分构造出的最大二叉树。
# 通过给定的数组构建最大二叉树, 并且输出这个树的根节点。
# (版本一) 基础版
class Solution:
    def constructMaximumBinaryTree(self, nums: List[int]) -> TreeNode:
        if len(nums) == 1:
            return TreeNode(nums[0])
        node = TreeNode(0)
        # 找到数组中最大的值和对应的下标
        maxValue = 0
        maxValueIndex = 0
        for i in range(len(nums)):
            if nums[i] > maxValue:
                maxValue = nums[i]
                maxValueIndex = i
        node.val = maxValue
        # 最大值所在的下标左区间 构造左子树
        if maxValueIndex > 0:
            new_list = nums[:maxValueIndex]
            node.left = self.constructMaximumBinaryTree(new_list)
        # 最大值所在的下标右区间 构造右子树
        if maxValueIndex < len(nums) - 1:
            new_list = nums[maxValueIndex+1:]
            node.right = self.constructMaximumBinaryTree(new_list)
        return node
# (版本二) 使用下标
class Solution:
    def traversal(self, nums: List[int], left: int, right: int) -> TreeNode:
        if left >= right:
            return None
        maxValueIndex = left
        for i in range(left + 1, right):
            if nums[i] > nums[maxValueIndex]:
                maxValueIndex = i
        root = TreeNode(nums[maxValueIndex])
        root.left = self.traversal(nums, left, maxValueIndex)
        root.right = self.traversal(nums, maxValueIndex + 1, right)
        return root

    def constructMaximumBinaryTree(self, nums: List[int]) -> TreeNode:
        return self.traversal(nums, 0, len(nums))
# (版本三) 使用切片
class Solution:
    def constructMaximumBinaryTree(self, nums: List[int]) -> TreeNode:
        if not nums:
            return None
        max_val = max(nums)
        max_index = nums.index(max_val)
        node = TreeNode(max_val)
        node.left = self.constructMaximumBinaryTree(nums[:max_index])
        node.right = self.constructMaximumBinaryTree(nums[max_index+1:])
        return node


# 617.合并二叉树
# 给定两个二叉树, 想象当你将它们中的一个覆盖到另一个上时, 两个二叉树的一些节点便会重叠。


# 700.二叉搜索树中的搜索
# 给定二叉搜索树(BST)的根节点和一个值。 你需要在BST中找到节点值等于给定值的节点。 返回以该节点为根的子树。 如果节点不存在, 则返回 NULL。





# 98.验证二叉搜索树
# 给定一个二叉树, 判断其是否是一个有效的二叉搜索树。
# 假设一个二叉搜索树具有如下特征：
# 节点的左子树只包含小于当前节点的数。
# 节点的右子树只包含大于当前节点的数。
# 所有左子树和右子树自身必须也是二叉搜索树。
# 迭代-中序遍历
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        stack = []
        cur = root
        pre = None
        while cur or stack:
            if cur: # 指针来访问节点, 访问到最底层
                stack.append(cur)
                cur = cur.left
            else: # 逐一处理节点
                cur = stack.pop()
                if pre and cur.val <= pre.val: # 比较当前节点和前节点的值的大小
                    return False
                pre = cur 
                cur = cur.right
        return True


# 530.二叉搜索树的最小绝对差
# 给你一棵所有节点为非负值的二叉搜索树, 请你计算树中任意两节点的差的绝对值的最小值。


# 501.二叉搜索树中的众数
# 给定一个有相同值的二叉搜索树(BST), 找出 BST 中的所有众数(出现频率最高的元素)。
# 假定 BST 有如下定义：
# 结点左子树中所含结点的值小于等于当前结点的值
# 结点右子树中所含结点的值大于等于当前结点的值
# 左子树和右子树都是二叉搜索树


# 236. 二叉树的最近公共祖先
# 给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
# 百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q, 最近公共祖先表示为一个结点 x, 满足 x 是 p、q 的祖先且 x 的深度尽可能大(一个节点也可以是它自己的祖先)。”
class Solution:
    """二叉树的最近公共祖先 递归法"""

    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root or root == p or root == q:
            return root
        
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        
        if left and right:
            return root
        if left:
            return left
        return right


# 235. 二叉搜索树的最近公共祖先
# 给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。
# 百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q, 最近公共祖先表示为一个结点 x, 满足 x 是 p、q 的祖先且 x 的深度尽可能大(一个节点也可以是它自己的祖先)。”


# 701.二叉搜索树中的插入操作
# 给定二叉搜索树(BST)的根节点和要插入树中的值, 将值插入二叉搜索树。 返回插入后二叉搜索树的根节点。 输入数据保证, 新值和原始二叉搜索树中的任意节点值都不同。
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
        # 返回更新后的以当前root为根节点的新树, 方便用于更新上一层的父子节点关系链

        # Base Case
        if not root: return TreeNode(val)

        # 单层递归逻辑:
        if val < root.val: 
            # 将val插入至当前root的左子树中合适的位置
            # 并更新当前root的左子树为包含目标val的新左子树
            root.left = self.insertIntoBST(root.left, val)

        if root.val < val:
            # 将val插入至当前root的右子树中合适的位置
            # 并更新当前root的右子树为包含目标val的新右子树
            root.right = self.insertIntoBST(root.right, val)

        # 返回更新后的以当前root为根节点的新树
        return root


# 450.删除二叉搜索树中的节点
# 给定一个二叉搜索树的根节点 root 和一个值 key, 删除二叉搜索树中的 key 对应的节点, 并保证二叉搜索树的性质不变。返回二叉搜索树(有可能被更新)的根节点的引用。
class Solution:
    def deleteNode(self, root: TreeNode, key: int) -> TreeNode:
        if not root: return root  #第一种情况：没找到删除的节点, 遍历到空节点直接返回了
        if root.val == key:  
            if not root.left and not root.right:  #第二种情况：左右孩子都为空(叶子节点), 直接删除节点,  返回NULL为根节点
                del root
                return None
            if not root.left and root.right:  #第三种情况：其左孩子为空, 右孩子不为空, 删除节点, 右孩子补位 , 返回右孩子为根节点
                tmp = root
                root = root.right
                del tmp
                return root
            if root.left and not root.right:  #第四种情况：其右孩子为空, 左孩子不为空, 删除节点, 左孩子补位, 返回左孩子为根节点
                tmp = root
                root = root.left
                del tmp
                return root
            else:  #第五种情况：左右孩子节点都不为空, 则将删除节点的左子树放到删除节点的右子树的最左面节点的左孩子的位置
                v = root.right
                while v.left:
                    v = v.left
                v.left = root.left
                tmp = root
                root = root.right
                del tmp
                return root
        if root.val > key: root.left = self.deleteNode(root.left,key)  #左递归
        if root.val < key: root.right = self.deleteNode(root.right,key)  #右递归
        return root


# 669. 修剪二叉搜索树
# 给定一个二叉搜索树, 同时给定最小边界L 和最大边界 R。通过修剪二叉搜索树, 使得所有节点的值在[L, R]中 (R>=L) 。你可能需要改变树的根节点, 所以结果应当返回修剪好的二叉搜索树的新的根节点。


# 108.将有序数组转换为二叉搜索树
# 将一个按照升序排列的有序数组, 转换为一棵高度平衡二叉搜索树。
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        '''
        构造二叉树：重点是选取数组最中间元素为分割点, 左侧是递归左区间;右侧是递归右区间
        必然是平衡树
        左闭右闭区间
        '''
        # 返回根节点
        root = self.traversal(nums, 0, len(nums)-1)
        return root

    def traversal(self, nums: List[int], left: int, right: int) -> TreeNode:
        # Base Case
        if left > right:
            return None
        
        # 确定左右界的中心, 防越界
        mid = left + (right - left) // 2
        # 构建根节点
        mid_root = TreeNode(nums[mid])
        # 构建以左右界的中心为分割点的左右子树
        mid_root.left = self.traversal(nums, left, mid-1)
        mid_root.right = self.traversal(nums, mid+1, right)

        # 返回由被传入的左右界定义的某子树的根节点
        return mid_root


# 538.把二叉搜索树转换为累加树
# 给出二叉 搜索 树的根节点, 该树的节点值各不相同, 请你将其转换为累加树(Greater Sum Tree), 使每个节点 node 的新值等于原树中大于或等于 node.val 的值之和。
# 提醒一下, 二叉搜索树满足下列约束条件：
# 节点的左子树仅包含键 小于 节点键的节点。 节点的右子树仅包含键 大于 节点键的节点。 左右子树也必须是二叉搜索树。
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def __init__(self):
        self.pre = TreeNode()

    def convertBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        '''
        倒序累加替换：  
        [2, 5, 13] -> [[2]+[1]+[0], [2]+[1], [2]] -> [20, 18, 13]
        '''
        self.traversal(root)
        return root

    def traversal(self, root: TreeNode) -> None:
        # 因为要遍历整棵树, 所以递归函数不需要返回值
        # Base Case
        if not root: 
            return None
        # 单层递归逻辑：中序遍历的反译 - 右中左
        self.traversal(root.right)  # 右

        # 中节点：用当前root的值加上pre的值
        root.val += self.pre.val    # 中
        self.pre = root             

        self.traversal(root.left)   # 左
