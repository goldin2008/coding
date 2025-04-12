"""
Binary Tree

> https://www.programiz.com/dsa/complete-binary-tree
> https://www.programiz.com/dsa/heap-data-structure
"""
# Checking if a binary tree is a complete binary tree in Python
class Node:
    def __init__(self, item):
        self.item = item
        self.left = None
        self.right = None


# Count the number of nodes
def count_nodes(root):
    if root is None:
        return 0
    return (1 + count_nodes(root.left) + count_nodes(root.right))


# Check if the tree is complete binary tree
def is_complete(root, index, numberNodes):
    # Check if the tree is empty
    if root is None:
        return True

    if index >= numberNodes:
        return False

    return (is_complete(root.left, 2 * index + 1, numberNodes)
            and is_complete(root.right, 2 * index + 2, numberNodes))


root = Node(1)
root.left = Node(2)
root.right = Node(3)
root.left.left = Node(4)
root.left.right = Node(5)
root.right.left = Node(6)

node_count = count_nodes(root)
index = 0

if is_complete(root, index, node_count):
    print("The tree is a complete binary tree")
else:
    print("The tree is not a complete binary tree")


# Max-Heap data structure in Python
def max_heapify(arr, n, i):
    largest = i          # Initialize largest as root
    left = 2 * i + 1     # Left child
    right = 2 * i + 2    # Right child

    # If left child exists and is greater than root
    if left < n and arr[left] > arr[largest]:
        largest = left

    # If right child exists and is greater than current largest
    if right < n and arr[right] > arr[largest]:
        largest = right

    # If largest is not root, swap and heapify the affected subtree
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]  # Swap
        max_heapify(arr, n, largest)  # Recursively heapify

def insert(array, newNum):
    size = len(array)
    if size == 0:
        array.append(newNum)
    else:
        array.append(newNum)
        # Build heap (rearrange array)
        for i in range((size//2)-1, -1, -1):
            heapify(array, size, i)

def deleteNode(array, num):
    size = len(array)
    i = 0
    for i in range(0, size):
        if num == array[i]:
            break
        
    array[i], array[size-1] = array[size-1], array[i]
    array.remove(num)
    # Build heap (rearrange array)
    for i in range((len(array)//2)-1, -1, -1):
        heapify(array, len(array), i)
    
arr = []

insert(arr, 3)
insert(arr, 4)
insert(arr, 9)
insert(arr, 5)
insert(arr, 2)

print ("Max-Heap array: " + str(arr))

deleteNode(arr, 4)
print("After deleting an element: " + str(arr))

"""
===================================================================================================
NOTE:
递归(深度和广度)只需要function call function, 用一个result储存结果;

DFS 深度(迭代)需要额外用stack来存遍历的node, 用while循环来遍历stack里面的所有node。
数据结构栈stack用list [], st = []
可用st.pop()获得item (last one(right) in the stack)
**深度遍历先遍历子node, 然后再处理node**. 用同一写法的话,都是先遍历(包括用None来先标记处理node)在处理.
"所以是preorder的话, 先处理node再添加子node; 如果不是preorder, 需要用None标记待处理node"
重点:要区分先处理还是后处理node

BFS 广度(迭代)需要额外用queue(FIFO)来存遍历的node, 用while循环来遍历queue里面的所有node。
数据结构队列queue用deque([]), q=collections.deque()
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

每次写递归，都按照这三要素来写，可以保证大家写出正确的递归算法
以下以前序遍历为例：
1.确定递归函数的参数和返回值
2.确定终止条件
3.确定单层递归的逻辑

二叉树中深度优先和广度优先遍历实现方式, 我们做二叉树相关题目, 经常会使用递归的方式来实现深度优先遍历, 也就是实现前中后序遍历, 使用递归是比较方便的。
栈其实就是递归的一种实现结构, 也就说前中后序遍历的逻辑其实都是可以借助**栈**使用**递归**的方式来实现的。
广度优先遍历的实现一般使用**队列**来实现, 这也是队列先进先出的特点所决定的, 因为需要先进先出的结构, 才能一层一层的来遍历二叉树。

有同学会把红黑树和二叉平衡搜索树弄分开了, 其实红黑树就是一种二叉平衡搜索树, 这两个树不是独立的, 所以C++中map、multimap、set、multiset的底层实现机制是二叉平衡搜索树, 再具体一点是红黑树。
morris遍历是二叉树遍历算法的超强进阶算法, morris遍历可以将非递归遍历中的空间复杂度降为O(1), 感兴趣大家就去查一查学习学习, 比较小众, 面试几乎不会考。我其实也没有研究过, 就不做过多介绍了。

现在已经讲过了几种二叉树了，二叉树，二叉平衡树，完全二叉树，二叉搜索树，后面还会有平衡二叉搜索树。 那么一些同学难免会有混乱了，我针对如下三个问题，帮大家在捋顺一遍：
- 1.平衡二叉搜索树是不是二叉搜索树和平衡二叉树的结合？
是的，是二叉搜索树和平衡二叉树的结合。
- 2.平衡二叉树与完全二叉树的区别在于底层节点的位置？
是的，完全二叉树底层必须是从左到右连续的，且次底层是满的。
- 3.堆是完全二叉树和排序的结合，而不是平衡二叉搜索树？
堆是一棵完全二叉树，同时保证父子节点的顺序关系（有序）。 但完全二叉树一定是平衡二叉树，堆的排序是父节点大于子节点，而搜索树是父节点大于左孩子，小于右孩子，所以堆不是平衡二叉搜索树。

在二叉树题目选择什么遍历顺序是不少同学头疼的事情,我们做了这么多二叉树的题目了,大体分分类。
1.涉及到二叉树的构造，无论普通二叉树还是二叉搜索树一定前序，都是先构造中节点。
2.求普通二叉树的属性，一般是后序，一般要通过递归函数的返回值做计算。
3.求二叉搜索树的属性，一定是中序了，要不白瞎了有序性了。
4.注意在普通二叉树的属性中，我用的是一般为后序，例如单纯求深度就用前序，二叉树：找所有路径 也用了前序，这是为了方便让父节点指向子节点。
"""

"""
1. 二叉树的理论基础
关于二叉树，你该了解这些！ ：二叉树的种类、存储方式、遍历方式、定义方式
二叉树的遍历方式
深度优先遍历
二叉树：前中后序递归法 ：递归三部曲初次亮相
二叉树：前中后序迭代法（一） ：通过栈模拟递归
二叉树：前中后序迭代法（二）统一风格
广度优先遍历
二叉树的层序遍历 ：通过队列模拟
"""
# 二叉树节点定义
class TreeNode:
    def __init__(self, val, left = None, right = None):
        self.val = val
        self.left = left
        self.right = right

# 中序遍历代码
void searchBST(TreeNode* cur) {
    if (cur == NULL) return ;
    searchBST(cur->left);       // 左
    （处理节点）                // 中
    searchBST(cur->right);      // 右
    return ;
}

"""
DFS 深度优先(前中后序遍历) 递归 Recursion
"""
#1 (Easy) 前序遍历-递归-LC144_二叉树的前序遍历
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
# 这第二种方法更好,因为func call func, 只用一个func就够了,不需要再造一个func了
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        if not root:
            return []

        left = self.preorderTraversal(root.left)
        right = self.preorderTraversal(root.right)

        return  [root.val] + left +  right


#2 (Easy) 中序遍历-递归-LC94_二叉树的中序遍历
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


#3 (Easy) 后序遍历-递归-LC145_二叉树的后序遍历
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
DFS 深度优先(前中后序遍历) 迭代 Iteration
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

class Solution:
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        
        st = [root]
        res = []
        while st:
            cur = st.pop()
            if cur != None:
                st.append(cur)
                st.append(None)

                if cur.right:
                    st.append(cur.right)

                if cur.left:
                    st.append(cur.left)
            else:
                cur = st.pop()
                res.append(cur.val)
        return res

"""
*** BFS 广度优先(层序遍历) 递归 Recursion
preorder: 先处理node, 再遍历添加子node
"""
#4 (Medium) 102.二叉树的层序遍历
    # 给你一个二叉树，请你返回其按 层序遍历 得到的节点值。 （即逐层地，从左到右访问所有节点）。
# Input: root = [3,9,20,null,null,15,7]
# Output: [[3],[9,20],[15,7]]
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        res = []
        def helper(root, depth):
            if not root: return []
            # if not root:
            #     return  # Works fine (implicit `None`)
            # # OR
            # if not root:
            #     pass  # Also valid, but less common

            # If the current depth (depth) equals the length of res, 
            # it means we are encountering this depth for the first time, 
            # so we append an empty list to res to start storing nodes at this level.
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
BFS 广度优先(层序遍历) 迭代 Iteration
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
#5 (Easy) 700.二叉搜索树中的搜索
    # 给定二叉搜索树（BST）的根节点和一个值。 你需要在BST中找到节点值等于给定值的节点。 返回以该节点为根的子树。 如果节点不存在，则返回 NULL。
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


#6 (Medium) 107.二叉树的层次遍历 II
    # 给定一个二叉树，返回其节点值自底向上的层次遍历。 （即按从叶子节点所在层到根节点所在的层，逐层从左向右遍历）
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# 迭代
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
        return result[::-1] # 反转结果


#7 (Medium) 199.二叉树的右视图
    # 给定一棵二叉树，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。
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


#8 (Easy) 637.二叉树的层平均值
    # 给定一个非空二叉树, 返回一个由每层节点平均值组成的数组。
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# 迭代
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


#9 (Medium) 429.N叉树的层序遍历
    # 给定一个 N 叉树，返回其节点值的层序遍历。 (即从左到右，逐层遍历)。
    # 例如，给定一个 3叉树 :
    # 429. N叉树的层序遍历
    # 返回其层序遍历:
    # [ [1], [3,2,4], [5,6] ]
# Definition for a Node.
# class Node:
#     def __init__(self, val=None, children=None):
#         self.val = val
#         self.children = children
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


#10 (Medium) 515.在每个树行中找最大值
    # 您需要在二叉树的每一行中找到最大的值。
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


#11 (Medium) 116.填充每个节点的下一个右侧节点指针
    # 给定一个完美二叉树，其所有叶子节点都在同一层，每个父节点都有两个子节点。二叉树定义如下：
    # struct Node {
    #   int val;
    #   Node *left;
    #   Node *right;
    #   Node *next;
    # }
    # 填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 NULL。
    # 初始状态下，所有 next 指针都被设置为 NULL。
# Definition for a Node.
# class Node:
#     def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
#         self.val = val
#         self.left = left
#         self.right = right
#         self.next = next
class Solution:
    def connect(self, root: 'Node') -> 'Node':
        if not root:
            return root
        
        queue = collections.deque([root])
        
        while queue:
            level_size = len(queue)
            prev = None # 记录上一个节点, each level, reset to None
            
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


#12 (Medium) 117.填充每个节点的下一个右侧节点指针II
    # 这道题目说是二叉树，但116题目说是完整二叉树，其实没有任何差别，一样的代码一样的逻辑一样的味道
# 层序遍历解法
# Definition for a Node.
# class Node:
#     def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
#         self.val = val
#         self.left = left
#         self.right = right
#         self.next = next
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


#13 (Easy) 104.二叉树的最大深度
    # 给定一个二叉树，找出其最大深度。
    # 二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。
    # 说明: 叶子节点是指没有子节点的节点。
    # 示例：
    # 给定二叉树 [3,9,20,null,null,15,7]，
    # 返回它的最大深度 3
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


#14 (Easy) 111.二叉树的最小深度
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


#15 (Easy) 226.翻转二叉树
    # 翻转一棵二叉树。
# 这道题目使用前序遍历和后序遍历都可以，唯独中序遍历不方便，因为中序遍历会把某些节点的左右孩子翻转了两次！建议拿纸画一画，就理解了
## *** 递归法：前序遍历
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
            if node.right:
                st.append(node.right) #右
            if node.left:
                st.append(node.left) #左

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
# 迭代法，伪中序遍历（结果是对的，看起来像是中序遍历，实际上它是前序遍历，
# 只不过把中间节点处理逻辑放到了中间。还是要用'统一写法'才是真正的中序遍历）：
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return None      
        stack = [root]        
        while stack:
            node = stack.pop()                   
            if node.right:
                stack.append(node.right)
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
# 迭代法，伪后序遍历（结果是对的，看起来像是后序遍历，实际上它是前序遍历，
# 只不过把中间节点处理逻辑放到了最后。还是要用'统一写法'才是真正的后序遍历）：
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return None      
        stack = [root]        
        while stack:
            node = stack.pop()                   
            if node.right:
                stack.append(node.right)  
            if node.left:
                stack.append(node.left)
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


"""
2. 求二叉树的属性
二叉树：是否对称
递归：后序，比较的是根节点的左子树与右子树是不是相互翻转
迭代：使用队列/栈将两个节点顺序放入容器中进行比较
二叉树：求最大深度
递归：后序，求根节点最大高度就是最大深度，通过递归函数的返回值做计算树的高度
迭代：层序遍历
二叉树：求最小深度
递归：后序，求根节点最小高度就是最小深度，注意最小深度的定义
迭代：层序遍历
二叉树：求有多少个节点
递归：后序，通过递归函数的返回值计算节点数量
迭代：层序遍历
二叉树：是否平衡
递归：后序，注意后序求高度和前序求深度，递归过程判断高度差
迭代：效率很低，不推荐
二叉树：找所有路径
递归：前序，方便让父节点指向子节点，涉及回溯处理根节点到叶子的所有路径
迭代：一个栈模拟递归，一个栈来存放对应的遍历路径
二叉树：递归中如何隐藏着回溯
详解二叉树：找所有路径 中递归如何隐藏着回溯
二叉树：求左叶子之和
递归：后序，必须三层约束条件，才能判断是否是左叶子。
迭代：直接模拟后序遍历
二叉树：求左下角的值
递归：顺序无所谓，优先左孩子搜索，同时找深度最大的叶子节点。
迭代：层序遍历找最后一行最左边
二叉树：求路径总和
递归:顺序无所谓,递归函数返回值为bool类型是为了搜索一条边,没有返回值是搜索整棵树。
迭代：栈里元素不仅要记录节点指针，还要记录从头结点到该节点的路径数值总和
"""
#16 (Easy) 101. 对称二叉树
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
# *** 迭代法: 使用队列
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
    # (跟使用队列没有区别,就是data structure变了,但是因为是放两个node,所以没区别)
# class Solution:
#     def isSymmetric(self, root: TreeNode) -> bool:
#         if not root:
#             return True
#         st = [] #这里改成了栈
#         st.append(root.left)
#         st.append(root.right)
#         while st:
#             rightNode = st.pop()
#             leftNode = st.pop()
#             if not leftNode and not rightNode:
#                 continue
#             if not leftNode or not rightNode or leftNode.val != rightNode.val:
#                 return False
#             st.append(leftNode.left)
#             st.append(rightNode.right)
#             st.append(leftNode.right)
#             st.append(rightNode.left)
#         return True
# 层次遍历
# class Solution:
#     def isSymmetric(self, root: TreeNode) -> bool:
#         if not root:
#             return True
        
#         queue = collections.deque([root.left, root.right])
        
#         while queue:
#             level_size = len(queue)
            
#             if level_size % 2 != 0:
#                 return False
            
#             level_vals = []
#             for i in range(level_size):
#                 node = queue.popleft()
#                 if node:
#                     level_vals.append(node.val)
#                     queue.append(node.left)
#                     queue.append(node.right)
#                 else:
#                     level_vals.append(None)
                    
#             if level_vals != level_vals[::-1]:
#                 return False
            
#         return True


#17 (Easy) 104.二叉树的最大深度
    # 给定一个二叉树, 找出其最大深度。
    # 二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。
    # 说明: 叶子节点是指没有子节点的节点。
    # 示例： 给定二叉树 [3,9,20,null,null,15,7]，
    # 返回它的最大深度 3
# *** 递归法
# 本题可以使用前序（中左右），也可以使用**后序遍历**（左右中），使用前序求的就是深度，使用后序求的是高度。
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
# class solution:
#     def maxdepth(self, root: treenode) -> int:
#         if not root:
#             return 0
#         return 1 + max(self.maxdepth(root.left), self.maxdepth(root.right))
# *** 层序遍历迭代法
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


#18 (Easy) 559.n叉树的最大深度
# 递归法, 后序遍历
class solution:
    def maxdepth(self, root: treenode) -> int:
        return self.getdepth(root)
        
    def getdepth(self, node):
        if not node:
            return 0
        leftdepth = self.getdepth(node.left) #左
        rightdepth = self.getdepth(node.right) #右
        depth = 1 + max(leftdepth, rightdepth) #中
        return depth
# class Solution:
#     def maxDepth(self, root: 'Node') -> int:
#         if not root:
#             return 0
        
#         max_depth = 1
        
#         for child in root.children:
#             max_depth = max(max_depth, self.maxDepth(child) + 1)
        
#         return max_depth
# *** 迭代法
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
# class solution:
#     def maxdepth(self, root: 'node') -> int:
#         if not root:
#             return 0
#         depth = 0
#         for i in range(len(root.children)):
#             depth = max(depth, self.maxdepth(root.children[i]))
#         return depth + 1
# 使用栈
# class Solution:
#     def maxDepth(self, root: 'Node') -> int:
#         if not root:
#             return 0
        
#         max_depth = 0
#         stack = [(root, 1)]
        
#         while stack:
#             node, depth = stack.pop()
#             max_depth = max(max_depth, depth)
#             for child in node.children:
#                 stack.append((child, depth + 1))
#         return max_depth


#19 (Easy) 111.二叉树的最小深度
    # 给定一个二叉树, 找出其最小深度。
    # 最小深度是从根节点到最近叶子节点的最短路径上的节点数量。
# 需要注意的是，只有当左右孩子都为空的时候，才说明遍历到最低点了。如果其中一个孩子不为空则不是最低点
# 递归法（版本一）后序遍历
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
        
        result = 1 + min(leftDepth, rightDepth)
        return result
# *** 递归法（版本二）
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
# class Solution:
#     def __init__(self):
#         self.result = float('inf')

#     def getDepth(self, node, depth):
#         if node is None:
#             return
#         if node.left is None and node.right is None:
#             self.result = min(self.result, depth)
#         if node.left:
#             self.getDepth(node.left, depth + 1)
#         if node.right:
#             self.getDepth(node.right, depth + 1)

#     def minDepth(self, root):
#         if root is None:
#             return 0
#         self.getDepth(root, 1)
#         return self.result
# *** 迭代法
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


#20 (Easy) 222.完全二叉树的节点个数
    # 给出一个完全二叉树，求出该树的节点个数。
    # 示例 1：
    # 输入：root = [1,2,3,4,5,6]
    # 输出：6
    # 示例 2：
    # 输入：root = []
    # 输出：0
    # 示例 3：
    # 输入：root = [1]
    # 输出：1
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
# *** 递归法：精简版
class Solution:
    def countNodes(self, root: TreeNode) -> int:
        if not root:
            return 0
        return self.countNodes(root.left) + self.countNodes(root.right) +1
# 迭代法
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
# class Solution:
#     def countNodes(self, root: TreeNode) -> int:
#         if not root:
#             return 0
#         left = root.left
#         right = root.right
#         leftDepth = 0 #这里初始为0是有目的的，为了下面求指数方便
#         rightDepth = 0
#         while left: #求左子树深度
#             left = left.left
#             leftDepth += 1
#         while right: #求右子树深度
#             right = right.right
#             rightDepth += 1
#         if leftDepth == rightDepth:
#             return (2 << leftDepth) - 1 #注意(2<<1) 相当于2^2，所以leftDepth初始为0
#         return self.countNodes(root.left) + self.countNodes(root.right) + 1
# # 完全二叉树写法2
# class Solution: # 利用完全二叉树特性
#     def countNodes(self, root: TreeNode) -> int:
#         if not root: return 0
#         count = 1
#         left = root.left; right = root.right
#         while left and right:
#             count+=1
#             left = left.left; right = right.right
#         if not left and not right: # 如果同时到底说明是满二叉树，反之则不是
#             return 2**count-1
#         return 1+self.countNodes(root.left)+self.countNodes(root.right)  
# # 完全二叉树写法3
# class Solution: # 利用完全二叉树特性
#     def countNodes(self, root: TreeNode) -> int:
#         if not root: return 0
#         count = 0
#         left = root.left; right = root.right
#         while left and right:
#             count+=1
#             left = left.left; right = right.right
#         if not left and not right: # 如果同时到底说明是满二叉树，反之则不是
#             return (2<<count)-1
#         return 1+self.countNodes(root.left)+self.countNodes(root.right) 


#21 (Easy) 110.平衡二叉树
    # 给定一个二叉树, 判断它是否是高度平衡的二叉树。
    # 本题中，一棵高度平衡二叉树定义为：一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过1。
    # 示例 1:
    # 给定二叉树 [3,9,20,null,null,15,7]
    # 返回 true 。
    # 示例 2:
    # 给定二叉树 [1,2,2,3,3,null,null,4,4]
    # 返回 false 。
# 求深度可以从上到下去查 所以需要**前序遍历**（中左右），而高度只能从下到上去查，所以只能**后序遍历**（左右中）
# 都知道回溯法其实就是递归，但是很少人用迭代的方式去实现回溯算法！
# 因为对于回溯算法已经是非常复杂的递归了，如果再用迭代的话，就是自己给自己找麻烦，效率也并不一定高。
# 讲了这么多二叉树题目的迭代法，有的同学会疑惑，迭代法中究竟什么时候用队列，什么时候用栈？
# 如果是模拟前中后序遍历就用栈，如果是适合层序遍历就用队列，当然还是其他情况，那么就是 先用队列试试行不行，不行就用栈。
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
# 迭代法
# class Solution:
#     def getDepth(self, cur):
#         st = []
#         if cur is not None:
#             st.append(cur)
#         depth = 0
#         result = 0
#         while st:
#             node = st[-1]
#             if node is not None:
#                 st.pop()
#                 st.append(node)                           # 中
#                 st.append(None)
#                 depth += 1
#                 if node.right:
#                     st.append(node.right)                 # 右
#                 if node.left:
#                     st.append(node.left)                   # 左

#             else:               
#                 node = st.pop()
#                 st.pop()
#                 depth -= 1
#             result = max(result, depth)
#         return result

#     def isBalanced(self, root):
#         st = []
#         if root is None:
#             return True
#         st.append(root)
#         while st:
#             node = st.pop()                                 # 中
#             if abs(self.getDepth(node.left) - self.getDepth(node.right)) > 1:
#                 return False
#             if node.right:
#                 st.append(node.right)                       # 右（空节点不入栈）
#             if node.left:
#                 st.append(node.left)                         # 左（空节点不入栈）
#         return True
# # 迭代法精简版
# class Solution:
#     def isBalanced(self, root: Optional[TreeNode]) -> bool:
#         if not root:
#             return True

#         height_map = {}
#         stack = [root]
#         while stack:
#             node = stack.pop()
#             if node:
#                 stack.append(node)
#                 stack.append(None)
#                 if node.left: stack.append(node.left)
#                 if node.right: stack.append(node.right)
#             else:
#                 real_node = stack.pop()
#                 left, right = height_map.get(real_node.left, 0), height_map.get(real_node.right, 0)
#                 if abs(left - right) > 1:
#                     return False
#                 height_map[real_node] = 1 + max(left, right)
#         return True


#22 (Easy) 257.二叉树的所有路径
    # 给定一个二叉树, 返回所有从根节点到叶子节点的路径。
    # 说明: 叶子节点是指没有子节点的节点。
# Definition for a binary tree node.
# 要知道递归和回溯就是一家的，本题也需要回溯。
# 回溯和递归是一一对应的，有一个递归，就要有一个回溯
# 所以回溯要和递归永远在一起，世界上最遥远的距离是你在花括号里，而我在花括号外！
# 回溯就隐藏在traversal(cur->left, path + "->", result);中的 path + "->"。
# 每次函数调用完，path依然是没有加上"->" 的，这就是回溯了。
# 所有路径不需要返回值，因为要记录所有的路径
# 递归法+回溯 (前序遍历)
# class Solution:
#     def binaryTreePaths(self, root):
#         result = []
#         path = []
#         if not root:
#             return result
#         self.traversal(root, path, result)
#         return result

#     def traversal(self, cur, path, result):
#         path.append(cur.val)  # 中
#         if not cur.left and not cur.right:  # 到达叶子节点
#             sPath = '->'.join(map(str, path))
#             result.append(sPath)
#             return
#         if cur.left:  # 左
#             self.traversal(cur.left, path, result)
#             path.pop()  # 回溯
#         if cur.right:  # 右
#             self.traversal(cur.right, path, result)
#             path.pop()  # 回溯
# 递归法+隐形回溯（版本一）
# from typing import List, Optional
# class Solution:
#     def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
#         if not root:
#             return []
#         result = []
#         self.traversal(root, [], result)
#         return result
    
#     def traversal(self, cur: TreeNode, path: List[int], result: List[str]) -> None:
#         if not cur:
#             return
#         path.append(cur.val)
#         if not cur.left and not cur.right:
#             result.append('->'.join(map(str, path)))
#         if cur.left:
#             self.traversal(cur.left, path[:], result)
#         if cur.right:
#             self.traversal(cur.right, path[:], result)
# *** 递归法+隐形回溯（版本二）前序遍历
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
# class Solution:

#     def binaryTreePaths(self, root: TreeNode) -> List[str]:
#         # 题目中节点数至少为1
#         stack, path_st, result = [root], [str(root.val)], []

#         while stack:
#             cur = stack.pop()
#             path = path_st.pop()
#             # 如果当前节点为叶子节点，添加路径到结果中
#             if not (cur.left or cur.right):
#                 result.append(path)
#             if cur.right:
#                 stack.append(cur.right)
#                 path_st.append(path + '->' + str(cur.right.val))
#             if cur.left:
#                 stack.append(cur.left)
#                 path_st.append(path + '->' + str(cur.left.val))

#         return result


#23 ??? (Easy) 404.左叶子之和
    # 计算给定二叉树的所有左叶子之和。
# 因为不能判断本节点是不是左叶子节点。
# 此时就要通过节点的父节点来判断其左孩子是不是左叶子了。
# 递归，后序遍历
# class Solution:
#     def sumOfLeftLeaves(self, root):
#         if root is None:
#             return 0
#         if root.left is None and root.right is None:
#             return 0
        
#         leftValue = self.sumOfLeftLeaves(root.left)  # 左
#         if root.left and not root.left.left and not root.left.right:  # 左子树是左叶子的情况
#             leftValue = root.left.val
            
#         rightValue = self.sumOfLeftLeaves(root.right)  # 右

#         sum_val = leftValue + rightValue  # 中
#         return sum_val
# 递归精简版
class Solution:
    def sumOfLeftLeaves(self, root):
        if root is None:
            return 0
        leftValue = 0
        if root.left is not None and root.left.left is None and root.left.right is None:
            leftValue = root.left.val
        return leftValue + self.sumOfLeftLeaves(root.left) + self.sumOfLeftLeaves(root.right)
# *** 迭代法
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


#24 (Medium) 513.找树左下角的值
    # 给定一个二叉树, 在树的最后一行找到最左边的值。
# （版本一）递归法 + 回溯
# class Solution:
#     def findBottomLeftValue(self, root: TreeNode) -> int:
#         self.max_depth = float('-inf')
#         self.result = None
#         self.traversal(root, 0)
#         return self.result
    
#     def traversal(self, node, depth):
#         if not node.left and not node.right:
#             if depth > self.max_depth:
#                 self.max_depth = depth
#                 self.result = node.val
#             return
        
#         if node.left:
#             depth += 1
#             self.traversal(node.left, depth)
#             depth -= 1
#         if node.right:
#             depth += 1
#             self.traversal(node.right, depth)
#             depth -= 1
# # （版本二）递归法+精简，层序遍历
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
# *** 迭代法
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


#25 (Easy) 112.路径总和
    # 给定一个二叉树和一个目标和, 判断该树中是否存在根节点到叶子节点的路径, 这条路径上所有节点值相加等于目标和。
    # 说明: 叶子节点是指没有子节点的节点。
    # 示例: 给定如下二叉树，以及目标和 sum = 22，
    # 返回 true, 因为存在目标和为 22 的根节点到叶子节点的路径 5->4->11->2。
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
# *** (版本一) 递归
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


"""
二叉树的修改与构造
翻转二叉树
递归：前序，交换左右孩子
迭代：直接模拟前序遍历
构造二叉树
递归：前序，重点在于找分割点，分左右区间构造
迭代：比较复杂，意义不大
构造最大的二叉树
递归：前序，分割点为数组最大值，分左右区间构造
迭代：比较复杂，意义不大
合并两个二叉树
递归：前序，同时操作两个树的节点，注意合并的规则
迭代：使用队列，类似层序遍历
"""
#27 (Medium) 105.从前序与中序遍历序列构造二叉树
    # 根据一棵树的前序遍历与中序遍历构造二叉树。
    # 注意: 你可以假设树中没有重复的元素。
    # 例如，给出
    # 前序遍历 preorder = [3,9,20,15,7] 中序遍历 inorder = [9,3,15,20,7] 返回如下的二叉树：
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
        # 如下代码中我坚持左闭右开的原则
        inorder_left = inorder[:separator_idx]
        inorder_right = inorder[separator_idx + 1:]

        # 第五步: 切割preorder数组. 得到preorder数组的左,右半边.
        # ⭐️ 重点1: 中序数组大小一定跟前序数组大小是相同的.
        # 如下代码中我坚持左闭右开的原则
        preorder_left = preorder[1:1 + len(inorder_left)]
        preorder_right = preorder[1 + len(inorder_left):]

        # 第六步: 递归
        root.left = self.buildTree(preorder_left, inorder_left)
        root.right = self.buildTree(preorder_right, inorder_right)
        # 第七步: 返回答案
        return root


#28 (Medium) 106.从中序与后序遍历序列构造二叉树
    # 根据一棵树的中序遍历与后序遍历构造二叉树。
    # 注意: 你可以假设树中没有重复的元素。
    # 例如，给出
    # 中序遍历 inorder = [9,3,15,20,7]
    # 后序遍历 postorder = [9,15,7,20,3] 返回如下的二叉树：
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
        # 如下代码中我坚持左闭右开的原则
        inorder_left = inorder[:separator_idx]
        inorder_right = inorder[separator_idx + 1:]

        # 第五步: 切割postorder数组. 得到postorder数组的左,右半边.
        # ⭐️ 重点1: 中序数组大小一定跟后序数组大小是相同的.
        # 如下代码中我坚持左闭右开的原则
        postorder_left = postorder[:len(inorder_left)]
        postorder_right = postorder[len(inorder_left): len(postorder) - 1]

        # 第六步: 递归
        root.left = self.buildTree(inorder_left, postorder_left)
        root.right = self.buildTree(inorder_right, postorder_right)
         # 第七步: 返回答案
        return root


#29 (Medium) 654.最大二叉树
    # 给定一个不含重复元素的整数数组。一个以此数组构建的最大二叉树定义如下：
    # 二叉树的根是数组中的最大元素。
    # 左子树是通过数组中最大值左边部分构造出的最大二叉树。
    # 右子树是通过数组中最大值右边部分构造出的最大二叉树。
    # 通过给定的数组构建最大二叉树, 并且输出这个树的根节点。
# 第一版终止条件，是遇到叶子节点就终止，因为空节点不会进入递归。
# 第二版相应的终止条件，是遇到空节点，也就是数组区间为0，就终止了。
# ??? 如何判断是不是要用叶子还是空节点作为判断依据
# 一般情况来说：如果让空节点（空指针）进入递归，就不加if，如果不让空节点进入递归，就加if限制一下， 终止条件也会相应的调整。
# 注意类似用数组构造二叉树的题目，每次分隔尽量不要定义新的数组，而是通过下标索引直接在原数组上操作，这样可以节约时间和空间上的开销。
# (版本一) 基础版
# class Solution:
#     def constructMaximumBinaryTree(self, nums: List[int]) -> TreeNode:
#         if len(nums) == 1:
#             return TreeNode(nums[0])
#         node = TreeNode(0)
#         # 找到数组中最大的值和对应的下标
#         maxValue = 0
#         maxValueIndex = 0
#         for i in range(len(nums)):
#             if nums[i] > maxValue:
#                 maxValue = nums[i]
#                 maxValueIndex = i
#         node.val = maxValue
#         # 最大值所在的下标左区间 构造左子树
#         if maxValueIndex > 0:
#             new_list = nums[:maxValueIndex]
#             node.left = self.constructMaximumBinaryTree(new_list)
#         # 最大值所在的下标右区间 构造右子树
#         if maxValueIndex < len(nums) - 1:
#             new_list = nums[maxValueIndex+1:]
#             node.right = self.constructMaximumBinaryTree(new_list)
#         return node
# (版本二) 使用下标
class Solution:
    def constructMaximumBinaryTree(self, nums: List[int]) -> TreeNode:
        return self.traversal(nums, 0, len(nums))

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
# *** (版本三) 使用切片
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


#30 (Easy) 617.合并二叉树
    # 给定两个二叉树, 想象当你将它们中的一个覆盖到另一个上时, 两个二叉树的一些节点便会重叠。
    # 你需要将他们合并为一个新的二叉树。合并的规则是如果两个节点重叠，那么将他们的值相加作为节点合并后的新值，否则不为 NULL 的节点将直接作为新二叉树的节点。
    # 注意: 合并必须从两个树的根节点开始。
# *** (版本一) 递归 - 前序 - 修改root1
class Solution:
    def mergeTrees(self, root1: TreeNode, root2: TreeNode) -> TreeNode:
        # 递归终止条件: 
        #  但凡有一个节点为空, 就立刻返回另外一个. 如果另外一个也为None就直接返回None. 
        if not root1: 
            return root2
        if not root2: 
            return root1
        # 上面的递归终止条件保证了代码执行到这里root1, root2都非空. 
        root1.val += root2.val # 中
        root1.left = self.mergeTrees(root1.left, root2.left) #左
        root1.right = self.mergeTrees(root1.right, root2.right) # 右
        
        return root1 # ⚠️ 注意: 本题我们重复使用了题目给出的节点而不是创建新节点. 节省时间, 空间. 
# (版本二) 递归 - 前序 - 新建root
# class Solution:
#     def mergeTrees(self, root1: TreeNode, root2: TreeNode) -> TreeNode:
#         # 递归终止条件: 
#         #  但凡有一个节点为空, 就立刻返回另外一个. 如果另外一个也为None就直接返回None. 
#         if not root1: 
#             return root2
#         if not root2: 
#             return root1
#         # 上面的递归终止条件保证了代码执行到这里root1, root2都非空. 
#         root = TreeNode() # 创建新节点
#         root.val += root1.val + root2.val# 中
#         root.left = self.mergeTrees(root1.left, root2.left) #左
#         root.right = self.mergeTrees(root1.right, root2.right) # 右
#         return root # ⚠️ 注意: 本题我们创建了新节点. 
# (版本三) 迭代
# class Solution:
#     def mergeTrees(self, root1: TreeNode, root2: TreeNode) -> TreeNode:
#         if not root1: 
#             return root2
#         if not root2: 
#             return root1

#         queue = deque()
#         queue.append(root1)
#         queue.append(root2)

#         while queue: 
#             node1 = queue.popleft()
#             node2 = queue.popleft()
#             # 更新queue
#             # 只有两个节点都有左节点时, 再往queue里面放.
#             if node1.left and node2.left: 
#                 queue.append(node1.left)
#                 queue.append(node2.left)
#             # 只有两个节点都有右节点时, 再往queue里面放.
#             if node1.right and node2.right: 
#                 queue.append(node1.right)
#                 queue.append(node2.right)

#             # 更新当前节点. 同时改变当前节点的左右孩子. 
#             node1.val += node2.val
#             if not node1.left and node2.left: 
#                 node1.left = node2.left
#             if not node1.right and node2.right: 
#                 node1.right = node2.right

#         return root1
# (版本四) 迭代 + 代码优化
# class Solution:
#     def mergeTrees(self, root1: TreeNode, root2: TreeNode) -> TreeNode:
#         if not root1:
#             return root2
#         if not root2:
#             return root1

#         queue = collections.deque()
#         queue.append((root1, root2))

#         while queue:
#             node1, node2 = queue.popleft()
#             node1.val += node2.val

#             if node1.left and node2.left:
#                 queue.append((node1.left, node2.left))
#             elif not node1.left:
#                 node1.left = node2.left

#             if node1.right and node2.right:
#                 queue.append((node1.right, node2.right))
#             elif not node1.right:
#                 node1.right = node2.right

#         return root1


"""
求二叉搜索树的属性
二叉搜索树中的搜索
递归：二叉搜索树的递归是有方向的
迭代：因为有方向，所以迭代法很简单
是不是二叉搜索树
递归：中序，相当于变成了判断一个序列是不是递增的
迭代：模拟中序，逻辑相同
求二叉搜索树的最小绝对差
递归：中序，双指针操作
迭代：模拟中序，逻辑相同
求二叉搜索树的众数
递归：中序，清空结果集的技巧，遍历一遍便可求众数集合
二叉搜索树转成累加树
递归：中序，双指针操作累加
迭代：模拟中序，逻辑相同
"""
# (Easy) 700.二叉搜索树中的搜索
    # 给定二叉搜索树(BST)的根节点和一个值。 你需要在BST中找到节点值等于给定值的节点。 返回以该节点为根的子树。 如果节点不存在, 则返回 NULL。
# （方法一） 递归
class Solution:
    def searchBST(self, root: TreeNode, val: int) -> TreeNode:
        # 为什么要有返回值: 
        #   因为搜索到目标节点就要立即return，
        #   这样才是找到节点就返回（搜索某一条边），如果不加return，就是遍历整棵树了.
        if not root or root.val == val: 
            return root

        if root.val > val: 
            return self.searchBST(root.left, val)

        if root.val < val: 
            return self.searchBST(root.right, val)
# （方法二）迭代
class Solution:
    def searchBST(self, root: TreeNode, val: int) -> TreeNode:
        while root:
            if val < root.val: root = root.left
            elif val > root.val: root = root.right
            else: return root
        return None


#31 (Medium) 98.验证二叉搜索树
    # 给定一个二叉树, 判断其是否是一个有效的二叉搜索树。
    # 假设一个二叉搜索树具有如下特征：
    # 节点的左子树只包含小于当前节点的数。
    # 节点的右子树只包含大于当前节点的数。
    # 所有左子树和右子树自身必须也是二叉搜索树。
# 迭代-中序遍历
# class Solution:
#     def isValidBST(self, root: TreeNode) -> bool:
#         stack = []
#         cur = root
#         pre = None # 记录前一个节点
#         while cur or stack:
#             if cur: # 指针来访问节点, 访问到最底层
#                 stack.append(cur)
#                 cur = cur.left # 左
#             else: # 逐一处理节点
#                 cur = stack.pop() # 中
#                 if pre and cur.val <= pre.val: # 比较当前节点和前节点的值的大小
#                     return False
#                 pre = cur # 保存前一个访问的结点
#                 cur = cur.right  右
#         return True
# 递归法（版本一）利用中序递增性质，转换成数组
# class Solution:
#     def __init__(self):
#         self.vec = []

#     def traversal(self, root):
#         if root is None:
#             return
#         self.traversal(root.left)
#         self.vec.append(root.val)  # 将二叉搜索树转换为有序数组
#         self.traversal(root.right)

#     def isValidBST(self, root):
#         self.vec = []  # 清空数组
#         self.traversal(root)
#         for i in range(1, len(self.vec)):
#             # 注意要小于等于，搜索树里不能有相同元素
#             if self.vec[i] <= self.vec[i - 1]:
#                 return False
#         return True
# # 递归法（版本二）设定极小值，进行比较
# class Solution:
#     def __init__(self):
#         self.maxVal = float('-inf')  # 因为后台测试数据中有int最小值

#     def isValidBST(self, root):
#         if root is None:
#             return True

#         left = self.isValidBST(root.left)
#         # 中序遍历，验证遍历的元素是不是从小到大
#         if self.maxVal < root.val:
#             self.maxVal = root.val
#         else:
#             return False
#         right = self.isValidBST(root.right)

#         return left and right
# *** 递归法（版本三）直接取该树的最小值，中序遍历
    # 同样本题也需要用pre节点记录cur节点的前一个节点。（这种写法一定要掌握）
class Solution:
    def __init__(self):
        self.pre = None  # 用来记录前一个节点

    def isValidBST(self, root):
        if root is None:
            return True

        left = self.isValidBST(root.left) # 左

        if self.pre is not None and self.pre.val >= root.val: # 中
            return False
        self.pre = root  # 记录前一个节点

        right = self.isValidBST(root.right) # 右
        return left and right


#32 (Easy) 530.二叉搜索树的最小绝对差
    # 给你一棵所有节点为非负值的二叉搜索树, 请你计算树中任意两节点的差的绝对值的最小值。
    # 提示：树中至少有 2 个节点。
# 同时要学会在递归遍历的过程中如何记录前后两个指针，这也是一个小技巧，学会了还是很受用的
# 递归法（版本一）利用中序递增，结合数组
# class Solution:
#     def __init__(self):
#         self.vec = []

#     def traversal(self, root):
#         if root is None:
#             return
#         self.traversal(root.left)
#         self.vec.append(root.val)  # 将二叉搜索树转换为有序数组
#         self.traversal(root.right)

#     def getMinimumDifference(self, root):
#         self.vec = []
#         self.traversal(root)
#         if len(self.vec) < 2:
#             return 0
#         result = float('inf')
#         for i in range(1, len(self.vec)):
#             # 统计有序数组的最小差值
#             result = min(result, self.vec[i] - self.vec[i - 1])
#         return result
# *** 递归法（版本二）利用中序递增，找到该树最小值，中序遍历
class Solution:
    def __init__(self):
        self.result = float('inf')
        self.pre = None

    def getMinimumDifference(self, root):
        self.traversal(root)
        return self.result
    
    def traversal(self, cur):
        if cur is None:
            return
        self.traversal(cur.left)  # 左
        if self.pre is not None:  # 中
            self.result = min(self.result, cur.val - self.pre.val)
        self.pre = cur  # 记录前一个
        self.traversal(cur.right)  # 右
# ??? 迭代法
class Solution:
    def getMinimumDifference(self, root):
        stack = []
        cur = root
        pre = None
        result = float('inf')

        while cur or stack:
            if cur:
                stack.append(cur)  # 将访问的节点放进栈
                cur = cur.left  # 左
            else:
                cur = stack.pop()
                if pre:  # 中
                    result = min(result, cur.val - pre.val)
                pre = cur
                cur = cur.right  # 右

        return result


#33 (Easy) 501.二叉搜索树中的众数
    # 给定一个有相同值的二叉搜索树(BST), 找出 BST 中的所有众数(出现频率最高的元素)。
    # 假定 BST 有如下定义：
    # 结点左子树中所含结点的值小于等于当前结点的值
    # 结点右子树中所含结点的值大于等于当前结点的值
    # 左子树和右子树都是二叉搜索树
    # 例如：
    # 给定 BST [1,null,2,2],
    # 返回[2].
    # 提示：如果众数超过1个，不需考虑输出顺序
    # 进阶：你可以不使用额外的空间吗？（假设由递归产生的隐式调用栈的开销不被计算在内）
# 递归法（版本一）利用字典
# from collections import defaultdict

# class Solution:
#     def searchBST(self, cur, freq_map):
#         if cur is None:
#             return
#         freq_map[cur.val] += 1  # 统计元素频率
#         self.searchBST(cur.left, freq_map)
#         self.searchBST(cur.right, freq_map)

#     def findMode(self, root):
#         freq_map = defaultdict(int)  # key:元素，value:出现频率
#         result = []
#         if root is None:
#             return result
#         self.searchBST(root, freq_map)
#         max_freq = max(freq_map.values())
#         for key, freq in freq_map.items():
#             if freq == max_freq:
#                 result.append(key)
#         return result
# *** 递归法（版本二）利用二叉搜索树性质
class Solution:
    def __init__(self):
        self.maxCount = 0  # 最大频率
        self.count = 0  # 统计频率
        self.pre = None
        self.result = []

    def findMode(self, root):
        self.count = 0
        self.maxCount = 0
        self.pre = None  # 记录前一个节点
        self.result = []

        self.searchBST(root)
        return self.result

    def searchBST(self, cur):
        if cur is None:
            return

        self.searchBST(cur.left)  # 左
        # 中
        if self.pre is None:  # 第一个节点
            self.count = 1
        elif self.pre.val == cur.val:  # 与前一个节点数值相同
            self.count += 1
        else:  # 与前一个节点数值不同
            self.count = 1
        self.pre = cur  # 更新上一个节点

        if self.count == self.maxCount:  # 如果与最大值频率相同，放进result中
            self.result.append(cur.val)

        if self.count > self.maxCount:  # 如果计数大于最大值频率
            self.maxCount = self.count  # 更新最大频率
            self.result = [cur.val]  # 很关键的一步，不要忘记清空result，之前result里的元素都失效了

        self.searchBST(cur.right)  # 右
        return
# 迭代法
    # 二叉搜索树用左中右中序遍历,因为中序遍历能从下到大排列
class Solution:
    def findMode(self, root):
        st = []
        cur = root
        pre = None
        maxCount = 0  # 最大频率
        count = 0  # 统计频率
        result = []

        while cur or st:
            if cur:  # 指针来访问节点，访问到最底层
                st.append(cur)  # 将访问的节点放进栈
                cur = cur.left  # 左

            else: # 中
                cur = st.pop()
                if pre is None:  # 第一个节点
                    count = 1
                elif pre.val == cur.val:  # 与前一个节点数值相同
                    count += 1
                else:  # 与前一个节点数值不同
                    count = 1

                if count == maxCount:  # 如果和最大值相同，放进result中
                    result.append(cur.val)

                if count > maxCount:  # 如果计数大于最大值频率
                    maxCount = count  # 更新最大频率
                    result = [cur.val]  # 很关键的一步，不要忘记清空result，之前result里的元素都失效了
                pre = cur

                cur = cur.right  # 右

        return result


"""
二叉树公共祖先问题
二叉树的公共祖先问题
递归：后序，回溯，找到左子树出现目标值，右子树节点目标值的节点。
迭代：不适合模拟回溯
二叉搜索树的公共祖先问题
递归：顺序无所谓，如果节点的数值在目标区间就是最近公共祖先
迭代：按序遍历
"""
#34 (Medium) 236.二叉树的最近公共祖先
    # 给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
    # 百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q, 最近公共祖先表示为一个结点 x, 满足 x 是 p、q 的祖先且 x 的深度尽可能大(一个节点也可以是它自己的祖先)。”
    # 例如，给定如下二叉树:  root = [3,5,1,6,2,0,8,null,null,7,4]
    # 示例 1: 输入: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1 输出: 3 解释: 节点 5 和节点 1 的最近公共祖先是节点 3。
    # 示例 2: 输入: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4 输出: 5 解释: 节点 5 和节点 4 的最近公共祖先是节点 5。因为根据定义最近公共祖先节点可以为节点本身。
    # 说明:
    # 所有节点的值都是唯一的。
    # p、q 为不同节点且均存在于给定的二叉树中。
搜索一条边的写法:
if (递归函数(root->left)) return ;
if (递归函数(root->right)) return ;

搜索整个树写法:
left = 递归函数(root->left);  // 左
right = 递归函数(root->right); // 右
left与right的逻辑处理;         // 中 

# class Solution:
#     """二叉树的最近公共祖先 递归法"""

#     def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
#         if not root or root == p or root == q:
#             return root
        
#         left = self.lowestCommonAncestor(root.left, p, q)
#         right = self.lowestCommonAncestor(root.right, p, q)
        
#         if left and right:
#             return root
#         if left:
#             return left
#         return right
# 递归法（版本一）
class Solution:
    def lowestCommonAncestor(self, root, p, q):
        if root == q or root == p or root is None:
            return root

        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)

        if left and right:
            return root

        if not left and right:
            return right
        elif left and not right:
            return left
        else: 
            return None
# *** 递归法（版本二）精简
class Solution:
    def lowestCommonAncestor(self, root, p, q):
        if root == q or root == p or root is None:
            return root

        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)

        if left and right:
            return root

        if not left:
            return right

        return left


#35 (Medium) 235.二叉搜索树的最近公共祖先
    # 给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。
    # 百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q, 最近公共祖先表示为一个结点 x, 满足 x 是 p、q 的祖先且 x 的深度尽可能大(一个节点也可以是它自己的祖先)。”
    # 例如，给定如下二叉搜索树:  root = [6,2,8,0,4,7,9,null,null,3,5]
    # 示例 1:
    # 输入: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
    # 输出: 6
    # 解释: 节点 2 和节点 8 的最近公共祖先是 6。
    # 示例 2:
    # 输入: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4
    # 输出: 2
    # 解释: 节点 2 和节点 4 的最近公共祖先是 2, 因为根据定义最近公共祖先节点可以为节点本身。
    # 说明:
    # 所有节点的值都是唯一的。
    # p、q 为不同节点且均存在于给定的二叉搜索树中。
# 递归法（版本一）
# class Solution:
#     def traversal(self, cur, p, q):
#         if cur is None:
#             return cur
#                                                         # 中
#         if cur.val > p.val and cur.val > q.val:           # 左
#             left = self.traversal(cur.left, p, q)
#             if left is not None:
#                 return left

#         if cur.val < p.val and cur.val < q.val:           # 右
#             right = self.traversal(cur.right, p, q)
#             if right is not None:
#                 return right

#         return cur

#     def lowestCommonAncestor(self, root, p, q):
#         return self.traversal(root, p, q)
# *** 迭代法（版本二）精简
class Solution:
    def lowestCommonAncestor(self, root, p, q):
        if root.val > p.val and root.val > q.val:
            return self.lowestCommonAncestor(root.left, p, q)
        elif root.val < p.val and root.val < q.val:
            return self.lowestCommonAncestor(root.right, p, q)
        else:
            return root
# *** 迭代法
class Solution:
    def lowestCommonAncestor(self, root, p, q):
        while root:
            if root.val > p.val and root.val > q.val:
                root = root.left
            elif root.val < p.val and root.val < q.val:
                root = root.right
            else:
                return root
        return None


"""
二叉搜索树的修改与构造
二叉搜索树中的插入操作
递归：顺序无所谓，通过递归函数返回值添加节点
迭代：按序遍历，需要记录插入父节点，这样才能做插入操作
二叉搜索树中的删除操作
递归：前序，想清楚删除非叶子节点的情况
迭代：有序遍历，较复杂
修剪二叉搜索树
递归：前序，通过递归函数返回值删除节点
迭代：有序遍历，较复杂
构造二叉搜索树
递归：前序，数组中间节点分割
迭代：较复杂，通过三个队列来模拟
"""
#36 (Medium) 701.二叉搜索树中的插入操作
    # 给定二叉搜索树(BST)的根节点和要插入树中的值, 将值插入二叉搜索树。 返回插入后二叉搜索树的根节点。 输入数据保证, 新值和原始二叉搜索树中的任意节点值都不同。
    # 注意，可能存在多种有效的插入方式，只要树在插入后仍保持为二叉搜索树即可。 你可以返回任意有效的结果。
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# class Solution:
#     def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
#         # 返回更新后的以当前root为根节点的新树, 方便用于更新上一层的父子节点关系链

#         # Base Case
#         if not root: return TreeNode(val)

#         # 单层递归逻辑:
#         if val < root.val: 
#             # 将val插入至当前root的左子树中合适的位置
#             # 并更新当前root的左子树为包含目标val的新左子树
#             root.left = self.insertIntoBST(root.left, val)

#         if root.val < val:
#             # 将val插入至当前root的右子树中合适的位置
#             # 并更新当前root的右子树为包含目标val的新右子树
#             root.right = self.insertIntoBST(root.right, val)

#         # 返回更新后的以当前root为根节点的新树
#         return root
# 递归法（版本一）
# class Solution:
#     def __init__(self):
#         self.parent = None

#     def traversal(self, cur, val):
#         if cur is None:
#             node = TreeNode(val)
#             if val > self.parent.val:
#                 self.parent.right = node
#             else:
#                 self.parent.left = node
#             return

#         self.parent = cur
#         if cur.val > val:
#             self.traversal(cur.left, val)
#         if cur.val < val:
#             self.traversal(cur.right, val)

#     def insertIntoBST(self, root, val):
#         self.parent = TreeNode(0)
#         if root is None:
#             return TreeNode(val)
#         self.traversal(root, val)
#         return root
# # 递归法（版本三）
# class Solution:
#     def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
#         if root is None or root.val == val:
#             return TreeNode(val)
#         elif root.val > val:
#             if root.left is None:
#                 root.left = TreeNode(val)
#             else:
#                 self.insertIntoBST(root.left, val)
#         elif root.val < val:
#             if root.right is None:
#                 root.right = TreeNode(val)
#             else:
#                 self.insertIntoBST(root.right, val)
#         return root
# *** 递归法（版本四）
    # 通过递归函数返回值完成了新加入节点的父子关系赋值操作了，下一层将加入节点返回，本层用root->left或者root->right将其接住
class Solution:
    def insertIntoBST(self, root, val):
        if root is None:
            node = TreeNode(val)
            return node

        if root.val > val:
            root.left = self.insertIntoBST(root.left, val)
        if root.val < val:
            root.right = self.insertIntoBST(root.right, val)

        return root
# *** 迭代法
class Solution:
    def insertIntoBST(self, root, val):
        if root is None:  # 如果根节点为空，创建新节点作为根节点并返回
            return TreeNode(val)

        parent = None  # 记录上一个节点，用于连接新节点
        cur = root
        while cur:
            parent = cur
            if val < cur.val:
                cur = cur.left
            else:
                cur = cur.right

        if val < parent.val:
            parent.left = TreeNode(val)  # 将新节点连接到父节点的左子树
        else:
            parent.right = TreeNode(val)  # 将新节点连接到父节点的右子树

        return root


#37 (Medium) 450.删除二叉搜索树中的节点
    # 给定一个二叉搜索树的根节点 root 和一个值 key, 删除二叉搜索树中的 key 对应的节点, 并保证二叉搜索树的性质不变。返回二叉搜索树(有可能被更新)的根节点的引用。
    # 一般来说，删除节点可分为两个步骤：
    # 首先找到需要删除的节点； 如果找到了，删除它。 说明： 要求算法时间复杂度为 $O(h)$，h 为树的高度。
# class Solution:
#     def deleteNode(self, root: TreeNode, key: int) -> TreeNode:
#         if not root: return root  #第一种情况：没找到删除的节点, 遍历到空节点直接返回了
#         if root.val == key:  
#             if not root.left and not root.right:  #第二种情况：左右孩子都为空(叶子节点), 直接删除节点,  返回NULL为根节点
#                 del root
#                 return None
#             if not root.left and root.right:  #第三种情况：其左孩子为空, 右孩子不为空, 删除节点, 右孩子补位 , 返回右孩子为根节点
#                 tmp = root
#                 root = root.right
#                 del tmp
#                 return root
#             if root.left and not root.right:  #第四种情况：其右孩子为空, 左孩子不为空, 删除节点, 左孩子补位, 返回左孩子为根节点
#                 tmp = root
#                 root = root.left
#                 del tmp
#                 return root
#             else:  #第五种情况：左右孩子节点都不为空, 则将删除节点的左子树放到删除节点的右子树的最左面节点的左孩子的位置
#                 v = root.right
#                 while v.left:
#                     v = v.left
#                 v.left = root.left
#                 tmp = root
#                 root = root.right
#                 del tmp
#                 return root
#         if root.val > key: root.left = self.deleteNode(root.left,key)  #左递归
#         if root.val < key: root.right = self.deleteNode(root.right,key)  #右递归
#         return root
# *** 递归法（版本一）
class Solution:
    def deleteNode(self, root, key):
        if root is None:
            return root
        if root.val == key:
            if not root.left and not root.right:
                return None
            elif not root.left:
                return root.right
            elif not root.right:
                return root.left
            else:
                cur = root.right
                while cur.left:
                    cur = cur.left
                cur.left = root.left
                return root.right

        if root.val > key:
            root.left = self.deleteNode(root.left, key)
        if root.val < key:
            root.right = self.deleteNode(root.right, key)
        return root
# 递归法（版本二）
# class Solution:
#     def deleteNode(self, root, key):
#         if root is None:  # 如果根节点为空，直接返回
#             return root
#         if root.val == key:  # 找到要删除的节点
#             if root.right is None:  # 如果右子树为空，直接返回左子树作为新的根节点
#                 return root.left
#             cur = root.right
#             while cur.left:  # 找到右子树中的最左节点
#                 cur = cur.left
#             root.val, cur.val = cur.val, root.val  # 将要删除的节点值与最左节点值交换
#         root.left = self.deleteNode(root.left, key)  # 在左子树中递归删除目标节点
#         root.right = self.deleteNode(root.right, key)  # 在右子树中递归删除目标节点
#         return root
# # 迭代法
# class Solution:
#     def deleteOneNode(self, target: TreeNode) -> TreeNode:
#         """
#         将目标节点（删除节点）的左子树放到目标节点的右子树的最左面节点的左孩子位置上
#         并返回目标节点右孩子为新的根节点
#         是动画里模拟的过程
#         """
#         if target is None:
#             return target
#         if target.right is None:
#             return target.left
#         cur = target.right
#         while cur.left:
#             cur = cur.left
#         cur.left = target.left
#         return target.right

#     def deleteNode(self, root: TreeNode, key: int) -> TreeNode:
#         if root is None:
#             return root
#         cur = root
#         pre = None  # 记录cur的父节点，用来删除cur
#         while cur:
#             if cur.val == key:
#                 break
#             pre = cur
#             if cur.val > key:
#                 cur = cur.left
#             else:
#                 cur = cur.right
#         if pre is None:  # 如果搜索树只有头结点
#             return self.deleteOneNode(cur)
#         # pre 要知道是删左孩子还是右孩子
#         if pre.left and pre.left.val == key:
#             pre.left = self.deleteOneNode(cur)
#         if pre.right and pre.right.val == key:
#             pre.right = self.deleteOneNode(cur)
#         return root


#38 (Medium) 669.修剪二叉搜索树
    # 给定一个二叉搜索树, 同时给定最小边界L 和最大边界 R。通过修剪二叉搜索树, 使得所有节点的值在[L, R]中 (R>=L) 。你可能需要改变树的根节点, 所以结果应当返回修剪好的二叉搜索树的新的根节点。
# *** 递归法（版本一）
class Solution:
    def trimBST(self, root: TreeNode, low: int, high: int) -> TreeNode:
        if root is None:
            return None
        if root.val < low:
            # 寻找符合区间 [low, high] 的节点
            return self.trimBST(root.right, low, high)
        if root.val > high:
            # 寻找符合区间 [low, high] 的节点
            return self.trimBST(root.left, low, high)
        root.left = self.trimBST(root.left, low, high)  # root.left 接入符合条件的左孩子
        root.right = self.trimBST(root.right, low, high)  # root.right 接入符合条件的右孩子
        return root
# 迭代法
# class Solution:
#     def trimBST(self, root: TreeNode, L: int, R: int) -> TreeNode:
#         if not root:
#             return None
        
#         # 处理头结点，让root移动到[L, R] 范围内，注意是左闭右闭
#         while root and (root.val < L or root.val > R):
#             if root.val < L:
#                 root = root.right  # 小于L往右走
#             else:
#                 root = root.left  # 大于R往左走
        
#         cur = root
        
#         # 此时root已经在[L, R] 范围内，处理左孩子元素小于L的情况
#         while cur:
#             while cur.left and cur.left.val < L:
#                 cur.left = cur.left.right
#             cur = cur.left
        
#         cur = root
        
#         # 此时root已经在[L, R] 范围内，处理右孩子大于R的情况
#         while cur:
#             while cur.right and cur.right.val > R:
#                 cur.right = cur.right.left
#             cur = cur.right
        
#         return root


#39 (Easy) 108.将有序数组转换为二叉搜索树
    # 将一个按照升序排列的有序数组, 转换为一棵高度平衡二叉搜索树。
    # 本题中，一个高度平衡二叉树是指一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1。
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# class Solution:
#     def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
#         '''
#         构造二叉树：重点是选取数组最中间元素为分割点, 左侧是递归左区间;右侧是递归右区间
#         必然是平衡树
#         左闭右闭区间
#         '''
#         # 返回根节点
#         root = self.traversal(nums, 0, len(nums)-1)
#         return root

#     def traversal(self, nums: List[int], left: int, right: int) -> TreeNode:
#         # Base Case
#         if left > right:
#             return None
        
#         # 确定左右界的中心, 防越界
#         mid = left + (right - left) // 2
#         # 构建根节点
#         mid_root = TreeNode(nums[mid])
#         # 构建以左右界的中心为分割点的左右子树
#         mid_root.left = self.traversal(nums, left, mid-1)
#         mid_root.right = self.traversal(nums, mid+1, right)

#         # 返回由被传入的左右界定义的某子树的根节点
#         return mid_root
# *** 递归法
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        root = self.traversal(nums, 0, len(nums) - 1)
        return root
    
    def traversal(self, nums: List[int], left: int, right: int) -> TreeNode:
        if left > right:
            return None
        
        mid = left + (right - left) // 2
        root = TreeNode(nums[mid])
        root.left = self.traversal(nums, left, mid - 1)
        root.right = self.traversal(nums, mid + 1, right)
        return root
# 迭代法
# from collections import deque
# class Solution:
#     def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
#         if len(nums) == 0:
#             return None
        
#         root = TreeNode(0)  # 初始根节点
#         nodeQue = deque()   # 放遍历的节点
#         leftQue = deque()   # 保存左区间下标
#         rightQue = deque()  # 保存右区间下标
        
#         nodeQue.append(root)               # 根节点入队列
#         leftQue.append(0)                  # 0为左区间下标初始位置
#         rightQue.append(len(nums) - 1)     # len(nums) - 1为右区间下标初始位置

#         while nodeQue:
#             curNode = nodeQue.popleft()
#             left = leftQue.popleft()
#             right = rightQue.popleft()
#             mid = left + (right - left) // 2

#             curNode.val = nums[mid]  # 将mid对应的元素给中间节点

#             if left <= mid - 1:  # 处理左区间
#                 curNode.left = TreeNode(0)
#                 nodeQue.append(curNode.left)
#                 leftQue.append(left)
#                 rightQue.append(mid - 1)

#             if right >= mid + 1:  # 处理右区间
#                 curNode.right = TreeNode(0)
#                 nodeQue.append(curNode.right)
#                 leftQue.append(mid + 1)
#                 rightQue.append(right)
#         return root


#40 (Medium) 538.把二叉搜索树转换为累加树
    # 给出二叉 搜索 树的根节点, 该树的节点值各不相同, 请你将其转换为累加树(Greater Sum Tree), 使每个节点 node 的新值等于原树中大于或等于 node.val 的值之和。
    # 提醒一下, 二叉搜索树满足下列约束条件：
    # 节点的左子树仅包含键 小于 节点键的节点。 节点的右子树仅包含键 大于 节点键的节点。 左右子树也必须是二叉搜索树。
    # 示例 1：
    # 输入：[4,1,6,0,2,5,7,null,null,null,3,null,null,null,8]
    # 输出：[30,36,21,36,35,26,15,null,null,null,33,null,null,null,8]
    # 示例 2：
    # 输入：root = [0,null,1]
    # 输出：[1,null,1]
    # 示例 3：
    # 输入：root = [1,0,2]
    # 输出：[3,3,2]
    # 示例 4：
    # 输入：root = [3,2,4,1]
    # 输出：[7,9,4,10]
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# class Solution:
#     def __init__(self):
#         self.pre = TreeNode()

#     def convertBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
#         '''
#         倒序累加替换：  
#         [2, 5, 13] -> [[2]+[1]+[0], [2]+[1], [2]] -> [20, 18, 13]
#         '''
#         self.traversal(root)
#         return root

#     def traversal(self, root: TreeNode) -> None:
#         # 因为要遍历整棵树, 所以递归函数不需要返回值
#         # Base Case
#         if not root: 
#             return None
#         # 单层递归逻辑：中序遍历的反译 - 右中左
#         self.traversal(root.right)  # 右

#         # 中节点：用当前root的值加上pre的值
#         root.val += self.pre.val    # 中
#         self.pre = root             

#         self.traversal(root.left)   # 左
# *** 递归法(版本一)
class Solution:
    def convertBST(self, root: TreeNode) -> TreeNode:
        self.pre = 0  # 记录前一个节点的数值
        self.traversal(root)
        return root
    def traversal(self, cur):
        if not cur:
            return        
        self.traversal(cur.right)
        cur.val += self.pre
        self.pre = cur.val
        self.traversal(cur.left)
# 递归法（版本二）
# class Solution:
#     def __init__(self):
#         self.count = 0

#     def convertBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
#         if root == None:
#             return 
#         '''
#         倒序累加替换：  
#         '''
#         # 右
#         self.convertBST(root.right)

#         # 中
#         # 中节点：用当前root的值加上pre的值
#         self.count += root.val

#         root.val = self.count 

#         # 左
#         self.convertBST(root.left)

#         return root
# 迭代法（版本一）
# class Solution:
#     def __init__(self):
#         self.pre = 0  # 记录前一个节点的数值
    
#     def traversal(self, root):
#         stack = []
#         cur = root
#         while cur or stack:
#             if cur:
#                 stack.append(cur)
#                 cur = cur.right  # 右
#             else:
#                 cur = stack.pop()  # 中
#                 cur.val += self.pre
#                 self.pre = cur.val
#                 cur = cur.left  # 左
    
#     def convertBST(self, root):
#         self.pre = 0
#         self.traversal(root)
#         return root
# 迭代法（版本二）
class Solution:
    def convertBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root: return root
        stack = []
        cur = root
        pre = 0
        while cur or stack:
            if cur:
                stack.append(cur)
                cur = cur.right
            else: 
                cur = stack.pop()
                cur.val+= pre
                pre = cur.val
                cur =cur.left
        return root


#X41 (Easy) Invert Binary Tree
    # Invert a binary tree and return its root. When a binary tree is inverted, it becomes the mirror image of itself.
from ds import TreeNode
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
"""
def invert_binary_tree_iterative(root: TreeNode) -> TreeNode:
    if not root:
        return None
    stack = [root]
    while stack:
        node = stack.pop()
        # Swap the left and right subtrees of the current node.
        node.left, node.right = node.right, node.left
        # Push the left and right subtrees onto the stack.
        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)
    return root

def invert_binary_tree_recursive(root: TreeNode) -> TreeNode:
    # Base case: If the node is null, there's nothing to invert.
    if not root:
        return None
    # Swap the left and right subtrees of the current node.
    root.left, root.right = root.right, root.left
    # Recursively invert the left and right subtrees.
    invert_binary_tree_recursive(root.left)
    invert_binary_tree_recursive(root.right)
    return root


#X42 (Easy) Balanced Binary Tree Validation
    # Determine if a binary tree is height-balanced, meaning no node's left 
    # subtree and right subtree have a height difference greater than 1.
def balanced_binary_tree_validation(root: TreeNode) -> bool:
    return get_height_imbalance(root) != -1

def get_height_imbalance(node: TreeNode) -> int:
    # Base case: if the node is null, its height is 0.
    if not node:
        return 0
    # Recursively get the height of the left and right subtrees. If
    # either subtree is imbalanced, propagate -1 up the tree.
    left_height = get_height_imbalance(node.left)
    right_height = get_height_imbalance(node.right)
    if left_height == -1 or right_height == -1:
        return -1
    # If the current node's subtree is imbalanced
    # (height difference > 1), return -1.
    if abs(left_height - right_height) > 1:
        return -1
    # Return the height of the current subtree.
    return 1 + max(left_height, right_height)


#X43 (Medium) Rightmost Nodes of a Binary Tree
    # Return an array containing the values of the rightmost nodes at each level of a binary tree.
from ds import TreeNode
from typing import List
from collections import deque
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
"""
def rightmost_nodes_of_a_binary_tree(root: TreeNode) -> List[int]:
    if not root:
        return []
    res = []
    queue = deque([root])
    while queue:
        level_size = len(queue)
        # Add all the non-null child nodes of the current level to the 
        # queue.
        for i in range(level_size):
            node = queue.popleft()
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
            # Record this level's last node to the result array.
            if i == level_size - 1:
                res.append(node.val)
    return res


#X44 (Medium) Widest Binary Tree Level
    # Return the width of the widest level in a binary tree, where the width of a level 
    # is defined as the distance between its leftmost and rightmost non-null nodes.
from collections import deque
from ds import TreeNode
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
"""
def widest_binary_tree_level(root: TreeNode) -> int:
    if not root:
        return 0
    max_width = 0
    queue = deque([(root, 0)])  # Stores (node, index) pairs.
    while queue:
        level_size = len(queue)
        # Set the 'leftmost_index' to the index of the first node in 
        # this level. Start 'rightmost_index' at the same point as 
        # 'leftmost_index' and update it as we traverse the level, 
        # eventually positioning it at the last node.
        leftmost_index = queue[0][1]
        rightmost_index = leftmost_index
        # Process all nodes at the current level.
        for _ in range(level_size):
            node, i = queue.popleft()
            if node.left:
                queue.append((node.left, 2*i + 1))
            if node.right:
                queue.append((node.right, 2*i + 2))
            rightmost_index = i
        max_width = max(max_width, rightmost_index - leftmost_index + 1)
    return max_width


#X45 (Medium) Binary Search Tree Validation
    # Verify whether a binary tree is a valid binary search tree (BST). 
    # A BST is a binary tree where each node meets the following criteria:
    # A node's left subtree contains only nodes of lower values than the node's value.
    # A node's right subtree contains only nodes of greater values than the node's value.
from ds import TreeNode
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
"""
def binary_search_tree_validation(root: TreeNode) -> bool:
    # Start validation at the root node. The root node can contain any 
    # value, so set the initial lower and upper bounds to -infinity and 
    # +infinity, respectively.
    return is_within_bounds(root, float('-inf'), float('inf'))

def is_within_bounds(node: TreeNode,
                     lower_bound: int, upper_bound: int) -> bool:
    # Base case: if the node is null, it satisfies the BST condition.
    if not node:
        return True
    # If the current node's value is not within the valid bounds, this 
    # tree is not a valid BST.
    if not lower_bound < node.val < upper_bound:
        return False
    # If the left subtree isn't a BST, this tree isn't a BST.
    if not is_within_bounds(node.left, lower_bound, node.val):
        return False
    # Otherwise, return true if the right subtree is also a BST.
    return is_within_bounds(node.right, node.val, upper_bound)


#X46 (Medium) Lowest Common Ancestor
    # Return the lowest common ancestor (LCA) of two nodes, p and q, in a binary tree. 
    # The LCA is defined as the lowest node that has both p and q as descendants. 
    # A node can be considered an ancestor of itself.
from ds import TreeNode
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
"""
def lowest_common_ancestor(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    dfs(root, p, q)
    return lca

def dfs(node: TreeNode, p: TreeNode, q: TreeNode) -> bool:
    global lca
    # Base case: a null node is neither 'p' nor 'q'.
    if not node:
        return False
    node_is_p_or_q = node == p or node == q
    # Recursively determine if the left and right subtrees contain 'p' 
    # or 'q'.
    left_contains_p_or_q = dfs(node.left, p, q)
    right_contains_p_or_q = dfs(node.right, p, q)
    # If two of the above three variables are true, the current node is 
    # the LCA.
    if (node_is_p_or_q + left_contains_p_or_q + right_contains_p_or_q == 2):
        lca = node
    # Return true if the current subtree contains 'p' or 'q'.
    return (node_is_p_or_q or left_contains_p_or_q or right_contains_p_or_q)


#X47 (Medium) Build Binary Tree From Preorder and Inorder Traversals
    # Construct a binary tree using arrays of values obtained after a preorder 
    # traversal and an inorder traversal of the tree.
from ds import TreeNode
from typing import List
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
"""
preorder_index = 0
inorder_indexes_map = {}

def build_binary_tree(preorder: List[int], inorder: List[int]) -> TreeNode:
    global inorder_indexes_map
    # Populate the hash map with the inorder values and their indexes.
    for i, val in enumerate(inorder):
        inorder_indexes_map[val] = i
    # Build the tree and return its root node.
    return build_subtree(0, len(inorder) - 1, preorder, inorder)

def build_subtree(left: int, right: int, preorder: List[int], inorder: List[int]) -> TreeNode:
    global preorder_index, inorder_indexes_map
    # Base case: if no elements are in this range, return None.
    if left > right:
        return None
    val = preorder[preorder_index]
    # Set 'inorder_index' to the index of the same value pointed at by
    # 'preorder_index'.
    inorder_index = inorder_indexes_map[val]
    node = TreeNode(val)
    # Advance 'preorder_index' so it points to the value of the next
    # node to be created.
    preorder_index += 1
    # Build the left and right subtrees and connect them to the current
    # node.
    node.left = build_subtree(
        left, inorder_index - 1, preorder, inorder
    )
    node.right = build_subtree(
        inorder_index + 1, right, preorder, inorder
    )
    return node


#X48 (Hard) Maximum Sum of a Continuous Path in a Binary Tree
    # Return the maximum sum of a continuous path in a binary tree. A path is defined by the following characteristics:
    # Consists of a sequence of nodes that can begin and end at any node in the tree
    # Each consecutive pair of nodes in the sequence is connected by an edge
    # The path must be a single continuous sequence of nodes that doesn't split into multiple paths
from ds import TreeNode
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
"""
max_sum = float('-inf')

def max_path_sum(root: TreeNode) -> int:
    global max_sum
    max_path_sum_helper(root)
    return max_sum

def max_path_sum_helper(node: TreeNode) -> int:
    global max_sum
    # Base case: null nodes have no path sum.
    if not node:
        return 0
    # Collect the maximum gain we can attain from the left and right 
    # subtrees, setting them to 0 if they're negative.
    left_sum = max(max_path_sum_helper(node.left), 0)
    right_sum = max(max_path_sum_helper(node.right), 0)
    # Update the overall maximum path sum if the current path sum is 
    # larger.
    max_sum = max(max_sum, node.val + left_sum + right_sum)
    # Return the maximum sum of a single, continuous path with the 
    # current node as an endpoint.
    return node.val + max(left_sum, right_sum)


#X49 (Medium) Binary Tree Symmetry
    # Determine if a binary tree is vertically symmetric. That is, 
    # the left subtree of the root node is a mirror of the right subtree.
from ds import TreeNode
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
"""
def binary_tree_symmetry(root: TreeNode) -> bool:
    if not root:
        return True  
    return compare_trees(root.left, root.right)


def compare_trees(node1: TreeNode, node2: TreeNode) -> bool:
    # Base case: if both nodes are null, they're symmetric.
    if not node1 and not node2:
        return True
    # If one node is null and the other isn't, they aren't symmetric.
    if not node1 or not node2:
        return False
    # If the values of the current nodes don't match, trees aren't symmetric.
    if node1.val != node2.val:
        return False
    # Compare the 'node1's left subtree with 'node2's right subtree. If these 
    # aren't symmetric, the whole tree is not symmetric.
    if not compare_trees(node1.left, node2.right):
        return False
    # Compare the 'node1's right subtree with 'node2's left subtree. 
    return compare_trees(node1.right, node2.left)


#X50 (Medium) Binary Tree Columns
    # Given the root of a binary tree, return a list of arrays where each array 
    # represents a vertical column of the tree. Nodes in the same column should 
    # be ordered from top to bottom. Nodes in the same row and column should be ordered from left to right.
from collections import defaultdict, deque
from ds import TreeNode
from typing import List
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
"""
def binary_tree_columns(root: TreeNode) -> List[List[int]]:
    if not root:
        return []
    column_map = defaultdict(list)
    leftmost_column = rightmost_column = 0
    queue = deque([(root, 0)])
    while queue:
        node, column = queue.popleft()
        if node:
            # Add the current node's value to its corresponding list in the hash 
            # map.
            column_map[column].append(node.val)
            leftmost_column = min(leftmost_column, column)
            rightmost_column = max(rightmost_column, column)
            # Add the current node's children to the queue with their respective
            # column ids.
            queue.append((node.left, column - 1))
            queue.append((node.right, column + 1))
    # Construct the output list by collecting values from each column in the hash   
    # map in the correct order.
    return [column_map[i] for i in range(leftmost_column, rightmost_column + 1)]


#X51 (Medium) Kth Smallest Number in a Binary Search Tree
    # Given the root of a binary search tree (BST) and an integer k, find the kth smallest node value.
from ds import TreeNode
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
"""
def kth_smallest_number_in_BST_iterative(root: TreeNode, k: int) -> int:
    stack = []
    node = root
    while stack or node:
        # Move to the leftmost node and add nodes to the stack as we go so they 
        # can be processed in future iterations.
        while node:
            stack.append(node)
            node = node.left
        # Pop the top node from the stack to process it, and decrement 'k'.
        node = stack.pop()
        k -= 1
        # If we have processed 'k' nodes, return the value of the 'k'th smallest 
        # node.
        if k == 0:
            return node.val
        # Move to the right subtree.
        node = node.right

def kth_smallest_number_in_BST_recursive(root: TreeNode, k: int) -> int:
    sorted_list = inorder(root)
    return sorted_list[k - 1]

# Inorder traversal function to attain a sorted list of nodes from the BST.
def inorder(node: TreeNode) -> List[int]:
    if not node:
        return []
    return inorder(node.left) + [node.val] + inorder(node.right)


#X52 (Medium) Serialize and Deserialize a Binary Tree
    # Write a function to serialize a binary tree into a string, and another 
    # function to deserialize that string back into the original binary tree structure.
from ds import TreeNode
from typing import List
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
"""
def serialize(root: TreeNode) -> str:
    # Perform a preorder traversal to add node values to a list, then convert the
    # list to a string.
    serialized_list = []
    preorder_serialize(root, serialized_list)
    # Convert the list to a string and separate each value using a comma 
    # delimiter.
    return ','.join(serialized_list)

# Helper function to perform serialization through preorder traversal.
def preorder_serialize(node, serialized_list) -> None:
    # Base case: mark null nodes as '#'.
    if node is None:
        serialized_list.append('#')
        return
    # Preorder traversal processes the current node first, then the left and right
    # children.
    serialized_list.append(str(node.val))
    preorder_serialize(node.left, serialized_list)
    preorder_serialize(node.right, serialized_list)

def deserialize(data: str) -> TreeNode:
    # Obtain the node values by splitting the string using the comma delimiter.
    node_values = iter(data.split(','))
    return build_tree(node_values)

# Helper function to construct the tree using preorder traversal.
def build_tree(values: List[str]) -> TreeNode:
    val = next(values)
    # Base case: '#' indicates a null node.
    if val == '#':
        return None
    # Use preorder traversal processes the current node first, then the left and 
    # right children.
    node = TreeNode(int(val))
    node.left = build_tree(values)
    node.right = build_tree(values)
    return node


# Binary Search
#X53 (Easy) Find the Insertion Index
    # You are given a sorted array that contains unique values, along with an integer target.
    # If the array contains the target value, return its index.
    # Otherwise, return the insertion index. This is the index where the target would 
    # be if it were inserted in order, maintaining the sorted sequence of the array.
from typing import List

def find_the_insertion_index(nums: List[int], target: int) -> int:
    left, right = 0, len(nums)
    while left < right:
        mid = (left + right) // 2
        # If the midpoint value is greater than or equal to the target, 
        # the lower bound is either at the midpoint, or to its left.
        if nums[mid] >= target:
            right = mid
        # The midpoint value is less than the target, indicating the 
        # lower bound is somewhere to the right.
        else:
            left = mid + 1
    return left


#X54 (Medium) First and Last Occurrences of a Number
    # Given an array of integers sorted in non-decreasing order, return 
    # the first and last indexes of a target number. If the target is not found, return [-1, -1] .
    # Example 1:
    # Input: nums = [1, 2, 3, 4, 4, 4, 5, 6, 7, 8, 9, 10, 11],
    #        target = 4
    # Output: [3, 5]
from typing import List

def first_and_last_occurrences_of_a_number(nums: List[int], target: int) -> List[int]:
    lower_bound = lower_bound_binary_search(nums, target)
    upper_bound = upper_bound_binary_search(nums, target)
    return [lower_bound, upper_bound]

def lower_bound_binary_search(nums: List[int], target: int) -> int:
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > target:
            right = mid - 1
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left if nums and nums[left] == target else -1

def upper_bound_binary_search(nums: List[int], target: int) -> int:
    left, right = 0, len(nums) - 1
    while left < right:
        # In upper-bound binary search, bias the midpoint to the right.
        mid = (left + right) // 2 + 1
        if nums[mid] > target:
            right = mid - 1
        elif nums[mid] < target:
            left = mid + 1
        else:
            left = mid
    # If the target doesn't exist in the array, then it's possible that
    # 'left = mid + 1' places the left pointer outside the array when
    # 'mid == n - 1'. So, we use the right pointer in the return 
    # statement instead. 
    return right if nums and nums[right] == target else -1


#X55 (Medium) Cutting Wood
    # You are given an array representing the heights of trees, and an integer k 
    # representing the total length of wood that needs to be cut.
    # For this task, a woodcutting machine is set to a certain height, H . 
    # The machine cuts off the top part of all trees taller than H, while trees 
    # shorter than H remain untouched. Determine the highest possible setting of 
    # the woodcutter (H) so that it cuts at least k meters of wood.
    # Assume the woodcutter cannot be set higher than the height of the tallest tree in the array.
def cutting_wood(heights: List[int], k: int) -> int:
    left, right = 0, max(heights)
    while left < right:
        # Bias the midpoint to the right during the upper-bound binary 
        # search.
        mid = (left + right) // 2 + 1
        if cuts_enough_wood(mid, k, heights):
            left = mid
        else:
            right = mid - 1
    return right

# Determine if the current value of 'H' cuts at least 'k' meters of 
# wood.
def cuts_enough_wood(H: int, k: int, heights: List[int]) -> bool:
    wood_collected = 0
    for height in heights:
        if height > H:
            wood_collected += (height - H)
    return wood_collected >= k


#X56 (Medium) Find the Target in a Rotated Sorted Array
    # A rotated sorted array is an array of numbers sorted in ascending order, 
    # in which a portion of the array is moved from the beginning to the end. 
    # For example, a possible rotation of [1, 2, 3, 4, 5] is [3, 4, 5, 1, 2] , 
    # where the first two numbers are moved to the end.
    # Given a rotated sorted array of unique numbers, return the index of a target value. 
    # If the target value is not present, return -1.
    # Example:
    # Input: nums = [8, 9, 1, 2, 3, 4, 5, 6, 7], target = 1
    # Output: 2
def find_the_target_in_a_rotated_sorted_array(nums: List[int], target: int) -> int:
   left, right = 0, len(nums) - 1
   while left < right:
       mid = (left + right) // 2
       if nums[mid] == target:
           return mid
       # If the left subarray [left : mid] is sorted, check if the 
       # target falls in this range. If it does, search the left 
       # subarray. Otherwise, search the right.
       elif nums[left] <= nums[mid]:
           if nums[left] <= target < nums[mid]:
               right = mid - 1
           else:
               left = mid + 1
       # If the right subarray [mid : right] is sorted, check if the
       # target falls in this range. If it does, search the right
       # subarray. Otherwise, search the left.
       else:
           if nums[mid] < target <= nums[right]:
               left = mid + 1
           else:
               right = mid - 1
   # If the target is found in the array, return it's index. Otherwise,
   # return -1.
   return left if nums and nums[left] == target else -1


#X57 (Hard) Find the Median From Two Sorted Arrays
    # Given two sorted integer arrays, find their median value as if they were merged into a single sorted sequence.
    # Example 1:
    # Input: nums1 = [0, 2, 5, 6, 8], nums2 = [1, 3, 7]
    # Output: 4.0
    # Explanation: Merging both arrays results in [0, 1, 2, 3, 5, 6, 7, 8], which has a median of (3 + 5) / 2 = 4.0.
def find_the_median_from_two_sorted_arrays(nums1: List[int], nums2: List[int]) -> float:
    # Optimization: ensure 'nums1' is the smaller array.
    if len(nums2) < len(nums1):
        nums1, nums2 = nums2, nums1
    m, n = len(nums1), len(nums2)
    half_total_len = (m + n) // 2
    left, right = 0, m - 1
    # A median always exists in a non-empty array, so continue binary search until
    # it’s found.
    while True:
        L1_index = (left + right) // 2
        L2_index = half_total_len - (L1_index + 1) - 1
        # Set to -infinity or +infinity if out of bounds.
        L1 = float('-inf') if L1_index < 0 else nums1[L1_index]
        R1 = float('inf') if L1_index >= m - 1 else nums1[L1_index + 1]
        L2 = float('-inf') if L2_index < 0 else nums2[L2_index]
        R2 = float('inf') if L2_index >= n - 1 else nums2[L2_index + 1]
        # If 'L1 > R2', then 'L1' is too far to the right. Narrow the search space
        # toward the left.
        if L1 > R2:
            right = L1_index - 1
        # If 'L2 > R1', then 'L1' is too far to the left. Narrow the search space
        # toward the right.
        elif L2 > R1:
            left = L1_index + 1
        # If both 'L1' and 'L2' are less than or equal to both 'R1' and 'R2', we 
        # found the correct slice.
        else:
            if (m + n) % 2 == 0:
                return (max(L1, L2) + min(R1, R2)) / 2.0
            else:
                return min(R1, R2)
            

#X58 (Medium) Matrix Search
    # Determine if a target value exists in a matrix. Each row of the matrix 
    # is sorted in non-decreasing order, and the first value of each row is 
    # greater than or equal to the last value of the previous row.
def matrix_search(matrix: List[List[int]], target: int) -> bool:
    m, n = len(matrix), len(matrix[0])
    left, right = 0, m * n - 1
    # Perform binary search to find the target.
    while left <= right:
        mid = (left + right) // 2
        r, c = mid // n, mid % n
        if matrix[r][c] == target:
            return True
        elif matrix[r][c] > target:
            right = mid - 1
        else:
            left = mid + 1
    return False


#X59 (Medium) Local Maxima in Array
    # A local maxima is a value greater than both its immediate neighbors. 
    # Return any local maxima in an array. You may assume that an element 
    # is always considered to be strictly greater than a neighbor that is outside the array.
def local_maxima_in_array(nums: List[int]) -> int:
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[mid + 1]:
            right = mid
        else:
            left = mid + 1
    return left


#X60 (Medium) Weighted Random Selection
    # Given an array of items, each with a corresponding weight, implement a 
    # function that randomly selects an item from the array, where the probability 
    # of selecting any item is proportional to its weight.
    # In other words, the probability of picking the item at index i is:
    # weights[i] / sum(weights).
    # Return the index of the selected item.
    # Example:
    # Input: weights = [3, 1, 2, 4]
import random
from typing import List

class WeightedRandomSelection:
    def __init__(self, weights: List[int]):
        self.prefix_sums = [weights[0]]
        for i in range(1, len(weights)):
            self.prefix_sums.append(self.prefix_sums[-1] + weights[i])

    def select(self) -> int:
        # Pick a random target between 1 and the largest endpoint on the number 
        # line.
        target = random.randint(1, self.prefix_sums[-1])
        left, right = 0, len(self.prefix_sums) - 1
        # Perform lower-bound binary search to find which endpoint (i.e., prefix 
        # sum value) corresponds to the target.
        while left < right:
            mid = (left + right) // 2
            if self.prefix_sums[mid] < target:
                left = mid + 1
            else:
                right = mid
        return left