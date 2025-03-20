"""

"""
#1 (Easy) 203.移除链表元素
    # 题意：删除链表中等于给定值 val 的所有节点。
    # 示例 1： 输入：head = [1,2,6,3,4,5,6], val = 6 输出：[1,2,3,4,5]
    # 示例 2： 输入：head = [], val = 1 输出：[]
    # 示例 3： 输入：head = [7,7,7,7], val = 7 输出：[]
"""
设置一个虚拟头结点在进行删除操作

双指针遍历linked list,用slow来定位中间位置
方法1
while fast.next and fast.next.next:
    fast = fast.next.next
    slow = slow.next

方法2
while fast and fast.next:
    slow = slow.next
    fast = fast.next.next

区别在slow停在的位置, 第1个要从slow.next断, 第2个要从slow断!!!
两种方法区别在linked list中node数目为偶数时slow的停留点不一样, 这时分割的时候要注意是用slow还是slow.next来分
node数目为奇数的时候,用两种方法slow的停留点都一样, 没有影响
"""
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeElements(self, head: ListNode, val: int) -> ListNode:
        # 设置一个虚拟头结点在进行删除操作,虚拟节点指向头结点
        dummy_head = ListNode(next=head) #添加一个虚拟节点
        cur = dummy_head #cur指向虚拟节点,用于遍历链表,最后返回dummy_head.next, 而不是head, 因为head可能被删除.
        while(cur.next!=None):
            if(cur.next.val == val):
                # 如果找到了对应值,就跳过next到下一个next.next
                # 本来应该是cur.next = cur.next
                cur.next = cur.next.next #删除cur.next节点
            else:
                # 如果没有找到对应值,就遍历下一个node
                cur = cur.next
                # return到dummy节点的next node
        return dummy_head.next


#2 (Medium) 707.设计链表
    # 题意：
    # 在链表类中实现这些功能：
    # get(index)：获取链表中第 index 个节点的值。如果索引无效，则返回-1。
    # addAtHead(val)：在链表的第一个元素之前添加一个值为 val 的节点。插入后，新节点将成为链表的第一个节点。
    # addAtTail(val)：将值为 val 的节点追加到链表的最后一个元素。
    # addAtIndex(index,val)：在链表中的第 index 个节点之前添加值为 val  的节点。如果 index 等于链表的长度，
    # 则该节点将附加到链表的末尾。如果 index 大于链表长度，则不会插入节点。如果index小于0，则在头部插入节点。
    # deleteAtIndex(index)：如果索引 index 有效，则删除链表中的第 index 个节点。
# （版本一）单链表法
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
        
class MyLinkedList:
    def __init__(self):
        self.dummy_head = ListNode()
        self.size = 0 # 设置一个链表长度的属性，便于后续操作，注意每次增和删的时候都要更新

    def get(self, index: int) -> int:
        if index < 0 or index >= self.size:
            return -1
        
        current = self.dummy_head.next
        for _ in range(index):
            current = current.next
        # while(index):
        #     current = current.next
        #     index -= 1

        return current.val

    def addAtHead(self, val: int) -> None:
        self.dummy_head.next = ListNode(val, self.dummy_head.next)
        # new_node = ListNode(val)
        # new_node.next = self.dummy_head.next
        # self.dummy_head.next = new_node
        self.size += 1

    def addAtTail(self, val: int) -> None:
        current = self.dummy_head
        while current.next:
            current = current.next
        current.next = ListNode(val)
        self.size += 1

    def addAtIndex(self, index: int, val: int) -> None:
        if index < 0 or index > self.size:
            return
        # if index < 0:
        #     self.addAtHead(val)
        #     return
        # elif index == self.size:
        #     self.addAtTail(val)
        #     return
        # elif index > self.size:
        #     return

        current = self.dummy_head
        for _ in range(index):
            current = current.next
        current.next = ListNode(val, current.next)
        # while(index):
        #     current = current.next
        #     index -= 1
        # node = ListNode(val)
        # node.next = current.next
        # current.next = node
        self.size += 1

    def deleteAtIndex(self, index: int) -> None:
        if index < 0 or index >= self.size:
            return
        
        current = self.dummy_head
        for _ in range(index):
            current = current.next
        # while(index):
        #     current = current.next
        #     index -= 1
        current.next = current.next.next
        self.size -= 1
# Your MyLinkedList object will be instantiated and called as such:
# obj = MyLinkedList()
# param_1 = obj.get(index)
# obj.addAtHead(val)
# obj.addAtTail(val)
# obj.addAtIndex(index,val)
# obj.deleteAtIndex(index)
# （版本二）双链表法
class ListNode:
    def __init__(self, val=0, prev=None, next=None):
        self.val = val
        self.prev = prev
        self.next = next

class MyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def get(self, index: int) -> int:
        if index < 0 or index >= self.size:
            return -1
        
        if index < self.size // 2:
            current = self.head
            for i in range(index):
                current = current.next
        else:
            current = self.tail
            for i in range(self.size - index - 1):
                current = current.prev
                
        return current.val

    def addAtHead(self, val: int) -> None:
        new_node = ListNode(val, None, self.head)
        if self.head:
            self.head.prev = new_node
        else:
            self.tail = new_node
        self.head = new_node
        self.size += 1

    def addAtTail(self, val: int) -> None:
        new_node = ListNode(val, self.tail, None)
        if self.tail:
            self.tail.next = new_node
        else:
            self.head = new_node
        self.tail = new_node
        self.size += 1

    def addAtIndex(self, index: int, val: int) -> None:
        if index < 0 or index > self.size:
            return
        
        if index == 0:
            self.addAtHead(val)
        elif index == self.size:
            self.addAtTail(val)
        else:
            if index < self.size // 2:
                current = self.head
                for i in range(index - 1):
                    current = current.next
            else:
                current = self.tail
                for i in range(self.size - index):
                    current = current.prev
            new_node = ListNode(val, current, current.next)
            current.next.prev = new_node
            current.next = new_node
            self.size += 1

    def deleteAtIndex(self, index: int) -> None:
        if index < 0 or index >= self.size:
            return
        
        if index == 0:
            self.head = self.head.next
            if self.head:
                self.head.prev = None
            else:
                self.tail = None
        elif index == self.size - 1:
            self.tail = self.tail.prev
            if self.tail:
                self.tail.next = None
            else:
                self.head = None
        else:
            if index < self.size // 2:
                current = self.head
                for i in range(index):
                    current = current.next
            else:
                current = self.tail
                for i in range(self.size - index - 1):
                    current = current.prev
            current.prev.next = current.next
            current.next.prev = current.prev
        self.size -= 1
# Your MyLinkedList object will be instantiated and called as such:
# obj = MyLinkedList()
# param_1 = obj.get(index)
# obj.addAtHead(val)
# obj.addAtTail(val)
# obj.addAtIndex(index,val)
# obj.deleteAtIndex(index)


#X3 (Easy) 206.Reverse Linked List
    # 题意：反转一个单链表。
    # 示例: 输入: 1->2->3->4->5->NULL 输出: 5->4->3->2->1->NULL
# ***（版本一）双指针法
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:  
        pre = None
        cur = head 
        while cur:
            temp = cur.next # 保存一下 cur的下一个节点，因为接下来要改变cur->next,同时也是下一次循环的cur
            cur.next = pre #反转
            #更新pre、cur指针
            pre = cur
            cur = temp
        return pre
# （版本二）递归法
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        return self.reverse(head, None)

    # 递归函数：输入当前节点cur和前一个节点pre，返回反转后的头结点. 每次只反转一个cur节点
    def reverse(self, cur: ListNode, pre: ListNode) -> ListNode:
        if cur == None:
            return pre # 返回反转后的头结点
        temp = cur.next # 保存一下 cur的下一个节点，因为接下来要改变cur->next,同时也是下一次循环的cur
        cur.next = pre
        return self.reverse(temp, cur)
# Linked List Reversal
    # Reverse a singly linked list.
from ds import ListNode
"""
Definition of ListNode:
class ListNode:
    def __init__(self, val=None, next=None):
        self.val = val
        self.next = next
"""
def linked_list_reversal(head: ListNode) -> ListNode:
    curr_node, prev_node = head, None
    # Reverse the direction of each node's pointer until 'curr_node' 
    # is null.
    while curr_node:
        next_node = curr_node.next
        curr_node.next = prev_node
        prev_node = curr_node
        curr_node = next_node
    # 'prev_node' will be pointing at the head of the reversed linked 
    # list.
    return prev_node


#4 (Medium) 24.两两交换链表中的节点
    # 给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。你不能只是单纯的改变节点内部的值，
    # 而是需要实际的进行节点交换。
# 递归版本
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    # 递归函数：输入当前节点head，返回交换后的头结点
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is None or head.next is None:
            return head # 如果链表为空或者只有一个节点

        # 待翻转的两个node分别是pre和cur
        pre = head # 保存第一个节点
        cur = head.next # 保存第二个节点
        next = head.next.next # 保存下一次递归的起点
        
        cur.next = pre  # 交换
        pre.next = self.swapPairs(next) # 将以next为head的后续链表两两交换

        return cur # 返回交换后的头结点
# *** 双指针法
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        dummy_head = ListNode(next=head)
        current = dummy_head
        
        # 必须有cur的下一个和下下个才能交换，否则说明已经交换结束了
        while current.next and current.next.next:
            temp = current.next # 防止节点修改
            temp1 = current.next.next.next # 防止节点修改, 保存下一次循环的起点
            
            current.next = current.next.next
            current.next.next = temp
            temp.next = temp1
            current = current.next.next
        return dummy_head.next


#X5 (Medium) 19.删除链表的倒数第N个节点
    # 给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。
    # 进阶：你能尝试使用一趟扫描实现吗？
    # 输入：head = [1,2,3,4,5], n = 2 输出：[1,2,3,5] 示例 2：
    # 输入：head = [1], n = 1 输出：[] 示例 3：
    # 输入：head = [1,2], n = 1 输出：[1]
# 双指针
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        # 创建一个虚拟节点，并将其下一个指针设置为链表的头部
        dummy_head = ListNode(0, head)
        
        # 创建两个指针，慢指针和快指针，并将它们初始化为虚拟节点
        slow = fast = dummy_head
        
        # 快指针比慢指针快 n+1 步
        for i in range(n+1):
            fast = fast.next
        
        # 移动两个指针，直到快速指针到达链表的末尾
        while fast:
            slow = slow.next
            fast = fast.next
        
        # 通过更新第 (n-1) 个节点的 next 指针删除第 n 个节点
        slow.next = slow.next.next
        
        return dummy_head.next
# Remove the Kth Last Node From a Linked List
    # Return the head of a singly linked list after removing the kth node from the end of it.
from ds import ListNode
"""
Definition of ListNode:
class ListNode:
    def __init__(self, val=None, next=None):
        self.val = val
        self.next = next
"""
def remove_kth_last_node(head: ListNode, k: int) -> ListNode:
    # A dummy node to ensure there's a node before 'head' in case we 
    # need to remove the head node.
    dummy = ListNode(-1)
    dummy.next = head
    trailer = leader = dummy
    # Advance 'leader' k steps ahead.  
    for _ in range(k):
        leader = leader.next
        # If k is larger than the length of the linked list, no node 
        # needs to be removed.
        if not leader:
            return head
    # Move 'leader' to the end of the linked list, keeping 'trailer'
    # k nodes behind.
    while leader.next:
        leader = leader.next
        trailer = trailer.next
    # Remove the kth node from the end.
    trailer.next = trailer.next.next
    return dummy.next


#X6 (Easy) 160.链表相交
    # 给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。
    # 如果两个链表没有交点，返回 null 。
# （版本一）求长度，同时出发
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        lenA, lenB = 0, 0
        cur = headA
        while cur:         # 求链表A的长度
            cur = cur.next 
            lenA += 1
        cur = headB 
        while cur:         # 求链表B的长度
            cur = cur.next 
            lenB += 1
        curA, curB = headA, headB
        if lenA > lenB:     # 让curB为最长链表的头，lenB为其长度
            curA, curB = curB, curA
            lenA, lenB = lenB, lenA 
        for _ in range(lenB - lenA):  # 让curA和curB在同一起点上（末尾位置对齐）
            curB = curB.next 
        while curA:         #  遍历curA 和 curB，遇到相同则直接返回
            if curA == curB:
                return curA
            else:
                curA = curA.next 
                curB = curB.next
        return None 
# （版本二）求长度，同时出发 （代码复用）
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        lenA = self.getLength(headA)
        lenB = self.getLength(headB)
        
        # 通过移动较长的链表，使两链表长度相等
        if lenA > lenB:
            headA = self.moveForward(headA, lenA - lenB)
        else:
            headB = self.moveForward(headB, lenB - lenA)
        
        # 将两个头向前移动，直到它们相交
        while headA and headB:
            if headA == headB:
                return headA
            headA = headA.next
            headB = headB.next
        
        return None
    
    def getLength(self, head: ListNode) -> int:
        length = 0
        while head:
            length += 1
            head = head.next
        return length
    
    def moveForward(self, head: ListNode, steps: int) -> ListNode:
        while steps > 0:
            head = head.next
            steps -= 1
        return head
# （版本三）求长度，同时出发 （代码复用 + 精简）
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        dis = self.getLength(headA) - self.getLength(headB)
        
        # 通过移动较长的链表，使两链表长度相等
        if dis > 0:
            headA = self.moveForward(headA, dis)
        else:
            headB = self.moveForward(headB, abs(dis))
        
        # 将两个头向前移动，直到它们相交
        while headA and headB:
            if headA == headB:
                return headA
            headA = headA.next
            headB = headB.next
        
        return None
    
    def getLength(self, head: ListNode) -> int:
        length = 0
        while head:
            length += 1
            head = head.next
        return length
    
    def moveForward(self, head: ListNode, steps: int) -> ListNode:
        while steps > 0:
            head = head.next
            steps -= 1
        return head
# （版本四）等比例法
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        # 处理边缘情况
        if not headA or not headB:
            return None
        
        # 在每个链表的头部初始化两个指针
        pointerA = headA
        pointerB = headB
        
        # 遍历两个链表直到指针相交
        while pointerA != pointerB:
            # 将指针向前移动一个节点
            pointerA = pointerA.next if pointerA else headB
            pointerB = pointerB.next if pointerB else headA
        
        # 如果相交，指针将位于交点节点，如果没有交点，值为None
        return pointerA
# Linked List Intersection
    # Return the node where two singly linked lists intersect. If the linked lists don't intersect, return null.
    # Example:
    # Output: Node 8
    # How the custom test cases work:
    # The input is designed to describe how the two input linked lists intersect. Here's how the skip inputs work:
    # skip_A: The number of nodes to skip in list A (from the head) to reach the intersection node.
    # skip_B: The number of nodes to skip in list B (from the head) to reach the intersection node.
    # For a linked list with no intersection, set skip_A and skip_B to the length of their respective linked lists, which effectively skips all nodes in each linked list.
from ds import ListNode
"""
Definition of ListNode:
class ListNode:
    def __init__(self, val=None, next=None):
        self.val = val
        self.next = next
"""
# The while loop continues until ptr_A and ptr_B point to the same node (intersection) 
# or both become None (no intersection).
def linked_list_intersection(head_A: ListNode, head_B: ListNode) -> ListNode:
    ptr_A, ptr_B = head_A, head_B
    # Traverse through list A with 'ptr_A' and list B with 'ptr_B' 
    # until they meet.
    while ptr_A != ptr_B:
        # Traverse list A -> list B by first traversing 'ptr_A' and 
        # then, upon reaching the end of list A, continue the 
        # traversal from the head of list B.
        ptr_A = ptr_A.next if ptr_A else head_B
        # Simultaneously, traverse list B -> list A.
        ptr_B = ptr_B.next if ptr_B else head_A
    # At this point, 'ptr_A' and 'ptr_B' either point to the 
    # intersection node or both are null if the lists do not 
    # intersect. Return either pointer.
    return ptr_A


#7 (Medium) 142.环形链表II
    # 题意： 给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。
    # 为了表示给定链表中的环，使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 
    # 如果 pos 是 -1，则在该链表中没有环。
    # 说明：不允许修改给定的链表
# （版本一）快慢指针法
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
    # This code implements Floyd's Cycle Detection Algorithm 
    # (also known as the Tortoise and Hare Algorithm) to detect a cycle in a linked list 
    # and find the node where the cycle begins. 
    # When slow and fast meet:
    # slow has traveled x + y steps.
    # fast has traveled x + y + nL steps 
    # (where n is the number of cycles fast has completed).
    # Since fast moves twice as fast as slow:2(x + y) = x + y + nL
class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        slow = head
        fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            # If there is a cycle, the slow and fast pointers will eventually meet
            if slow == fast:
                # Move one of the pointers back to the start of the list
                slow = head
                while slow != fast:
                    slow = slow.next
                    fast = fast.next
                return slow
        # If there is no cycle, return None
        return None
# （版本二）集合法
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        visited = set()
        
        while head:
            if head in visited:
                return head
            visited.add(head)
            head = head.next
        
        return None


#X8 (Medium/Hard) 146.LRU Cache
    # Design and implement a data structure for the Least Recently Used (LRU) cache that 
    # supports the following operations:
    # LRUCache(capacity: int): Initialize an LRU cache with the specified capacity.
    # get(key: int) -> int: Return the value associated with a key. Return -1 if the key doesn't exist.
    # put(key: int, value: int) -> None: Add a key and its value to the cache. If adding the key would 
    # result in the cache exceeding its size capacity, evict the least recently used element. 
    # If the key already exists in the cache, update its value.
    # Example:
    # Input: [
    #   put(1, 100),
    #   put(2, 250),
    #   get(2),
    #   put(4, 300),
    #   put(3, 200),
    #   get(4),
    #   get(1),
    # ],
    #   capacity = 3
    # Output: [250, 300, -1]
    # Explanation:
    # put(1, 100)  # cache is[1: 100]
    # put(2, 250)  # cache is[1: 100, 2: 250]
    # get(2)       # return 250
    # put(4, 300)  # cache is[1: 100, 2: 250, 4: 300]t
    # put(3, 200)  # cache is[2: 250, 4: 300, 3: 200]
    # get(4)       # return 300
    # get(1)       # key 1 was evicted when adding key 3 due to the capacity
    #              # limit: return -1
# Approach 1: Doubly Linked List
# class DoublyLinkedListNode:
#     def __init__(self, key: int, val: int):
#         self.key = key
#         self.val = val
#         self.next = self.prev = None

# class LRUCache:
#     def __init__(self, capacity: int):
#         self.capacity = capacity
#         # A hash map that maps keys to nodes.
#         self.hashmap = {}
#         # Initialize the head and tail dummy nodes and connect them to 
#         # each other to establish a basic two-node doubly linked list.
#         self.head = DoublyLinkedListNode(-1, -1)
#         self.tail = DoublyLinkedListNode(-1, -1)
#         self.head.next = self.tail
#         self.tail.prev = self.head

#     def get(self, key: int) -> int:
#         if key not in self.hashmap:
#             return -1
#         # To make this key the most recently used, remove its node and 
#         # re-add it to the tail of the linked list.
#         self.remove_node(self.hashmap[key])
#         self.add_to_tail(self.hashmap[key])
#         return self.hashmap[key].val

#     def put(self, key: int, value: int) -> None:
#         # If a node with this key already exists, remove it from the 
#         # linked list.
#         if key in self.hashmap:
#             self.remove_node(self.hashmap[key])
#         node = DoublyLinkedListNode(key, value)
#         self.hashmap[key] = node
#         # Remove the least recently used node from the cache if adding 
#         # this new node will result in an overflow.
#         if len(self.hashmap) > self.capacity:
#             del self.hashmap[self.head.next.key]
#             self.remove_node(self.head.next)
#         self.add_to_tail(node)

#     def add_to_tail(self, node: DoublyLinkedListNode) -> None:
#         prev_node = self.tail.prev
#         node.prev = prev_node
#         node.next = self.tail
#         prev_node.next = node
#         self.tail.prev = node

#     def remove_node(self, node: DoublyLinkedListNode) -> None:
#         node.prev.next = node.next
#         node.next.prev = node.prev

class ListNode:
    def __init__(self, key: int, value: int):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.dic = {} # A hash map that maps keys to nodes.
        self.capacity = capacity
        # Initialize the head and tail dummy nodes and connect them to 
        # each other to establish a basic two-node doubly linked list.
        self.head = ListNode(-1, -1)
        self.tail = ListNode(-1, -1)
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key: int) -> int:
        if key not in self.dic:
            return -1
        node = self.dic[key]
        self.remove(node)
        self.add(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        # If a node with this key already exists, remove it from the linked list.
        # node = ListNode(key, value)
        if key in self.dic:
            # self.dic[key] = node
            old = self.dic[key]
            self.remove(old)

        node = ListNode(key, value)
        self.dic[key] = node
        self.add(node)
        # Remove the least recently used node from the cache if adding 
        # this new node will result in an overflow.
        if len(self.dic) > self.capacity:
            del_node = self.head.next
            del self.dic[del_node.key]
            self.remove(del_node)

    def add(self, node: ListNode) -> None:
        # self.dic[node.key] = node
        node.next = self.tail
        node.prev = self.tail.prev
        self.tail.prev.next = node
        self.tail.prev = node

    def remove(self, node: ListNode) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev


#X9 (Easy) Palindromic Linked List
    # Given the head of a singly linked list, determine if it's a palindrome.
# Time complexity : O(n)
# Space complexity : O(1)
# Approach 1: Find the Middle and Reverse the Second Half
def palindromic_linked_list(head: ListNode) -> bool:
    # Find the middle of the linked list and then reverse the second half of the
    # linked list starting at this midpoint.
    mid = find_middle(head)
    second_head = reverse_list(mid)
    # Compare the first half and the reversed second half of the list
    ptr1, ptr2 = head, second_head
    res = True
    while ptr2:
        if ptr1.val != ptr2.val:
            res = False
        ptr1, ptr2 = ptr1.next, ptr2.next
    return res

# From the 'Reverse Linked List' problem.
def reverse_list(head: ListNode) -> ListNode:
    prevNode, currNode = None, head
    while currNode:
        nextNode = currNode.next
        currNode.next = prevNode
        prevNode = currNode
        currNode = nextNode
    return prevNode

# From the 'Linked List Midpoint' problem.
def find_middle(head: ListNode) -> ListNode:
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow


#X10 (Medium) Flatten a Multi-Level Linked List
    # In a multi-level linked list, each node has a next pointer and child pointer. 
    # The next pointer connects to the subsequent node in the same linked list, 
    # while the child pointer points to the head of a new linked list under it. 
    # This creates multiple levels of linked lists. If a node does not have a child list, its child attribute is set to null.
    # Flatten the multi-level linked list into a single-level linked list by linking the end of each level to the start of the next one.
    # How the custom test cases work
    # The input is a nested list representation of a multi-level linked list. 
    # Each element in the list represents a node which includes a value and a child list, 
    # with the entire data structure using the following structure:
    # [[val_1, [child_list_1]], [val_2, [child_list_2]], ...]
    # Each child list follows this same structure.
from ds import MultiLevelListNode
"""
Definition of MultiLevelListNode:
class MultiLevelListNode:
    def __init__(self, val, next, child):
        self.val = val
        self.next = next
        self.child = child
"""
def flatten_multi_level_list(head: MultiLevelListNode) -> MultiLevelListNode:
    if not head:
        return None
    tail = head
    # Find the tail of the linked list at the first level.
    while tail.next:
        tail = tail.next
    curr = head
    # Process each node at the current level. If a node has a child linked list,
    # append it to the tail and then update the tail to the end of the extended 
    # linked list. Continue until all nodes at the current level are processed.
    while curr:
        if curr.child:
            tail.next = curr.child
            # Disconnect the child linked list from the current node.
            curr.child = None
            while tail.next:
                tail = tail.next
        curr = curr.next
    return head


#11 (Hard) 23.Merge k Sorted Lists
    # You are given an array of k linked-lists lists, each linked-list is sorted in ascending order.
    # Merge all the linked-lists into one sorted linked-list and return it.
    # Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# Approach 2: Compare one by one
# Compare every k nodes (head of every linked list) and get the node with the smallest value.
# Extend the final sorted linked list with the selected nodes.
# Time complexity : O(kN)
# Space complexity : O(1)
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        # Step 1: Initialize an empty result linked-list
        result = ListNode(None)
        curr = result

        # Step 2: Initialize a list of pointers to the k linked-lists
        ptrs = [l for l in lists if l is not None]

        # Step 3: Iterate through the k linked-lists simultaneously
        while ptrs:
            # Step 4: Find the linked-list with the smallest first element
            min_node = ptrs[0]
            min_idx = 0
            for i in range(1, len(ptrs)):
                if ptrs[i].val < min_node.val:
                    min_node = ptrs[i]
                    min_idx = i

            # Step 5: Append the smallest element to the result linked-list
            # curr.next = min_node
            curr.next = ListNode(min_node.val)
            curr = curr.next

            # Step 6: Move the pointer of the linked-list with the smallest element to the next node
            ptrs[min_idx] = ptrs[min_idx].next

            # Step 7: Remove any linked-list that has been exhausted
            ptrs = [l for l in ptrs if l is not None]

        # Step 8: Return the result linked-list
        return result.next
# Approach 3: Optimize Approach 2 by Priority Queue
# Almost the same as the one above but optimize the comparison process by priority queue. You can refer here for more information about it.
# Time complexity : O(Nlogk) where k is the number of linked lists.
# Space complexity : O(k) The code above present applies in-place method which cost O(1) space. And the priority queue (often implemented with heaps) costs O(k) space (it's far less than N in most situations).
class HeapNode:
    def __init__(self, node):
        self.node = node

    def __lt__(self, other):
        # Define comparison based on ListNode's value
        return self.node.val < other.node.val

class Solution:
    def mergeKLists(
        self, lists: List[Optional[ListNode]]
    ) -> Optional[ListNode]:
        dummy = ListNode(0)
        current = dummy
        heap = []

        # Initialize the heap
        for l in lists:
            if l:
                heapq.heappush(heap, HeapNode(l))

        # Extract the minimum node and add its next node to the heap
        while heap:
            heap_node = heapq.heappop(heap)
            node = heap_node.node
            current.next = node
            current = current.next
            if node.next:
                heapq.heappush(heap, HeapNode(node.next))

        return dummy.next
# Approach 5: Merge with Divide And Conquer
    # The merging process is done in-place, minimizing additional space usage.
    # Time complexity : O(Nlogk) where k is the number of linked lists.
    # Space complexity : O(1)
    # Definition for singly-linked list.
lists = [
    ListNode(1, ListNode(4, ListNode(5))),
    ListNode(1, ListNode(3, ListNode(4))),
    ListNode(2, ListNode(6))
]
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        amount = len(lists)
        interval = 1
        # Merge lists in pairs, doubling the interval each time
        while interval < amount:
            # No, lists in Python do not handle out-of-range indices gracefully 
            # like string slicing does. If you try to access or modify a list 
            # using an out-of-range index, Python will raise an IndexError. 
            # NOT LIKE #2 Easy 541. 反转字符串II in String sectoin

            # amount - interval is the last index of the first list 
            for i in range(0, amount - interval, interval * 2): # Ends at (amount - interval) to avoid out-of-bounds errors.
                lists[i] = self.merge2Lists(lists[i], lists[i + interval])
            interval *= 2 # This increases the distance between the lists being merged

        # Return the final merged list (first element of the list)
        return lists[0] if amount > 0 else None

    def merge2Lists(self, l1, l2):
        head = point = ListNode(0) # Create a dummy node to serve as the head of the merged list
        while l1 and l2:
            if l1.val <= l2.val:
                point.next = l1
                l1 = l1.next
            else:
                point.next = l2
                l2 = l2.next
            point = point.next

        if not l1:
            point.next = l2
        else:
            point.next = l1

        return head.next