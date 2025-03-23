"""
# Comparison: `list` vs `deque`

| Feature                  | **`list`**                              | **`deque`**                              |
|--------------------------|-----------------------------------------|------------------------------------------|
| **Import**               | Built-in (no import needed)             | Requires `from collections import deque` |
| **Memory Allocation**    | Contiguous memory block                | Linked memory blocks (more efficient for frequent insertions/deletions at ends) |
| **Append/Pop at Start**  | Slow (\(O(n)\))                        | Fast (\(O(1)\))                         |
| **Append/Pop at End**    | Fast (\(O(1)\))                        | Fast (\(O(1)\))                         |
| **Insert/Delete in Middle** | Slow (\(O(n)\))                      | Slow (\(O(n)\))                         |
| **Index Access**         | Fast (\(O(1)\))                        | Slow (\(O(n)\))                         |
| **Use Case**             | General-purpose, random access         | Queue/stack operations, frequent insertions/deletions at ends |

```python
my_list = [1, 2, 3]
my_list.append(4)       # Fast: O(1)
my_list.pop()           # Fast: O(1)
my_list.insert(0, 0)    # Slow: O(n)
print(my_list[1])       # Fast: O(1)

from collections import deque

my_deque = deque([1, 2, 3])
my_deque.append(4)       # Fast: O(1)
my_deque.appendleft(0)   # Fast: O(1)
my_deque.pop()           # Fast: O(1)
my_deque.popleft()       # Fast: O(1)
print(my_deque[1])       # Slow: O(n)
```
"""
#X1 (Medium) 232.用栈实现队列
    # 使用栈实现队列的下列操作：
    # push(x) -- 将一个元素放入队列的尾部。
    # pop() -- 从队列首部移除元素。
    # peek() -- 返回队列首部的元素。
    # empty() -- 返回队列是否为空。
    # MyQueue queue = new MyQueue();
    # queue.push(1);
    # queue.push(2);
    # queue.peek();  // 返回 1
    # queue.pop();   // 返回 1
    # queue.empty(); // 返回 false
    # 说明:
    # 你只能使用标准的栈操作 -- 也就是只有 push to top, peek/pop from top, size, 和 is empty 操作是合法的。
    # 你所使用的语言也许不支持栈。你可以使用 list 或者 deque(双端队列)来模拟一个栈,只要是标准的栈操作即可。
    # 假设所有操作都是有效的 (例如,一个空的队列不会调用 pop 或者 peek 操作)。
class MyQueue:
    def __init__(self):
        """
        in主要负责push,out主要负责pop
        """
        self.stack_in = []
        self.stack_out = []

    def push(self, x: int) -> None:
        """
        有新元素进来,就往in里面push
        """
        self.stack_in.append(x)

    def pop(self) -> int:
        """
        Removes the element from in front of queue and returns that element.
        """
        if self.empty():
            return None
        
        if self.stack_out:
            return self.stack_out.pop()
        else:
            for i in range(len(self.stack_in)):
                self.stack_out.append(self.stack_in.pop())
            return self.stack_out.pop()

    def peek(self) -> int:
        """
        Get the front element.
        """
        ans = self.pop()
        self.stack_out.append(ans)
        return ans

    def empty(self) -> bool:
        """
        只要in或者out有元素,说明队列不为空
        """
        return not (self.stack_in or self.stack_out)
# Implement a Queue using Stacks
    # Implement a queue using the stack data structure. Include the following functions:
    # enqueue(x: int) -> None: adds x to the end of the queue.
    # dequeue() -> int: removes and returns the element from the front of the queue.
    # peek() -> int: returns the front element of the queue.
    # You may not use any other data structures to implement the queue.
    # Example
    # Input: [enqueue(1), enqueue(2), dequeue(), enqueue(3), peek()]
    # Output: [1, 2]
# *****
class Queue:
    def __init__(self):
        self.enqueue_stack = []
        self.dequeue_stack = []

    def enqueue(self, x: int) -> None:
        self.enqueue_stack.append(x)

    def transfer_enqueue_to_dequeue(self) -> None:
        # If the dequeue stack is empty, push all elements from the enqueue stack
        # onto the dequeue stack. This ensures the top of the dequeue stack
        # contains the most recent value.
        if not self.dequeue_stack:
            while self.enqueue_stack:
                self.dequeue_stack.append(self.enqueue_stack.pop())

    def dequeue(self) -> int:
        self.transfer_enqueue_to_dequeue()
        # Pop and return the value at the top of the dequeue stack.
        return self.dequeue_stack.pop() if self.dequeue_stack else None

    def peek(self) -> int:
        self.transfer_enqueue_to_dequeue()
        return self.dequeue_stack[-1] if self.dequeue_stack else None

    
#2 (Easy) 225. 用队列实现栈
    # 使用队列实现栈的下列操作：
    # push(x) -- 元素 x 入栈
    # pop() -- 移除栈顶元素
    # top() -- 获取栈顶元素
    # empty() -- 返回栈是否为空
    # 注意:
    # 你只能使用队列的基本操作-- 也就是 push to back, peek/pop from front, size, 和 is empty 这些操作是合法的。
    # 你所使用的语言也许不支持队列。 你可以使用 list 或者 deque(双端队列)来模拟一个队列 , 只要是标准的队列操作即可。
    # 你可以假设所有操作都是有效的(例如, 对一个空的栈不会调用 pop 或者 top 操作)。
from collections import deque
class MyStack:
    def __init__(self):
        """
        Python普通的Queue或SimpleQueue没有类似于peek的功能
        也无法用索引访问,在实现top的时候较为困难。

        用list可以,但是在使用pop(0)的时候时间复杂度为O(n)
        因此这里使用双向队列,我们保证只执行popleft()和append(),因为deque可以用索引访问,可以实现和peek相似的功能

        in - 存所有数据
        out - 仅在pop的时候会用到
        """
        self.queue_in = deque()
        self.queue_out = deque()

    def push(self, x: int) -> None:
        """
        直接append即可
        """
        self.queue_in.append(x)

    def pop(self) -> int:
        """
        1. 首先确认不空
        2. 因为队列的特殊性,FIFO,所以我们只有在pop()的时候才会使用queue_out
        3. 先把queue_in中的所有元素(除了最后一个),依次出列放进queue_out
        4. 交换in和out,此时out里只有一个元素
        5. 把out中的pop出来,即是原队列的最后一个
        
        tip: 这不能像栈实现队列一样,因为另一个queue也是FIFO,如果执行pop()它不能像
        stack一样从另一个pop(),所以干脆in只用来存数据,pop()的时候两个进行交换
        """
        if self.empty():
            return None

        for i in range(len(self.queue_in) - 1):
            self.queue_out.append(self.queue_in.popleft())
        
        self.queue_in, self.queue_out = self.queue_out, self.queue_in    # 交换in和out,这也是为啥in只用来存
        return self.queue_out.popleft()

    def top(self) -> int:
        """
        写法一：
        1. 首先确认不空
        2. 我们仅有in会存放数据,所以返回第一个即可(这里实际上用到了栈)
        写法二：
        1. 首先确认不空
        2. 因为队列的特殊性,FIFO,所以我们只有在pop()的时候才会使用queue_out
        3. 先把queue_in中的所有元素(除了最后一个),依次出列放进queue_out
        4. 交换in和out,此时out里只有一个元素
        5. 把out中的pop出来,即是原队列的最后一个,并使用temp变量暂存
        6. 把temp追加到queue_in的末尾
        """
        # 写法一：
        # if self.empty():
        #     return None
        
        # return self.queue_in[-1]    # 这里实际上用到了栈,因为直接获取了queue_in的末尾元素

        # 写法二：
        if self.empty():
            return None

        for i in range(len(self.queue_in) - 1):
            self.queue_out.append(self.queue_in.popleft())
        
        self.queue_in, self.queue_out = self.queue_out, self.queue_in 
        temp = self.queue_out.popleft()   
        self.queue_in.append(temp)
        return temp

    def empty(self) -> bool:
        """
        因为只有in存了数据,只要判断in是不是有数即可
        """
        return len(self.queue_in) == 0
# *** 优化,使用一个队列实现
class MyStack:
    def __init__(self):
        self.que = deque()

    def push(self, x: int) -> None:
        self.que.append(x)

    def pop(self) -> int:
        if self.empty():
            return None
        for i in range(len(self.que)-1):
            self.que.append(self.que.popleft())
        return self.que.popleft()

    def top(self) -> int:
        # 写法一：
        # if self.empty():
        #     return None
        # return self.que[-1]

        # 写法二：
        if self.empty():
            return None
        for i in range(len(self.que)-1):
            self.que.append(self.que.popleft())
        temp = self.que.popleft()
        self.que.append(temp)
        return temp

    def empty(self) -> bool:
        return not self.que
# *****
class MyStack:
    def __init__(self):
        self.que = deque()

    def push(self, x: int) -> None:
        self.que.append(x)

    def transform(self) -> None:
        if self.que:
            for _ in range(len(self.que)-1):
                self.que.append(self.que.popleft())

    def pop(self) -> int:
        self.transform()
        return self.que.popleft() if self.que else None

    def top(self) -> int:
        self.transform()
        ans = self.que.popleft()
        self.que.append(ans)
        return ans if self.que else None

    def empty(self) -> bool:
        return not self.que


#X3 (Easy) 20. 有效的括号
    # 给定一个只包括 '(',')','{','}','[',']' 的字符串,判断字符串是否有效。
    # 有效字符串需满足：
    # 左括号必须用相同类型的右括号闭合。
    # 左括号必须以正确的顺序闭合。
    # 注意空字符串可被认为是有效字符串。
    # 示例 1:
    # 输入: "()"
    # 输出: true
    # 示例 2:
    # 输入: "()[]{}"
    # 输出: true
    # 示例 3:
    # 输入: "(]"
    # 输出: false
    # 示例 4:
    # 输入: "([)]"
    # 输出: false
    # 示例 5:
    # 输入: "{[]}"
    # 输出: true
# *** 方法一,仅使用栈,更省空间
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        
        for item in s:
            if item == '(':
                stack.append(')')
            elif item == '[':
                stack.append(']')
            elif item == '{':
                stack.append('}')
            elif not stack or stack[-1] != item:
                return False
            else:
                stack.pop()
        
        return True if not stack else False
# 方法二,使用字典
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        mapping = {
            '(': ')',
            '[': ']',
            '{': '}'
        }
        for item in s:
            if item in mapping.keys():
                stack.append(mapping[item])
            elif not stack or stack[-1] != item: 
                return False
            else: 
                stack.pop()
        return True if not stack else False
# Valid Parenthesis Expression
    # Given a string representing an expression of parentheses containing the characters 
    # '(', ')', '[', ']', '{', or '}', determine if the expression forms a valid sequence of parentheses.
    # A sequence of parentheses is valid if every opening parenthesis has a corresponding closing parenthesis, 
    # and no closing parenthesis appears before its matching opening parenthesis.
    # Example 1:
    # Input: s = '([]{})'
    # Output: True
    # Example 2:
    # Input: s = '([]{)}'
    # Output: False
def valid_parenthesis_expression(s: str) -> bool:
    parentheses_map = {'(': ')', '{': '}', '[': ']'}
    stack = []
    for c in s:
        # If the current character is an opening parenthesis, push it 
        # onto the stack.
        if c in parentheses_map:
            stack.append(c)
        # If the current character is a closing parenthesis, check if 
        # it closes the opening parenthesis at the top of the stack.
        else:
            if stack and parentheses_map[stack[-1]] == c:
                stack.pop()
            else:
                return False
    # If the stack is empty, all opening parentheses were successfully 
    # closed.
    return not stack


#X4 (Easy) 1047. 删除字符串中的所有相邻重复项
    # 给出由小写字母组成的字符串 S,重复项删除操作会选择两个相邻且相同的字母,并删除它们。
    # 在 S 上反复执行重复项删除操作,直到无法继续删除。
    # 在完成所有重复项删除操作后返回最终的字符串。答案保证唯一。
    # 示例：
    # 输入："abbaca"
    # 输出："ca"
    # 解释：例如,在 "abbaca" 中,我们可以删除 "bb" 由于两字母相邻且相同,这是此时唯一可以执行删除操作的重复项。之后我们得到字符串 "aaca",其中又只有 "aa" 可以执行重复项删除操作,所以最后的字符串为 "ca"。
# *** 方法一,使用栈
class Solution:
    def removeDuplicates(self, s: str) -> str:
        res = list()
        for item in s:
            if res and res[-1] == item:
                res.pop()
            else:
                res.append(item)
        return "".join(res)  # 字符串拼接
# 方法二,使用双指针模拟栈,如果不让用栈可以作为备选方法。
class Solution:
    def removeDuplicates(self, s: str) -> str:
        res = list(s)
        slow = fast = 0
        length = len(res)

        while fast < length:
            # 如果一样直接换,不一样会把后面的填在slow的位置
            res[slow] = res[fast]
            
            # 如果发现和前一个一样,就退一格指针
            if slow > 0 and res[slow] == res[slow - 1]:
                slow -= 1
            else:
                slow += 1
            fast += 1
            
        return ''.join(res[0: slow])
# Repeated Removal of Adjacent Duplicates
# Given a string, continually perform the following operation: remove a pair of adjacent duplicates from the string. 
# Continue performing this operation until the string no longer contains pairs of adjacent duplicates. Return the final string.
# Example 1:
# Input: s = 'aacabba'
# Output: 'c'
# Example 2:
# Input: s = 'aaa'
# Output: 'a'
def repeated_removal_of_adjacent_duplicates(s: str) -> str:
    stack = []
    for c in s:
        # If the current character is the same as the top character on the stack,
        # a pair of adjacent duplicates has been formed. So, pop the top character 
        # from the stack.
        if stack and c == stack[-1]:
            stack.pop()
        # Otherwise, push the current character onto the stack.
        else:
            stack.append(c)
    # Return the remaining characters as a string.
    return ''.join(stack)


#5 (Medium) 150.逆波兰表达式求值
    # 根据 逆波兰表示法,求表达式的值。
    # 有效的运算符包括 + ,  - ,  * ,  / 。每个运算对象可以是整数,也可以是另一个逆波兰表达式。
    # 说明：
    # 整数除法只保留整数部分。 给定逆波兰表达式总是有效的。换句话说,表达式总会得出有效数值且不存在除数为 0 的情况。
    # 示例 1：
    # 输入: ["2", "1", "+", "3", " * "]
    # 输出: 9
    # 解释: 该算式转化为常见的中缀算术表达式为：((2 + 1) * 3) = 9
    # 示例 2：
    # 输入: ["4", "13", "5", "/", "+"]
    # 输出: 6
    # 解释: 该算式转化为常见的中缀算术表达式为：(4 + (13 / 5)) = 6
    # 示例 3：
    # 输入: ["10", "6", "9", "3", "+", "-11", " * ", "/", " * ", "17", "+", "5", "+"]
    # 输出: 22
    # 解释:该算式转化为常见的中缀算术表达式为：

    # ((10 * (6 / ((9 + 3) * -11))) + 17) + 5       
    # = ((10 * (6 / (12 * -11))) + 17) + 5       
    # = ((10 * (6 / -132)) + 17) + 5     
    # = ((10 * 0) + 17) + 5     
    # = (0 + 17) + 5    
    # = 17 + 5    
    # = 22    
    # 逆波兰表达式：是一种后缀表达式,所谓后缀就是指运算符写在后面。
    # 平常使用的算式则是一种中缀表达式,如 ( 1 + 2 ) * ( 3 + 4 ) 。
    # 该算式的逆波兰表达式写法为 ( ( 1 2 + ) ( 3 4 + ) * ) 。
    # 逆波兰表达式主要有以下两个优点：
    # 去掉括号后表达式无歧义,上式即便写成 1 2 + 3 4 + * 也可以依据次序计算出正确结果。
    # 适合用栈操作运算：遇到数字则入栈；遇到运算符则取出栈顶两个数字进行计算,并将结果压入栈中。
# Approach 1: Reducing the List In-place
# Time Complexity : O(n^2)
# Space Complexity : O(1)
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        cur = 0

        while len(tokens) > 1:
            # Move the current position pointer to the next operator.
            while tokens[cur] not in "+-*/":
                cur += 1

            # Extract the operator and numbers from the list.
            operator = tokens[cur]
            num1 = int(tokens[cur - 2])
            num2 = int(tokens[cur - 1])

            # Calculate the result to overwrite the operator with.
            if operator == "+":
                tokens[cur] = num1 + num2
            elif operator == "-":
                tokens[cur] = num1 - num2
            elif operator == "*":
                tokens[cur] = num1 * num2
            else:
                tokens[cur] = int(num1 / num2)

            # Remove the numbers and move the pointer to the position
            # after the new number we just added.
            tokens.pop(cur - 2)
            tokens.pop(cur - 2)
            cur -= 1

        return int(tokens[0])
# *** Approach 2: Evaluate with Stack
# Time Complexity : O(n)
# Space Complexity : O(n)
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        stack = []

        for token in tokens:
            if token not in "+-/*":
                stack.append(int(token))
                continue

            num2 = stack.pop()
            num1 = stack.pop()

            result = 0
            if token == "+":
                result = num1 + num2
            elif token == "-":
                result = num1 - num2
            elif token == "*":
                result = num1 * num2
            else:
                result = int(num1 / num2)

            stack.append(result)

        return stack.pop()


from operator import add, sub, mul
class Solution:
    op_map = {'+': add, '-': sub, '*': mul, '/': lambda x, y: int(x / y)}
    
    def evalRPN(self, tokens: List[str]) -> int:
        stack = []
        for token in tokens:
            if token not in {'+', '-', '*', '/'}:
                stack.append(int(token))
            else:
                op2 = stack.pop()
                op1 = stack.pop()
                stack.append(self.op_map[token](op1, op2))  # 第一个出来的在运算符后面
        return stack.pop()
# 另一种可行,但因为使用eval相对较慢的方法:
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        stack = []
        for item in tokens:
            if item not in {"+", "-", "*", "/"}:
                stack.append(item)
            else:
                first_num, second_num = stack.pop(), stack.pop()
                stack.append(
                    int(eval(f'{second_num} {item} {first_num}'))   # 第一个出来的在运算符后面
                )
        return int(stack.pop()) # 如果一开始只有一个数,那么会是字符串形式的


#X6 (Hard) 239. 滑动窗口最大值
    # 给定一个数组 nums,有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。
    # 滑动窗口每次只向右移动一位。返回滑动窗口中的最大值。
    # 进阶：
    # 你能在线性时间复杂度内解决此题吗？
from collections import deque
class MyQueue: #单调队列(从大到小
    def __init__(self):
        self.queue = collections.deque() #这里需要使用deque实现单调队列,直接使用list会超时
    
    #每次弹出的时候,比较当前要弹出的数值是否等于队列出口元素的数值,如果相等则弹出。
    #同时pop之前判断队列当前是否为空。
    def pop(self, value):
        if self.queue and value == self.queue[0]:
            self.queue.popleft() #list.pop(n)时间复杂度为O(n),这里需要使用collections.deque()

    #如果push的数值大于入口元素的数值,那么就将队列后端的数值弹出,直到push的数值小于等于队列入口元素的数值为止。
    #这样就保持了队列里的数值是单调从大到小的了。
    def push(self, value):
        while self.queue and value > self.queue[-1]:
            self.queue.pop()
        self.queue.append(value)
        
    #查询当前队列里的最大值 直接返回队列前端也就是front就可以了。
    def front(self):
        return self.queue[0]

class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        que = MyQueue()
        result = []
        for i in range(k): #先将前k的元素放进队列
            que.push(nums[i])
        result.append(que.front()) #result 记录前k的元素的最大值
        for i in range(k, len(nums)):
            que.pop(nums[i - k]) #滑动窗口移除最前面元素
            que.push(nums[i]) #滑动窗口前加入最后面的元素
            result.append(que.front()) #记录对应的最大值
        return result
# (Hard) Maximums of Sliding Window
    # There's a sliding window of size k that slides through an integer array from left to right. 
    # Create a new array that records the largest number found in each window as it slides through.
    # Example:
    # Input: nums = [3, 2, 4, 1, 2, 1, 1], k = 4
    # Output: [4, 4, 4, 2]
# ***** Sliding Window Maximum
from collections import deque
from typing import List

def maximums_of_sliding_window(nums: List[int], k: int) -> List[int]:
    res = []
    dq = deque()
    left = right = 0
    while right < len(nums):
        # 1) Ensure the values of the deque maintain a monotonic decreasing order
        # by removing candidates <= the current candidate.
        while dq and dq[-1][0] <= nums[right]:
            dq.pop()
        # *** 2) Add the current candidate and its index to the deque.
        dq.append((nums[right], right))
        # If the window is of length 'k', record the maximum of the window.
        if right - left + 1 == k:
            # 3) Remove values whose indexes occur outside the window.
            if dq and dq[0][1] < left:
                dq.popleft()
            # The maximum value of this window is the leftmost value in the 
            # deque.
            res.append(dq[0][0])
            # Slide the window by advancing both 'left' and 'right'. The right  
            # pointer always gets advanced so we just need to advance 'left'.
            left += 1
        right += 1
    return res


#7 (Medium) 347.前 K 个高频元素
    # 给定一个非空的整数数组,返回其中出现频率前 k 高的元素。
    # 示例 1:
    # 输入: nums = [1,1,1,2,2,3], k = 2
    # 输出: [1,2]
    # 示例 2:
    # 输入: nums = [1], k = 1
    # 输出: [1]
    # 提示：
    # 你可以假设给定的 k 总是合理的,且 1 ≤ k ≤ 数组中不相同的元素的个数。
    # 你的算法的时间复杂度必须优于 $O(n \log n)$ , n 是数组的大小。
    # 题目数据保证答案唯一,换句话说,数组中前 k 个高频元素的集合是唯一的。
    # 你可以按任意顺序返回答案。
# 然后是对频率进行排序，这里我们可以使用一种 容器适配器就是优先级队列。
# 什么是优先级队列呢？
# 其实就是一个披着队列外衣的堆，因为优先级队列对外接口只是从队头取元素，从队尾添加元素，再无其他取元素的方式，看起来就是一个队列。
# 缺省情况下priority_queue利用max-heap（大顶堆）完成对元素的排序，这个大顶堆是以vector为表现形式的complete binary tree（完全二叉树）。
# 所以大家经常说的大顶堆（堆头是最大元素），小顶堆（堆头是最小元素），如果懒得自己实现的话，就直接用priority_queue（优先级队列）就可以了，
# 底层实现都是一样的，从小到大排就是小顶堆，从大到小排就是大顶堆。
# 本题我们就要使用优先级队列来对部分频率进行排序。
# 为什么不用快排呢， 使用快排要将map转换为vector的结构，然后对整个数组进行排序， 
# 而这种场景下，我们其实只需要维护k个有序的序列就可以了，所以使用优先级队列是最优的。
# 定义一个大小为k的大顶堆，在每次移动更新大顶堆的时候，每次弹出都把最大的元素弹出去了，那么怎么保留下来前K个高频元素呢。
# 而且使用大顶堆就要把所有元素都进行排序，那能不能只排序k个元素呢？
# 所以我们要用小顶堆，因为要统计最大前k个元素，只有小顶堆每次将最小的元素弹出，最后小顶堆里积累的才是前k个最大元素。
#时间复杂度：O(nlogk)
#空间复杂度：O(n)
# The heapq module in Python implements a min-heap by default.
# 小顶堆, 也叫做最小堆
# Min-Heap
import heapq
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        #要统计元素出现频率
        map_ = {} #nums[i]:对应出现的次数
        for n in nums:
            map_[n] = map_.get(n, 0) + 1
        
        #对频率排序
        #定义一个小顶堆,大小为k
        pri_que = [] #小顶堆
        
        #用固定大小为k的小顶堆,扫描所有频率的数值
        for key, freq in map_.items():
            heapq.heappush(pri_que, (freq, key))
            if len(pri_que) > k: #如果堆的大小大于了K,则队列弹出,保证堆的大小一直为k
                heapq.heappop(pri_que)
        
        #找出前K个高频元素,因为小顶堆先弹出的是最小的,所以倒序来输出到数组
        result = [0] * k
        for i in range(k-1, -1, -1):
            result[i] = heapq.heappop(pri_que)[1]

        return result
# 如果要使用大顶堆,可以把频率取反,就可以模拟出大顶堆的效果
# *** Max-Heap
import heapq
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        hashmap = {}
        for n in nums:
            hashmap[n] = hashmap.get(n, 0) + 1
        
        max_heap = []
        for key, freq in hashmap.items():
            heapq.heappush(max_heap, (-freq, key))
        
        res = [0] * k
        for i in range(k):
            res[i] = heapq.heappop(max_heap)[1]

        return res


#X8 (Hard) 224.Evaluate Expression
    # Given a string representing a mathematical expression containing integers, parentheses, 
    # addition, and subtraction operators, evaluate and return the result of the expression.
# (Hard) 224. Basic Calculator
    # Given a string s representing a valid expression, implement a basic calculator 
    # to evaluate it, and return the result of the evaluation.
    # Note: You are not allowed to use any built-in function which evaluates strings 
    # as mathematical expressions, such as eval().
    # Example 1:
    # Input: s = "1 + 1"
    # Output: 2
    # Example 2:
    # Input: s = " 2-1 + 2 "
    # Output: 3
    # Example 3:
    # Input: s = "(1+(4+5+2)-3)+(6+8)"
    # Output: 23
    # Constraints:
    # 1 <= s.length <= 3 * 105
    # s consists of digits, '+', '-', '(', ')', and ' '.
    # s represents a valid expression.
    # '+' is not used as a unary operation (i.e., "+1" and "+(2 + 3)" is invalid).
    # '-' could be used as a unary operation (i.e., "-1" and "-(2 + 3)" is valid).
    # There will be no two consecutive operators in the input.
    # Every number and running calculation will fit in a signed 32-bit integer.
# Approach 1: Stack and String Reversal
# Time Complexity: O(n)
# Space Complexity: O(n)
class Solution:
    def calculate(self, s: str) -> int:
        stack = []
        curr_num, sign, res = 0, 1, 0
        for c in s:
            if c.isdigit():
                curr_num = curr_num * 10 + int(c)
            elif c == '+' or c == '-':
                res += curr_num * sign
                sign = -1 if c == '-' else 1
                curr_num = 0
            elif c == '(':
                stack.append(res)
                stack.append(sign)
                res, sign = 0, 1
            elif c == ')':
                res += sign * curr_num
                res *= stack.pop()
                res += stack.pop()
                curr_num = 0
        return res + curr_num * sign
# Explain in detail
def evaluate_expression(s: str) -> int:
    stack = []
    curr_num, sign, res = 0, 1, 0
    for c in s:
        if c.isdigit():
            curr_num = curr_num * 10 + int(c)
        # If the current character is an operator, add 'curr_num' to 
        # the result after multiplying it by its sign.
        elif c == '+' or c == '-':
            res += curr_num * sign
            # Update the sign and reset 'curr_num'.
            sign = -1 if c == '-' else 1
            curr_num = 0
        # If the current character is an opening parenthesis, a new 
        # nested expression is starting. 
        elif c == '(':
            # Save the current 'res' and 'sign' values by pushing them 
            # onto the stack, then reset their values to start 
            # calculating the new nested expression.
            stack.append(res)
            stack.append(sign)
            res, sign = 0, 1
        # If the current character is a closing parenthesis, a nested 
        # expression has ended.
        elif c == ')':
            # Finalize the result of the current nested expression.
            res += sign * curr_num
            # Apply the sign of the current nested  expression's result 
            # before adding this result to the result of the outer 
            # expression.
            res *= stack.pop()
            res += stack.pop()
            curr_num = 0
    # Finalize the result of the overall expression.
    return res + curr_num * sign

class Solution:
    def calculate(self, s: str) -> int:
        stack = []
        operand = 0 # current number
        res = 0 # For the on-going result
        sign = 1 # 1 means positive, -1 means negative  

        for c in s:
            if c.isdigit():
                # Forming operand, since it could be more than one digit
                operand = (operand * 10) + int(c)
            elif c == '+':
                # Evaluate the expression to the left,
                # with result, sign, operand
                res += sign * operand
                # Save the recently encountered '+' sign
                sign = 1
                # Reset operand
                operand = 0
            elif c == '-':
                res += sign * operand
                sign = -1
                operand = 0
            elif c == '(':
                # Push the result and sign on to the stack, for later
                # We push the result first, then sign
                stack.append(res)
                stack.append(sign)
                # Reset operand and result, as if new evaluation begins for the new sub-expression
                sign = 1
                res = 0
            elif c == ')':
                # Evaluate the expression to the left
                # with result, sign and operand
                res += sign * operand
                # ')' marks end of expression within a set of parenthesis
                # Its result is multiplied with sign on top of stack
                # as stack.pop() is the sign before the parenthesis
                res *= stack.pop() # stack pop 1, sign
                # Then add to the next operand on the top.
                # as stack.pop() is the result calculated before this parenthesis
                # (operand on stack) + (sign on stack * (result from parenthesis))
                res += stack.pop() # stack pop 2, operand
                # Reset the operand
                operand = 0
        return res + sign * operand


#X9 (Medium) Next Largest Number to the Right
    # Given an integer array nums, return an output array res where, 
    # for each value nums[i], res[i] is the first number to the right that's larger than nums[i]. 
    # If no larger number exists to the right of nums[i], set res[i] to ‐1.
    # Example:
    # Input: nums = [5, 2, 4, 6, 1]
    # Output: [6, 4, 6, -1, -1]
def next_largest_number_to_the_right(nums: List[int]) -> List[int]:
    res = [0] * len(nums)
    stack = []
    # Find the next largest number of each element, starting with the 
    # rightmost element.
    for i in range(len(nums) - 1, -1, -1):
        # Pop values from the top of the stack until the current 
        # value's next largest number is at the top.
        while stack and stack[-1] <= nums[i]:
            stack.pop()
        # Record the current value's next largest number, which is at 
        # the top of the stack. If the stack is empty, record -1.
        res[i] = stack[-1] if stack else -1
        stack.append(nums[i])
    return res


#10 (Medium) 227. Basic Calculator II
    # Given a string s which represents an expression, evaluate this expression and 
    # return its value. 
    # The integer division should truncate toward zero.
    # You may assume that the given expression is always valid. All intermediate 
    # results will be in the range of [-231, 231 - 1].
    # Note: You are not allowed to use any built-in function which evaluates 
    # strings as mathematical expressions, such as eval().
    # Example 1:
    # Input: s = "3+2*2"
    # Output: 7
    # Example 2:
    # Input: s = " 3/2 "
    # Output: 1
    # Example 3:
    # Input: s = " 3+5 / 2 "
    # Output: 5
    # Constraints:
    # 1 <= s.length <= 3 * 105
    # s consists of integers and operators ('+', '-', '*', '/') separated by some number of spaces.
    # s represents a valid expression.
    # All the integers in the expression are non-negative integers in the range [0, 231 - 1].
    # The answer is guaranteed to fit in a 32-bit integer.
# Use Stack
# time complexity: O(n)
# space complexity: O(n)
class Solution:
    def calculate(self, s: str) -> int:
        stack = []
        curr_num = 0
        operator = '+'
        
        for i, c in enumerate(s):
            if c.isdigit():
                curr_num = curr_num * 10 + int(c)
            
            if (not c.isdigit() and c != ' ') or i == len(s) - 1:
                if operator == '+':
                    stack.append(curr_num)
                elif operator == '-':
                    stack.append(-curr_num)
                elif operator == '*':
                    stack.append(stack.pop() * curr_num)
                elif operator == '/':
                    # Handle division truncating toward zero
                    top = stack.pop()
                    if top // curr_num < 0 and top % curr_num != 0:
                        stack.append(top // curr_num + 1)
                    else:
                        stack.append(top // curr_num)
                
                # Update the operator and reset curr_num
                operator = c
                curr_num = 0
        
        # Sum all the numbers in the stack to get the final result
        return sum(stack)
# NO Stack
# time complexity: O(n)
# space complexity: O(1)
class Solution:
    def calculate(self, s: str) -> int:
        curr_num = 0
        prev_num = 0
        res = 0
        operator = '+'
        
        for i, c in enumerate(s):
            if c.isdigit():
                curr_num = curr_num * 10 + int(c)
            
            if (not c.isdigit() and c != ' ') or i == len(s) - 1:
                if operator == '+':
                    res += prev_num
                    prev_num = curr_num
                elif operator == '-':
                    res += prev_num
                    prev_num = -curr_num
                elif operator == '*':
                    prev_num *= curr_num
                elif operator == '/':
                    # Handle division truncating toward zero
                    if prev_num < 0:
                        prev_num = -(-prev_num // curr_num)
                    else:
                        prev_num = prev_num // curr_num
                
                # Update the operator and reset curr_num
                operator = c
                curr_num = 0
        
        # Add the last prev_num to the result
        res += prev_num
        return res
# Another way
# Time Complexity: O(n)
# Space Complexity: O(1)
class Solution:
    def calculate(self, s: str) -> int:
        inner, outer, result, opt = 0, 0, 0, '+'
        for c in s + '+':
            if c == ' ': continue
            if c.isdigit():
                inner = 10 * inner + int(c)
                continue
            if opt == '+':
                result += outer
                outer = inner
            elif opt == '-':
                result += outer
                outer = -inner
            elif opt == '*':
                outer = outer * inner
            elif opt == '/':
                outer = int(outer / inner)
            inner, opt = 0, c
        return result + outer


# 11 (Hard) 282. Expression Add Operators
    # Given a string num that contains only digits and an integer target, 
    # return all possibilities to insert the binary operators '+', '-', 
    # and/or '*' between the digits of num so that the resultant expression 
    # evaluates to the target value.
    # Note that operands in the returned expressions should not contain leading zeros.
    # Example 1:
    # Input: num = "123", target = 6
    # Output: ["1*2*3","1+2+3"]
    # Explanation: Both "1*2*3" and "1+2+3" evaluate to 6.
    # Example 2:
    # Input: num = "232", target = 8
    # Output: ["2*3+2","2+3*2"]
    # Explanation: Both "2*3+2" and "2+3*2" evaluate to 8.
    # Example 3:
    # Input: num = "3456237490", target = 9191
    # Output: []
    # Explanation: There are no expressions that can be created from "3456237490" to evaluate to 9191.
# Approach 1: Backtracking
    # We can use backtracking to explore all possible combinations of inserting operators.
# time complexity: O(4^n)
# space complexity: O(n)
    # index: The current position in the num string.
    # prev_operand: The value of the previous operand (used for multiplication).
    # current_operand: The current number being formed by extending digits.
    # value: The current value of the expression being evaluated.
    # string: A list representing the current expression (used to build the final string).
class Solution:
    def addOperators(self, num: 'str', target: 'int') -> 'List[str]':
        def recurse(index, prev_operand, current_operand, value, string):
            # print(index)
            # Done processing all the digits in num
            if index == len(num):
                # If the final value == target expected AND
                # no operand is left unprocessed
                if value == target and current_operand == 0:
                    answers.append("".join(string[1:]))
                return

            # Extending the current operand by one digit
            current_operand = current_operand*10 + int(num[index])
            str_op = str(current_operand)

            # To avoid cases where we have 1 + 05 or 1 * 05 since 05 won't be a
            # valid operand. Hence this check
            if current_operand > 0:
                # NO OP recursion
                # print(f"recurse({index + 1}, {prev_operand}, {current_operand}, {value}, {string})")
                recurse(index + 1, prev_operand, current_operand, value, string)

            # print("index: ", index)
            # ADDITION
            string.append('+'); string.append(str_op)
            # print(string)
            # print(f"recurse({index + 1}, {current_operand}, {0}, {value + current_operand}, {string})")
            recurse(index + 1, current_operand, 0, value + current_operand, string)
            string.pop();string.pop()
            # print("# END ADDITION")
            # print(string)

            # Can subtract or multiply only if there are some previous operands
            if string:
                # SUBTRACTION
                string.append('-'); string.append(str_op)
                recurse(index + 1, -current_operand, 0, value - current_operand, string)
                string.pop();string.pop()

                # MULTIPLICATION
                string.append('*'); string.append(str_op)
                recurse(index + 1, current_operand * prev_operand, 0, value - prev_operand + (current_operand * prev_operand), string)
                string.pop();string.pop()

        answers = []
        recurse(0, 0, 0, 0, [])    
        return answers
# Approach 2: Backtracking
    # The same as above but with a slightly different implementation.
# time complexity: O(4^n)
# space complexity: O(n)
class Solution:
    def addOperators(self, num: str, target: int) -> List[str]:
        def backtrack(index, prev_op, cur_op, value, string):
            # Base case: If we've processed all digits
            if index == len(num):
                # If the current value matches the target and no digits are left
                if value == target and cur_op == 0:
                    result.append("".join(string))
                return
            
            # Extend the current operand by adding the current digit
            cur_op = cur_op * 10 + int(num[index])
            str_op = str(cur_op)
            
            # If the current operand is not zero, we can continue to extend it
            if cur_op > 0:
                backtrack(index + 1, prev_op, cur_op, value, string)
            
            # If this is the first operand, we can't add an operator
            if not string:
                backtrack(index + 1, cur_op, 0, value + cur_op, [str_op])
            else:
                # Try adding '+'
                backtrack(index + 1, cur_op, 0, value + cur_op, string + ['+', str_op])
                # Try adding '-'
                backtrack(index + 1, -cur_op, 0, value - cur_op, string + ['-', str_op])
                # Try adding '*'
                backtrack(index + 1, prev_op * cur_op, 0, value - prev_op + (prev_op * cur_op), string + ['*', str_op])
        
        result = []
        backtrack(0, 0, 0, 0, [])
        return result


# 12 (Hard) 772. Basic Calculator III
    # Implement a basic calculator to evaluate a simple expression string.
    # The expression string contains only non-negative integers, '+', '-', 
    # '*', '/' operators, and open '(' and closing parentheses ')'. 
    # The integer division should truncate toward zero.
    # You may assume that the given expression is always valid. All intermediate 
    # results will be in the range of [-231, 231 - 1].
    # Note: You are not allowed to use any built-in function which evaluates 
    # strings as mathematical expressions, such as eval().
    # Example 1:
    # Input: s = "1+1"
    # Output: 2
    # Example 2:
    # Input: s = "6-4/2"
    # Output: 4
    # Example 3:
    # Input: s = "2*(5+5*2)/3+(6/2+8)"
    # Output: 21
    # Constraints:
    # 1 <= s <= 104
    # s consists of digits, '+', '-', '*', '/', '(', and ')'.
    # s is a valid expression.
# Approach 1: Stack
# time complexity: O(n)
# space complexity: O(n)
    # stack: Used to store intermediate results and operators.
    # curr: Used to build multi-digit numbers.
    # previous_operator: Tracks the last operator encountered.
    # s += "@": Appends a sentinel character (@) to the end of the string to 
    # ensure the last number is processed.
# s = "3*(4+5)-6"
# Initialization:
# stack = []
# curr = 0
# previous_operator = "+"
# s = "3*(4+5)-6@"
class Solution:
    def calculate(self, s: str) -> int:
        def evaluate(x, y, operator):
            if operator == "+":
                return x
            if operator == "-":
                return -x
            if operator == "*":
                return x * y
            if operator == "/":
                return int(x / y)  # Truncate division toward zero
        
        stack = []
        curr = 0
        previous_operator = "+"
        s += "@"  # Add a sentinel character to handle the last number
        
        for c in s:
            if c.isdigit():
                # Build the current number
                curr = curr * 10 + int(c)
            elif c == "(":
                # Push the current operator onto the stack and reset it to '+'
                stack.append(previous_operator)
                previous_operator = "+"
            else:
                # Evaluate the current number with the previous operator
                if previous_operator in "*/":
                    # For '*' or '/', evaluate with the top of the stack
                    stack.append(evaluate(stack.pop(), curr, previous_operator))
                else:
                    # For '+' or '-', evaluate with 0 (no previous operand)
                    stack.append(evaluate(curr, 0, previous_operator))
                
                # Reset the current number and update the operator
                curr = 0
                previous_operator = c
                if c == ")":
                    while type(stack[-1]) == int:
                        curr += stack.pop()
                    previous_operator = stack.pop()

        return sum(stack)
# Approach 2: Solve Isolated Expressions With Recursion
# time complexity: O(n)
# space complexity: O(n)
class Solution:
    def calculate(self, s: str) -> int:
        def evaluate(x, y, operator):
            if operator == "+":
                return x
            if operator == "-":
                return -x
            if operator == "*":
                return x * y
            return int(x / y)
        
        def solve(i):
            stack = []
            curr = 0
            previous_operator = "+"
            
            while i[0] < len(s):
                c = s[i[0]]
                if c == "(":
                    i[0] += 1
                    curr = solve(i)
                elif c.isdigit():
                    curr = curr * 10 + int(c)
                else:
                    if previous_operator in "*/":
                        stack.append(evaluate(stack.pop(), curr, previous_operator))
                    else:
                        stack.append(evaluate(curr, 0, previous_operator))
                     
                    if c == ")":
                        break
                    
                    curr = 0
                    previous_operator = c
                    
                i[0] += 1
            
            return sum(stack)

        s += "@"
        return solve([0])
# # Approach 3: Recursive Descent Parsing
# # time complexity: O(n)
# # space complexity: O(n)
# class Solution:
#     def calculate(self, s: str) -> int:
#         def helper(s: str, index: int) -> (int, int):
#             curr_num = 0
#             prev_num = 0
#             res = 0
#             operator = '+'
#             i = index
            
#             while i < len(s):
#                 c = s[i]
                
#                 if c.isdigit():
#                     curr_num = curr_num * 10 + int(c)
#                 elif c == '(':
#                     # Recursively calculate the result inside the parentheses
#                     curr_num, i = helper(s, i + 1)
#                 elif c == ')':
#                     # End of the current sub-expression
#                     break
#                 elif c != ' ':
#                     # Handle the previous operator
#                     if operator == '+':
#                         res += prev_num
#                         prev_num = curr_num
#                     elif operator == '-':
#                         res += prev_num
#                         prev_num = -curr_num
#                     elif operator == '*':
#                         prev_num *= curr_num
#                     elif operator == '/':
#                         # Handle division truncating toward zero
#                         if prev_num < 0:
#                             prev_num = -(-prev_num // curr_num)
#                         else:
#                             prev_num = prev_num // curr_num
                    
#                     # Update the operator and reset curr_num
#                     operator = c
#                     curr_num = 0
                
#                 i += 1
            
#             # Add the last prev_num to the result
#             res += prev_num
#             return res, i
        
#         # Start the recursive calculation from the beginning of the string
#         result, _ = helper(s, 0)
#         return result
# # Approach 4: Stack and Recursion
# # time complexity: O(n)
# # space complexity: O(n)
# class Solution:
#     def calculate(self, s: str) -> int:
#         def evaluate(operator: str, num: int, stack: list):
#             if operator == '+':
#                 stack.append(num)
#             elif operator == '-':
#                 stack.append(-num)
#             elif operator == '*':
#                 stack.append(stack.pop() * num)
#             elif operator == '/':
#                 # Handle division truncating toward zero
#                 prev_num = stack.pop()
#                 if prev_num < 0:
#                     stack.append(-(-prev_num // num))
#                 else:
#                     stack.append(prev_num // num)
        
#         stack = []
#         curr_num = 0
#         operator = '+'
#         i = 0
        
#         while i < len(s):
#             c = s[i]
            
#             if c.isdigit():
#                 # Build the current number
#                 curr_num = curr_num * 10 + int(c)
#             elif c == '(':
#                 # Push the current result and operator onto the stack
#                 stack.append((operator, len(stack)))
#                 operator = '+'
#                 curr_num = 0
#             elif c == ')':
#                 # Evaluate the current expression inside the parentheses
#                 evaluate(operator, curr_num, stack)
#                 # Sum all the numbers in the stack until the matching '('
#                 temp = 0
#                 while len(stack) > stack[-1][1]:
#                     temp += stack.pop()
#                 # Pop the operator and reset the stack pointer
#                 op, ptr = stack.pop()
#                 evaluate(op, temp, stack)
#                 curr_num = 0
#             elif c != ' ':
#                 # Evaluate the previous operator
#                 evaluate(operator, curr_num, stack)
#                 operator = c
#                 curr_num = 0
            
#             i += 1
        
#         # Evaluate the last number
#         evaluate(operator, curr_num, stack)
        
#         # Sum all the numbers in the stack to get the final result
#         return sum(stack)


# Heaps
# The heapq module in Python implements a min-heap by default.
# When the input consists of (key, value) pairs, the heapq module will order the elements based on the key. 
# This is because heapq compares tuples lexicographically, meaning it first compares the keys, 
# and if the keys are equal, it compares the values.
"""
The `heapq` module provides an implementation of the **heap queue algorithm**, 
also known as the **priority queue algorithm**. It is useful for efficiently 
managing a collection of items with priorities.
---
## **Features of `heapq`**
- **Min-Heap by Default**: The smallest element is always at the root.
- **Efficient Operations**:
  - Insertion: \(O(\log n)\)
  - Extraction of the smallest element: \(O(\log n)\)
  - Access to the smallest element: \(O(1)\)
- **In-Place Operations**: Works directly on a list, transforming it into a heap.
- **No Dedicated Heap Class**: Uses standard Python lists.
"""

# Min-Heap Behavior
"""
import heapq

# Create an empty heap
heap = []

# Push elements into the heap
heapq.heappush(heap, 5)
heapq.heappush(heap, 2)
heapq.heappush(heap, 10)
heapq.heappush(heap, 1)

print(heap)  # Output: [1, 2, 10, 5] (smallest element at the root)

# Pop elements from the heap
print(heapq.heappop(heap))  # Output: 1 (smallest element)
print(heapq.heappop(heap))  # Output: 2 (next smallest element)
print(heapq.heappop(heap))  # Output: 5
print(heapq.heappop(heap))  # Output: 10
"""

# Max-Heap with heapq
"""
import heapq

# Create an empty heap
heap = []

# Push elements into the heap (negate values to simulate max-heap)
heapq.heappush(heap, -5)
heapq.heappush(heap, -2)
heapq.heappush(heap, -10)
heapq.heappush(heap, -1)

print(heap)  # Output: [-10, -5, -2, -1] (largest element at the root)

# Pop elements from the heap (negate values to get original numbers)
print(-heapq.heappop(heap))  # Output: 10 (largest element)
print(-heapq.heappop(heap))  # Output: 5 (next largest element)
print(-heapq.heappop(heap))  # Output: 2
print(-heapq.heappop(heap))  # Output: 1
"""
#X13 (Medium) 692.K Most Frequent Strings
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
"""
The heapq.heapify function is used to transform a regular list into a heap in-place. 
It rearranges the elements of the list to satisfy the heap property, 
which is essential for efficient heap operations like heappush, heappop, etc.
"""
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

# ***** Max-Heap
import heapq
class Solution:
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        hashmap = {}
        max_heap = []

        for w in words:
            hashmap[w] = hashmap.get(w, 0) + 1
        
        for key, freq in hashmap.items():
            heapq.heappush(max_heap, (-freq, key))
        
        res = [""] * k
        for i in range(k):
            res[i] = heapq.heappop(max_heap)[1]
        
        return res


#X14 (Medium) Combine Sorted Linked Lists
    # Given k singly linked lists, each sorted in ascending order, combine them 
    # into one sorted linked list.
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


#X15 (Hard) Median of an Integer Stream
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


#X16 (Medium) Sort a K-Sorted Array
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