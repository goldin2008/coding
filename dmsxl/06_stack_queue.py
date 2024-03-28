"""

"""
#1 232.用栈实现队列
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
    # 你所使用的语言也许不支持栈。你可以使用 list 或者 deque（双端队列）来模拟一个栈，只要是标准的栈操作即可。
    # 假设所有操作都是有效的 （例如，一个空的队列不会调用 pop 或者 peek 操作）。
class MyQueue:

    def __init__(self):
        """
        in主要负责push，out主要负责pop
        """
        self.stack_in = []
        self.stack_out = []


    def push(self, x: int) -> None:
        """
        有新元素进来，就往in里面push
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
        只要in或者out有元素，说明队列不为空
        """
        return not (self.stack_in or self.stack_out)


#2 225. 用队列实现栈
    # 使用队列实现栈的下列操作：
    # push(x) -- 元素 x 入栈
    # pop() -- 移除栈顶元素
    # top() -- 获取栈顶元素
    # empty() -- 返回栈是否为空
    # 注意:
    # 你只能使用队列的基本操作-- 也就是 push to back, peek/pop from front, size, 和 is empty 这些操作是合法的。
    # 你所使用的语言也许不支持队列。 你可以使用 list 或者 deque（双端队列）来模拟一个队列 , 只要是标准的队列操作即可。
    # 你可以假设所有操作都是有效的（例如, 对一个空的栈不会调用 pop 或者 top 操作）。
from collections import deque
class MyStack:

    def __init__(self):
        """
        Python普通的Queue或SimpleQueue没有类似于peek的功能
        也无法用索引访问，在实现top的时候较为困难。

        用list可以，但是在使用pop(0)的时候时间复杂度为O(n)
        因此这里使用双向队列，我们保证只执行popleft()和append()，因为deque可以用索引访问，可以实现和peek相似的功能

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
        2. 因为队列的特殊性，FIFO，所以我们只有在pop()的时候才会使用queue_out
        3. 先把queue_in中的所有元素（除了最后一个），依次出列放进queue_out
        4. 交换in和out，此时out里只有一个元素
        5. 把out中的pop出来，即是原队列的最后一个
        
        tip：这不能像栈实现队列一样，因为另一个queue也是FIFO，如果执行pop()它不能像
        stack一样从另一个pop()，所以干脆in只用来存数据，pop()的时候两个进行交换
        """
        if self.empty():
            return None

        for i in range(len(self.queue_in) - 1):
            self.queue_out.append(self.queue_in.popleft())
        
        self.queue_in, self.queue_out = self.queue_out, self.queue_in    # 交换in和out，这也是为啥in只用来存
        return self.queue_out.popleft()

    def top(self) -> int:
        """
        写法一：
        1. 首先确认不空
        2. 我们仅有in会存放数据，所以返回第一个即可（这里实际上用到了栈）
        写法二：
        1. 首先确认不空
        2. 因为队列的特殊性，FIFO，所以我们只有在pop()的时候才会使用queue_out
        3. 先把queue_in中的所有元素（除了最后一个），依次出列放进queue_out
        4. 交换in和out，此时out里只有一个元素
        5. 把out中的pop出来，即是原队列的最后一个，并使用temp变量暂存
        6. 把temp追加到queue_in的末尾
        """
        # 写法一：
        # if self.empty():
        #     return None
        
        # return self.queue_in[-1]    # 这里实际上用到了栈，因为直接获取了queue_in的末尾元素

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
        因为只有in存了数据，只要判断in是不是有数即可
        """
        return len(self.queue_in) == 0
# 优化，使用一个队列实现
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


#3 20. 有效的括号
    # 给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串，判断字符串是否有效。
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
# 方法一，仅使用栈，更省空间
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
# 方法二，使用字典
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


#4 1047. 删除字符串中的所有相邻重复项
    # 给出由小写字母组成的字符串 S，重复项删除操作会选择两个相邻且相同的字母，并删除它们。
    # 在 S 上反复执行重复项删除操作，直到无法继续删除。
    # 在完成所有重复项删除操作后返回最终的字符串。答案保证唯一。
    # 示例：
    # 输入："abbaca"
    # 输出："ca"
    # 解释：例如，在 "abbaca" 中，我们可以删除 "bb" 由于两字母相邻且相同，这是此时唯一可以执行删除操作的重复项。之后我们得到字符串 "aaca"，其中又只有 "aa" 可以执行重复项删除操作，所以最后的字符串为 "ca"。
# 方法一，使用栈
class Solution:
    def removeDuplicates(self, s: str) -> str:
        res = list()
        for item in s:
            if res and res[-1] == item:
                res.pop()
            else:
                res.append(item)
        return "".join(res)  # 字符串拼接
# 方法二，使用双指针模拟栈，如果不让用栈可以作为备选方法。
class Solution:
    def removeDuplicates(self, s: str) -> str:
        res = list(s)
        slow = fast = 0
        length = len(res)

        while fast < length:
            # 如果一样直接换，不一样会把后面的填在slow的位置
            res[slow] = res[fast]
            
            # 如果发现和前一个一样，就退一格指针
            if slow > 0 and res[slow] == res[slow - 1]:
                slow -= 1
            else:
                slow += 1
            fast += 1
            
        return ''.join(res[0: slow])


#5 150. 逆波兰表达式求值
    # 根据 逆波兰表示法，求表达式的值。
    # 有效的运算符包括 + ,  - ,  * ,  / 。每个运算对象可以是整数，也可以是另一个逆波兰表达式。
    # 说明：
    # 整数除法只保留整数部分。 给定逆波兰表达式总是有效的。换句话说，表达式总会得出有效数值且不存在除数为 0 的情况。
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
    # 逆波兰表达式：是一种后缀表达式，所谓后缀就是指运算符写在后面。
    # 平常使用的算式则是一种中缀表达式，如 ( 1 + 2 ) * ( 3 + 4 ) 。
    # 该算式的逆波兰表达式写法为 ( ( 1 2 + ) ( 3 4 + ) * ) 。
    # 逆波兰表达式主要有以下两个优点：
    # 去掉括号后表达式无歧义，上式即便写成 1 2 + 3 4 + * 也可以依据次序计算出正确结果。
    # 适合用栈操作运算：遇到数字则入栈；遇到运算符则取出栈顶两个数字进行计算，并将结果压入栈中。
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
# 另一种可行，但因为使用eval相对较慢的方法:
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
        return int(stack.pop()) # 如果一开始只有一个数，那么会是字符串形式的


#6 239. 滑动窗口最大值
    # 给定一个数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。
    # 返回滑动窗口中的最大值。
    # 进阶：
    # 你能在线性时间复杂度内解决此题吗？
from collections import deque
class MyQueue: #单调队列（从大到小
    def __init__(self):
        self.queue = deque() #这里需要使用deque实现单调队列，直接使用list会超时
    
    #每次弹出的时候，比较当前要弹出的数值是否等于队列出口元素的数值，如果相等则弹出。
    #同时pop之前判断队列当前是否为空。
    def pop(self, value):
        if self.queue and value == self.queue[0]:
            self.queue.popleft()#list.pop()时间复杂度为O(n),这里需要使用collections.deque()
            
    #如果push的数值大于入口元素的数值，那么就将队列后端的数值弹出，直到push的数值小于等于队列入口元素的数值为止。
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


#7 347.前 K 个高频元素
    # 给定一个非空的整数数组，返回其中出现频率前 k 高的元素。
    # 示例 1:
    # 输入: nums = [1,1,1,2,2,3], k = 2
    # 输出: [1,2]
    # 示例 2:
    # 输入: nums = [1], k = 1
    # 输出: [1]
    # 提示：
    # 你可以假设给定的 k 总是合理的，且 1 ≤ k ≤ 数组中不相同的元素的个数。
    # 你的算法的时间复杂度必须优于 $O(n \log n)$ , n 是数组的大小。
    # 题目数据保证答案唯一，换句话说，数组中前 k 个高频元素的集合是唯一的。
    # 你可以按任意顺序返回答案。
#时间复杂度：O(nlogk)
#空间复杂度：O(n)
import heapq
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        #要统计元素出现频率
        map_ = {} #nums[i]:对应出现的次数
        for i in range(len(nums)):
            map_[nums[i]] = map_.get(nums[i], 0) + 1
        
        #对频率排序
        #定义一个小顶堆，大小为k
        pri_que = [] #小顶堆
        
        #用固定大小为k的小顶堆，扫描所有频率的数值
        for key, freq in map_.items():
            heapq.heappush(pri_que, (freq, key))
            if len(pri_que) > k: #如果堆的大小大于了K，则队列弹出，保证堆的大小一直为k
                heapq.heappop(pri_que)
        
        #找出前K个高频元素，因为小顶堆先弹出的是最小的，所以倒序来输出到数组
        result = [0] * k
        for i in range(k-1, -1, -1):
            result[i] = heapq.heappop(pri_que)[1]
        return result
