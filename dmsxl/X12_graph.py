"""
广搜 (bfs) 是一圈一圈的搜索过程, 和深搜 (dfs) 是一条路跑到黑然后再回溯。

#
DFS
正是因为dfs搜索可一个方向, 并需要回溯, 所以用递归的方式来实现是最方便的。
1. 确认递归函数, 参数
通常我们递归的时候, 我们递归搜索需要了解哪些参数, 其实也可以在写递归函数的时候, 发现需要什么参数, 再去补充就可以。
一般情况, 深搜需要 二维数组数组结构保存所有路径, 需要一维数组保存单一路径, 这种保存结果的数组, 我们可以定义一个全局变量, 避免让我们的函数参数过多。
例如这样：
vector<vector<int>> result; // 保存符合条件的所有路径
vector<int> path; // 起点到终点的路径
void dfs (图, 目前搜索的节点)  

2. 确认终止条件
终止条件很重要, 很多同学写dfs的时候, 之所以容易死循环, 栈溢出等等这些问题, 都是因为终止条件没有想清楚。
if (终止条件) {
    存放结果;
    return;
}

3. 处理目前搜索节点出发的路径
一般这里就是一个for循环的操作, 去遍历 目前搜索节点 所能到的所有节点。
for (选择：本节点所连接的其他节点) {
    处理节点;
    dfs(图, 选择的节点); // 递归
    回溯, 撤销处理结果
}

BFS
广搜的搜索方式就适合于解决两个点之间的最短路径问题。
因为广搜是从起点出发, 以起始点为中心一圈一圈进行搜索, 一旦遇到终点, 记录之前走过的节点就是一条最短路。
当然, 也有一些问题是广搜 和 深搜都可以解决的, 例如岛屿问题, 这类问题的特征就是不涉及具体的遍历方式, 只要能把相邻且相同属性的节点标记上就行。

"""
# 回溯算法
void backtracking(参数) {
    if (终止条件) {
        存放结果;
        return;
    }
    for (选择：本层集合中元素（树中节点孩子的数量就是集合的大小）) {
        处理节点;
        backtracking(路径, 选择列表); // 递归
        回溯, 撤销处理结果
    }
}
# dfs的代码框架
void dfs(参数) {
    if (终止条件) {
        存放结果;
        return;
    }

    for (选择：本节点所连接的其他节点) {
        处理节点;
        dfs(图, 选择的节点); // 递归
        回溯, 撤销处理结果
    }
}

"""
最短路算法总结篇
至此已经讲解了四大最短路算法, 分别是Dijkstra、Bellman_ford、SPFA 和 Floyd。

针对这四大最短路算法，我用了七篇长文才彻底讲清楚，分别是：

dijkstra朴素版
dijkstra堆优化版
Bellman_ford
Bellman_ford 队列优化算法 (又名SPFA)
bellman_ford 算法判断负权回路
bellman_ford之单源有限最短路
Floyd 算法精讲
启发式搜索: A * 算法
"""


#1 (Medium) 797.所有可能的路径
    # 给你一个有 n 个节点的 有向无环图（DAG）, 请你找出所有从节点 0 到节点 n-1 的路径并输出（不要求按特定顺序）
    # graph[i] 是一个从节点 i 可以访问的所有节点的列表（即从节点 i 到节点 graph[i][j]存在一条有向边）。
    # 提示：
    # n == graph.length
    # 2 <= n <= 15
    # 0 <= graph[i][j] < n
    # graph[i][j] != i（即不存在自环）
    # graph[i] 中的所有元素 互不相同
    # 保证输入为 有向无环图（DAG）
# 本题是比较基础的深度优先搜索模板题, 这种有向图路径问题, 最合适使用深搜, 当然本题也可以使用广搜, 但广搜相对来说就麻烦了一些, 需要记录一下路径。
# 而深搜和广搜都适合解决颜色类的问题, 例如岛屿系列, 其实都是 遍历+标记, 所以使用哪种遍历都是可以的。
class Solution:
    def __init__(self):
        self.result = []
        self.path = [0]

    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        if not graph: return []

        self.dfs(graph, 0)
        return self.result
    
    def dfs(self, graph, root: int):
        if root == len(graph) - 1:  # 成功找到一条路径时
            # ***Python的list是mutable类型***
            # ***回溯中必须使用Deep Copy***
            self.result.append(self.path[:]) 
            return
        
        for node in graph[root]:   # 遍历节点n的所有后序节点
            self.path.append(node)
            self.dfs(graph, node)
            self.path.pop() # 回溯

# 【输入示例】
# 5 5
# 1 3
# 3 5
# 1 2
# 2 4
# 4 5
# 【输出示例】
# 1 3 5
# 1 2 4 5  

# 邻接矩阵写法
def dfs(graph, x, n, path, result):
    if x == n:
        result.append(path.copy())
        return
    for i in range(1, n + 1):
        if graph[x][i] == 1:
            path.append(i)
            dfs(graph, i, n, path, result)
            path.pop()

def main():
    n, m = map(int, input().split())
    graph = [[0] * (n + 1) for _ in range(n + 1)]

    for _ in range(m):
        s, t = map(int, input().split())
        graph[s][t] = 1

    result = []
    dfs(graph, 1, n, [1], result)

    if not result:
        print(-1)
    else:
        for path in result:
            print(' '.join(map(str, path)))

if __name__ == "__main__":
    main()

# 邻接表写法
from collections import defaultdict

result = []  # 收集符合条件的路径
path = []  # 1节点到终点的路径

def dfs(graph, x, n):
    if x == n:  # 找到符合条件的一条路径
        result.append(path.copy())
        return
    for i in graph[x]:  # 找到 x指向的节点
        path.append(i)  # 遍历到的节点加入到路径中来
        dfs(graph, i, n)  # 进入下一层递归
        path.pop()  # 回溯，撤销本节点

def main():
    n, m = map(int, input().split())

    graph = defaultdict(list)  # 邻接表
    for _ in range(m):
        s, t = map(int, input().split())
        graph[s].append(t)

    path.append(1)  # 无论什么路径已经是从1节点出发
    dfs(graph, 1, n)  # 开始遍历

    # 输出结果
    if not result:
        print(-1)
    for pa in result:
        print(' '.join(map(str, pa)))

if __name__ == "__main__":
    main()



#2 广搜的使用场景, 广搜的过程以及广搜的代码框架
    # 其实, 我们仅仅需要一个容器, 能保存我们要遍历过的元素就可以, 那么用队列, 还是用栈, 甚至用数组, 都是可以的。
    # 用队列的话, 就是保证每一圈都是一个方向去转, 例如统一顺时针或者逆时针。
    # 因为队列是先进先出, 加入元素和弹出元素的顺序是没有改变的。
    # 如果用栈的话, 就是第一圈顺时针遍历, 第二圈逆时针遍历, 第三圈有顺时针遍历。
    # 因为栈是先进后出, 加入元素和弹出元素的顺序改变了。
    # 那么广搜需要注意 转圈搜索的顺序吗？ 不需要！
    # 所以用队列, 还是用栈都是可以的, 但大家都习惯用队列了, 所以下面的讲解用我也用队列来讲, 只不过要给大家说清楚, 并不是非要用队列, 用栈也可以。
# from collections import deque

dir = [(0, 1), (1, 0), (-1, 0), (0, -1)] # 创建方向元素

def bfs(grid, visited, x, y):
    queue = collections.deque() # 初始化队列
    queue.append((x, y)) # 放入第一个元素/起点
    visited[x][y] = True # 标记为访问过的节点
  
    while queue: # 遍历队列里的元素
        curx, cury = queue.popleft() # 取出第一个元素
        
        for dx, dy in dir: # 遍历四个方向
            nextx, nexty = curx + dx, cury + dy
        
            if nextx < 0 or nextx >= len(grid) or nexty < 0 or nexty >= len(grid[0]): # 越界了, 直接跳过
                continue
                
            if not visited[nextx][nexty]: # 如果节点没被访问过  
                queue.append((nextx, nexty)) # 加入队列
                visited[nextx][nexty] = True # 标记为访问过的节点


#3 (Medium) 200.岛屿数量 DFS solution
    # 给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格, 请你计算网格中岛屿的数量。
    # 岛屿总是被水包围, 并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。
    # 此外, 你可以假设该网格的四条边均被水包围。
# 本题思路, 是用遇到一个没有遍历过的节点陆地, 计数器就加一, 然后把该节点陆地所能遍历到的陆地都标记上。
# 在遇到标记过的陆地节点和海洋节点的时候直接跳过。 这样计数器就是最终岛屿的数量。
# 那么如果把节点陆地所能遍历到的陆地都标记上呢, 就可以使用 DFS, BFS或者并查集。

# 这里大家应该能看出区别了, 无疑就是版本一中 调用dfs 的条件判断 放在了 版本二 的 终止条件位置上。
# 版本一的写法是 ：下一个节点是否能合法已经判断完了, 只要调用dfs就是可以合法的节点。
# 版本二的写法是：不管节点是否合法, 上来就dfs, 然后在终止条件的地方进行判断, 不合法再return。
# 理论上来讲, 版本一的效率更高一些, 因为避免了 没有意义的递归调用, 在调用dfs之前, 就做合法性判断。 
# 但从写法来说, 可能版本二 更利于理解一些。（不过其实都差不太多）
# 很多同学看了同一道题目, 都是dfs, 写法却不一样, 有时候有终止条件, 有时候连终止条件都没有, 其实这就是根本原因, 两种写法而已。
# *** DFS 版本一
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])
        visited = [[False] * n for _ in range(m)]
        dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # 四个方向
        result = 0

        def dfs(x, y):
            for d in dirs:
                nextx = x + d[0]
                nexty = y + d[1]
                if nextx < 0 or nextx >= m or nexty < 0 or nexty >= n:  # 越界了, 直接跳过
                    continue
                if not visited[nextx][nexty] and grid[nextx][nexty] == '1':  # 没有访问过的同时是陆地的
                    visited[nextx][nexty] = True
                    dfs(nextx, nexty)
        
        for i in range(m):
            for j in range(n):
                if not visited[i][j] and grid[i][j] == '1':
                    visited[i][j] = True
                    result += 1  # 遇到没访问过的陆地, +1
                    dfs(i, j)  # 将与其链接的陆地都标记上 true

        return result
# 版本二
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])
        visited = [[False] * n for _ in range(m)]
        dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # 四个方向
        result = 0

        def dfs(x, y):
            ### Difference starts here ###
            if visited[x][y] or grid[x][y] == '0':
                return  # 终止条件：访问过的节点 或者 遇到海水
            visited[x][y] = True
            ### Difference ends here ###

            for d in dirs:
                nextx = x + d[0]
                nexty = y + d[1]
                if nextx < 0 or nextx >= m or nexty < 0 or nexty >= n:  # 越界了, 直接跳过
                    continue
                dfs(nextx, nexty)
        
        for i in range(m):
            for j in range(n):
                if not visited[i][j] and grid[i][j] == '1':
                    result += 1  # 遇到没访问过的陆地, +1
                    dfs(i, j)  # 将与其链接的陆地都标记上 true

        return result
# 我们用三个状态去标记每一个格子
# 0 代表海水
# 1 代表陆地
# 2 代表已经访问的陆地
# class Solution:
#     def numIslands(self, grid: List[List[str]]) -> int:
#         res = 0

#         for i in range(len(grid)):
#             for j in range(len(grid[0])):
#                 if grid[i][j] == "1":
#                     res += 1
#                     self.traversal(grid, i, j)
#         return res

#     def traversal(self, grid, i, j):
#         m = len(grid)
#         n = len(grid[0])

#         if i < 0 or j < 0 or i >= m or j >= n:
#             return   # 越界了
#         elif grid[i][j] == "2" or grid[i][j] == "0":
#             return 

#         grid[i][j] = "2"
#         self.traversal(grid, i - 1, j) # 往上走
#         self.traversal(grid, i + 1, j) # 往下走
#         self.traversal(grid, i, j - 1) # 往左走
#         self.traversal(grid, i, j + 1) # 往右走

# 题目描述：
# 给定一个由 1（陆地）和 0（水）组成的矩阵，你需要计算岛屿的数量。岛屿由水平方向或垂直方向上相邻的陆地连接而成，并且四周都是水域。你可以假设矩阵外均被水包围。
# 输入描述：
# 第一行包含两个整数 N, M，表示矩阵的行数和列数。
# 后续 N 行，每行包含 M 个数字，数字为 1 或者 0。
# 输出描述：
# 输出一个整数，表示岛屿的数量。如果不存在岛屿，则输出 0。
# 输入示例：
# 4 5
# 1 1 0 0 0
# 1 1 0 0 0
# 0 0 1 0 0
# 0 0 0 1 1
# 输出示例：
# 3
# 版本一
# 理论上来讲，版本一的效率更高一些，因为避免了 没有意义的递归调用，在调用dfs之前，就做合法性判断。 
# 但从写法来说，可能版本二 更利于理解一些。（不过其实都差不太多）
direction = [[0, 1], [1, 0], [0, -1], [-1, 0]]  # 四个方向：上、右、下、左


def dfs(grid, visited, x, y):
    """
    对一块陆地进行深度优先遍历并标记
    """
    for i, j in direction:
        next_x = x + i
        next_y = y + j
        # 下标越界，跳过
        if next_x < 0 or next_x >= len(grid) or next_y < 0 or next_y >= len(grid[0]):
            continue
        # 未访问的陆地，标记并调用深度优先搜索
        if not visited[next_x][next_y] and grid[next_x][next_y] == 1:
            visited[next_x][next_y] = True
            dfs(grid, visited, next_x, next_y)


if __name__ == '__main__':  
    # 版本一
    n, m = map(int, input().split())
    
    # 邻接矩阵
    grid = []
    for i in range(n):
        grid.append(list(map(int, input().split())))
    
    # 访问表
    visited = [[False] * m for _ in range(n)]
    
    res = 0
    for i in range(n):
        for j in range(m):
            # 判断：如果当前节点是陆地，res+1并标记访问该节点，使用深度搜索标记相邻陆地。
            if grid[i][j] == 1 and not visited[i][j]:
                res += 1
                visited[i][j] = True
                dfs(grid, visited, i, j)
    
    print(res)
# 版本二
direction = [[0, 1], [1, 0], [0, -1], [-1, 0]]  # 四个方向：上、右、下、左


def dfs(grid, visited, x, y):
    """
    对一块陆地进行深度优先遍历并标记
    """
    # 与版本一的差别，在调用前增加判断终止条件
    if visited[x][y] or grid[x][y] == 0:
        return
    visited[x][y] = True

    for i, j in direction:
        next_x = x + i
        next_y = y + j
        # 下标越界，跳过
        if next_x < 0 or next_x >= len(grid) or next_y < 0 or next_y >= len(grid[0]):
            continue
        # 由于判断条件放在了方法首部，此处直接调用dfs方法
        dfs(grid, visited, next_x, next_y)


if __name__ == '__main__':
    # 版本二
    n, m = map(int, input().split())

    # 邻接矩阵
    grid = []
    for i in range(n):
        grid.append(list(map(int, input().split())))

    # 访问表
    visited = [[False] * m for _ in range(n)]

    res = 0
    for i in range(n):
        for j in range(m):
            # 判断：如果当前节点是陆地，res+1并标记访问该节点，使用深度搜索标记相邻陆地。
            if grid[i][j] == 1 and not visited[i][j]:
                res += 1
                dfs(grid, visited, i, j)

    print(res)


#4 (Medium) 200.岛屿数量 BFS solution
# 不少同学用广搜做这道题目的时候, 超时了。 这里有一个广搜中很重要的细节：
# 根本原因是只要 加入队列就代表 走过, 就需要标记, 而不是从队列拿出来的时候再去标记走过。
# 很多同学可能感觉这有区别吗？
# 如果从队列拿出节点, 再去标记这个节点走过, 就会发生下图所示的结果, 会导致很多节点重复加入队列
class Solution:
    def __init__(self):
        self.dirs = [[0, 1], [1, 0], [-1, 0], [0, -1]] 
        
    def numIslands(self, grid: List[List[str]]) -> int:
        m = len(grid)
        n = len(grid[0])
        visited = [[False]*n for _ in range(m)]
        res = 0

        for i in range(m):
            for j in range(n):
                if visited[i][j] == False and grid[i][j] == '1':
                    res += 1
                    self.bfs(grid, i, j, visited)  # Call bfs within this condition
        return res

    def bfs(self, grid, i, j, visited):
        q = deque()
        q.append((i,j))
        visited[i][j] = True
        while q:
            x, y = q.popleft()
            # visited[nextx][nexty] = True // 从队列中取出在标记走过, 错误, 会导致很多节点重复加入队列
            for d in dirs:
                nextx = x + d[0]
                nexty = y + d[1]
                if nextx < 0 or nextx >= m or nexty < 0 or nexty >= n:  # 越界了, 直接跳过
                    continue
                # bfs
                if visited[nextx][nexty]:  # 如果已经访问过了, 跳过
                    continue
                if grid[nextx][nexty] == '0':
                    continue
                q.append((nextx, nexty))
                visited[nextx][nexty] = True # 只要加入队列立刻标记

from collections import deque
directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
def bfs(grid, visited, x, y):
    que = deque([])
    que.append([x,y])
    visited[x][y] = True
    while que:
        cur_x, cur_y = que.popleft()
        for i, j in directions:
            next_x = cur_x + i
            next_y = cur_y + j
            if next_y < 0 or next_x < 0 or next_x >= len(grid) or next_y >= len(grid[0]):
                continue
            if not visited[next_x][next_y] and grid[next_x][next_y] == 1: 
                visited[next_x][next_y] = True
                que.append([next_x, next_y])


def main():
    n, m = map(int, input().split())
    grid = []
    for i in range(n):
        grid.append(list(map(int, input().split())))
    visited = [[False] * m for _ in range(n)]
    res = 0
    for i in range(n):
        for j in range(m):
            if grid[i][j] == 1 and not visited[i][j]:
                res += 1
                bfs(grid, visited, i, j)
    print(res)

if __name__ == "__main__":
    main()


#X17 (Medium) 200.Count Islands
    # Given a binary matrix representing 1s as land and 0s as water, return the number of islands.
    # An island is formed by connecting adjacent lands 4-directionally (up, down, left, and right).
from typing import List

def count_islands(matrix: List[List[int]]) -> int:
    if not matrix:
        return 0
    count = 0
    for r in range(len(matrix)):
        for c in range(len(matrix[0])):
            # If a land cell is found, perform DFS to explore the full 
            # island, and include this island in our count.
            if matrix[r][c] == 1:
                dfs(r, c, matrix)
                count += 1
    return count

def dfs(r: int, c: int, matrix: List[List[int]]) -> None:
    # Mark the current land cell as visited.
    matrix[r][c] = -1
    # Define direction vectors for up, down, left, and right.
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # Recursively call DFS on each neighboring land cell to continue 
    # exploring this island.
    for d in dirs:
        next_r, next_c = r + d[0], c + d[1]
        if is_within_bounds(next_r, next_c, matrix) and matrix[next_r][next_c] == 1:
            dfs(next_r, next_c, matrix)

def is_within_bounds(r: int, c: int, matrix: List[List[int]]) -> bool:
    return 0 <= r < len(matrix) and 0 <= c < len(matrix[0])
# 输入示例
# 4 5
# 1 1 0 0 0
# 1 1 0 0 0
# 0 0 1 0 0
# 0 0 0 1 1
# 输出示例
# 4
# 大家通过注释可以发现，两种写法，版本一，在主函数遇到陆地就计数为1，接下来的相邻陆地都在dfs中计算。
# 版本二 在主函数遇到陆地 计数为0，也就是不计数，陆地数量都去dfs里做计算。
# 这也是为什么大家看了很多 dfs的写法 ，发现写法怎么都不一样呢？ 其实这就是根本原因。

# DFS
# 四个方向
position = [[0, 1], [1, 0], [0, -1], [-1, 0]]
count = 0


def dfs(grid, visited, x, y):
    """
    深度优先搜索，对一整块陆地进行标记
    """
    global count  # 定义全局变量，便于传递count值
    for i, j in position:
        cur_x = x + i
        cur_y = y + j
        # 下标越界，跳过
        if cur_x < 0 or cur_x >= len(grid) or cur_y < 0 or cur_y >= len(grid[0]):
            continue
        if not visited[cur_x][cur_y] and grid[cur_x][cur_y] == 1:
            visited[cur_x][cur_y] = True
            count += 1
            dfs(grid, visited, cur_x, cur_y)


n, m = map(int, input().split())
# 邻接矩阵
grid = []
for i in range(n):
    grid.append(list(map(int, input().split())))
# 访问表
visited = [[False] * m for _ in range(n)]

result = 0  # 记录最终结果
for i in range(n):
    for j in range(m):
        if grid[i][j] == 1 and not visited[i][j]:
            count = 1
            visited[i][j] = True
            dfs(grid, visited, i, j)
            result = max(count, result)

print(result)
# BFS
from collections import deque

position = [[0, 1], [1, 0], [0, -1], [-1, 0]]  # 四个方向
count = 0


def bfs(grid, visited, x, y):
    """
    广度优先搜索对陆地进行标记
    """
    global count  # 声明全局变量
    que = deque()
    que.append([x, y])
    while que:
        cur_x, cur_y = que.popleft()
        for i, j in position:
            next_x = cur_x + i
            next_y = cur_y + j
            # 下标越界，跳过
            if next_x < 0 or next_x >= len(grid) or next_y < 0 or next_y >= len(grid[0]):
                continue
            if grid[next_x][next_y] == 1 and not visited[next_x][next_y]:
                visited[next_x][next_y] = True
                count += 1
                que.append([next_x, next_y])


n, m = map(int, input().split())
# 邻接矩阵
grid = []
for i in range(n):
    grid.append(list(map(int, input().split())))
visited = [[False] * m for _ in range(n)]  # 访问表

result = 0  # 记录最终结果
for i in range(n):
    for j in range(m):
        if grid[i][j] == 1 and not visited[i][j]:
            count = 1
            visited[i][j] = True
            bfs(grid, visited, i, j)
            res = max(result, count)

print(result)


# 102. 沉没孤岛
# 题目描述：
# 给定一个由 1（陆地）和 0（水）组成的矩阵，岛屿指的是由水平或垂直方向上相邻的陆地单元格组成的区域，
# 且完全被水域单元格包围。孤岛是那些位于矩阵内部、所有单元格都不接触边缘的岛屿。
# 现在你需要将所有孤岛“沉没”，即将孤岛中的所有陆地单元格（1）转变为水域单元格（0）。
# 输入描述：
# 第一行包含两个整数 N, M，表示矩阵的行数和列数。
# 之后 N 行，每行包含 M 个数字，数字为 1 或者 0，表示岛屿的单元格。
# 输出描述
# 输出将孤岛“沉没”之后的岛屿矩阵。
# 输入示例：
# 4 5
# 1 1 0 0 0
# 1 1 0 0 0
# 0 0 1 0 0
# 0 0 0 1 1
# 输出示例：
# 1 1 0 0 0
# 1 1 0 0 0
# 0 0 0 0 0
# 0 0 0 1 1
# 深搜版
def dfs(grid, x, y):
    grid[x][y] = 2
    directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]  # 四个方向
    for dx, dy in directions:
        nextx, nexty = x + dx, y + dy
        # 超过边界
        if nextx < 0 or nextx >= len(grid) or nexty < 0 or nexty >= len(grid[0]):
            continue
        # 不符合条件，不继续遍历
        if grid[nextx][nexty] == 0 or grid[nextx][nexty] == 2:
            continue
        dfs(grid, nextx, nexty)

def main():
    n, m = map(int, input().split())
    grid = [[int(x) for x in input().split()] for _ in range(n)]

    # 步骤一：
    # 从左侧边，和右侧边 向中间遍历
    for i in range(n):
        if grid[i][0] == 1:
            dfs(grid, i, 0)
        if grid[i][m - 1] == 1:
            dfs(grid, i, m - 1)

    # 从上边和下边 向中间遍历
    for j in range(m):
        if grid[0][j] == 1:
            dfs(grid, 0, j)
        if grid[n - 1][j] == 1:
            dfs(grid, n - 1, j)

    # 步骤二、步骤三
    for i in range(n):
        for j in range(m):
            if grid[i][j] == 1:
                grid[i][j] = 0
            if grid[i][j] == 2:
                grid[i][j] = 1

    # 打印结果
    for row in grid:
        print(' '.join(map(str, row)))

if __name__ == "__main__":
    main()

# 广搜版
from collections import deque

n, m = list(map(int, input().split()))
g = []
for _ in range(n):
    row = list(map(int,input().split()))
    g.append(row)
    
directions = [(1,0),(-1,0),(0,1),(0,-1)]
count = 0

def bfs(r,c,mode):
    global count 
    q = deque()
    q.append((r,c))
    count += 1
    
    while q:
        r, c = q.popleft()
        if mode:
            g[r][c] = 2
            
        for di in directions:
            next_r = r + di[0]
            next_c = c + di[1]
            if next_c < 0 or next_c >= m or next_r < 0 or next_r >= n:
                continue
            if g[next_r][next_c] == 1:
                q.append((next_r,next_c))
                if mode:
                    g[r][c] = 2
                    
                count += 1

for i in range(n):
    if g[i][0] == 1: bfs(i,0,True)
    if g[i][m-1] == 1: bfs(i, m-1,True)
    
for j in range(m):
    if g[0][j] == 1: bfs(0,j,1)
    if g[n-1][j] == 1: bfs(n-1,j,1)

for i in range(n):
    for j in range(m):
        if g[i][j] == 2:
            g[i][j] = 1
        else:
            g[i][j] = 0
            
for row in g:
    print(" ".join(map(str, row)))


#5 (Medium) 695.岛屿的最大面积
    # 给你一个大小为 m x n 的二进制矩阵 grid 。
    # 岛屿 是由一些相邻的 1 (代表土地) 构成的组合, 这里的「相邻」要求两个 1 必须在 水平或者竖直的四个方向上 相邻。你可以假设 grid 的四个边缘都被 0（代表水）包围着。
    # 岛屿的面积是岛上值为 1 的单元格的数目。
    # 计算并返回 grid 中最大的岛屿面积。如果没有岛屿, 则返回面积为 0 。
    # 输入：grid = [[0,0,1,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,1,1,0,1,0,0,0,0,0,0,0,0],[0,1,0,0,1,1,0,0,1,0,1,0,0],[0,1,0,0,1,1,0,0,1,1,1,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,1,1,0,0,0,0]]
    # 输出：6
    # 解释：答案不应该是 11 , 因为岛屿只能包含水平或垂直这四个方向上的 1 。
# BFS
class Solution:
    def __init__(self):
        self.count = 0

    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        # 与200.独立岛屿不同的是：此题grid列表内是int！！！

        # BFS
        if not grid: return 0

        m, n = len(grid), len(grid[0])
        visited = [[False for i in range(n)] for j in range(m)]

        result = 0
        for i in range(m):
            for j in range(n):
                if not visited[i][j] and grid[i][j] == 1:
                    # 每一个新岛屿
                    self.count = 0
                    self.bfs(grid, visited, i, j)
                    result = max(result, self.count)

        return result

    def bfs(self, grid, visited, i, j):
        self.count += 1
        visited[i][j] = True    

        queue = collections.deque([(i, j)])
        while queue:
            x, y = queue.popleft()
            for new_x, new_y in [(x + 1, y), (x - 1, y), (x, y - 1), (x, y + 1)]:
                if 0 <= new_x < len(grid) and 0 <= new_y < len(grid[0]) and not visited[new_x][new_y] and grid[new_x][new_y] == 1:
                    visited[new_x][new_y] = True
                    self.count += 1
                    queue.append((new_x, new_y))
#DFS
# 大家通过注释可以发现, 两种写法, 版本一, 在主函数遇到陆地就计数为1, 接下来的相邻陆地都在dfs中计算。 版本二 在主函数遇到陆地 计数为0, 也就是不计数, 陆地数量都去dfs里做计算。
# 这也是为什么大家看了很多, dfs的写法, 发现写法怎么都不一样呢？ 其实这就是根本原因。
# 这里其实涉及到dfs的两种写法。
# 写法一, dfs只处理下一个节点, 即在主函数遇到岛屿就计数为1, dfs处理接下来的相邻陆地

# 写法二, dfs处理当前节点, 即即在主函数遇到岛屿就计数为0, dfs处理接下来的全部陆地
class Solution:
    def __init__(self):
        self.count = 0

    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        # DFS
        if not grid: return 0

        m, n = len(grid), len(grid[0])
        visited = [[False for _ in range(n)] for _ in range(m)]

        result = 0
        for i in range(m):
            for j in range(n):
                if not visited[i][j] and grid[i][j] == 1:
                    self.count = 0
                    self.dfs(grid, visited, i, j)
                    result = max(result, self.count)
        return result
        
    def dfs(self, grid, visited, x, y):
        if visited[x][y] or grid[x][y] == 0:
            return 
        visited[x][y] = True
        self.count += 1
        for new_x, new_y in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
            if 0 <= new_x < len(grid) and 0 <= new_y < len(grid[0]): 
                self.dfs(grid, visited, new_x, new_y)


#6 (Medium) 1020.飞地的数量
    # 给你一个大小为 m x n 的二进制矩阵 grid , 其中 0 表示一个海洋单元格、1 表示一个陆地单元格。
    # 一次 移动 是指从一个陆地单元格走到另一个相邻（上、下、左、右）的陆地单元格或跨过 grid 的边界。
    # 返回网格中 无法 在任意次数的移动中离开网格边界的陆地单元格的数量。
    # 输入：grid = [[0,0,0,0],[1,0,1,0],[0,1,1,0],[0,0,0,0]]
    # 输出：3
    # 解释：有三个 1 被 0 包围。一个 1 没有被包围, 因为它在边界上。
    # 输入：grid = [[0,1,1,0],[0,0,1,0],[0,0,1,0],[0,0,0,0]]
    # 输出：0
    # 解释：所有 1 都在边界上或可以到达边界。
# DFS 深度优先遍历
class Solution:
    def __init__(self):
        self.position = [[-1, 0], [0, 1], [1, 0], [0, -1]]	# 四个方向

    def numEnclaves(self, grid: List[List[int]]) -> int:
        rowSize, colSize, ans = len(grid), len(grid[0]), 0
        # 标记数组记录每个值为 1 的位置是否可以到达边界, 可以为 True, 反之为 False
        visited = [[False for _ in range(colSize)] for _ in range(rowSize)]
        # 搜索左边界和右边界, 对值为 1 的位置进行深度优先遍历
        for row in range(rowSize):
            if grid[row][0] == 1:
                visited[row][0] = True
                self.dfs(grid, row, 0, visited)
            if grid[row][colSize - 1] == 1:
                visited[row][colSize - 1] = True
                self.dfs(grid, row, colSize - 1, visited)
        # 搜索上边界和下边界, 对值为 1 的位置进行深度优先遍历, 但是四个角不需要, 因为上面遍历过了
        for col in range(1, colSize - 1):
            if grid[0][col] == 1:
                visited[0][col] = True
                self.dfs(grid, 0, col, visited)
            if grid[rowSize - 1][col] == 1:
                visited[rowSize - 1][col] = True
                self.dfs(grid, rowSize - 1, col, visited)
        # 找出矩阵中值为 1 但是没有被标记过的位置, 记录答案
        for row in range(rowSize):
            for col in range(colSize):
                if grid[row][col] == 1 and not visited[row][col]:
                    ans += 1
        return ans

    # 深度优先遍历, 把可以通向边缘部分的 1 全部标记成 true
    def dfs(self, grid: List[List[int]], row: int, col: int, visited: List[List[bool]]) -> None:
        for current in self.position:
            newRow, newCol = row + current[0], col + current[1]
            # 索引下标越界
            if newRow < 0 or newRow >= len(grid) or newCol < 0 or newCol >= len(grid[0]): 
                continue
            # 当前位置值不是 1 或者已经被访问过了
            if grid[newRow][newCol] == 0 or visited[newRow][newCol]: continue

            visited[newRow][newCol] = True
            self.dfs(grid, newRow, newCol, visited)

# BFS 广度优先遍历
class Solution:
    def __init__(self):
        self.position = [[-1, 0], [0, 1], [1, 0], [0, -1]]	# 四个方向

    def numEnclaves(self, grid: List[List[int]]) -> int:
        rowSize, colSize, ans = len(grid), len(grid[0]), 0
        # 标记数组记录每个值为 1 的位置是否可以到达边界, 可以为 True, 反之为 False
        visited = [[False for _ in range(colSize)] for _ in range(rowSize)]
        queue = deque()		# 队列
        # 搜索左侧边界和右侧边界查找 1 存入队列
        for row in range(rowSize):
            if grid[row][0] == 1:
                visited[row][0] = True
                queue.append([row, 0])
            if grid[row][colSize - 1] == 1:
                visited[row][colSize - 1] = True
                queue.append([row, colSize - 1])
        # 搜索上边界和下边界查找 1 存入队列, 但是四个角不用遍历, 因为上面已经遍历到了
        for col in range(1, colSize - 1):
            if grid[0][col] == 1:
                visited[0][col] = True
                queue.append([0, col])
            if grid[rowSize - 1][col] == 1:
                visited[rowSize - 1][col] = True
                queue.append([rowSize - 1, col])

        self.bfs(grid, queue, visited)	# 广度优先遍历

        # 找出矩阵中值为 1 但是没有被标记过的位置, 记录答案
        for row in range(rowSize):
            for col in range(colSize):
                if grid[row][col] == 1 and not visited[row][col]:
                    ans += 1
        return ans

    # 广度优先遍历, 把可以通向边缘部分的 1 全部标记成 true
    def bfs(self, grid: List[List[int]], queue: deque, visited: List[List[bool]]) -> None:
        while queue:
            curPos = queue.popleft()
            for current in self.position:
                row, col = curPos[0] + current[0], curPos[1] + current[1]
                # 索引下标越界
                if row < 0 or row >= len(grid) or col < 0 or col >= len(grid[0]): continue
                # 当前位置值不是 1 或者已经被访问过了
                if grid[row][col] == 0 or visited[row][col]: continue

                visited[row][col] = True
                queue.append([row, col])


#7 (Medium) 130.被围绕的区域
    # 给你一个 m x n 的矩阵 board , 由若干字符 'X' 和 'O' , 找到所有被 'X' 围绕的区域, 并将这些区域里所有的 'O' 用 'X' 填充。
    # 输入：board = [["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]
    # 输出：[["X","X","X","X"],["X","X","X","X"],["X","X","X","X"],["X","O","X","X"]]
    # 解释：被围绕的区间不会存在于边界上, 换句话说, 任何边界上的 'O' 都不会被填充为 'X'。 任何不在边界上, 或不与边界上的 'O' 相连的 'O' 最终都会被填充为 'X'。如果两个元素在水平或垂直方向相邻, 则称它们是“相连”的。
# 步骤一：深搜或者广搜将地图周边的'O'全部改成'A'
# 步骤二：在遍历地图, 将'O'全部改成'X'（地图中间的'O'改成了'X'）, 将'A'改回'O'（保留的地图周边的'O'）
# DFS 深度优先遍历
class Solution:
    dir_list = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        row_size = len(board)
        column_size = len(board[0])
        visited = [[False] * column_size for _ in range(row_size)]
        # 从边缘开始, 将边缘相连的O改成A。然后遍历所有, 将A改成O, O改成X
        # 第一行和最后一行
        for i in range(column_size):
            if board[0][i] == "O" and not visited[0][i]:
                self.dfs(board, 0, i, visited)
            if board[row_size-1][i] == "O" and not visited[row_size-1][i]:
                self.dfs(board, row_size-1, i, visited)

        # 第一列和最后一列
        for i in range(1, row_size-1):
            if board[i][0] == "O" and not visited[i][0]:
                self.dfs(board, i, 0, visited)
            if board[i][column_size-1] == "O" and not visited[i][column_size-1]:
                self.dfs(board, i, column_size-1, visited)
        
        for i in range(row_size):
            for j in range(column_size):
                if board[i][j] == "A":
                    board[i][j] = "O"
                elif board[i][j] == "O":
                    board[i][j] = "X"

    def dfs(self, board, x, y, visited):
        if visited[x][y] or board[x][y] == "X":
            return
        visited[x][y] = True
        board[x][y] = "A"
        for i in range(4):
            new_x = x + self.dir_list[i][0]
            new_y = y + self.dir_list[i][1]
            if new_x >= len(board) or new_y >= len(board[0]) or new_x < 0 or new_y < 0:
                continue
            self.dfs(board, new_x, new_y, visited)
# 101. 孤岛的总面积
# 题目描述
# 给定一个由 1（陆地）和 0（水）组成的矩阵，岛屿指的是由水平或垂直方向上相邻的陆地单元格组成的区域，
# 且完全被水域单元格包围。孤岛是那些位于矩阵内部、所有单元格都不接触边缘的岛屿。
# 现在你需要计算所有孤岛的总面积，岛屿面积的计算方式为组成岛屿的陆地的总数。
# 输入描述
# 第一行包含两个整数 N, M，表示矩阵的行数和列数。之后 N 行，每行包含 M 个数字，数字为 1 或者 0。
# 输出描述
# 输出一个整数，表示所有孤岛的总面积，如果不存在孤岛，则输出 0。
# 输入示例
# 4 5
# 1 1 0 0 0
# 1 1 0 0 0
# 0 0 1 0 0
# 0 0 0 1 1
# 输出示例：
# 1

# 深搜版
position = [[1, 0], [0, 1], [-1, 0], [0, -1]]
count = 0

def dfs(grid, x, y):
    global count
    grid[x][y] = 0
    count += 1
    for i, j in position:
        next_x = x + i
        next_y = y + j
        if next_x < 0 or next_y < 0 or next_x >= len(grid) or next_y >= len(grid[0]):
            continue
        if grid[next_x][next_y] == 1: 
            dfs(grid, next_x, next_y)
                
n, m = map(int, input().split())

# 邻接矩阵
grid = []
for i in range(n):
    grid.append(list(map(int, input().split())))

# 清除边界上的连通分量
for i in range(n):
    if grid[i][0] == 1: 
        dfs(grid, i, 0)
    if grid[i][m - 1] == 1: 
        dfs(grid, i, m - 1)

for j in range(m):
    if grid[0][j] == 1: 
        dfs(grid, 0, j)
    if grid[n - 1][j] == 1: 
        dfs(grid, n - 1, j)
    
count = 0 # 将count重置为0
# 统计内部所有剩余的连通分量
for i in range(n):
    for j in range(m):
        if grid[i][j] == 1:
            dfs(grid, i, j)
            
print(count)
# 广搜版
from collections import deque

# 处理输入
n, m = list(map(int, input().split()))
g = []
for _ in range(n):
    row = list(map(int, input().split()))
    g.append(row)

# 定义四个方向、孤岛面积（遍历完边缘后会被重置）
directions = [[0,1], [1,0], [-1,0], [0,-1]]
count = 0

# 广搜
def bfs(r, c):
    global count
    q = deque()
    q.append((r, c))
    g[r][c] = 0
    count += 1

    while q:
        r, c = q.popleft()
        for di in directions:
            next_r = r + di[0]
            next_c = c + di[1]
            if next_c < 0 or next_c >= m or next_r < 0 or next_r >= n:
                continue
            if g[next_r][next_c] == 1:
                q.append((next_r, next_c))
                g[next_r][next_c] = 0
                count += 1


for i in range(n):
    if g[i][0] == 1: 
        bfs(i, 0)
    if g[i][m-1] == 1: 
        bfs(i, m-1)

for i in range(m):
    if g[0][i] == 1: 
        bfs(0, i)
    if g[n-1][i] == 1: 
        bfs(n-1, i)

count = 0
for i in range(n):
    for j in range(m):
        if g[i][j] == 1: 
            bfs(i, j)

print(count)

direction = [[1, 0], [-1, 0], [0, 1], [0, -1]]
result = 0

###
# 深度搜尋
def dfs(grid, y, x):
    grid[y][x] = 0
    global result 
    result += 1
    
    for i, j in direction:
        next_x = x + j
        next_y = y + i
        if (next_x < 0 or next_y < 0 or
            next_x >= len(grid[0]) or next_y >= len(grid)
        ):
            continue
        if grid[next_y][next_x] == 1 and not visited[next_y][next_x]:
            visited[next_y][next_x] = True
            dfs(grid, next_y, next_x)    


# 讀取輸入值
n, m = map(int, input().split())
grid = []
visited = [[False] * m for _ in range(n)]

for i in range(n):
    grid.append(list(map(int, input().split())))

# 處理邊界
for j in range(m):
    # 上邊界
    if grid[0][j] == 1 and not visited[0][j]: 
        visited[0][j] = True
        dfs(grid, 0, j)
    # 下邊界
    if grid[n - 1][j] == 1 and not visited[n - 1][j]: 
        visited[n - 1][j] = True
        dfs(grid, n - 1, j)
    
for i in range(n):
    # 左邊界
    if grid[i][0] == 1 and not visited[i][0]: 
        visited[i][0] = True
        dfs(grid, i, 0)
    # 右邊界
    if grid[i][m - 1] == 1 and not visited[i][m - 1]: 
        visited[i][m - 1] = True
        dfs(grid, i, m - 1)
    
# 計算孤島總面積
result = 0  # 初始化，避免使用到處理邊界時所產生的累加值

for i in range(n):
    for j in range(m):
        if grid[i][j] == 1 and not visited[i][j]:
            visited[i][j] = True
            dfs(grid, i, j)

# 輸出孤島的總面積
print(result)


#8 (Medium) 417.太平洋大西洋水流问题
    # 有一个 m × n 的矩形岛屿, 与 太平洋 和 大西洋 相邻。 “太平洋” 处于大陆的左边界和上边界, 而 “大西洋” 处于大陆的右边界和下边界。
    # 这个岛被分割成一个由若干方形单元格组成的网格。给定一个 m x n 的整数矩阵 heights ,  heights[r][c] 表示坐标 (r, c) 上单元格 高于海平面的高度 。
    # 岛上雨水较多, 如果相邻单元格的高度 小于或等于 当前单元格的高度, 雨水可以直接向北、南、东、西流向相邻单元格。水可以从海洋附近的任何单元格流入海洋。
    # 返回网格坐标 result 的 2D 列表 , 其中 result[i] = [ri, ci] 表示雨水从单元格 (ri, ci) 流动 既可流向太平洋也可流向大西洋 。
    # 示例 1：
    # 输入: heights = [[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]
    # 输出: [[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]]
    # 示例 2：
    # 输入: heights = [[2,1],[1,2]]
    # 输出: [[0,0],[0,1],[1,0],[1,1]]
    # 提示：
    # m == heights.length
    # n == heights[r].length
    # 1 <= m, n <= 200
    # 0 <= heights[r][c] <= 10^5
# 那么我们可以 反过来想, 从太平洋边上的节点 逆流而上, 将遍历过的节点都标记上。 
# 从大西洋的边上节点 逆流而长, 将遍历过的节点也标记上。 
# 然后两方都标记过的节点就是既可以流太平洋也可以流大西洋的节点。

# 所以 调用dfs函数, 只要参数传入的是 数组pacific, 那么地图中 每一个节点其实就遍历一次, 无论你调用多少次。
# 同理, 调用 dfs函数, 只要 参数传入的是 数组atlantic, 地图中每个节点也只会遍历一次。
# 所以, 以下这段代码的时间复杂度是 2 * n * m。 地图用每个节点就遍历了两次, 参数传入pacific的时候遍历一次, 参数传入atlantic的时候遍历一次。
# 那么本题整体的时间复杂度其实是： 2 * n * m + n * m , 所以最终时间复杂度为 O(n * m) 。
# 空间复杂度为：O(n * m) 这个就不难理解了。开了几个 n * m 的数组。

# DFS 深度优先遍历
class Solution:
    def __init__(self):
        self.position = [[-1, 0], [0, 1], [1, 0], [0, -1]]	# 四个方向

    # heights：题目给定的二维数组,  row：当前位置的行号,  col：当前位置的列号
    # sign：记录是哪一条河, 两条河中可以一个为 0, 一个为 1
    # visited：记录这个位置可以到哪条河
    def dfs(self, heights: List[List[int]], row: int, col: int, sign: int, visited: List[List[List[int]]]):
        for current in self.position:
            curRow, curCol = row + current[0], col + current[1]
            # 索引下标越界
            if curRow < 0 or curRow >= len(heights) or curCol < 0 or curCol >= len(heights[0]): continue
            # 不满足条件或者已经被访问过
            if heights[curRow][curCol] < heights[row][col] or visited[curRow][curCol][sign]: continue
            visited[curRow][curCol][sign] = True
            self.dfs(heights, curRow, curCol, sign, visited)

    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        rowSize, colSize = len(heights), len(heights[0])
        # visited 记录 [row, col] 位置是否可以到某条河, 可以为 true, 反之为 false；
        # 假设太平洋的标记为 1, 大西洋为 0
        # ans 用来保存满足条件的答案
        ans, visited = [], [[[False for _ in range(2)] for _ in range(colSize)] for _ in range(rowSize)]
        # 第一列和最后一列
        for row in range(rowSize):
            visited[row][0][1] = True
            visited[row][colSize - 1][0] = True
            self.dfs(heights, row, 0, 1, visited)
            self.dfs(heights, row, colSize - 1, 0, visited)
        # 第一行和最后一行
        for col in range(0, colSize):
            visited[0][col][1] = True
            visited[rowSize - 1][col][0] = True
            self.dfs(heights, 0, col, 1, visited)
            self.dfs(heights, rowSize - 1, col, 0, visited)
        for row in range(rowSize):
            for col in range(colSize):
                # 如果该位置即可以到太平洋又可以到大西洋, 就放入答案数组
                if visited[row][col][0] and visited[row][col][1]:
                    ans.append([row, col])
        return ans
# BFS 广度优先遍历
class Solution:
    def __init__(self):
        self.position = [[-1, 0], [0, 1], [1, 0], [0, -1]]

    # heights：题目给定的二维数组, visited：记录这个位置可以到哪条河
    def bfs(self, heights: List[List[int]], queue: deque, visited: List[List[List[int]]]):
        while queue:
            curPos = queue.popleft()
            for current in self.position:
                row, col, sign = curPos[0] + current[0], curPos[1] + current[1], curPos[2]
                # 越界
                if row < 0 or row >= len(heights) or col < 0 or col >= len(heights[0]): continue
                # 不满足条件或已经访问过
                if heights[row][col] < heights[curPos[0]][curPos[1]] or visited[row][col][sign]: continue
                visited[row][col][sign] = True
                queue.append([row, col, sign])

    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        rowSize, colSize = len(heights), len(heights[0])
        # visited 记录 [row, col] 位置是否可以到某条河, 可以为 true, 反之为 false；
        # 假设太平洋的标记为 1, 大西洋为 0
        # ans 用来保存满足条件的答案
        ans, visited = [], [[[False for _ in range(2)] for _ in range(colSize)] for _ in range(rowSize)]
        # 队列, 保存的数据为 [行号, 列号, 标记]
        # 假设太平洋的标记为 1, 大西洋为 0
        queue = deque()
        for row in range(rowSize):
            visited[row][0][1] = True
            visited[row][colSize - 1][0] = True
            queue.append([row, 0, 1])
            queue.append([row, colSize - 1, 0])
        for col in range(0, colSize):
            visited[0][col][1] = True
            visited[rowSize - 1][col][0] = True
            queue.append([0, col, 1])
            queue.append([rowSize - 1, col, 0])

        self.bfs(heights, queue, visited)	# 广度优先遍历

        for row in range(rowSize):
            for col in range(colSize):
                # 如果该位置即可以到太平洋又可以到大西洋, 就放入答案数组
                if visited[row][col][0] and visited[row][col][1]:
                    ans.append([row, col])
        return ans
# 103. 水流问题
# 题目描述：
# 现有一个 N × M 的矩阵，每个单元格包含一个数值，这个数值代表该位置的相对高度。
# 矩阵的左边界和上边界被认为是第一组边界，而矩阵的右边界和下边界被视为第二组边界。
# 矩阵模拟了一个地形，当雨水落在上面时，水会根据地形的倾斜向低处流动，
# 但只能从较高或等高的地点流向较低或等高并且相邻（上下左右方向）的地点。
# 我们的目标是确定那些单元格，从这些单元格出发的水可以达到第一组边界和第二组边界。
# 输入描述：
# 第一行包含两个整数 N 和 M，分别表示矩阵的行数和列数。
# 后续 N 行，每行包含 M 个整数，表示矩阵中的每个单元格的高度。
# 输出描述：
# 输出共有多行，每行输出两个整数，用一个空格隔开，表示可达第一组边界和第二组边界的单元格的坐标，
# 输出顺序任意。
# 输入示例：
# 5 5
# 1 3 1 2 4
# 1 2 1 3 2
# 2 4 7 2 1
# 4 5 6 1 1
# 1 4 1 2 1
# 输出示例：
# 0 4
# 1 3
# 2 2
# 3 0
# 3 1
# 3 2
# 4 0
# 4 1
first = set()
second = set()
directions = [[-1, 0], [0, 1], [1, 0], [0, -1]]

def dfs(i, j, graph, visited, side):
    if visited[i][j]:
        return
    
    visited[i][j] = True
    side.add((i, j))
    
    for x, y in directions:
        new_x = i + x
        new_y = j + y
        if (
            0 <= new_x < len(graph)
            and 0 <= new_y < len(graph[0])
            and int(graph[new_x][new_y]) >= int(graph[i][j])
        ):
            dfs(new_x, new_y, graph, visited, side)

def main():
    global first
    global second
    
    N, M = map(int, input().strip().split())
    graph = []
    for _ in range(N):
        row = input().strip().split()
        graph.append(row)
    
    # 是否可到达第一边界
    visited = [[False] * M for _ in range(N)]
    for i in range(M):
        dfs(0, i, graph, visited, first)
    for i in range(N):
        dfs(i, 0, graph, visited, first)
    
    # 是否可到达第二边界
    visited = [[False] * M for _ in range(N)]
    for i in range(M):
        dfs(N - 1, i, graph, visited, second)
    for i in range(N):
        dfs(i, M - 1, graph, visited, second)

    # 可到达第一边界和第二边界
    res = first & second
    
    for x, y in res:
        print(f"{x} {y}")
    
    
if __name__ == "__main__":
    main()


#9 ??? (Hard) 827.最大人工岛
    # 给你一个大小为 n x n 二进制矩阵 grid 。最多 只能将一格 0 变成 1 。
    # 返回执行此操作后, grid 中最大的岛屿面积是多少？
    # 岛屿 由一组上、下、左、右四个方向相连的 1 形成。
    # 示例 1:
    # 输入: grid = [[1, 0], [0, 1]]
    # 输出: 3
    # 解释: 将一格0变成1, 最终连通两个小岛得到面积为 3 的岛屿。
    # 示例 2:
    # 输入: grid = [[1, 1], [1, 0]]
    # 输出: 4
    # 解释: 将一格0变成1, 岛屿的面积扩大为 4。
    # 示例 3:
    # 输入: grid = [[1, 1], [1, 1]]
    # 输出: 4
    # 解释: 没有0可以让我们变成1, 面积依然为 4。
class Solution:
    def largestIsland(self, grid: List[List[int]]) -> int:
        visited = set()    #标记访问过的位置
        m, n = len(grid), len(grid[0])
        res = 0
        island_size = 0     #用于保存当前岛屿的尺寸
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]] #四个方向
        islands_size = defaultdict(int)  #保存每个岛屿的尺寸

        def dfs(island_num, r, c):
            visited.add((r, c))
            grid[r][c] = island_num     #访问过的位置标记为岛屿编号
            nonlocal island_size
            island_size += 1
            for i in range(4):
                nextR = r + directions[i][0]
                nextC = c + directions[i][1]
                if (nextR not in range(m) or     #行坐标越界
                    nextC not in range(n) or     #列坐标越界
                    (nextR, nextC) in visited):  #坐标已访问
                    continue
                if grid[nextR][nextC] == 1:      #遇到有效坐标, 进入下一个层搜索
                    dfs(island_num, nextR, nextC)

        island_num = 2             #初始岛屿编号设为2,  因为grid里的数据有0和1,  所以从2开始编号
        all_land = True            #标记是否整个地图都是陆地
        for r in range(m):
            for c in range(n):
                if grid[r][c] == 0:
                    all_land = False    #地图里不全是陆地
                if (r, c) not in visited and grid[r][c] == 1:
                    island_size = 0     #遍历每个位置前重置岛屿尺寸为0
                    dfs(island_num, r, c)
                    islands_size[island_num] = island_size #保存当前岛屿尺寸
                    island_num += 1     #下一个岛屿编号加一
        if all_land:
            return m * n     #如果全是陆地,  返回地图面积

        count = 0            #某个位置0变成1后当前岛屿尺寸
        #因为后续计算岛屿面积要往四个方向遍历, 但某2个或3个方向的位置可能同属于一个岛, 
        #所以为避免重复累加, 把已经访问过的岛屿编号加入到这个集合
        visited_island = set() #保存访问过的岛屿
        for r in range(m):
            for c in range(n):
                if grid[r][c] == 0:
                    count = 1        #把由0转换为1的位置计算到面积里
                    visited_island.clear()   #遍历每个位置前清空集合
                    for i in range(4):
                        nearR = r + directions[i][0]
                        nearC = c + directions[i][1]
                        if nearR not in range(m) or nearC not in range(n): #周围位置越界
                            continue
                        if grid[nearR][nearC] in visited_island:  #岛屿已访问
                            continue
                        count += islands_size[grid[nearR][nearC]] #累加连在一起的岛屿面积
                        visited_island.add(grid[nearR][nearC])    #标记当前岛屿已访问
                    res = max(res, count) 
        return res
# 104.建造最大岛屿
# 题目描述：
# 给定一个由 1（陆地）和 0（水）组成的矩阵，你最多可以将矩阵中的一格水变为一块陆地，
# 在执行了此操作之后，矩阵中最大的岛屿面积是多少。
# 岛屿面积的计算方式为组成岛屿的陆地的总数。岛屿是被水包围，
# 并且通过水平方向或垂直方向上相邻的陆地连接而成的。你可以假设矩阵外均被水包围。
# 输入描述：
# 第一行包含两个整数 N, M，表示矩阵的行数和列数。之后 N 行，每行包含 M 个数字，
# 数字为 1 或者 0，表示岛屿的单元格。
# 输出描述：
# 输出一个整数，表示最大的岛屿面积。
# 输入示例：
# 4 5
# 1 1 0 0 0
# 1 1 0 0 0
# 0 0 1 0 0
# 0 0 0 1 1
# 输出示例
# 6
# BFS
from typing import List
from collections import defaultdict

class Solution:
    def __init__(self):
        self.direction = [(1,0),(-1,0),(0,1),(0,-1)]
        self.res = 0
        self.count = 0
        self.idx = 1
        self.count_area = defaultdict(int)

    def max_area_island(self, grid: List[List[int]]) -> int:
        if not grid or len(grid) == 0 or len(grid[0]) == 0:
            return 0

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    self.count = 0
                    self.idx += 1
                    self.dfs(grid,i,j)
        # print(grid)
        self.check_area(grid)
        # print(self.count_area)
        
        if self.check_largest_connect_island(grid=grid):
            return self.res + 1
        return max(self.count_area.values())
    
    def dfs(self,grid,row,col):
        grid[row][col] = self.idx
        self.count += 1
        for dr,dc in self.direction:
            _row = dr + row 
            _col = dc + col 
            if 0<=_row<len(grid) and 0<=_col<len(grid[0]) and grid[_row][_col] == 1:
                self.dfs(grid,_row,_col)
        return

    def check_area(self,grid):
        m, n = len(grid), len(grid[0])
        for row in range(m):
            for col in range(n):
                  self.count_area[grid[row][col]] = self.count_area.get(grid[row][col],0) + 1
        return

    def check_largest_connect_island(self,grid):
        m, n = len(grid), len(grid[0])
        has_connect = False
        for row in range(m):
            for col in range(n):
                if grid[row][col] == 0:
                    has_connect = True
                    area = 0
                    visited = set()
                    for dr, dc in self.direction:
                        _row = row + dr 
                        _col = col + dc
                        if 0<=_row<len(grid) and 0<=_col<len(grid[0]) and grid[_row][_col] != 0 and grid[_row][_col] not in visited:
                            visited.add(grid[_row][_col])
                            area += self.count_area[grid[_row][_col]]
                            self.res = max(self.res, area)
                            
        return has_connect




def main():
    m, n = map(int, input().split())
    grid = []

    for i in range(m):
        grid.append(list(map(int,input().split())))

    
    sol = Solution()
    print(sol.max_area_island(grid))

if __name__ == '__main__':
    main()

import collections

directions = [[-1, 0], [0, 1], [0, -1], [1, 0]]
area = 0

def dfs(i, j, grid, visited, num):
    global area
    
    if visited[i][j]:
        return

    visited[i][j] = True
    grid[i][j] = num  # 标记岛屿号码
    area += 1
    
    for x, y in directions:
        new_x = i + x
        new_y = j + y
        if (
            0 <= new_x < len(grid)
            and 0 <= new_y < len(grid[0])
            and grid[new_x][new_y] == "1"
        ):
            dfs(new_x, new_y, grid, visited, num)
    

def main():
    global area
    
    N, M = map(int, input().strip().split())
    grid = []
    for i in range(N):
        grid.append(input().strip().split())
    visited = [[False] * M for _ in range(N)]
    rec = collections.defaultdict(int)
    
    cnt = 2
    for i in range(N):
        for j in range(M):
            if grid[i][j] == "1":
                area = 0
                dfs(i, j, grid, visited, cnt)
                rec[cnt] = area  # 纪录岛屿面积
                cnt += 1
    
    res = 0
    for i in range(N):
        for j in range(M):
            if grid[i][j] == "0":
                max_island = 1  # 将水变为陆地，故从1开始计数
                v = set()
                for x, y in directions:
                    new_x = i + x
                    new_y = j + y
                    if (
                        0 <= new_x < len(grid)
                        and 0 <= new_y < len(grid[0])
                        and grid[new_x][new_y] != "0"
                        and grid[new_x][new_y] not in v  # 岛屿不可重复
                    ):
                        max_island += rec[grid[new_x][new_y]]
                        v.add(grid[new_x][new_y])
                res = max(res, max_island)

    if res == 0:
        return max(rec.values())  # 无水的情况
    return res
    
if __name__ == "__main__":
    print(main())


#12 (Easy) 463.岛屿的周长
    # 给定一个 row x col 的二维网格地图 grid , 其中：grid[i][j] = 1 表示陆地,  grid[i][j] = 0 表示水域。
    # 网格中的格子 水平和垂直 方向相连（对角线方向不相连）。整个网格被水完全包围, 但其中恰好有一个岛屿（或者说, 一个或多个表示陆地的格子相连组成的岛屿）。
    # 岛屿中没有“湖”（“湖” 指水域在岛屿内部且不和岛屿周围的水相连）。格子是边长为 1 的正方形。网格为长方形, 且宽度和高度均不超过 100 。计算这个岛屿的周长。
    # 输入：grid = [[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]]
    # 输出：16
    # 解释：它的周长是上面图片中的 16 个黄色的边
    # 示例 2：
    # 输入：grid = [[1]]
    # 输出：4
    # 示例 3：
    # 输入：grid = [[1,0]]
    # 输出：4
    # 提示：
    # row == grid.length
    # col == grid[i].length
    # 1 <= row, col <= 100
    # grid[i][j] 为 0 或 1
# 解法一：遍历每一个空格, 遇到岛屿, 计算其上下左右的情况, 遇到水域或者出界的情况, 就可以计算边了。
class Solution:
    def __init__(self):
        self.direction = [[0, 1], [1, 0], [-1, 0], [0, -1]]

    def islandPerimeter(self, grid):
        result = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    for k in range(4):  # Four directions: up, down, left, right
                        x = i + self.direction[k][0]
                        y = j + self.direction[k][1]  # Calculate adjacent coordinates x,y
                        if (x < 0                      # i is on the boundary
                                or x >= len(grid)      # i is on the boundary
                                or y < 0               # j is on the boundary
                                or y >= len(grid[0])   # j is on the boundary
                                or grid[x][y] == 0):  # Position x,y is water
                            result += 1
        return result
# 解法二：计算出总的岛屿数量, 因为有一对相邻两个陆地, 边的总数就减2, 那么在计算出相邻岛屿的数量就可以了。result = 岛屿数量 * 4 - cover * 2;
class Solution:
    def islandPerimeter(self, grid):
        land_count = 0  # Number of land cells
        neighbor_count = 0  # Number of neighboring land cells
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    land_count += 1
                    # Count neighboring land cells above
                    if i - 1 >= 0 and grid[i - 1][j] == 1:
                        neighbor_count += 1
                    # Count neighboring land cells to the left
                    if j - 1 >= 0 and grid[i][j - 1] == 1:
                        neighbor_count += 1
                    # Why not count cells below and to the right? To avoid duplicate counting
        return land_count * 4 - neighbor_count * 2
# *** 
# 扫描每个cell,如果当前位置为岛屿 grid[i][j] == 1,  从当前位置判断四边方向, 如果边界或者是水域, 证明有边界存在, res矩阵的对应cell加一。
class Solution:
    def islandPerimeter(self, grid: List[List[int]]) -> int:

        m = len(grid)
        n = len(grid[0])

        # 创建res二维素组记录答案
        res = [[0] * n for j in range(m)]

        for i in range(m):
            for j in range(len(grid[i])):
                # 如果当前位置为水域, 不做修改或reset res[i][j] = 0
                if grid[i][j] == 0:
                    res[i][j] = 0
                # 如果当前位置为陆地, 往四个方向判断, update res[i][j]
                # 只要上下左右四个边, 都会有一个边
                elif grid[i][j] == 1:
                    if i == 0 or (i > 0 and grid[i-1][j] == 0): # 上边
                        res[i][j] += 1
                    if j == 0 or (j >0 and grid[i][j-1] == 0): # 左边
                        res[i][j] += 1
                    if i == m-1 or (i < m-1 and grid[i+1][j] == 0): # 下边
                        res[i][j] += 1
                    if j == n-1 or (j < n-1 and grid[i][j+1] == 0): # 右边
                        res[i][j] += 1

        # 最后求和res矩阵, 这里其实不一定需要矩阵记录, 可以设置一个variable res 记录边长, 舍矩阵无非是更加形象而已
        ans = sum([sum(row) for row in res])

        return ans
# 106. 岛屿的周长
# 题目描述
# 给定一个由 1（陆地）和 0（水）组成的矩阵，岛屿是被水包围，
# 并且通过水平方向或垂直方向上相邻的陆地连接而成的。
# 你可以假设矩阵外均被水包围。在矩阵中恰好拥有一个岛屿，假设组成岛屿的陆地边长都为 1，
# 请计算岛屿的周长。岛屿内部没有水域。
# 输入描述
# 第一行包含两个整数 N, M，表示矩阵的行数和列数。之后 N 行，每行包含 M 个数字，
# 数字为 1 或者 0，表示岛屿的单元格。
# 输出描述
# 输出一个整数，表示岛屿的周长。
# 输入示例
# 5 5
# 0 0 0 0 0
# 0 1 0 1 0
# 0 1 1 1 0
# 0 1 1 1 0
# 0 0 0 0 0
# 输出示例
# 14
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    # 读取 n 和 m
    n = int(data[0])
    m = int(data[1])
    
    # 初始化 grid
    grid = []
    index = 2
    for i in range(n):
        grid.append([int(data[index + j]) for j in range(m)])
        index += m
    
    sum_land = 0    # 陆地数量
    cover = 0       # 相邻数量

    for i in range(n):
        for j in range(m):
            if grid[i][j] == 1:
                sum_land += 1
                # 统计上边相邻陆地
                if i - 1 >= 0 and grid[i - 1][j] == 1:
                    cover += 1
                # 统计左边相邻陆地
                if j - 1 >= 0 and grid[i][j - 1] == 1:
                    cover += 1
                # 不统计下边和右边，避免重复计算
    
    result = sum_land * 4 - cover * 2
    print(result)

if __name__ == "__main__":
    main()


#10 (Hard) 127.单词接龙
    # 字典 wordList 中从单词 beginWord 和 endWord 的 转换序列 是一个按下述规格形成的序列：
    # 序列中第一个单词是 beginWord 。
    # 序列中最后一个单词是 endWord 。
    # 每次转换只能改变一个字母。
    # 转换过程中的中间单词必须是字典 wordList 中的单词。
    # 给你两个单词 beginWord 和 endWord 和一个字典 wordList , 找到从 beginWord 到 endWord 的 最短转换序列 中的 单词数目 。如果不存在这样的转换序列, 返回 0。
    # 示例 1：
    # 输入：beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
    # 输出：5
    # 解释：一个最短转换序列是 "hit" -> "hot" -> "dot" -> "dog" -> "cog", 返回它的长度 5。
    # 示例 2：
    # 输入：beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log"]
    # 输出：0
    # 解释：endWord "cog" 不在字典中, 所以无法进行转换。
# 首先题目中并没有给出点与点之间的连线, 而是要我们自己去连, 条件是字符只能差一个, 所以判断点与点之间的关系, 要自己判断是不是差一个字符, 如果差一个字符, 那就是有链接。
# 然后就是求起点和终点的最短路径长度, 这里无向图求最短路, 广搜最为合适, 广搜只要搜到了终点, 那么一定是最短的路径。因为广搜就是以起点中心向四周扩散的搜索。
# 本题如果用深搜, 会比较麻烦, 要在到达终点的不同路径中选则一条最短路。 而广搜只要达到终点, 一定是最短路。
# 另外需要有一个注意点：
# 本题是一个无向图, 需要用标记位, 标记着节点是否走过, 否则就会死循环！
# 本题给出集合是数组型的, 可以转成set结构, 查找更快一些
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        wordSet = set(wordList)
        if len(wordSet)== 0 or endWord not in wordSet:
            return 0
        mapping = {beginWord:1}
        queue = deque([beginWord]) 
        while queue:
            word = queue.popleft()
            path = mapping[word]
            for i in range(len(word)):
                word_list = list(word)
                for j in range(26):
                    word_list[i] = chr(ord('a')+j)
                    newWord = "".join(word_list)
                    if newWord == endWord:
                        return path+1
                    if newWord in wordSet and newWord not in mapping:
                        mapping[newWord] = path+1
                        queue.append(newWord)                      
        return 0

# 110. 字符串接龙
    # 题目描述
    # 字典 strList 中从字符串 beginStr 和 endStr 的转换序列是一个按下述规格形成的序列：
    # 序列中第一个字符串是 beginStr。
    # 序列中最后一个字符串是 endStr。
    # 每次转换只能改变一个位置的字符（例如 ftr 可以转化 fty ，但 ftr 不能转化 frx）。
    # 转换过程中的中间字符串必须是字典 strList 中的字符串。
    # beginStr 和 endStr 不在 字典 strList 中
    # 字符串中只有小写的26个字母
    # 给你两个字符串 beginStr 和 endStr 和一个字典 strList，找到从 beginStr 到 endStr 的最短转换序列中的字符串数目。如果不存在这样的转换序列，返回 0。
    # 输入描述
    # 第一行包含一个整数 N，表示字典 strList 中的字符串数量。 第二行包含两个字符串，用空格隔开，分别代表 beginStr 和 endStr。 后续 N 行，每行一个字符串，代表 strList 中的字符串。
    # 输出描述
    # 输出一个整数，代表从 beginStr 转换到 endStr 需要的最短转换序列中的字符串数量。如果不存在这样的转换序列，则输出 0。
    # 输入示例
    # 6
    # abc def
    # efc
    # dbc
    # ebc
    # dec
    # dfc
    # yhn
    # 输出示例
    # 4
    # 提示信息
    # 从 startStr 到 endStr，在 strList 中最短的路径为 abc -> dbc -> dec -> def，所以输出结果为 4
    # 数据范围：
    # 2 <= N <= 500
def judge(s1,s2):
    count=0
    for i in range(len(s1)):
        if s1[i]!=s2[i]:
            count+=1
    return count==1

if __name__=='__main__':
    n=int(input())
    beginstr,endstr=map(str,input().split())
    if beginstr==endstr:
        print(0)
        exit()
    strlist=[]
    for i in range(n):
        strlist.append(input())
    
    # use bfs
    visit=[False for i in range(n)]
    queue=[[beginstr,1]]
    while queue:
        str,step=queue.pop(0)
        if judge(str,endstr):
            print(step+1)
            exit()
        for i in range(n):
            if visit[i]==False and judge(strlist[i],str):
                visit[i]=True
                queue.append([strlist[i],step+1])
    print(0)


#11 ??? (Medium) 841.钥匙和房间
    # 有 N 个房间, 开始时你位于 0 号房间。每个房间有不同的号码：0, 1, 2, ..., N-1, 并且房间里可能有一些钥匙能使你进入下一个房间。
    # 在形式上, 对于每个房间 i 都有一个钥匙列表 rooms[i], 每个钥匙 rooms[i][j] 由 [0,1, ..., N-1] 中的一个整数表示, 其中 N = rooms.length。 钥匙 rooms[i][j] = v 可以打开编号为 v 的房间。
    # 最初, 除 0 号房间外的其余所有房间都被锁住。
    # 你可以自由地在房间之间来回走动。
    # 如果能进入每个房间返回 true, 否则返回 false。
    # 示例 1：
    # 输入: [[1],[2],[3],[]]
    # 输出: true
    # 解释: 我们从 0 号房间开始, 拿到钥匙 1。 之后我们去 1 号房间, 拿到钥匙 2。 然后我们去 2 号房间, 拿到钥匙 3。 最后我们去了 3 号房间。 由于我们能够进入每个房间, 我们返回 true。
    # 示例 2：
    # 输入：[[1,3],[3,0,1],[2],[0]]
    # 输出：false
    # 解释：我们不能进入 2 号房间。
# 本题是一个有向图搜索全路径的问题。 只能用深搜（DFS）或者广搜（BFS）来搜。不能用并查集的方式去解决。
# DFS 深度搜索优先
class Solution:
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        visited = [False for i in range(len(rooms))]

        self.dfs(0, rooms, visited)

        # 检查是否都访问到了
        for i in range(len(visited)):
            if not visited[i] :
                return False
        return True

    def dfs(self, key: int, rooms: List[List[int]]  , visited : List[bool] ) :
        if visited[key] :
            return
        visited[key] = True
        keys = rooms[key]
        for i in range(len(keys)) :
            # 深度优先搜索遍历
            self.dfs(keys[i], rooms, visited)
# BFS 广度搜索优先
class Solution:
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        visited = [False] * len(rooms)
        self.bfs(rooms, 0, visited)
        
        for room in visited:
            if room == False:
                return False
        return True
    
    def bfs(self, rooms, index, visited):
        q = collections.deque()
        q.append(index)

        visited[0] = True

        while len(q) != 0:
            index = q.popleft()
            for nextIndex in rooms[index]:
                if visited[nextIndex] == False:
                    q.append(nextIndex)
                    visited[nextIndex] = True

# 105.有向图的完全联通
    # 【题目描述】
    # 给定一个有向图，包含 N 个节点，节点编号分别为 1，2，...，N。现从 1 号节点开始，如果可以从 1 号节点的边可以到达任何节点，则输出 1，否则输出 -1。
    # 【输入描述】
    # 第一行包含两个正整数，表示节点数量 N 和边的数量 K。 后续 K 行，每行两个正整数 s 和 t，表示从 s 节点有一条边单向连接到 t 节点。
    # 【输出描述】
    # 如果可以从 1 号节点的边可以到达任何节点，则输出 1，否则输出 -1。
    # 【输入示例】
    # 4 4
    # 1 2
    # 2 1
    # 1 3
    # 2 4
    # 【输出示例】
    # 1
    # 【提示信息】
    # 从 1 号节点可以到达任意节点，输出 1。
    # 数据范围：
    # 1 <= N <= 100；
    # 1 <= K <= 2000。
# BFS算法
import collections

path = set()  # 纪录 BFS 所经过之节点

def bfs(root, graph):
    global path
    
    que = collections.deque([root])
    while que:
        cur = que.popleft()
        path.add(cur)
        
        for nei in graph[cur]:
            que.append(nei)
        graph[cur] = []
    return

def main():
    N, K = map(int, input().strip().split())
    graph = collections.defaultdict(list)
    for _ in range(K):
        src, dest = map(int, input().strip().split())
        graph[src].append(dest)
    
    bfs(1, graph)
    if path == {i for i in range(1, N + 1)}:
        return 1
    return -1
        

if __name__ == "__main__":
    print(main())
# DFS
def dfs(graph, key, visited):
    for neighbor in graph[key]:
        if not visited[neighbor]:  # Check if the next node is not visited
            visited[neighbor] = True
            dfs(graph, neighbor, visited)

def main():
    import sys
    input = sys.stdin.read
    data = input().split()

    n = int(data[0])
    m = int(data[1])
    
    graph = [[] for _ in range(n + 1)]
    index = 2
    for _ in range(m):
        s = int(data[index])
        t = int(data[index + 1])
        graph[s].append(t)
        index += 2

    visited = [False] * (n + 1)
    visited[1] = True  # Process node 1 beforehand
    dfs(graph, 1, visited)

    for i in range(1, n + 1):
        if not visited[i]:
            print(-1)
            return
    
    print(1)

if __name__ == "__main__":
    main()




# // 并查集初始化
void init() {
    for (int i = 0; i < n; ++i) {
        father[i] = i;
    }
}

# # // 并查集里寻根的过程
# int find(int u) {
#     if (u == father[u]) return u; // 如果根就是自己, 直接返回
#     else return find(father[u]); // 如果根不是自己, 就根据数组下标一层一层向下找
# }
# // 并查集里寻根的过程
# 除了根节点其他所有节点都挂载根节点下, 这样我们在寻根的时候就很快, 只需要一步, 
# 如果我们想达到这样的效果, 就需要 路径压缩, 将非根节点的所有节点直接指向根节点。 那么在代码层面如何实现呢？
# 我们只需要在递归的过程中, 让 father[u] 接住 递归函数 find(father[u]) 的返回结果。
# 因为 find 函数向上寻找根节点, father[u] 表述 u 的父节点, 那么让 father[u] 直接获取 find函数 返回的根节点, 这样就让节点 u 的父节点 变成根节点。
int find(int u) {
    if (u == father[u]) return u;
    else return father[u] = find(father[u]); // 路径压缩
}

# // 判断 u 和 v是否找到同一个根
bool isSame(int u, int v) {
    u = find(u);
    v = find(v);
    return u == v;
}

# // 将v, u 这条边加入并查集
void join(int u, int v) {
    u = find(u); // 寻找u的根
    v = find(v); // 寻找v的根
    if (u == v) return; // 如果发现根相同, 则说明在一个集合, 不用两个节点相连直接返回
    father[v] = u;
}

# // 将v->u 这条边加入并查集
void join(int u, int v) {
    u = find(u); // 寻找u的根
    v = find(v); // 寻找v的根

    if (rank[u] <= rank[v]) father[u] = v; // rank小的树合入到rank大的树
    else father[v] = u;

    if (rank[u] == rank[v] && u != v) rank[v]++; // 如果两棵树高度相同, 则v的高度+1因为, 方面 if (rank[u] <= rank[v]) father[u] = v; 注意是 <=
}


#13 (Easy) 1971.寻找图中是否存在路径
    # 有一个具有 n个顶点的 双向 图, 其中每个顶点标记从 0 到 n - 1（包含 0 和 n - 1）。图中的边用一个二维整数数组 edges 表示, 其中 edges[i] = [ui, vi] 表示顶点 ui 和顶点 vi 之间的双向边。 每个顶点对由 最多一条 边连接, 并且没有顶点存在与自身相连的边。
    # 请你确定是否存在从顶点 start 开始, 到顶点 end 结束的 有效路径 。
    # 给你数组 edges 和整数 n、start和end, 如果从 start 到 end 存在 有效路径 , 则返回 true, 否则返回 false 。
# PYTHON并查集解法如下：
class Solution:
    def validPath(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        p = [i for i in range(n)]

        def find(i):
            if p[i] != i:
                p[i] = find(p[i])
            return p[i]

        for u, v in edges:
            p[find(u)] = find(v)
        return find(source) == find(destination)


#14 (Medium) 684.冗余连接
    # 树可以看成是一个连通且 无环 的 无向 图。
    # 给定往一棵 n 个节点 (节点值 1～n) 的树中添加一条边后的图。添加的边的两个顶点包含在 1 到 n 中间, 且这条附加的边不属于树中已存在的边。图的信息记录于长度为 n 的二维数组 edges , edges[i] = [ai, bi] 表示图中在 ai 和 bi 之间存在一条边。
    # 请找出一条可以删去的边, 删除后可使得剩余部分是一个有着 n 个节点的树。如果有多个答案, 则返回数组 edges 中最后出现的边。
class Solution:
    def __init__(self):
        """
        初始化
        """
        self.n = 1005
        self.father = [i for i in range(self.n)]

    def find(self, u):
        """
        并查集里寻根的过程
        """
        if u == self.father[u]:
            return u
        self.father[u] = self.find(self.father[u])
        return self.father[u]

    def join(self, u, v):
        """
        将v->u 这条边加入并查集
        """
        u = self.find(u)
        v = self.find(v)
        if u == v : return
        self.father[v] = u
        pass

    def same(self, u, v ):
        """
        判断 u 和 v是否找到同一个根, 本题用不上
        """
        u = self.find(u)
        v = self.find(v)
        return u == v

    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        for i in range(len(edges)):
            if self.same(edges[i][0], edges[i][1]) :
                return edges[i]
            else :
                self.join(edges[i][0], edges[i][1])
        return []

# *** Python简洁写法：
class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        n = len(edges)
        p = [i for i in range(n+1)]
        def find(i):
            if p[i] != i:
                p[i] = find(p[i])
            return p[i]
        for u, v in edges:
            if p[find(u)] == find(v):
                return [u, v]
            p[find(u)] = find(v)


#15 ??? (Hard) 685.冗余连接II
# ??? 为什么需要分两种情况: 一种看入度2的情况, 另一种看是不是成环
    # 在本问题中, 有根树指满足以下条件的 有向 图。该树只有一个根节点, 所有其他节点都是该根节点的后继。该树除了根节点之外的每一个节点都有且只有一个父节点, 而根节点没有父节点。
    # 输入一个有向图, 该图由一个有着 n 个节点（节点值不重复, 从 1 到 n）的树及一条附加的有向边构成。附加的边包含在 1 到 n 中的两个不同顶点间, 这条附加的边不属于树中已存在的边。
    # 结果图是一个以边组成的二维数组 edges 。 每个元素是一对 [ui, vi], 用以表示 有向 图中连接顶点 ui 和顶点 vi 的边, 其中 ui 是 vi 的一个父节点。
    # 返回一条能删除的边, 使得剩下的图是有 n 个节点的有根树。若有多个答案, 返回最后出现在给定二维数组的答案。
# 我们要实现两个最为关键的函数：
# isTreeAfterRemoveEdge() 判断删一个边之后是不是树了
# getRemoveEdge 确定图中一定有了有向环, 那么要找到需要删除的那条边
# 此时应该是用到并查集了, 并查集为什么可以判断 一个图是不是树呢？
# 因为如果两个点所在的边在添加图之前如果就可以在并查集里找到了相同的根, 那么这条边添加上之后 这个图一定不是树了
class Solution:
    def __init__(self):
        self.n = 1010
        self.father = [i for i in range(self.n)]

    def find(self, u: int):
        """
        并查集里寻根的过程
        """
        if u == self.father[u]:
            return u
        self.father[u] = self.find(self.father[u])
        return self.father[u]

    def join(self, u: int, v: int):
        """
        将v->u 这条边加入并查集
        """
        u = self.find(u)
        v = self.find(v)
        if u == v : return
        self.father[v] = u
        pass

    def same(self, u: int, v: int ):
        """
        判断 u 和 v是否找到同一个根, 本题用不上
        """
        u = self.find(u)
        v = self.find(v)
        return u == v

    def init_father(self):
        self.father = [i for i in range(self.n)]
        pass

    def getRemoveEdge(self, edges: List[List[int]]) -> List[int]:
        """
        在有向图里找到删除的那条边, 使其变成树
        """

        self.init_father()
        for i in range(len(edges)):
            if self.same(edges[i][0], edges[i][1]): # 构成有向环了, 就是要删除的边
                return edges[i]
            self.join(edges[i][0], edges[i][1]);
        return []

    def isTreeAfterRemoveEdge(self, edges: List[List[int]], deleteEdge: int) -> bool:
        """
        删一条边之后判断是不是树
        """

        self.init_father()
        for i in range(len(edges)):
            if i == deleteEdge: continue
            if self.same(edges[i][0], edges[i][1]): #  构成有向环了, 一定不是树
                return False
            self.join(edges[i][0], edges[i][1]);
        return True

    def findRedundantDirectedConnection(self, edges: List[List[int]]) -> List[int]:
        inDegree = [0 for i in range(self.n)]

        for i in range(len(edges)):
            inDegree[ edges[i][1] ] += 1

        # 找入度为2的节点所对应的边, 注意要倒序, 因为优先返回最后出现在二维数组中的答案
        towDegree = []
        # i代表第i条边
        for i in range(len(edges))[::-1]:
            if inDegree[edges[i][1]] == 2 :
                towDegree.append(i)

        # 处理图中情况1 和 情况2
        # 如果有入度为2的节点, 那么一定是两条边里删一个, 看删哪个可以构成树
        if len(towDegree) > 0:
            if(self.isTreeAfterRemoveEdge(edges, towDegree[0])) :
                return edges[towDegree[0]]
            return edges[towDegree[1]]

        # 明确没有入度为2的情况, 那么一定有有向环, 找到构成环的边返回就可以了
        return self.getRemoveEdge(edges)


#16 prim算法精讲
    # 题目描述：
    # 在世界的某个区域，有一些分散的神秘岛屿，每个岛屿上都有一种珍稀的资源或者宝藏。国王打算在这些岛屿上建公路，方便运输。
    # 不同岛屿之间，路途距离不同，国王希望你可以规划建公路的方案，如何可以以最短的总公路距离将所有岛屿联通起来。
    # 给定一张地图，其中包括了所有的岛屿，以及它们之间的距离。以最小化公路建设长度，确保可以链接到所有岛屿。
    # 输入描述：
    # 第一行包含两个整数V和E，V代表顶点数，E代表边数。顶点编号是从1到V。例如：V=2，一个有两个顶点，分别是1和2。
    # 接下来共有E行，每行三个整数v1，v2和val，v1和v2为边的起点和终点，val代表边的权值。
    # 输出描述：
    # 输出联通所有岛屿的最小路径总距离
    # 输入示例：
    # 7 11
    # 1 2 1
    # 1 3 1
    # 1 5 2
    # 2 6 1
    # 2 4 2
    # 2 3 2
    # 3 4 1
    # 4 5 1
    # 5 6 2
    # 5 7 1
    # 6 7 1
    # 输出示例：
    # 6
# 接收输入
v, e = list(map(int, input().strip().split()))
# 按照常规的邻接矩阵存储图信息，不可达的初始化为10001
graph = [[10001] * (v+1) for _ in range(v+1)]
for _ in range(e):
    x, y, w = list(map(int, input().strip().split()))
    graph[x][y] = w
    graph[y][x] = w

# 定义加入生成树的标记数组和未加入生成树的最近距离
visited = [False] * (v + 1)
minDist = [10001] * (v + 1)

# 循环 n - 1 次，建立 n - 1 条边
# 从节点视角来看：每次选中一个节点加入树，更新剩余的节点到树的最短距离，
# 这一步其实蕴含了确定下一条选取的边，计入总路程 ans 的计算
for _ in range(1, v + 1):
    min_val = 10002
    cur = -1
    for j in range(1, v + 1):
        if visited[j] == False and minDist[j] < min_val:
            cur = j
            min_val = minDist[j]
    visited[cur] = True
    for j in range(1, v + 1):
        if visited[j] == False and minDist[j] > graph[cur][j]:
            minDist[j] = graph[cur][j]

ans = 0
for i in range(2, v + 1):
    ans += minDist[i]
print(ans)

def prim(v, e, edges):
    import sys
    import heapq

    # 初始化邻接矩阵，所有值初始化为一个大值，表示无穷大
    grid = [[10001] * (v + 1) for _ in range(v + 1)]

    # 读取边的信息并填充邻接矩阵
    for edge in edges:
        x, y, k = edge
        grid[x][y] = k
        grid[y][x] = k

    # 所有节点到最小生成树的最小距离
    minDist = [10001] * (v + 1)

    # 记录节点是否在树里
    isInTree = [False] * (v + 1)

    # Prim算法主循环
    for i in range(1, v):
        cur = -1
        minVal = sys.maxsize

        # 选择距离生成树最近的节点
        for j in range(1, v + 1):
            if not isInTree[j] and minDist[j] < minVal:
                minVal = minDist[j]
                cur = j

        # 将最近的节点加入生成树
        isInTree[cur] = True

        # 更新非生成树节点到生成树的距离
        for j in range(1, v + 1):
            if not isInTree[j] and grid[cur][j] < minDist[j]:
                minDist[j] = grid[cur][j]

    # 统计结果
    result = sum(minDist[2:v+1])
    return result

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().split()
    
    v = int(data[0])
    e = int(data[1])
    
    edges = []
    index = 2
    for _ in range(e):
        x = int(data[index])
        y = int(data[index + 1])
        k = int(data[index + 2])
        edges.append((x, y, k))
        index += 3

    result = prim(v, e, edges)
    print(result)


#17 kruskal算法精讲
    # 题目描述：
    # 在世界的某个区域，有一些分散的神秘岛屿，每个岛屿上都有一种珍稀的资源或者宝藏。国王打算在这些岛屿上建公路，方便运输。
    # 不同岛屿之间，路途距离不同，国王希望你可以规划建公路的方案，如何可以以最短的总公路距离将 所有岛屿联通起来。
    # 给定一张地图，其中包括了所有的岛屿，以及它们之间的距离。以最小化公路建设长度，确保可以链接到所有岛屿。
    # 输入描述：
    # 第一行包含两个整数V 和 E，V代表顶点数，E代表边数 。顶点编号是从1到V。例如：V=2，一个有两个顶点，分别是1和2。
    # 接下来共有 E 行，每行三个整数 v1，v2 和 val，v1 和 v2 为边的起点和终点，val代表边的权值。
    # 输出描述：
    # 输出联通所有岛屿的最小路径总距离
    # 输入示例：
    # 7 11
    # 1 2 1
    # 1 3 1
    # 1 5 2
    # 2 6 1
    # 2 4 2
    # 2 3 2
    # 3 4 1
    # 4 5 1
    # 5 6 2
    # 5 7 1
    # 6 7 1
    # 输出示例：
    # 6
class Edge:
    def __init__(self, l, r, val):
        self.l = l
        self.r = r
        self.val = val

n = 10001
father = list(range(n))

def init():
    global father
    father = list(range(n))

def find(u):
    if u != father[u]:
        father[u] = find(father[u])
    return father[u]

def join(u, v):
    u = find(u)
    v = find(v)
    if u != v:
        father[v] = u

def kruskal(v, edges):
    edges.sort(key=lambda edge: edge.val)
    init()
    result_val = 0

    for edge in edges:
        x = find(edge.l)
        y = find(edge.r)
        if x != y:
            result_val += edge.val
            join(x, y)

    return result_val

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().split()

    v = int(data[0])
    e = int(data[1])

    edges = []
    index = 2
    for _ in range(e):
        v1 = int(data[index])
        v2 = int(data[index + 1])
        val = int(data[index + 2])
        edges.append(Edge(v1, v2, val))
        index += 3

    result_val = kruskal(v, edges)
    print(result_val)


#18 拓扑排序精讲
    # 题目描述：
    # 某个大型软件项目的构建系统拥有 N 个文件，文件编号从 0 到 N - 1，在这些文件中，某些文件依赖于其他文件的内容，这意味着如果文件 A 依赖于文件 B，则必须在处理文件 A 之前处理文件 B （0 <= A, B <= N - 1）。请编写一个算法，用于确定文件处理的顺序。
    # 输入描述：
    # 第一行输入两个正整数 N, M。表示 N 个文件之间拥有 M 条依赖关系。
    # 后续 M 行，每行两个正整数 S 和 T，表示 T 文件依赖于 S 文件。
    # 输出描述：
    # 输出共一行，如果能处理成功，则输出文件顺序，用空格隔开。
    # 如果不能成功处理（相互依赖），则输出 -1。
    # 输入示例：
    # 5 4
    # 0 1
    # 0 2
    # 1 3
    # 2 4
    # 输出示例：
    # 0 1 2 3 4
    # 数据范围：
    # 0 <= N <= 10 ^ 5
    # 1 <= M <= 10 ^ 9
from collections import deque, defaultdict

def topological_sort(n, edges):
    inDegree = [0] * n # inDegree 记录每个文件的入度
    umap = defaultdict(list) # 记录文件依赖关系

    # 构建图和入度表
    for s, t in edges:
        inDegree[t] += 1
        umap[s].append(t)

    # 初始化队列，加入所有入度为0的节点
    queue = deque([i for i in range(n) if inDegree[i] == 0])
    result = []

    while queue:
        cur = queue.popleft()  # 当前选中的文件
        result.append(cur)
        for file in umap[cur]:  # 获取该文件指向的文件
            inDegree[file] -= 1  # cur的指向的文件入度-1
            if inDegree[file] == 0:
                queue.append(file)

    if len(result) == n:
        print(" ".join(map(str, result)))
    else:
        print(-1)


if __name__ == "__main__":
    n, m = map(int, input().split())
    edges = [tuple(map(int, input().split())) for _ in range(m)]
    topological_sort(n, edges)


#19 dijkstra（朴素版）精讲
    # 卡码网：47. 参加科学大会(opens new window)
    # 【题目描述】
    # 小明是一位科学家，他需要参加一场重要的国际科学大会，以展示自己的最新研究成果。
    # 小明的起点是第一个车站，终点是最后一个车站。然而，途中的各个车站之间的道路状况、交通拥堵程度以及可能的自然因素（如天气变化）等不同，这些因素都会影响每条路径的通行时间。
    # 小明希望能选择一条花费时间最少的路线，以确保他能够尽快到达目的地。
    # 【输入描述】
    # 第一行包含两个正整数，第一个正整数 N 表示一共有 N 个公共汽车站，第二个正整数 M 表示有 M 条公路。
    # 接下来为 M 行，每行包括三个整数，S、E 和 V，代表了从 S 车站可以单向直达 E 车站，并且需要花费 V 单位的时间。
    # 【输出描述】
    # 输出一个整数，代表小明从起点到终点所花费的最小时间。
    # 输入示例
    # 7 9
    # 1 2 1
    # 1 3 4
    # 2 3 2
    # 2 4 5
    # 3 4 2
    # 4 5 3
    # 2 6 4
    # 5 7 4
    # 6 7 9
    # 输出示例：12
import sys

def dijkstra(n, m, edges, start, end):
    # 初始化邻接矩阵
    grid = [[float('inf')] * (n + 1) for _ in range(n + 1)]
    for p1, p2, val in edges:
        grid[p1][p2] = val

    # 初始化距离数组和访问数组
    minDist = [float('inf')] * (n + 1)
    visited = [False] * (n + 1)

    minDist[start] = 0  # 起始点到自身的距离为0

    for _ in range(1, n + 1):  # 遍历所有节点
        minVal = float('inf')
        cur = -1

        # 选择距离源点最近且未访问过的节点
        for v in range(1, n + 1):
            if not visited[v] and minDist[v] < minVal:
                minVal = minDist[v]
                cur = v

        if cur == -1:  # 如果找不到未访问过的节点，提前结束
            break

        visited[cur] = True  # 标记该节点已被访问

        # 更新未访问节点到源点的距离
        for v in range(1, n + 1):
            if not visited[v] and grid[cur][v] != float('inf') and minDist[cur] + grid[cur][v] < minDist[v]:
                minDist[v] = minDist[cur] + grid[cur][v]

    return -1 if minDist[end] == float('inf') else minDist[end]

if __name__ == "__main__":
    input = sys.stdin.read
    data = input().split()
    n, m = int(data[0]), int(data[1])
    edges = []
    index = 2
    for _ in range(m):
        p1 = int(data[index])
        p2 = int(data[index + 1])
        val = int(data[index + 2])
        edges.append((p1, p2, val))
        index += 3
    start = 1  # 起点
    end = n    # 终点

    result = dijkstra(n, m, edges, start, end)
    print(result)


#20 dijkstra（堆优化版）精讲
    # 卡码网：47. 参加科学大会(opens new window)
    # 【题目描述】
    # 小明是一位科学家，他需要参加一场重要的国际科学大会，以展示自己的最新研究成果。
    # 小明的起点是第一个车站，终点是最后一个车站。然而，途中的各个车站之间的道路状况、交通拥堵程度以及可能的自然因素（如天气变化）等不同，这些因素都会影响每条路径的通行时间。
    # 小明希望能选择一条花费时间最少的路线，以确保他能够尽快到达目的地。
    # 【输入描述】
    # 第一行包含两个正整数，第一个正整数 N 表示一共有 N 个公共汽车站，第二个正整数 M 表示有 M 条公路。
    # 接下来为 M 行，每行包括三个整数，S、E 和 V，代表了从 S 车站可以单向直达 E 车站，并且需要花费 V 单位的时间。
    # 【输出描述】
    # 输出一个整数，代表小明从起点到终点所花费的最小时间。
    # 输入示例
    # 7 9
    # 1 2 1
    # 1 3 4
    # 2 3 2
    # 2 4 5
    # 3 4 2
    # 4 5 3
    # 2 6 4
    # 5 7 4
    # 6 7 9
    # 输出示例：12
import heapq

class Edge:
    def __init__(self, to, val):
        self.to = to
        self.val = val

def dijkstra(n, m, edges, start, end):
    grid = [[] for _ in range(n + 1)]

    for p1, p2, val in edges:
        grid[p1].append(Edge(p2, val))

    minDist = [float('inf')] * (n + 1)
    visited = [False] * (n + 1)

    pq = []
    heapq.heappush(pq, (0, start))
    minDist[start] = 0

    while pq:
        cur_dist, cur_node = heapq.heappop(pq)

        if visited[cur_node]:
            continue

        visited[cur_node] = True

        for edge in grid[cur_node]:
            if not visited[edge.to] and cur_dist + edge.val < minDist[edge.to]:
                minDist[edge.to] = cur_dist + edge.val
                heapq.heappush(pq, (minDist[edge.to], edge.to))

    return -1 if minDist[end] == float('inf') else minDist[end]

# 输入
n, m = map(int, input().split())
edges = [tuple(map(int, input().split())) for _ in range(m)]
start = 1  # 起点
end = n    # 终点

# 运行算法并输出结果
result = dijkstra(n, m, edges, start, end)
print(result)


#21 Bellman_ford 算法精讲
    # 卡码网：94. 城市间货物运输 I(opens new window)
    # 题目描述
    # 某国为促进城市间经济交流，决定对货物运输提供补贴。共有 n 个编号为 1 到 n 的城市，通过道路网络连接，网络中的道路仅允许从某个城市单向通行到另一个城市，不能反向通行。
    # 网络中的道路都有各自的运输成本和政府补贴，道路的权值计算方式为：运输成本 - 政府补贴。
    # 权值为正表示扣除了政府补贴后运输货物仍需支付的费用；权值为负则表示政府的补贴超过了支出的运输成本，实际表现为运输过程中还能赚取一定的收益。
    # 请找出从城市 1 到城市 n 的所有可能路径中，综合政府补贴后的最低运输成本。
    # 如果最低运输成本是一个负数，它表示在遵循最优路径的情况下，运输过程中反而能够实现盈利。
    # 城市 1 到城市 n 之间可能会出现没有路径的情况，同时保证道路网络中不存在任何负权回路。
    # 负权回路是指一系列道路的总权值为负，这样的回路使得通过反复经过回路中的道路，理论上可以无限地减少总成本或无限地增加总收益。
    # 输入描述
    # 第一行包含两个正整数，第一个正整数 n 表示该国一共有 n 个城市，第二个整数 m 表示这些城市中共有 m 条道路。
    # 接下来为 m 行，每行包括三个整数，s、t 和 v，表示 s 号城市运输货物到达 t 号城市，道路权值为 v（单向图）。
    # 输出描述
    # 如果能够从城市 1 到连通到城市 n， 请输出一个整数，表示运输成本。如果该整数是负数，则表示实现了盈利。如果从城市 1 没有路径可达城市 n，请输出 "unconnected"。
    # 输入示例：
    # 6 7
    # 5 6 -2
    # 1 2 1
    # 5 3 1
    # 2 5 2
    # 2 4 -3
    # 4 6 4
    # 1 3 5
def main():
    n, m = map(int, input().strip().split())
    edges = []
    for _ in range(m):
        src, dest, weight = map(int, input().strip().split())
        edges.append([src, dest, weight])
    
    minDist = [float("inf")] * (n + 1)
    minDist[1] = 0  # 起点处距离为0
    
    for i in range(1, n):
        updated = False
        for src, dest, weight in edges:
            if minDist[src] != float("inf") and minDist[src] + weight < minDist[dest]:
                minDist[dest] = minDist[src] + weight
                updated = True
        if not updated:  # 若边不再更新，即停止回圈
            break
    
    if minDist[-1] == float("inf"):  # 返还终点权重
        return "unconnected"
    return minDist[-1]
    
if __name__ == "__main__":
    print(main())


#22 Bellman_ford 队列优化算法（又名SPFA）
    # 卡码网：94. 城市间货物运输 I(opens new window)
    # 题目描述
    # 某国为促进城市间经济交流，决定对货物运输提供补贴。共有 n 个编号为 1 到 n 的城市，通过道路网络连接，网络中的道路仅允许从某个城市单向通行到另一个城市，不能反向通行。
    # 网络中的道路都有各自的运输成本和政府补贴，道路的权值计算方式为：运输成本 - 政府补贴。
    # 权值为正表示扣除了政府补贴后运输货物仍需支付的费用；权值为负则表示政府的补贴超过了支出的运输成本，实际表现为运输过程中还能赚取一定的收益。
    # 请找出从城市 1 到城市 n 的所有可能路径中，综合政府补贴后的最低运输成本。
    # 如果最低运输成本是一个负数，它表示在遵循最优路径的情况下，运输过程中反而能够实现盈利。
    # 城市 1 到城市 n 之间可能会出现没有路径的情况，同时保证道路网络中不存在任何负权回路。
    # 负权回路是指一系列道路的总权值为负，这样的回路使得通过反复经过回路中的道路，理论上可以无限地减少总成本或无限地增加总收益。
    # 输入描述
    # 第一行包含两个正整数，第一个正整数 n 表示该国一共有 n 个城市，第二个整数 m 表示这些城市中共有 m 条道路。
    # 接下来为 m 行，每行包括三个整数，s、t 和 v，表示 s 号城市运输货物到达 t 号城市，道路权值为 v（单向图）。
    # 输出描述
    # 如果能够从城市 1 到连通到城市 n， 请输出一个整数，表示运输成本。如果该整数是负数，则表示实现了盈利。如果从城市 1 没有路径可达城市 n，请输出 "unconnected"。
    # 输入示例：
    # 6 7
    # 5 6 -2
    # 1 2 1
    # 5 3 1
    # 2 5 2
    # 2 4 -3
    # 4 6 4
    # 1 3 5
import collections

def main():
    n, m = map(int, input().strip().split())
    edges = [[] for _ in range(n + 1)]
    for _ in range(m):
        src, dest, weight = map(int, input().strip().split())
        edges[src].append([dest, weight])
    
    minDist = [float("inf")] * (n + 1)
    minDist[1] = 0
    que = collections.deque([1])
    visited = [False] * (n + 1)
    visited[1] = True
    
    while que:
        cur = que.popleft()
        visited[cur] = False
        for dest, weight in edges[cur]:
            if minDist[cur] != float("inf") and minDist[cur] + weight < minDist[dest]:
                minDist[dest] = minDist[cur] + weight
                if visited[dest] == False:
                    que.append(dest)
                    visited[dest] = True
    
    if minDist[-1] == float("inf"):
        return "unconnected"
    return minDist[-1]

if __name__ == "__main__":
    print(main())


#23 bellman_ford之判断负权回路
    # 卡码网：95. 城市间货物运输 II(opens new window)
    # 【题目描述】
    # 某国为促进城市间经济交流，决定对货物运输提供补贴。共有 n 个编号为 1 到 n 的城市，通过道路网络连接，网络中的道路仅允许从某个城市单向通行到另一个城市，不能反向通行。
    # 网络中的道路都有各自的运输成本和政府补贴，道路的权值计算方式为：运输成本 - 政府补贴。权值为正表示扣除了政府补贴后运输货物仍需支付的费用；
    # 权值为负则表示政府的补贴超过了支出的运输成本，实际表现为运输过程中还能赚取一定的收益。
    # 然而，在评估从城市 1 到城市 n 的所有可能路径中综合政府补贴后的最低运输成本时，存在一种情况：图中可能出现负权回路。
    # 负权回路是指一系列道路的总权值为负，这样的回路使得通过反复经过回路中的道路，理论上可以无限地减少总成本或无限地增加总收益。
    # 为了避免货物运输商采用负权回路这种情况无限的赚取政府补贴，算法还需检测这种特殊情况。
    # 请找出从城市 1 到城市 n 的所有可能路径中，综合政府补贴后的最低运输成本。同时能够检测并适当处理负权回路的存在。
    # 城市 1 到城市 n 之间可能会出现没有路径的情况
    # 【输入描述】
    # 第一行包含两个正整数，第一个正整数 n 表示该国一共有 n 个城市，第二个整数 m 表示这些城市中共有 m 条道路。
    # 接下来为 m 行，每行包括三个整数，s、t 和 v，表示 s 号城市运输货物到达 t 号城市，道路权值为 v。
    # 【输出描述】
    # 如果没有发现负权回路，则输出一个整数，表示从城市 1 到城市 n 的最低运输成本（包括政府补贴）。
    # 如果该整数是负数，则表示实现了盈利。如果发现了负权回路的存在，则输出 "circle"。如果从城市 1 无法到达城市 n，则输出 "unconnected"。
    # 输入示例
    # 4 4
    # 1 2 -1
    # 2 3 1
    # 3 1 -1
    # 3 4 1
    # 输出示例
    # circle
# Bellman-Ford方法求解含有负回路的最短路问题
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    index = 0
    
    n = int(data[index])
    index += 1
    m = int(data[index])
    index += 1
    
    grid = []
    for i in range(m):
        p1 = int(data[index])
        index += 1
        p2 = int(data[index])
        index += 1
        val = int(data[index])
        index += 1
        # p1 指向 p2，权值为 val
        grid.append([p1, p2, val])

    start = 1  # 起点
    end = n    # 终点

    minDist = [float('inf')] * (n + 1)
    minDist[start] = 0
    flag = False

    for i in range(1, n + 1):  # 这里我们松弛n次，最后一次判断负权回路
        for side in grid:
            from_node = side[0]
            to = side[1]
            price = side[2]
            if i < n:
                if minDist[from_node] != float('inf') and minDist[to] > minDist[from_node] + price:
                    minDist[to] = minDist[from_node] + price
            else:  # 多加一次松弛判断负权回路
                if minDist[from_node] != float('inf') and minDist[to] > minDist[from_node] + price:
                    flag = True

    if flag:
        print("circle")
    elif minDist[end] == float('inf'):
        print("unconnected")
    else:
        print(minDist[end])

if __name__ == "__main__":
    main()

# SPFA方法求解含有负回路的最短路问题
from collections import deque
from math import inf

def main():
    n, m = [int(i) for i in input().split()]
    graph = [[] for _ in range(n+1)]
    min_dist = [inf for _ in range(n+1)]
    count = [0 for _ in range(n+1)]  # 记录节点加入队列的次数
    for _ in range(m):
        s, t, v = [int(i) for i in input().split()]
        graph[s].append([t, v])
        
    min_dist[1] = 0  # 初始化
    count[1] = 1
    d = deque([1])
    flag = False
    
    while d:  # 主循环
        cur_node = d.popleft()
        for next_node, val in graph[cur_node]:
            if min_dist[next_node] > min_dist[cur_node] + val:
                min_dist[next_node] = min_dist[cur_node] + val
                count[next_node] += 1
                if next_node not in d:
                    d.append(next_node)
                if count[next_node] == n:  # 如果某个点松弛了n次，说明有负回路
                    flag = True
        if flag:
            break
            
    if flag:
        print("circle")
    else:
        if min_dist[-1] == inf:
            print("unconnected")
        else:
            print(min_dist[-1])


if __name__ == "__main__":
    main()


#24 bellman_ford之单源有限最短路
    # 卡码网：96. 城市间货物运输 III(opens new window)
    # 【题目描述】
    # 某国为促进城市间经济交流，决定对货物运输提供补贴。共有 n 个编号为 1 到 n 的城市，通过道路网络连接，网络中的道路仅允许从某个城市单向通行到另一个城市，不能反向通行。
    # 网络中的道路都有各自的运输成本和政府补贴，道路的权值计算方式为：运输成本 - 政府补贴。
    # 权值为正表示扣除了政府补贴后运输货物仍需支付的费用；
    # 权值为负则表示政府的补贴超过了支出的运输成本，实际表现为运输过程中还能赚取一定的收益。
    # 请计算在最多经过 k 个城市的条件下，从城市 src 到城市 dst 的最低运输成本。
    # 【输入描述】
    # 第一行包含两个正整数，第一个正整数 n 表示该国一共有 n 个城市，第二个整数 m 表示这些城市中共有 m 条道路。
    # 接下来为 m 行，每行包括三个整数，s、t 和 v，表示 s 号城市运输货物到达 t 号城市，道路权值为 v。
    # 最后一行包含三个正整数，src、dst、和 k，src 和 dst 为城市编号，从 src 到 dst 经过的城市数量限制。
    # 【输出描述】
    # 输出一个整数，表示从城市 src 到城市 dst 的最低运输成本，如果无法在给定经过城市数量限制下找到从 src 到 dst 的路径，则输出 "unreachable"，表示不存在符合条件的运输方案。
    # 输入示例：
    # 6 7
    # 1 2 1
    # 2 4 -3
    # 2 5 2
    # 1 3 5
    # 3 5 1
    # 4 6 4
    # 5 6 -2
    # 2 6 1
    # 输出示例：
    # 0
#本题为单源有限最短路问题，同样是 kama94.城市间货物运输I 延伸题目。


#25 Floyd 算法精讲
    # 卡码网：97. 小明逛公园(opens new window)
    # 【题目描述】
    # 小明喜欢去公园散步，公园内布置了许多的景点，相互之间通过小路连接，小明希望在观看景点的同时，能够节省体力，走最短的路径。
    # 给定一个公园景点图，图中有 N 个景点（编号为 1 到 N），以及 M 条双向道路连接着这些景点。每条道路上行走的距离都是已知的。
    # 小明有 Q 个观景计划，每个计划都有一个起点 start 和一个终点 end，表示他想从景点 start 前往景点 end。由于小明希望节省体力，他想知道每个观景计划中从起点到终点的最短路径长度。 请你帮助小明计算出每个观景计划的最短路径长度。
    # 【输入描述】
    # 第一行包含两个整数 N, M, 分别表示景点的数量和道路的数量。
    # 接下来的 M 行，每行包含三个整数 u, v, w，表示景点 u 和景点 v 之间有一条长度为 w 的双向道路。
    # 接下里的一行包含一个整数 Q，表示观景计划的数量。
    # 接下来的 Q 行，每行包含两个整数 start, end，表示一个观景计划的起点和终点。
    # 【输出描述】
    # 对于每个观景计划，输出一行表示从起点到终点的最短路径长度。如果两个景点之间不存在路径，则输出 -1。
    # 【输入示例】
    # 7 3 1 2 4 2 5 6 3 6 8 2 1 2 2 3
    # 【输出示例】
    # 4 -1
    # 【提示信息】
    # 从 1 到 2 的路径长度为 4，2 到 3 之间并没有道路。
    # 1 <= N, M, Q <= 1000.
# 基于三维数组的Floyd
if __name__ == '__main__':
    max_int = 10005  # 设置最大路径，因为边最大距离为10^4

    n, m = map(int, input().split())

    grid = [[[max_int] * (n+1) for _ in range(n+1)] for _ in range(n+1)]  # 初始化三维dp数组

    for _ in range(m):
        p1, p2, w = map(int, input().split())
        grid[p1][p2][0] = w
        grid[p2][p1][0] = w

    # 开始floyd
    for k in range(1, n+1):
        for i in range(1, n+1):
            for j in range(1, n+1):
                grid[i][j][k] = min(grid[i][j][k-1], grid[i][k][k-1] + grid[k][j][k-1])

    # 输出结果
    z = int(input())
    for _ in range(z):
        start, end = map(int, input().split())
        if grid[start][end][n] == max_int:
            print(-1)
        else:
            print(grid[start][end][n])

# 基于二维数组的Floyd
if __name__ == '__main__':
    max_int = 10005  # 设置最大路径，因为边最大距离为10^4

    n, m = map(int, input().split())

    grid = [[max_int]*(n+1) for _ in range(n+1)]  # 初始化二维dp数组

    for _ in range(m):
        p1, p2, val = map(int, input().split())
        grid[p1][p2] = val
        grid[p2][p1] = val

    # 开始floyd
    for k in range(1, n+1):
        for i in range(1, n+1):
            for j in range(1, n+1):
                grid[i][j] = min(grid[i][j], grid[i][k] + grid[k][j])

    # 输出结果
    z = int(input())
    for _ in range(z):
        start, end = map(int, input().split())
        if grid[start][end] == max_int:
            print(-1)
        else:
            print(grid[start][end])


#26 A * 算法精讲 （A star算法）
    # 卡码网：126. 骑士的攻击(opens new window)
    # 题目描述
    # 在象棋中，马和象的移动规则分别是“马走日”和“象走田”。现给定骑士的起始坐标和目标坐标，要求根据骑士的移动规则，计算从起点到达目标点所需的最短步数。
    # 骑士移动规则如图，红色是起始位置，黄色是骑士可以走的地方。
    # 棋盘大小 1000 x 1000（棋盘的 x 和 y 坐标均在 [1, 1000] 区间内，包含边界）
    # 输入描述
    # 第一行包含一个整数 n，表示测试用例的数量。
    # 接下来的 n 行，每行包含四个整数 a1, a2, b1, b2，分别表示骑士的起始位置 (a1, a2) 和目标位置 (b1, b2)。
    # 输出描述
    # 输出共 n 行，每行输出一个整数，表示骑士从起点到目标点的最短路径长度。
    # 输入示例
    # 6
    # 5 2 5 4
    # 1 1 2 2
    # 1 1 8 8
    # 1 1 8 7
    # 2 1 3 3
    # 4 6 4 6




#X27 (Medium) 133.Graph Deep Copy
    # Given a reference to a node within an undirected graph, create a deep copy (clone) of the graph. 
    # The copied graph must be completely independent of the original one. 
    # This means you need to make new nodes for the copied graph instead of 
    # reusing any nodes from the original graph.
from ds import GraphNode

"""
Definition of GraphNode:
class GraphNode:
    def __init__(self, val):
        self.val = val
        self.neighbors = []
"""
def graph_deep_copy(node: GraphNode) -> GraphNode:
    if not node:
        return None
    return dfs(node)

def dfs(node: GraphNode, clone_map = {}) -> GraphNode:
    # If this node was already cloned, then return this previously  
    # cloned node.
    if node in clone_map:
        return clone_map[node]
    # Clone the current node.
    cloned_node = GraphNode(node.val)
    # Store the current clone to ensure it doesn't need to be created 
    # again in future DFS calls.
    clone_map[node] = cloned_node
    # Iterate through the neighbors of the current node to connect  
    # their clones to the current cloned node.
    for neighbor in node.neighbors:
        cloned_neighbor = dfs(neighbor, clone_map)
        cloned_node.neighbors.append(cloned_neighbor)
    return cloned_node


#X28 (Medium) Matrix Infection
    # You are given a matrix where each cell is either:
    # 0: Empty
    # 1: Uninfected
    # 2: Infected
    # With each passing second, every infected cell (2) infects its uninfected neighboring 
    # cells (1) that are 4-directionally adjacent. Determine the number of seconds 
    # required for all uninfected cells to become infected. If this is impossible, return ‐1.
from collections import deque
from typing import List


def matrix_infection(matrix: List[List[int]]) -> int:
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    queue = deque()
    ones = seconds = 0
    # Count the total number of uninfected cells and add each infected 
    # cell to the queue to represent level 0 of the level-order  
    # traversal.
    for r in range(len(matrix)):
        for c in range(len(matrix[0])):
            if matrix[r][c] == 1:
                ones += 1
            elif matrix[r][c] == 2:
                queue.append((r, c))
    # Use level-order traversal to determine how long it takes to 
    # infect the uninfected cells.
    while queue and ones > 0:
        # 1 second passes with each level of the matrix that's explored.
        seconds += 1
        for _ in range(len(queue)):
            r, c = queue.popleft()
            # Infect any neighboring 1s and add them to the queue to be 
            # processed in the next level.
            for d in dirs:
                next_r, next_c = r + d[0], c + d[1]
                if is_within_bounds(next_r, next_c, matrix) and matrix[next_r][next_c] == 1:
                    matrix[next_r][next_c] = 2
                    ones -= 1
                    queue.append((next_r, next_c))
    # If there are still uninfected cells left, return -1. Otherwise, 
    # return the time passed.
    return seconds if ones == 0 else -1

def is_within_bounds(r: int, c: int, matrix: List[List[int]]) -> bool:
    return 0 <= r < len(matrix) and 0 <= c < len(matrix[0])


#X29 (Medium) 785.Bipartite Graph Validation
    # Given an undirected graph, determine if it's bipartite. A graph is bipartite if the nodes 
    # can be colored in one of two colors, so that no two adjacent nodes are the same color.
    # The input is presented as an adjacency list, where graph[i] is a list of all nodes adjacent to node i.
def bipartite_graph_validation(graph: List[List[int]]) -> bool:
    colors = [0] * len(graph)
    # Determine if each graph component is bipartite.
    for i in range(len(graph)):
        if colors[i] == 0 and not dfs(i, 1, graph, colors):
            return False
    return True

def dfs(node: int, color: int, graph: List[List[int]], colors: List[int]) -> bool:
    colors[node] = color
    for neighbor in graph[node]:
        # If the current neighbor has the same color as the current 
        # node, the graph is not bipartite.
        if colors[neighbor] == color:
            return False
        # If the current neighbor is not colored, color it with the 
        # other color and continue the DFS.
        if colors[neighbor] == 0 and not dfs(neighbor, -color, graph, colors):
            return False
    return True


#X30 (Medium) Longest Increasing Path
    # Find the longest strictly increasing path in a matrix of positive integers. 
    # A path is a sequence of cells where each one is 4-directionally adjacent 
    # (up, down, left, or right) to the previous one.
def longest_increasing_path(matrix: List[List[int]]) -> int:
    if not matrix:
        return 0
    res = 0
    m, n = len(matrix), len(matrix[0])
    memo = [[0] * n for _ in range(m)]
    # Find the longest increasing path starting at each cell. The
    # maximum of these is equal to the overall longest increasing
    # path.
    for r in range(m):
        for c in range(n):
            res = max(res, dfs(r, c, matrix, memo))
    return res

def dfs(r: int, c: int, matrix: List[List[int]], memo: List[List[int]]) -> int:
    if memo[r][c] != 0:
        return memo[r][c]
    max_path = 1
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # The longest path starting at the current cell is equal to the
    # longest path of its larger neighboring cells, plus 1.
    for d in dirs:
        next_r, next_c = r + d[0], c + d[1]
        if is_within_bounds(next_r, next_c, matrix) and matrix[next_r][next_c] > matrix[r][c]:
            max_path = max(max_path, 1 + dfs(next_r, next_c, matrix, memo))
    memo[r][c] = max_path
    return max_path

def is_within_bounds(r: int, c: int, matrix: List[List[int]]) -> bool:
    return 0 <= r < len(matrix) and 0 <= c < len(matrix[0])


#X31 (Hard) Shortest Transformation Sequence
    # Given two words, start and end, and a dictionary containing an array of words, 
    # return the length of the shortest transformation sequence to transform start to end. 
    # A transformation sequence is a series of words in which:
    # Each word differs from the preceding word by exactly one letter.
    # Each word in the sequence exists in the dictionary.
    # If no such transformation sequence exists, return 0.
    # Example:
    # Input: start = 'red', end = 'hit',
    #        dictionary = [
    #             'red', 'bed', 'hat', 'rod', 'rad', 'rat', 'hit', 'bad', 'bat'
    #        ]
    # Output: 5
from collections import deque
from typing import List

def shortest_transformation_sequence(start: str, end: str, dictionary: List[str]) -> int:
    dictionary_set = set(dictionary)
    if start not in dictionary_set or end not in dictionary_set:
        return 0
    if start == end:
        return 1
    lower_case_alphabet = 'abcdefghijklmnopqrstuvwxyz'
    queue = deque([start])
    visited = set([start])
    dist = 0
    # Use level-order traversal to find the shortest path from the 
    # start word to the end word.
    while queue:
        for _ in range(len(queue)):
            curr_word = queue.popleft()
            # If we found the end word, we've reached it via the 
            # shortest path.
            if curr_word == end:
                return dist + 1
            # Generate all possible words that have a one-letter 
            # difference to the current word.
            for i in range(len(curr_word)):
                for c in lower_case_alphabet:
                    next_word = curr_word[:i] + c + curr_word[i+1:]
                    # If 'next_word' exists in the dictionary, it's a 
                    # neighbor of the current word. If unvisited, add it
                    # to the queue to be processed in the next level.
                    if next_word in dictionary_set and next_word not in visited:
                        visited.add(next_word)
                        queue.append(next_word)
        dist += 1
    # If there is no way to reach the end node, then no path exists.
    return 0

from collections import deque
from typing import List

def shortest_transformation_sequence_optimized(start: str, end: str, dictionary: List[str]) -> int:
    dictionary_set = set(dictionary)
    if start not in dictionary_set or end not in dictionary_set:
        return 0
    if start == end:
        return 1
    start_queue = deque([start])
    start_visited = {start}
    end_queue = deque([end])
    end_visited = {end}
    level_start = level_end = 0
    # Perform a level-order traversal from the start word and another 
    # from the end word.
    while start_queue and end_queue:
        # Explore the next level of the traversal that starts from the 
        # start word. If it meets the other traversal, the shortest 
        # path between 'start' and 'end' has been found.
        level_start += 1
        if explore_level(start_queue, start_visited, end_visited, dictionary_set):
            return level_start + level_end + 1
        # Explore the next level of the traversal that starts from the  
        # end word.
        level_end += 1
        if explore_level(end_queue, end_visited, start_visited, dictionary_set):
            return level_start + level_end + 1
    # If the traversals never met, then no path exists.
    return 0

# This function explores the next level in the level-order traversal 
# and checks if two searches meet.
def explore_level(queue, visited, other_visited, dictionary_set) -> bool:
    lower_case_alphabet = 'abcdefghijklmnopqrstuvwxyz'
    for _ in range(len(queue)):
        current_word = queue.popleft()
        for i in range(len(current_word)):
            for c in lower_case_alphabet:
                next_word = current_word[:i] + c + current_word[i + 1:]
                # If 'next_word' has been visited during the other
                # traversal, it means both searches have met.
                if next_word in other_visited:
                    return True
                if next_word in dictionary_set and next_word not in visited:
                    visited.add(next_word)
                    queue.append(next_word)
    # If no word has been visited by the other traversal, the searches 
    # have not met yet.
    return False


#X32 (Hard) Merging Communities
    # There are n people numbered from 0 to n - 1 , with each person initially 
    # belonging to a separate community. When two people from different communities connect, 
    # their communities merge into a single community.
    # Your goal is to write two functions:
    # connect(x: int, y: int) -> None: Connects person x with person y and merges their communities.
    # get_community_size(x: int) -> int: Returns the size of the community which person x belongs to.
class UnionFind:
    def __init__(self, size: int):
        self.parent = [i for i in range(size)]
        self.size = [1] * size

    def union(self, x: int, y: int) -> None:
        rep_x, rep_y = self.find(x), self.find(y)
        if rep_x != rep_y:
            # If 'rep_x' represents a larger community, connect
            # 'rep_y 's community to it.
            if self.size[rep_x] > self.size[rep_y]:
                self.parent[rep_y] = rep_x
                self.size[rep_x] += self.size[rep_y]
                # Otherwise, connect 'rep_x's community to 'rep_y'.
            else:
                self.parent[rep_x] = rep_y
                self.size[rep_y] += self.size[rep_x]
        
    def find(self, x: int) -> int:
        if x == self.parent[x]:
            return x
        # Path compression.
        self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def get_size(self, x: int) -> int:
        return self.size[self.find(x)]


class MergingCommunities:
    def __init__(self, n: int):
        self.uf = UnionFind(n)

    def connect(self, x: int, y: int) -> None:
        self.uf.union(x, y)

    def get_community_size(self, x: int) -> int:
        return self.uf.get_size(x)


#X33 (Medium) 207.Prerequisites
    # Given an integer n representing the number of courses labeled from 0 to n - 1, 
    # and an array of prerequisite pairs, determine if it's possible to enroll in all courses.
    # Each prerequisite is represented as a pair [a, b], indicating that course a must be taken before course b.
from collections import defaultdict, deque
from typing import List


def prerequisites(n: int, prerequisites: List[List[int]]) -> bool:
    graph = defaultdict(list)
    in_degrees = [0] * n
    # Represent the graph as an adjacency list and record the in-
    # degree of each course.
    for prerequisite, course in prerequisites:
        graph[prerequisite].append(course)
        in_degrees[course] += 1
    queue = deque()
    # Add all courses with an in-degree of 0 to the queue.
    for i in range(n):
        if in_degrees[i] == 0:
            queue.append(i)
    enrolled_courses = 0
    # Perform topological sort.
    while queue:
        node = queue.popleft()
        enrolled_courses += 1
        for neighbor in graph[node]:
            in_degrees[neighbor] -= 1
            # If the in-degree of a neighboring course becomes 0, add 
            # it to the queue.
            if in_degrees[neighbor] == 0:
                queue.append(neighbor)
    # Return true if we've successfully enrolled in all courses.
    return enrolled_courses == n

class Solution:
    def canFinish(self, numCourses, prerequisites):
        indegree = [0] * numCourses
        adj = [[] for _ in range(numCourses)]

        for prerequisite in prerequisites:
            adj[prerequisite[1]].append(prerequisite[0])
            indegree[prerequisite[0]] += 1

        queue = deque()
        for i in range(numCourses):
            if indegree[i] == 0:
                queue.append(i)

        nodesVisited = 0
        while queue:
            node = queue.popleft()
            nodesVisited += 1

            for neighbor in adj[node]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)

        return nodesVisited == numCourses
# Directed Graph Cycle Detection:
# A cycle exists if a node is revisited within the current recursion path (i.e., during the same DFS traversal).
# Two Tracking Arrays:
# visit[node]: Marks if a node has been fully processed (all descendants checked).
# inStack[node]: Marks if a node is part of the current recursion path.
# Case 1: inStack[node] = True
# Explanation:
# If a node B is revisited while it’s still in the recursion stack (i.e., inStack[B] = True), it means we’ve encountered a path like A → B → ... → A, forming a cycle.

# Case 2: visit[node] = True
# Explanation:
# If a node B is already visited (visit[B] = True) but not in the current stack (inStack[B] = False), it means B was fully processed in a previous DFS call. Since no cycle was found earlier, we skip reprocessing it.

# If node is already present in inStack, we have a cycle. We return true.
# If node is already visited, we return false because we already visited this node and didn't find a cycle earlier.
class Solution:
    def dfs(self, node, adj, visit, inStack):
        # If the node is already in the stack, we have a cycle.
        if inStack[node]:
            return True
        if visit[node]:
            return False
        # Mark the current node as visited and part of current recursion stack.
        visit[node] = True
        inStack[node] = True
        for neighbor in adj[node]:
            if self.dfs(neighbor, adj, visit, inStack):
                return True
        # Remove the node from the stack.
        inStack[node] = False
        return False

    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        adj = [[] for _ in range(numCourses)]
        for prerequisite in prerequisites:
            adj[prerequisite[1]].append(prerequisite[0])

        visit = [False] * numCourses
        inStack = [False] * numCourses
        for i in range(numCourses):
            if self.dfs(i, adj, visit, inStack):
                return False
        return True


#X34 (Hard) Shortest Path
    # Given an integer n representing nodes labeled from 0 to n - 1 in an undirected graph, 
    # and an array of non-negative weighted edges, return an array where each index i 
    # contains the shortest path length from a specified start node to node i. 
    # If a node is unreachable, set its distance to -1.
    # Each edge is represented by a triplet of positive integers: the start node, the end node, and the weight of the edge.
from collections import defaultdict
import heapq
from typing import List


def shortest_path(n: int, edges: List[int], start: int) -> List[int]:
    graph = defaultdict(list)
    distances = [float('inf')] * n
    distances[start] = 0
    # Represent the graph as an adjacency list.
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))
    min_heap = [(0, start)]  # (distance, node)
    # Use Dijkstra's algorithm to find the shortest path between the start node
    # and all other nodes.
    while min_heap:
        curr_dist, curr_node = heapq.heappop(min_heap)
        # If the current distance to this node is greater than the recorded
        # distance, we've already found the shortest distance to this node.
        if curr_dist > distances[curr_node]:
            continue
        # Update the distances of the neighboring nodes.
        for neighbor, weight in graph[curr_node]:
            neighbor_dist = curr_dist + weight
            # Only update the distance if we find a shorter path to this 
            # neighbor.
            if neighbor_dist < distances[neighbor]:
                distances[neighbor] = neighbor_dist
                heapq.heappush(min_heap, (neighbor_dist, neighbor))
    # Convert all infinity values to -1, representing unreachable nodes.
    return [-1 if dist == float('inf') else dist for dist in distances]


#X35 (Medium) Connect the Dots
    # Given a set of points on a plane, determine the minimum cost to connect all these points.
    # The cost of connecting two points is equal to the Manhattan distance between them, 
    # which is calculated as |x1 - x2| + |y1 - y2| for two points (x1, y1) and (x2, y2).
from typing import List

class UnionFind:
    def __init__(self, size):
        self.parent = [i for i in range(size)]
        self.size = [1] * size
 
    def union(self, x, y) -> bool:
        rep_x, rep_y = self.find(x), self.find(y)
        if rep_x != rep_y:
            if self.size[rep_x] > self.size[rep_y]:
                self.parent[rep_y] = rep_x
                self.size[rep_x] += self.size[rep_y]
            else:
                self.parent[rep_x] = rep_y
                self.size[rep_y] += self.size[rep_x]
            # Return True if both groups were merged.
            return True
        # Return False if the points belong to the same group.
        return False 


    def find(self, x) -> int:
        if x == self.parent[x]:
            return x
        self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

def connect_the_dots(points: List[List[int]]) -> int:
    n = len(points)
    # Create and populate a list of all possible edges.
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            # Manhattan distance.
            cost = (abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1]))
            edges.append((cost, i, j))
    # Sort the edges by their cost in ascending order.
    edges.sort()
    uf = UnionFind(n)
    total_cost = edges_added = 0
    # Use Kruskal's algorithm to create the MST and identify its minimum cost.
    for cost, p1, p2 in edges:
        # If the points are not already connected (i.e. their representatives are
        # not the same), connect them, and add the cost to the total cost.
        if uf.union(p1, p2):
            total_cost += cost
            edges_added += 1
            # If n - 1 edges have been added to the MST, the MST is complete.
            if edges_added == n - 1:
                return total_cost