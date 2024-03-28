"""

"""
#1 797.所有可能的路径
    # 给你一个有 n 个节点的 有向无环图（DAG），请你找出所有从节点 0 到节点 n-1 的路径并输出（不要求按特定顺序）
    # graph[i] 是一个从节点 i 可以访问的所有节点的列表（即从节点 i 到节点 graph[i][j]存在一条有向边）。
    # 提示：
    # n == graph.length
    # 2 <= n <= 15
    # 0 <= graph[i][j] < n
    # graph[i][j] != i（即不存在自环）
    # graph[i] 中的所有元素 互不相同
    # 保证输入为 有向无环图（DAG）
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

# 2
from collections import deque

dir = [(0, 1), (1, 0), (-1, 0), (0, -1)] # 创建方向元素

def bfs(grid, visited, x, y):
  
  queue = deque() # 初始化队列
  queue.append((x, y)) # 放入第一个元素/起点
  visited[x][y] = True # 标记为访问过的节点
  
  while queue: # 遍历队列里的元素
  
    curx, cury = queue.popleft() # 取出第一个元素
    
    for dx, dy in dir: # 遍历四个方向
    
      nextx, nexty = curx + dx, cury + dy
      
      if nextx < 0 or nextx >= len(grid) or nexty < 0 or nexty >= len(grid[0]): # 越界了，直接跳过
        continue
        
      if not visited[nextx][nexty]: # 如果节点没被访问过  
        queue.append((nextx, nexty)) # 加入队列
        visited[nextx][nexty] = True # 标记为访问过的节点


#3 200. 岛屿数量
    # 给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
    # 岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。
    # 此外，你可以假设该网格的四条边均被水包围。
# DFS 版本一
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
                if nextx < 0 or nextx >= m or nexty < 0 or nexty >= n:  # 越界了，直接跳过
                    continue
                if not visited[nextx][nexty] and grid[nextx][nexty] == '1':  # 没有访问过的同时是陆地的
                    visited[nextx][nexty] = True
                    dfs(nextx, nexty)
        
        for i in range(m):
            for j in range(n):
                if not visited[i][j] and grid[i][j] == '1':
                    visited[i][j] = True
                    result += 1  # 遇到没访问过的陆地，+1
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
            if visited[x][y] or grid[x][y] == '0':
                return  # 终止条件：访问过的节点 或者 遇到海水
            visited[x][y] = True
            for d in dirs:
                nextx = x + d[0]
                nexty = y + d[1]
                if nextx < 0 or nextx >= m or nexty < 0 or nexty >= n:  # 越界了，直接跳过
                    continue
                dfs(nextx, nexty)
        
        for i in range(m):
            for j in range(n):
                if not visited[i][j] and grid[i][j] == '1':
                    result += 1  # 遇到没访问过的陆地，+1
                    dfs(i, j)  # 将与其链接的陆地都标记上 true

        return result
# 我们用三个状态去标记每一个格子
# 0 代表海水
# 1 代表陆地
# 2 代表已经访问的陆地
class Solution:
    def traversal(self, grid, i, j):
        m = len(grid)
        n = len(grid[0])

        if i < 0 or j < 0 or i >= m or j >= n:
            return   # 越界了
        elif grid[i][j] == "2" or grid[i][j] == "0":
            return 

        grid[i][j] = "2"
        self.traversal(grid, i - 1, j) # 往上走
        self.traversal(grid, i + 1, j) # 往下走
        self.traversal(grid, i, j - 1) # 往左走
        self.traversal(grid, i, j + 1) # 往右走

    def numIslands(self, grid: List[List[str]]) -> int:
        res = 0


        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == "1":
                    res += 1
                    self.traversal(grid, i, j)

        return res


#4 BFS solution
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
            for k in range(4):
                next_i = x + self.dirs[k][0]
                next_j = y + self.dirs[k][1]

                if next_i < 0 or next_i >= len(grid):
                    continue 
                if next_j < 0 or next_j >= len(grid[0]):
                    continue
                if visited[next_i][next_j]:
                    continue
                if grid[next_i][next_j] == '0':
                    continue
                q.append((next_i, next_j))
                visited[next_i][next_j] = True


#5 695. 岛屿的最大面积
    # 给你一个大小为 m x n 的二进制矩阵 grid 。
    # 岛屿 是由一些相邻的 1 (代表土地) 构成的组合，这里的「相邻」要求两个 1 必须在 水平或者竖直的四个方向上 相邻。你可以假设 grid 的四个边缘都被 0（代表水）包围着。
    # 岛屿的面积是岛上值为 1 的单元格的数目。
    # 计算并返回 grid 中最大的岛屿面积。如果没有岛屿，则返回面积为 0 。
    # 输入：grid = [[0,0,1,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,1,1,0,1,0,0,0,0,0,0,0,0],[0,1,0,0,1,1,0,0,1,0,1,0,0],[0,1,0,0,1,1,0,0,1,1,1,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,1,1,0,0,0,0]]
    # 输出：6
    # 解释：答案不应该是 11 ，因为岛屿只能包含水平或垂直这四个方向上的 1 。
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


#6 1020. 飞地的数量
    # 给你一个大小为 m x n 的二进制矩阵 grid ，其中 0 表示一个海洋单元格、1 表示一个陆地单元格。
    # 一次 移动 是指从一个陆地单元格走到另一个相邻（上、下、左、右）的陆地单元格或跨过 grid 的边界。
    # 返回网格中 无法 在任意次数的移动中离开网格边界的陆地单元格的数量。
    # 输入：grid = [[0,0,0,0],[1,0,1,0],[0,1,1,0],[0,0,0,0]]
    # 输出：3
    # 解释：有三个 1 被 0 包围。一个 1 没有被包围，因为它在边界上。
    # 输入：grid = [[0,1,1,0],[0,0,1,0],[0,0,1,0],[0,0,0,0]]
    # 输出：0
    # 解释：所有 1 都在边界上或可以到达边界。
# DFS 深度优先遍历
class Solution:
    def __init__(self):
        self.position = [[-1, 0], [0, 1], [1, 0], [0, -1]]	# 四个方向

    # 深度优先遍历，把可以通向边缘部分的 1 全部标记成 true
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

    def numEnclaves(self, grid: List[List[int]]) -> int:
        rowSize, colSize, ans = len(grid), len(grid[0]), 0
        # 标记数组记录每个值为 1 的位置是否可以到达边界，可以为 True，反之为 False
        visited = [[False for _ in range(colSize)] for _ in range(rowSize)]
        # 搜索左边界和右边界，对值为 1 的位置进行深度优先遍历
        for row in range(rowSize):
            if grid[row][0] == 1:
                visited[row][0] = True
                self.dfs(grid, row, 0, visited)
            if grid[row][colSize - 1] == 1:
                visited[row][colSize - 1] = True
                self.dfs(grid, row, colSize - 1, visited)
        # 搜索上边界和下边界，对值为 1 的位置进行深度优先遍历，但是四个角不需要，因为上面遍历过了
        for col in range(1, colSize - 1):
            if grid[0][col] == 1:
                visited[0][col] = True
                self.dfs(grid, 0, col, visited)
            if grid[rowSize - 1][col] == 1:
                visited[rowSize - 1][col] = True
                self.dfs(grid, rowSize - 1, col, visited)
        # 找出矩阵中值为 1 但是没有被标记过的位置，记录答案
        for row in range(rowSize):
            for col in range(colSize):
                if grid[row][col] == 1 and not visited[row][col]:
                    ans += 1
        return ans
# BFS 广度优先遍历
class Solution:
    def __init__(self):
        self.position = [[-1, 0], [0, 1], [1, 0], [0, -1]]	# 四个方向

    # 广度优先遍历，把可以通向边缘部分的 1 全部标记成 true
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


    def numEnclaves(self, grid: List[List[int]]) -> int:
        rowSize, colSize, ans = len(grid), len(grid[0]), 0
        # 标记数组记录每个值为 1 的位置是否可以到达边界，可以为 True，反之为 False
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
        # 搜索上边界和下边界查找 1 存入队列，但是四个角不用遍历，因为上面已经遍历到了
        for col in range(1, colSize - 1):
            if grid[0][col] == 1:
                visited[0][col] = True
                queue.append([0, col])
            if grid[rowSize - 1][col] == 1:
                visited[rowSize - 1][col] = True
                queue.append([rowSize - 1, col])
        self.bfs(grid, queue, visited)	# 广度优先遍历
        # 找出矩阵中值为 1 但是没有被标记过的位置，记录答案
        for row in range(rowSize):
            for col in range(colSize):
                if grid[row][col] == 1 and not visited[row][col]:
                    ans += 1
        return ans


#7 130. 被围绕的区域
    # 给你一个 m x n 的矩阵 board ，由若干字符 'X' 和 'O' ，找到所有被 'X' 围绕的区域，并将这些区域里所有的 'O' 用 'X' 填充。
    # 输入：board = [["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]
    # 输出：[["X","X","X","X"],["X","X","X","X"],["X","X","X","X"],["X","O","X","X"]]
    # 解释：被围绕的区间不会存在于边界上，换句话说，任何边界上的 'O' 都不会被填充为 'X'。 任何不在边界上，或不与边界上的 'O' 相连的 'O' 最终都会被填充为 'X'。如果两个元素在水平或垂直方向相邻，则称它们是“相连”的。
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
        # 从边缘开始，将边缘相连的O改成A。然后遍历所有，将A改成O，O改成X
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


#8 417. 太平洋大西洋水流问题
    # 有一个 m × n 的矩形岛屿，与 太平洋 和 大西洋 相邻。 “太平洋” 处于大陆的左边界和上边界，而 “大西洋” 处于大陆的右边界和下边界。
    # 这个岛被分割成一个由若干方形单元格组成的网格。给定一个 m x n 的整数矩阵 heights ， heights[r][c] 表示坐标 (r, c) 上单元格 高于海平面的高度 。
    # 岛上雨水较多，如果相邻单元格的高度 小于或等于 当前单元格的高度，雨水可以直接向北、南、东、西流向相邻单元格。水可以从海洋附近的任何单元格流入海洋。
    # 返回网格坐标 result 的 2D 列表 ，其中 result[i] = [ri, ci] 表示雨水从单元格 (ri, ci) 流动 既可流向太平洋也可流向大西洋 。
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
# DFS 深度优先遍历
class Solution:
    def __init__(self):
        self.position = [[-1, 0], [0, 1], [1, 0], [0, -1]]	# 四个方向

    # heights：题目给定的二维数组， row：当前位置的行号， col：当前位置的列号
    # sign：记录是哪一条河，两条河中可以一个为 0，一个为 1
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
        # visited 记录 [row, col] 位置是否可以到某条河，可以为 true，反之为 false；
        # 假设太平洋的标记为 1，大西洋为 0
        # ans 用来保存满足条件的答案
        ans, visited = [], [[[False for _ in range(2)] for _ in range(colSize)] for _ in range(rowSize)]
        for row in range(rowSize):
            visited[row][0][1] = True
            visited[row][colSize - 1][0] = True
            self.dfs(heights, row, 0, 1, visited)
            self.dfs(heights, row, colSize - 1, 0, visited)
        for col in range(0, colSize):
            visited[0][col][1] = True
            visited[rowSize - 1][col][0] = True
            self.dfs(heights, 0, col, 1, visited)
            self.dfs(heights, rowSize - 1, col, 0, visited)
        for row in range(rowSize):
            for col in range(colSize):
                # 如果该位置即可以到太平洋又可以到大西洋，就放入答案数组
                if visited[row][col][0] and visited[row][col][1]:
                    ans.append([row, col])
        return ans
# BFS 广度优先遍历
class Solution:
    def __init__(self):
        self.position = [[-1, 0], [0, 1], [1, 0], [0, -1]]

    # heights：题目给定的二维数组，visited：记录这个位置可以到哪条河
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
        # visited 记录 [row, col] 位置是否可以到某条河，可以为 true，反之为 false；
        # 假设太平洋的标记为 1，大西洋为 0
        # ans 用来保存满足条件的答案
        ans, visited = [], [[[False for _ in range(2)] for _ in range(colSize)] for _ in range(rowSize)]
        # 队列，保存的数据为 [行号, 列号, 标记]
        # 假设太平洋的标记为 1，大西洋为 0
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
                # 如果该位置即可以到太平洋又可以到大西洋，就放入答案数组
                if visited[row][col][0] and visited[row][col][1]:
                    ans.append([row, col])
        return ans


#9 827.最大人工岛
    # 给你一个大小为 n x n 二进制矩阵 grid 。最多 只能将一格 0 变成 1 。
    # 返回执行此操作后，grid 中最大的岛屿面积是多少？
    # 岛屿 由一组上、下、左、右四个方向相连的 1 形成。
    # 示例 1:
    # 输入: grid = [[1, 0], [0, 1]]
    # 输出: 3
    # 解释: 将一格0变成1，最终连通两个小岛得到面积为 3 的岛屿。
    # 示例 2:
    # 输入: grid = [[1, 1], [1, 0]]
    # 输出: 4
    # 解释: 将一格0变成1，岛屿的面积扩大为 4。
    # 示例 3:
    # 输入: grid = [[1, 1], [1, 1]]
    # 输出: 4
    # 解释: 没有0可以让我们变成1，面积依然为 4。
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
                if grid[nextR][nextC] == 1:      #遇到有效坐标，进入下一个层搜索
                    dfs(island_num, nextR, nextC)

        island_num = 2             #初始岛屿编号设为2， 因为grid里的数据有0和1， 所以从2开始编号
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
            return m * n     #如果全是陆地， 返回地图面积

        count = 0            #某个位置0变成1后当前岛屿尺寸
        #因为后续计算岛屿面积要往四个方向遍历，但某2个或3个方向的位置可能同属于一个岛，
        #所以为避免重复累加，把已经访问过的岛屿编号加入到这个集合
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


#10 127. 单词接龙
    # 字典 wordList 中从单词 beginWord 和 endWord 的 转换序列 是一个按下述规格形成的序列：
    # 序列中第一个单词是 beginWord 。
    # 序列中最后一个单词是 endWord 。
    # 每次转换只能改变一个字母。
    # 转换过程中的中间单词必须是字典 wordList 中的单词。
    # 给你两个单词 beginWord 和 endWord 和一个字典 wordList ，找到从 beginWord 到 endWord 的 最短转换序列 中的 单词数目 。如果不存在这样的转换序列，返回 0。
    # 示例 1：
    # 输入：beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
    # 输出：5
    # 解释：一个最短转换序列是 "hit" -> "hot" -> "dot" -> "dog" -> "cog", 返回它的长度 5。
    # 示例 2：
    # 输入：beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log"]
    # 输出：0
    # 解释：endWord "cog" 不在字典中，所以无法进行转换。
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


#11 841.钥匙和房间
    # 有 N 个房间，开始时你位于 0 号房间。每个房间有不同的号码：0，1，2，...，N-1，并且房间里可能有一些钥匙能使你进入下一个房间。
    # 在形式上，对于每个房间 i 都有一个钥匙列表 rooms[i]，每个钥匙 rooms[i][j] 由 [0,1，...，N-1] 中的一个整数表示，其中 N = rooms.length。 钥匙 rooms[i][j] = v 可以打开编号为 v 的房间。
    # 最初，除 0 号房间外的其余所有房间都被锁住。
    # 你可以自由地在房间之间来回走动。
    # 如果能进入每个房间返回 true，否则返回 false。
    # 示例 1：
    # 输入: [[1],[2],[3],[]]
    # 输出: true
    # 解释: 我们从 0 号房间开始，拿到钥匙 1。 之后我们去 1 号房间，拿到钥匙 2。 然后我们去 2 号房间，拿到钥匙 3。 最后我们去了 3 号房间。 由于我们能够进入每个房间，我们返回 true。
    # 示例 2：
    # 输入：[[1,3],[3,0,1],[2],[0]]
    # 输出：false
    # 解释：我们不能进入 2 号房间。
# DFS 深度搜索优先
class Solution:
    def dfs(self, key: int, rooms: List[List[int]]  , visited : List[bool] ) :
        if visited[key] :
            return
        visited[key] = True
        keys = rooms[key]
        for i in range(len(keys)) :
            # 深度优先搜索遍历
            self.dfs(keys[i], rooms, visited)

    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        visited = [False for i in range(len(rooms))]

        self.dfs(0, rooms, visited)

        # 检查是否都访问到了
        for i in range(len(visited)):
            if not visited[i] :
                return False
        return True
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


#12 463. 岛屿的周长
    # 给定一个 row x col 的二维网格地图 grid ，其中：grid[i][j] = 1 表示陆地， grid[i][j] = 0 表示水域。
    # 网格中的格子 水平和垂直 方向相连（对角线方向不相连）。整个网格被水完全包围，但其中恰好有一个岛屿（或者说，一个或多个表示陆地的格子相连组成的岛屿）。
    # 岛屿中没有“湖”（“湖” 指水域在岛屿内部且不和岛屿周围的水相连）。格子是边长为 1 的正方形。网格为长方形，且宽度和高度均不超过 100 。计算这个岛屿的周长。
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
# 扫描每个cell,如果当前位置为岛屿 grid[i][j] == 1， 从当前位置判断四边方向，如果边界或者是水域，证明有边界存在，res矩阵的对应cell加一。
class Solution:
    def islandPerimeter(self, grid: List[List[int]]) -> int:

        m = len(grid)
        n = len(grid[0])

        # 创建res二维素组记录答案
        res = [[0] * n for j in range(m)]

        for i in range(m):
            for j in range(len(grid[i])):
                # 如果当前位置为水域，不做修改或reset res[i][j] = 0
                if grid[i][j] == 0:
                    res[i][j] = 0
                # 如果当前位置为陆地，往四个方向判断，update res[i][j]
                elif grid[i][j] == 1:
                    if i == 0 or (i > 0 and grid[i-1][j] == 0):
                        res[i][j] += 1
                    if j == 0 or (j >0 and grid[i][j-1] == 0):
                        res[i][j] += 1
                    if i == m-1 or (i < m-1 and grid[i+1][j] == 0):
                        res[i][j] += 1
                    if j == n-1 or (j < n-1 and grid[i][j+1] == 0):
                        res[i][j] += 1

        # 最后求和res矩阵，这里其实不一定需要矩阵记录，可以设置一个variable res 记录边长，舍矩阵无非是更加形象而已
        ans = sum([sum(row) for row in res])

        return ans


#13 1971. 寻找图中是否存在路径
    # 有一个具有 n个顶点的 双向 图，其中每个顶点标记从 0 到 n - 1（包含 0 和 n - 1）。图中的边用一个二维整数数组 edges 表示，其中 edges[i] = [ui, vi] 表示顶点 ui 和顶点 vi 之间的双向边。 每个顶点对由 最多一条 边连接，并且没有顶点存在与自身相连的边。
    # 请你确定是否存在从顶点 start 开始，到顶点 end 结束的 有效路径 。
    # 给你数组 edges 和整数 n、start和end，如果从 start 到 end 存在 有效路径 ，则返回 true，否则返回 false 。
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


#14 684.冗余连接
    # 树可以看成是一个连通且 无环 的 无向 图。
    # 给定往一棵 n 个节点 (节点值 1～n) 的树中添加一条边后的图。添加的边的两个顶点包含在 1 到 n 中间，且这条附加的边不属于树中已存在的边。图的信息记录于长度为 n 的二维数组 edges ，edges[i] = [ai, bi] 表示图中在 ai 和 bi 之间存在一条边。
    # 请找出一条可以删去的边，删除后可使得剩余部分是一个有着 n 个节点的树。如果有多个答案，则返回数组 edges 中最后出现的边。
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
        判断 u 和 v是否找到同一个根，本题用不上
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
#Python简洁写法：
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


#15 685.冗余连接II
    # 在本问题中，有根树指满足以下条件的 有向 图。该树只有一个根节点，所有其他节点都是该根节点的后继。该树除了根节点之外的每一个节点都有且只有一个父节点，而根节点没有父节点。
    # 输入一个有向图，该图由一个有着 n 个节点（节点值不重复，从 1 到 n）的树及一条附加的有向边构成。附加的边包含在 1 到 n 中的两个不同顶点间，这条附加的边不属于树中已存在的边。
    # 结果图是一个以边组成的二维数组 edges 。 每个元素是一对 [ui, vi]，用以表示 有向 图中连接顶点 ui 和顶点 vi 的边，其中 ui 是 vi 的一个父节点。
    # 返回一条能删除的边，使得剩下的图是有 n 个节点的有根树。若有多个答案，返回最后出现在给定二维数组的答案。
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
        判断 u 和 v是否找到同一个根，本题用不上
        """
        u = self.find(u)
        v = self.find(v)
        return u == v

    def init_father(self):
        self.father = [i for i in range(self.n)]
        pass

    def getRemoveEdge(self, edges: List[List[int]]) -> List[int]:
        """
        在有向图里找到删除的那条边，使其变成树
        """

        self.init_father()
        for i in range(len(edges)):
            if self.same(edges[i][0], edges[i][1]): # 构成有向环了，就是要删除的边
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
            if self.same(edges[i][0], edges[i][1]): #  构成有向环了，一定不是树
                return False
            self.join(edges[i][0], edges[i][1]);
        return True

    def findRedundantDirectedConnection(self, edges: List[List[int]]) -> List[int]:
        inDegree = [0 for i in range(self.n)]

        for i in range(len(edges)):
            inDegree[ edges[i][1] ] += 1

        # 找入度为2的节点所对应的边，注意要倒序，因为优先返回最后出现在二维数组中的答案
        towDegree = []
        for i in range(len(edges))[::-1]:
            if inDegree[edges[i][1]] == 2 :
                towDegree.append(i)

        # 处理图中情况1 和 情况2
        # 如果有入度为2的节点，那么一定是两条边里删一个，看删哪个可以构成树
        if len(towDegree) > 0:
            if(self.isTreeAfterRemoveEdge(edges, towDegree[0])) :
                return edges[towDegree[0]]
            return edges[towDegree[1]]

        # 明确没有入度为2的情况，那么一定有有向环，找到构成环的边返回就可以了
        return self.getRemoveEdge(edges)
