# import OS module
import os

def get_file_list(): 
    # Get the list of all files and directories
    path = "./leetcode/"
    dir_list = os.listdir(path) 
    print("Files and directories in '", path, "' :")
    # prints all files
    print(len(dir_list))

    print("Python Program to print list the files in a directory.")
    Direc = path
    files = os.listdir(Direc)
    files = [f for f in files if os.path.isfile(Direc+'/'+f)] #Filtering only the files.
    # print(*files, sep="\n")
    print(len(files))

    return files


import re

def count_file(files):
    patterns = ['_AR_', '_LL_', '_HM_', '_ST_',
               '_PP_', '_QU_', '_BS_', '_BT_',
               '_GD_', '_MS_', '_DP_', '_EX_']

    # result = re.match(pattern, test_string)
    # indices = [i for i, x in enumerate(files) if re.search(pattern, x)]
    # indices = [x for i, x in enumerate(files) if re.findall(pattern, x)]
    tot = 0
    for p in patterns:
        indices = [x for x in files if re.findall(p, x)]
        print(p, len(indices))
        tot += len(indices)
    
    print('Total: ', tot)

#01 数组 AR | 5 (5)
#02 链表 LL | 7 (7)
#03 哈希表 HM | 8 (8)
#04 字符串 ST | 5 (7)
#05 双指针 PP | 9 (10)
#06 栈和队列 QU | 7 (7)

#07 二叉树 BS| 29 (40)
#08 回溯 BT | 15 (16)
#09 贪心 GD | 18 (17)
#10 动态规划 DP | 37 (41)

#11 单调栈 MS | 5 (5)
#12 图 GR | (15)
#13 额外 EX | 35 (36)
#14 总数 | 180 (214)

if __name__ == "__main__":
    files = get_file_list()
    count_file(files)
