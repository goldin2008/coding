# 912. Sort an Array

# Comparison Based Sort
# 1 Selection Sort
class Solution:
    def selection_sort(self, lst: List[int]) -> None:
        """
        Mutates lst so that it is sorted via selecting the minimum element and
        swapping it with the corresponding index
        """
        for i in range(len(lst)):
            min_index = i
            for j in range(i + 1, len(lst)):
                # Update minimum index
                if lst[j] < lst[min_index]:
                    min_index = j

            # Swap current index with minimum element in rest of list
            lst[min_index], lst[i] = lst[i], lst[min_index]
# 2 Bubble Sort
class Solution:
    def bubble_sort(self, lst: List[int]) -> None:
        """
        Mutates lst so that it is sorted via swapping adjacent elements until
        the entire lst is sorted.
        """
        has_swapped = True
        # if no swap occurred, lst is sorted
        while has_swapped:
            has_swapped = False
            for i in range(len(lst) - 1):
                if lst[i] > lst[i + 1]:
                    # Swap adjacent elements
                    lst[i], lst[i + 1] = lst[i + 1], lst[i]
                    has_swapped = True          
# 3 Insertion Sort
class Solution:
    def insertion_sort(self, lst: List[int]) -> None:
        """
        Mutates elements in lst by inserting out of place elements into appropriate
        index repeatedly until lst is sorted
        """
        for i in range(1, len(lst)):
            current_index = i

            while current_index > 0 and lst[current_index - 1] > lst[current_index]:
                # Swap elements that are out of order
                lst[current_index], lst[current_index - 1] = lst[current_index - 1], lst[current_index]
                current_index -= 1
# 4 Heap Sort
class Solution:
    def heap_sort(self, lst: List[int]) -> None:
        """
        Mutates elements in lst by utilizing the heap data structure
        """
        def max_heapify(heap_size, index):
            left, right = 2 * index + 1, 2 * index + 2
            largest = index
            if left < heap_size and lst[left] > lst[largest]:
                largest = left
            if right < heap_size and lst[right] > lst[largest]:
                largest = right
            if largest != index:
                lst[index], lst[largest] = lst[largest], lst[index]
                max_heapify(heap_size, largest)

        # heapify original lst
        for i in range(len(lst) // 2 - 1, -1, -1):
            max_heapify(len(lst), i)

        # use heap to sort elements
        for i in range(len(lst) - 1, 0, -1):
            # swap last element with first element
            lst[i], lst[0] = lst[0], lst[i]
            # note that we reduce the heap size by 1 every iteration
            max_heapify(i, 0)

              
# Approach 1:
# 5 Merge Sort
https://www.geeksforgeeks.org/merge-sort/

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        temp_arr = [0] * len(nums)
        
        # Function to merge two sub-arrays in sorted order.
        def merge(left: int, mid: int, right: int):
            # Calculate the start and sizes of two halves.
            start1 = left
            start2 = mid + 1
            n1 = mid - left + 1
            n2 = right - mid

            # Copy elements of both halves into a temporary array.
            for i in range(n1):
                temp_arr[start1 + i] = nums[start1 + i]
            for i in range(n2):
                temp_arr[start2 + i] = nums[start2 + i]

            # Merge the sub-arrays 'in tempArray' back into the original array 'arr' in sorted order.
            i, j, k = 0, 0, left
            while i < n1 and j < n2:
                if temp_arr[start1 + i] <= temp_arr[start2 + j]:
                    nums[k] = temp_arr[start1 + i]
                    i += 1
                else:
                    nums[k] = temp_arr[start2 + j]
                    j += 1
                k += 1

            # Copy remaining elements
            while i < n1:
                nums[k] = temp_arr[start1 + i]
                i += 1
                k += 1
            while j < n2:
                nums[k] = temp_arr[start2 + j]
                j += 1
                k += 1

        # Recursive function to sort an array using merge sort
        def merge_sort(left: int, right: int):
            if left >= right:
                return
            mid = (left + right) // 2
            # Sort first and second halves recursively.
            merge_sort(left, mid)
            merge_sort(mid + 1, right)
            # Merge the sorted halves.
            merge(left, mid, right)
    
        merge_sort(0, len(nums) - 1)
        return nums
    
# Approach 2:
# 6 Heap Sort
https://www.geeksforgeeks.org/heap-sort/

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        # Function to heapify a subtree (in top-down order) rooted at index i.
        def heapify(n: int, i: int):
            # Initialize largest as root 'i'.
            largest = i;
            left = 2 * i + 1
            right = 2 * i + 2
            # If left child is larger than root.
            if left < n and nums[left] > nums[largest]:
                largest = left
            # If right child is larger than largest so far.
            if right < n and nums[right] > nums[largest]:
                largest = right
            # If largest is not root swap root with largest element
            # Recursively heapify the affected sub-tree (i.e. move down).
            if largest != i:
                nums[i], nums[largest] =  nums[largest], nums[i]
                heapify(n, largest)

        def heap_sort():
            n = len(nums)
            # Build heap; heapify (top-down) all elements except leaf nodes.
            for i in range(n // 2 - 1, -1, -1):
                heapify(n, i)
            # Traverse elements one by one, to move current root to end, and
            for i in range(n - 1, -1, -1):
                nums[0], nums[i] = nums[i], nums[0]
                # call max heapify on the reduced heap.
                heapify(i, 0)

        heap_sort()
        return nums


# NON-Comparison Based Sort
# Approach 3:
# 7 Counting Sort
https://www.geeksforgeeks.org/counting-sort/

class Solution:
def sortArray(self, nums: List[int]) -> List[int]:
    def counting_sort():
        # Create the counting hash map.
        counts = {}
        # Find the minimum and maximum values in the array.
        minVal, maxVal = min(nums), max(nums)
        # Update element's count in the hash map.
        for val in nums:
            counts[val] = counts.get(val, 0) + 1

        index = 0
        # Place each element in its correct position in the array.
        for val in range(minVal, maxVal + 1, 1):
            # Append all 'val's together if they exist.
            while counts.get(val, 0) > 0:
                nums[index] = val
                index += 1
                counts[val] -= 1

    counting_sort()
    return nums

# Approach 4: Radix Sort
# 8 Radix sort function.
class Solution:
    def radix_sort(self, nums: List[int]) -> List[int]:
        # Find the absolute maximum element to find max number of digits.
        max_element = nums[0]
        for val in nums:
            max_element = max(abs(val), max_element)

        max_digits = 0
        while max_element > 0:
            max_digits += 1
            max_element = max_element // 10

        place_value = 1
        
        # Bucket sort function for each place value digit.
        def bucket_sort():
            buckets = [[] for i in range(10)]
            # Store the respective number based on it's digit.
            for val in nums:
                digit = abs(val) / place_value
                digit = int(digit % 10)
                buckets[digit].append(val)

            # Overwrite 'nums' in sorted order of current place digits.
            index = 0
            for digit in range(10):
                for val in buckets[digit]:
                    nums[index] = val
                    index += 1

        # Radix sort, least significant digit place to most significant.      
        for _ in range(max_digits):
            bucket_sort()
            place_value *= 10

        # Seperate out negatives and reverse them. 
        positives = [val for val in nums if val >= 0]
        negatives = [val for val in nums if val < 0]
        negatives.reverse()

        # Final 'arr' will be 'negative' elements, then 'positive' elements.
        return negatives + positives
            
    def sortArray(self, nums: List[int]) -> List[int]:  
        return self.radix_sort(nums)                                                      

# 9 Bucket Sort
class Solution:
    def bucket_sort(self, lst: List[int], K) -> None:
        """
        Sorts a list of integers using K buckets
        """
        buckets = [[] for _ in range(K)]

        # place elements into buckets
        shift = min(lst)
        max_val = max(lst) - shift
        bucket_size = max(1, max_val / K)
        for i, elem in enumerate(lst):
            # same as K * lst[i] / max(lst)
            index = (elem - shift) // bucket_size
            # edge case for max value
            if index == K:
                # put the max value in the last bucket
                buckets[K - 1].append(elem)
            else:
                buckets[index].append(elem)

        # sort individual buckets
        for bucket in buckets:
            bucket.sort()

        # convert sorted buckets into final output
        sorted_array = []
        for bucket in buckets:
            sorted_array.extend(bucket)

        # common practice to mutate original array with sorted elements
        # perfectly fine to just return sorted_array instead
        for i in range(len(lst)):
            lst[i] = sorted_array[i]
