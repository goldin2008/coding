"""
Sorting

> https://www.geeksforgeeks.org/analysis-of-different-sorting-techniques/
> https://realpython.com/sorting-algorithms-python/
> https://www.geeksforgeeks.org/selection-sort/
> https://www.geeksforgeeks.org/python-program-for-heap-sort/
"""
from random import randint
from timeit import repeat

def run_sorting_algorithm(algorithm, array):
    # Set up the context and prepare the call to the specified
    # algorithm using the supplied array. Only import the
    # algorithm function if it's not the built-in `sorted()`.
    setup_code = f"from __main__ import {algorithm}" \
        if algorithm != "sorted" else ""

    stmt = f"{algorithm}({array})"

    # Execute the code ten different times and return the time
    # in seconds that each execution took
    # stmt: This will take the code you want to measure the execution time
    # setup: This will have setup details that need to be executed before stmt
    # number: The stmt will execute as per the number is given here.
    times = repeat(setup=setup_code, stmt=stmt, repeat=3, number=10)

    # Finally, display the name of the algorithm and the
    # minimum time it took to run
    print(f"Algorithm: {algorithm}. Minimum execution time: {min(times)}")


"""
Comparison based sorting

1. Bubble Sort
Bubble Sort is a simple sorting algorithm that repeatedly steps through a list, compares adjacent elements, 
and swaps them if they are in the wrong order. The algorithm gets its name because smaller elements "bubble" 
to the top of the list (beginning) like bubbles in water.

Right end in the list is the sorted.

Time Complexity
Worst-case: O(n²) — When the list is in reverse order.
Best-case: O(n) — When the list is already sorted (with the swapped optimization).
Average-case: O(n²).

Space Complexity
Bubble Sort is an in-place sorting algorithm, meaning it requires only a constant amount of additional space (O(1)).
"""
How Bubble Sort Works
1. Start at the beginning of the list.
2. Compare the first two elements. If the first is greater than the second, swap them.
3. Move to the next pair of elements and repeat the comparison and swap.
4. Continue this process until you reach the end of the list. This completes one pass.
5. Repeat the process for multiple passes until no more swaps are needed (the list is sorted).
# Solution 1
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):  # Number of passes
        swapped = False  # Flag to check if any swaps occurred
        for j in range(0, n-i-1):  # Compare adjacent elements
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]  # Swap
                swapped = True
        if not swapped:  # If no swaps occurred, the list is sorted
            break
    return arr

# Solution 2
def bubble_sort(array):
    n = len(array)
    for i in range(n):
        # Create a flag that will allow the function to
        # terminate early if there's nothing left to sort
        already_sorted = True

        # Start looking at each item of the list one by one,
        # comparing it with its adjacent value. With each
        # iteration, the portion of the array that you look at
        # shrinks because the remaining items have already been
        # sorted.
        for j in range(n - i - 1):
            if array[j] > array[j + 1]:
                # If the item you're looking at is greater than its
                # adjacent value, then swap them
                array[j], array[j + 1] = array[j + 1], array[j]

                # Since you had to swap two elements,
                # set the `already_sorted` flag to `False` so the
                # algorithm doesn't finish prematurely
                already_sorted = False

        # If there were no swaps during the last iteration,
        # the array is already sorted, and you can terminate
        if already_sorted:
            break
    return array

"""
2. Insertion Sort

Insertion Sort is a simple and intuitive sorting algorithm that builds the final sorted list one element at a time. 
It works similarly to how you might sort a hand of playing cards: you pick one card at a time and insert it 
into its correct position in the sorted part of your hand.

Left end in the list is the sorted.

Time Complexity
Worst-case: O(n²) — When the list is in reverse order.
Best-case: O(n) — When the list is already sorted.
Average-case: O(n²).

Space Complexity
Insertion Sort is an in-place sorting algorithm, meaning it requires only a constant amount of additional space (O(1)).
"""
How Insertion Sort Works
1. Start with the second element (assume the first element is already sorted).
2. Compare the current element with the elements in the sorted part of the list.
3. Shift all larger elements to the right to make space for the current element.
4. Insert the current element into its correct position in the sorted part.
5. Repeat until the entire list is sorted.

# Solution 1
def insertion_sort(arr):
    for i in range(1, len(arr)):  # Start from the second element
        key = arr[i]  # Current element to be inserted
        j = i - 1  # Start comparing with the previous element
        while j >= 0 and key < arr[j]:  # Shift elements to the right
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key  # Insert the key in the correct position
    return arr

# Solution 2
def insertion_sort(array):
    # Loop from the second element of the array until
    # the last element
    for i in range(1, len(array)):
        # This is the element we want to position in its
        # correct place
        key_item = array[i]

        # Initialize the variable that will be used to
        # find the correct position of the element referenced
        # by `key_item`
        j = i - 1

        # Run through the list of items (the left
        # portion of the array) and find the correct position
        # of the element referenced by `key_item`. Do this only
        # if `key_item` is smaller than its adjacent values.
        while j >= 0 and array[j] > key_item:
            # Shift the value one position to the left
            # and reposition j to point to the next element
            # (from right to left)
            array[j + 1] = array[j]
            j -= 1

        # When you finish shifting the elements, you can position
        # `key_item` in its correct location
        array[j + 1] = key_item
    return array

"""
3. Selection Sort

Selection Sort is a simple and intuitive sorting algorithm that works by repeatedly selecting 
the smallest (or largest, depending on the order) element from the unsorted portion of the list 
and swapping it with the first unsorted element. This process continues until the entire list is sorted.

Time Complexity
Worst-case: O(n²) — When the list is in reverse order.
Best-case: O(n²) — Even if the list is already sorted, the algorithm still performs all comparisons.
Average-case: O(n²).

Space Complexity
Selection Sort is an in-place sorting algorithm, meaning it requires only a constant amount of additional space (O(1)).
"""
How Selection Sort Works
1. Divide the list into two parts: the sorted part (initially empty) and the unsorted part (initially the entire list).
2. Find the smallest element in the unsorted part.
3. Swap it with the first element of the unsorted part.
4. Move the boundary between the sorted and unsorted parts one element to the right.
5. Repeat until the entire list is sorted.

def selection_sort(arr):
    n = len(arr)
    for i in range(n):  # Number of passes
        min_idx = i  # Assume the first unsorted element is the smallest
        for j in range(i + 1, n):  # Find the smallest element in the unsorted part
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]  # Swap
    return arr

"""
4. Merge Sort

Merge Sort is a highly efficient, stable, and comparison-based sorting algorithm 
that uses the divide-and-conquer approach. It works by recursively dividing the 
list into smaller sublists, sorting them, and then merging the sorted sublists to produce the final sorted list.

merge() has a linear runtime. This leads to a runtime complexity of O(n).
The second step splits the input array recursively and calls merge() for each half.
Since the array is halved until a single element remains, the total number of
halving operations performed by this function is log2n. Since merge() is called
for each half, we get a total runtime of O(nlog2n).
Pros:
Cons: Both bubble sort and insertion sort beat merge sort when sorting a ten-element list.
merge sort use much more memory than bubble sort and insertion sort, which are both able
to sort the list in place. Due to this limitation, you may not want to use merge sort 
to sort large lists in memory-constrained hardware.

Time Complexity
Worst-case: O(n log n) — The list is always divided into two halves, and merging takes linear time.
Best-case: O(n log n) — Even if the list is already sorted, the algorithm still performs all divisions and merges.
Average-case: O(n log n).

Space Complexity
Merge Sort requires additional space for the temporary arrays used during merging, so its space complexity is O(n).

> https://www.programiz.com/dsa/merge-sort
"""
How Merge Sort Works
1. Divide: Split the list into two halves.
2. Conquer: Recursively sort each half.
3. Combine: Merge the two sorted halves into a single sorted list.

# Solution 1
def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2  # Find the middle point
        left_half = arr[:mid]  # Divide the list into two halves
        right_half = arr[mid:]

        merge_sort(left_half)  # Recursively sort the left half
        merge_sort(right_half)  # Recursively sort the right half

        # Merge the sorted halves
        i = j = k = 0
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        # Check if any elements are left in the left half
        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        # Check if any elements are left in the right half
        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1

    return arr

# Solution 2
def merge(left, right):
    # If the first array is empty, then nothing needs
    # to be merged, and you can return the second array as the result
    if len(left) == 0:
        return right

    # If the second array is empty, then nothing needs
    # to be merged, and you can return the first array as the result
    if len(right) == 0:
        return left

    result = []
    index_left = index_right = 0

    # Now go through both arrays until all the elements
    # make it into the resultant array
    while len(result) < len(left) + len(right):
        # The elements need to be sorted to add them to the
        # resultant array, so you need to decide whether to get
        # the next element from the first or the second array
        if left[index_left] <= right[index_right]:
            result.append(left[index_left])
            index_left += 1
        else:
            result.append(right[index_right])
            index_right += 1

        # If you reach the end of either array, then you can
        # add the remaining elements from the other array to
        # the result and break the loop
        if index_right == len(right):
            result += left[index_left:]
            break

        if index_left == len(left):
            result += right[index_right:]
            break

    return result

def merge_sort(array):
    # If the input array contains fewer than two elements,
    # then return it as the result of the function
    if len(array) < 2:
        return array

    midpoint = len(array) // 2

    # Sort the array by recursively splitting the input
    # into two equal halves, sorting each half and merging them
    # together into the final result
    return merge(
        left=merge_sort(array[:midpoint]),
        right=merge_sort(array[midpoint:]))

# Solution 3
def mergeSort(array):
    if len(array) > 1:

        #  r is the point where the array is divided into two subarrays
        r = len(array)//2
        L = array[:r]
        M = array[r:]

        # Sort the two halves
        mergeSort(L)
        mergeSort(M)

        i = j = k = 0

        # Until we reach either end of either L or M, pick larger among
        # elements L and M and place them in the correct position at A[p..r]
        while i < len(L) and j < len(M):
            if L[i] < M[j]:
                array[k] = L[i]
                i += 1
            else:
                array[k] = M[j]
                j += 1
            k += 1

        # When we run out of elements in either L or M,
        # pick up the remaining elements and put in A[p..r]
        while i < len(L):
            array[k] = L[i]
            i += 1
            k += 1

        while j < len(M):
            array[k] = M[j]
            j += 1
            k += 1

"""
5. Quicksort

Quicksort is a highly efficient sorting algorithm that uses a divide-and-conquer approach 
to sort elements. It was developed by Tony Hoare in 1960. 

Time Complexity
Best/Average Case: O(nlogn) - Occurs when the pivot divides the array into roughly equal halves.
Worst Case: O(n^2) - Occurs when the pivot is the smallest or largest element (e.g., already sorted array and pivot is always the first or last element).

Space Complexity
O(logn) for the recursion stack in the best/average case.
O(n) in the worst case (unbalanced partitions).

> https://www.programiz.com/dsa/quick-sort
"""
How Quicksort Works
1. Choose a Pivot: Select an element from the array to serve as the pivot. The choice of pivot can vary (e.g., first element, last element, middle element, or a random element).
2. Partitioning:
  - Rearrange the array so that all elements smaller than the pivot are on its left, and all elements greater than the pivot are on its right.
  - The pivot is now in its correct sorted position.
3. Recursively Sort:
  - Apply the same process recursively to the sub-arrays on the left and right of the pivot.
4. Base Case: The recursion stops when the sub-arrays have one or zero elements, as they are already sorted.

# Solution 1
def quicksort(array):
    if len(array) <= 1:
        return array
    pivot = choose_pivot(array)  # Choose a pivot
    left = [x for x in array if x < pivot]
    middle = [x for x in array if x == pivot]
    right = [x for x in array if x > pivot]
    return quicksort(left) + middle + quicksort(right)

# In-Place Quicksort
# Quicksort can be implemented in-place (without extra space) by swapping elements 
# during partitioning. Here's a Python implementation:
def quicksort_inplace(arr, low, high):
    if low < high:
        pivot_index = partition(arr, low, high)
        quicksort_inplace(arr, low, pivot_index - 1)
        quicksort_inplace(arr, pivot_index + 1, high)

def partition(arr, low, high):
    pivot = arr[high]  # Choose the last element as the pivot
    i = low - 1  # Index of the smaller element
    for j in range(low, high):
        if arr[j] < pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]  # Place the pivot in the correct position
    return i + 1

# Solution 2
from random import randint

def quicksort(array):
    # If the input array contains fewer than two elements,
    # then return it as the result of the function
    if len(array) < 2:
        return array

    low, same, high = [], [], []

    # Select your `pivot` element randomly
    pivot = array[randint(0, len(array) - 1)]

    for item in array:
        # Elements that are smaller than the `pivot` go to
        # the `low` list. Elements that are larger than
        # `pivot` go to the `high` list. Elements that are
        # equal to `pivot` go to the `same` list.
        if item < pivot:
            low.append(item)
        elif item == pivot:
            same.append(item)
        elif item > pivot:
            high.append(item)

    # The final result combines the sorted `low` list
    # with the `same` list and the sorted `high` list
    return quicksort(low) + same + quicksort(high)

# Solution 3
# function to find the partition position
def partition(array, low, high):

  # choose the rightmost element as pivot
  pivot = array[high]

  # pointer for greater element
  i = low - 1

  # traverse through all elements
  # compare each element with pivot
  for j in range(low, high):
    if array[j] <= pivot:
      # if element smaller than pivot is found
      # swap it with the greater element pointed by i
      i = i + 1

      # swapping element at i with element at j
      array[i], array[j] = array[j], array[i]

  # swap the pivot element with the greater element specified by i
  array[i + 1], array[high] = array[high], array[i + 1]

  # return the position from where partition is done
  return i + 1

# function to perform quicksort
def quickSort(array, low, high):
  if low < high:
    # find pivot element such that
    # element smaller than pivot are on the left
    # element greater than pivot are on the right
    pi = partition(array, low, high)

    # recursive call on the left of pivot
    quickSort(array, low, pi - 1)

    # recursive call on the right of pivot
    quickSort(array, pi + 1, high)

"""
6. Heapsort

Heapsort is a comparison-based sorting algorithm that uses a binary heap data structure to sort elements. 
It was invented by J. W. J. Williams in 1964 and later refined by Robert W. Floyd. 
Heapsort is known for its efficiency and in-place sorting capability.

Time Complexity
Heap Construction: O(n) - Building the heap from an unsorted array.
Sorting: O(nlogn) - Extracting elements and restoring the heap.
Overall: O(nlogn).

Space Complexity O(1) - Heapsort is an in-place algorithm.
> https://www.programiz.com/dsa/heap-sort
"""
How Heapsort Works
Heapsort consists of two main phases:
1. Build a Max-Heap:
  - Convert the input array into a max-heap, where the largest element is at the root.
  - This ensures that the largest element is always at the top of the heap.
2. Extract Elements from the Heap:
  - Repeatedly remove the root (largest element) and place it at the end of the array.
  - Restore the heap property after each removal.

Steps in Detail
1. Build the Max-Heap:
  - Start from the last non-leaf node and work upwards.
  - For each node, ensure that it satisfies the max-heap property (parent is greater than or equal to its children).
2. Sort the Array:
  - Swap the root (largest element) with the last element in the heap.
  - Reduce the heap size by 1 (effectively removing the largest element from the heap).
  - Restore the max-heap property for the remaining elements.
  - Repeat until the heap is empty.

def heapify(arr, n, i):
    # 1. Find largest among root and children
    
    largest = i  # Initialize largest as root
    l = 2 * i + 1  # left = 2*i + 1
    r = 2 * i + 2  # right = 2*i + 2
 
    # See if left child of root exists and is
    # greater than root
    if l < n and arr[l] > arr[i]:
        largest = l
 
    # See if right child of root exists and is
    # greater than root
    if r < n and arr[r] > arr[largest]:
        largest = r
    
    # 2. If root is not largest, swap with largest and continue heapifying
    # Change root, if needed
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i] # swap
        # 3. Heapify the root.
        heapify(arr, n, largest)

# The main function to sort an array of given size
def heapsort(arr):
    n = len(arr)
    # Build a maxheap.
    # Since last parent will be at ((n//2)-1) we can start at that location.
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
 
    # One by one extract elements
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]  # swap
        heapify(arr, i, 0) # Heapify root element

"""
Non-comparison based sorting

7. Radix Sort

Radix Sort is a non-comparative sorting algorithm that sorts numbers by processing 
individual digits or groups of digits. It works by distributing elements into "buckets" 
based on the value of each digit, starting from the least significant digit (LSD) or 
the most significant digit (MSD). Radix Sort is efficient for sorting integers or 
strings with fixed-length keys.

Time Complexity
Best/Average/Worst Case: O(d*(n+k)), 
where: n = number of elements. d = number of digits in the largest number,
k = range of input values

Space Complexity
O(n+b), where: n = number of elements. b = base (usually 10 for decimal numbers).

Time Complexity: Counting sort is a linear time sorting algorithm that sort in O(n+k) time when elements are in the range from 1 to k.
Space Complexity: O(n+k)
> https://www.geeksforgeeks.org/radix-sort/
"""
How Radix Sort Works
1. Identify the Maximum Number:
  - Determine the number of digits (k) in the largest number in the input array.
2. Sort by Each Digit:
  - Starting from the least significant digit (LSD) or the most significant digit (MSD), sort the numbers into buckets based on the current digit.
  - After processing each digit, combine the buckets back into a single array.
3. Repeat for All Digits:
  - Continue the process for each digit until all digits have been processed.

Steps in Detail
1. Find the Maximum Number:
  - Traverse the array to find the largest number and determine the number of digits (k).
2. Initialize Buckets:
  - Create 10 buckets (for digits 0–9) to hold numbers during sorting.
3. Sort by Digits:
  - For each digit (from LSD to MSD):
    - Distribute numbers into buckets based on the current digit.
    - Combine the buckets back into the array in order.
4. Repeat Until All Digits Are Processed:
  - Continue the process until all digits have been processed.

# Python program for implementation of Radix Sort
# A function to do counting sort of arr[] according to
# the digit represented by exp.
def countingSort(arr, exp1):
    n = len(arr)
 
    # The output array elements that will have sorted arr
    output = [0] * (n)
 
    # initialize count array as 0
    count = [0] * (10)
 
    # Count occurrences of each digit
    # Store count of occurrences in count[]
    for i in range(n):
        index = (arr[i] // exp) % 10
        count[index] += 1
 
    # Change count[i] so that count[i] now contains actual
    # position of this digit in output array
    for i in range(1, 10):
        count[i] += count[i - 1]
 
    # Build the output array
    # Traverse the input array in reverse order and place each element 
    # in its correct position in the output array using the count array.
    i = n - 1
    while i >= 0:
        index = arr[i] // exp1
        output[count[index % 10] - 1] = arr[i]
        count[index % 10] -= 1
        i -= 1
 
    # Copying the output array to arr[],
    # so that arr now contains sorted numbers
    i = 0
    for i in range(0, len(arr)):
        arr[i] = output[i]
 
# Method to do Radix Sort
def radix_sort(arr):
 
    # Find the maximum number to know number of digits
    max1 = max(arr)
 
    # Do counting sort for every digit. Note that instead
    # of passing digit number, exp is passed. exp is 10^i
    # where i is current digit number
    exp = 1
    while max1 / exp >= 1:
        countingSort(arr, exp)
        exp *= 10

"""
8. Counting Sort

Counting Sort is a non-comparative sorting algorithm that works by counting 
the occurrences of each element in the input array and using this information 
to place the elements in the correct position in the output array. It is efficient 
for sorting integers or elements with a limited range of values.

Time Complexity
Best/Average/Worst Case: O(n+k), where:
n = number of elements. k = range of input values.
Space Complexity
O(n+k): Requires additional space for the count and output arrays.

> https://www.geeksforgeeks.org/counting-sort/
"""
How Counting Sort Works
1. Count Occurrences:
  - Create a count array to store the frequency of each element in the input array.
2. Compute Positions:
  - Modify the count array to store the cumulative count, which determines the position of each element in the output array.
3. Build the Output Array:
  - Place each element in its correct position in the output array using the count array.
4. Copy Back to Original Array:
  - Copy the sorted elements from the output array back to the original array.

Steps in Detail
1. Find the Range:
  - Determine the minimum and maximum values in the input array to define the range (k).
2. Initialize the Count Array:
  - Create a count array of size k+1 (to include all possible values).
3. Count Frequencies:
  - Traverse the input array and increment the count for each element.
4. Compute Cumulative Counts:
  - Modify the count array to store the cumulative count, which represents the position of each element in the output array.
5. Build the Output Array:
  - Traverse the input array in reverse order (to maintain stability) and place each element in its correct position in the output array.
6. Copy Back:
  - Copy the sorted elements from the output array back to the original array.

# Python program for counting sort
# The main function that sort the given string arr[] in 
# alphabetical order
def count_sort(arr):
    # Find the range of input values
    max_val = max(arr)
    min_val = min(arr)
    range_of_elements = max_val - min_val + 1

    # Initialize count and output arrays
    # Create a count array to store count of individual
    # elements and initialize count array as 0
    count = [0] * range_of_elements
    output = [0] * len(arr)
  
    # Store count of each character
    # Count the frequency of each element
    for num in arr:
        count[num - min_val] += 1
  
    # Change count_arr[i] so that count_arr[i] now contains actual
    # position of this element in output array
    # Compute cumulative counts
    for i in range(1, len(count)):
        count[i] += count[i - 1]
  
    # Build the output character array
    for i in range(len(arr) - 1, -1, -1):
        output[count[arr[i] - min_val] - 1] = arr[i]
        count[arr[i] - min_val] -= 1
  
    # Copy the output array to arr, so that arr now
    # contains sorted characters
    for i in range(len(arr)):
        arr[i] = output[i]
  
    return arr

"""
9. Bucket Sort

Bucket Sort is a sorting algorithm that works by distributing elements into a number of 
"buckets" and then sorting the elements within each bucket. It is most effective when 
the input is uniformly distributed over a range, as it allows the elements to be evenly 
distributed across the buckets.

Time Complexity
Best Case: 
O(n+k) - When the input is uniformly distributed and each bucket contains a small number of elements.
Average Case: 
O(n+k) - For uniformly distributed data.
Worst Case: 
O(n^2) - When all elements are placed in a single bucket, and an 
O(n^2) sorting algorithm is used for sorting the bucket.

Space Complexity
O(n+k): Requires additional space for the buckets and the output array.

> https://www.programiz.com/dsa/bucket-sort
> https://www.geeksforgeeks.org/bucket-sort-2/
"""
How Bucket Sort Works
1. Create Buckets:
  - Divide the range of input values into k equally spaced intervals (buckets).
2. Distribute Elements:
  - Place each element into the appropriate bucket based on its value.
3. Sort Individual Buckets:
  - Sort the elements within each bucket using another sorting algorithm (e.g., Insertion Sort).
4. Combine Buckets:
  - Concatenate the sorted buckets to produce the final sorted array.

Steps in Detail
1. Determine the Range:
  - Find the minimum and maximum values in the input array to define the range.
2. Create Buckets:
  - Divide the range into k intervals (buckets) of equal size.
3. Distribute Elements:
  - Traverse the input array and place each element into the appropriate bucket.
4. Sort Buckets:
  - Sort the elements within each bucket using a stable sorting algorithm.
5. Combine Buckets:
  - Concatenate the sorted buckets to produce the final sorted array.

def bucket_sort(arr):
    # Determine the range of input values
    min_val = min(arr)
    max_val = max(arr)
    range_of_elements = max_val - min_val

    # Create buckets
    num_buckets = len(arr)
    buckets = [[] for _ in range(num_buckets)]

    # Distribute elements into buckets
    for num in arr:
        index = int((num - min_val) / range_of_elements * (num_buckets - 1))
        buckets[index].append(num)

    # Sort individual buckets
    for bucket in buckets:
        bucket.sort()

    # Combine buckets into the final sorted array
    sorted_arr = []
    for bucket in buckets:
        sorted_arr.extend(bucket)

    return sorted_arr

# Solution 2 
def insertion_sort(arr):
    for i in range(1, len(arr)):  # Start from the second element
        key = arr[i]  # Current element to be inserted
        j = i - 1  # Start comparing with the previous element
        while j >= 0 and key < arr[j]:  # Shift elements to the right
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key  # Insert the key in the correct position
    return arr

def bucket_sort(x):
    arr = []
    slot_num = 10 # 10 means 10 slots, each
                  # slot's size is 0.1
    for i in range(slot_num):
        arr.append([])
          
    # Put array elements in different buckets 
    for j in x:
        index_b = int(slot_num * j) 
        arr[index_b].append(j)
      
    # Sort individual buckets 
    for i in range(slot_num):
        arr[i] = insertion_sort(arr[i])
          
    # concatenate the result
    k = 0
    for i in range(slot_num):
        for j in range(len(arr[i])):
            x[k] = arr[i][j]
            k += 1
    return x

"""
10. Timsort

Timsort is a hybrid sorting algorithm derived from Merge Sort and Insertion Sort. 
It was designed by Tim Peters in 2002 for use in Python's sorted() and list.sort() functions. 
Timsort is highly efficient for real-world data, as it is optimized to handle various types of input, 
including already sorted or partially sorted data.

the complexity of Timsort is O(n log2n)
Timsort performs exceptionally well on already-sorted or close-to-sorted lists,
leading to a best-case scenario of O(n).
In this case, Timsort clearly beats merge sort and matches the best-case scenario for Quicksort.
But the worst case for Timsort is also O(n log2n), which surpasses Quicksort's O(n2).

Time Complexity
Best Case: O(n) - When the input is already sorted or nearly sorted.
Average Case: O(nlogn) - For random data.
Worst Case: O(nlogn) - Even for highly unstructured data.

Space Complexity
Timsort requires O(n) additional space for merging runs.
"""
How Timsort Works
Timsort operates in two main phases:
1. Divide the Input into Runs:
  - A "run" is a subsequence of the input that is already sorted (either non-decreasing or strictly decreasing).
  - Timsort identifies runs and extends them to a minimum size (called minrun) using Insertion Sort.
2. Merge the Runs:
  - Use a modified Merge Sort to merge the runs efficiently.
  - The merging process is optimized to minimize the number of comparisons and memory usage.

def insertion_sort(array, left=0, right=None):
    if right is None:
        right = len(array) - 1

    # Loop from the element indicated by
    # `left` until the element indicated by `right`
    for i in range(left + 1, right + 1):
        # This is the element we want to position in its
        # correct place
        key_item = array[i]

        # Initialize the variable that will be used to
        # find the correct position of the element referenced
        # by `key_item`
        j = i - 1

        # Run through the list of items (the left
        # portion of the array) and find the correct position
        # of the element referenced by `key_item`. Do this only
        # if the `key_item` is smaller than its adjacent values.
        while j >= left and array[j] > key_item:
            # Shift the value one position to the left
            # and reposition `j` to point to the next element
            # (from right to left)
            array[j + 1] = array[j]
            j -= 1

        # When you finish shifting the elements, position
        # the `key_item` in its correct location
        array[j + 1] = key_item

    return array

def timsort(array):
    min_run = 32
    n = len(array)

    # Start by slicing and sorting small portions of the
    # input array. The size of these slices is defined by
    # your `min_run` size.
    for i in range(0, n, min_run):
        insertion_sort(array, i, min((i + min_run - 1), n - 1))

    # Now you can start merging the sorted slices.
    # Start from `min_run`, doubling the size on
    # each iteration until you surpass the length of
    # the array.
    size = min_run
    while size < n:
        # Determine the arrays that will
        # be merged together
        for start in range(0, n, size * 2):
            # Compute the `midpoint` (where the first array ends
            # and the second starts) and the `endpoint` (where
            # the second array ends)
            midpoint = start + size - 1
            end = min((start + size * 2 - 1), (n-1))

            # Merge the two subarrays.
            # The `left` array should go from `start` to
            # `midpoint + 1`, while the `right` array should
            # go from `midpoint + 1` to `end + 1`.
            merged_array = merge(
                left=array[start:midpoint + 1],
                right=array[midpoint + 1:end + 1])

            # Finally, put the merged array back into
            # your array
            array[start:start + len(merged_array)] = merged_array

        # Each iteration should double the size of your arrays
        size *= 2

    return array

ARRAY_LENGTH = 10000

if __name__ == "__main__":
    # Generate an array of `ARRAY_LENGTH` items consisting
    # of random integer values between 0 and 999

    # array = [randint(0, 1000) for i in range(ARRAY_LENGTH)]
    array = [randint(0, 1000) for i in range(ARRAY_LENGTH)]

    # Call the function using the name of the sorting algorithm
    # and the array you just created
    run_sorting_algorithm(algorithm="sorted", array=array)

    # Call each of the functions
    run_sorting_algorithm(algorithm="bubble_sort", array=array)
# Algorithm: bubble_sort. Minimum execution time: 73.21720498399998

    run_sorting_algorithm(algorithm="selection_sort", array=array)
# Algorithm: selection_sort. Minimum execution time: 31.74471645899996

    run_sorting_algorithm(algorithm="insertion_sort", array=array)
# Algorithm: insertion_sort. Minimum execution time: 56.71029764299999

    run_sorting_algorithm(algorithm="merge_sort", array=array)
# Algorithm: merge_sort. Minimum execution time: 0.6195857160000173

    run_sorting_algorithm(algorithm="heapsort", array=array)
# Algorithm: heapsort. Minimum execution time: 0.48962458400001196

    run_sorting_algorithm(algorithm="quicksort", array=array)
# Algorithm: quicksort. Minimum execution time: 0.11675417600002902

    run_sorting_algorithm(algorithm="timsort", array=array)
# Algorithm: timsort. Minimum execution time: 0.39657199999999193

    run_sorting_algorithm(algorithm="radix_sort", array=array)
# Algorithm: radix_sort. Minimum execution time: 0.15927529199996115

    run_sorting_algorithm(algorithm="count_sort", array=array)
# Algorithm: count_sort. Minimum execution time: 0.04350587499999392

    # run_sorting_algorithm(algorithm="bucket_sort", array=array)
# Algorithm: bucket_sort. Minimum execution time: 0.1



# 912. Sort an Array
https://leetcode.com/explore/learn/card/sorting/

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

#X11 148. Sort List
    # Given the head of a linked list, return the list after sorting it in ascending order.
from ds import ListNode
"""
Definition of ListNode:
class ListNode:
    def __init__(self, val=None, next=None):
        self.val = val
        self.next = next
"""
def sort_linked_list(head: ListNode) -> ListNode:
    # If the linked list is empty or has only one element, it's already
    # sorted.
    if not head or not head.next:
        return head
    # Split the linked list into halves using the fast and slow pointer
    # technique.
    second_head = split_list(head) 
    # Recursively sort both halves.
    first_half_sorted = sort_linked_list(head)
    second_half_sorted = sort_linked_list(second_head)
    # Merge the sorted sublists.
    return merge(first_half_sorted, second_half_sorted)

def split_list(head: ListNode) -> ListNode:
    slow = fast = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    second_head = slow.next
    slow.next = None
    return second_head

def merge(l1: ListNode, l2: ListNode) -> ListNode:
    dummy = ListNode(0)
    # This pointer will be used to append nodes to the tail of the 
    # merged linked list.
    tail = dummy
    # Continually append the node with the smaller value from each 
    # linked list to the merged linked list until one of the linked 
    # lists has no more nodes to merge.
    while l1 and l2:
        if l1.val < l2.val:
            tail.next = l1
            l1 = l1.next
        else:
            tail.next = l2
            l2 = l2.next
        tail = tail.next
    # One of the two linked lists could still have nodes remaining.
    # Attach those nodes to the end of the merged linked list.
    tail.next = l1 or l2
    return dummy.next

#X12 912. Sort an Array
    # Given an array of integers nums, sort the array in ascending order and return it.
    # You must solve the problem without using any built-in functions in O(nlog(n)) 
    # time complexity and with the smallest space complexity possible.
def sort_array_counting_sort(nums: List[int]) -> List[int]:
    if not nums:
        return []
    res = []
    # Count occurrences of each element in 'nums'.
    counts = [0] * (max(nums) + 1)
    for num in nums:
        counts[num] += 1
    # Build the sorted array by appending each index 'i' to it a total 
    # of 'counts[i]' times.
    for i, count in enumerate(counts):
        res.extend([i] * count)
    return res

def sort_array(nums: List[int]) -> List[int]:
    quicksort(nums, 0, len(nums) - 1)
    return nums

def quicksort(nums: List[int], left: int, right: int) -> None:
    # Base case: if the subarray has 0 or 1 element, it's already 
    # sorted.
    if left >= right:
        return
    # Partition the array and retrieve the pivot index.
    pivot_index = partition(nums, left, right)
    # Call quicksort on the left and right parts to recursively sort
    # them.
    quicksort(nums, left, pivot_index - 1)
    quicksort(nums, pivot_index + 1, right)

def partition(nums: List[int], left: int, right: int) -> int:
    pivot = nums[right]
    lo = left
    # Move all numbers less than the pivot to the left, which
    # consequently positions all numbers greater than or equal to the
    # pivot to the right.
    for i in range(left, right):
        if nums[i] < pivot:
            nums[lo], nums[i] = nums[i], nums[lo]
            lo += 1
    # After partitioning, 'lo' will be positioned where the pivot should
    # be. So, swap the pivot number with the number at the 'lo' pointer.
    nums[lo], nums[right] = nums[right], nums[lo]
    return lo

def quicksort_optimized(nums: List[int], left: int, right: int) -> None:
    if left >= right:
        return
    # Choose a pivot at a random index.
    random_index = random.randint(left, right)
    # Swap the randomly chosen pivot with the rightmost element to 
    # position the pivot at the rightmost index.
    nums[random_index], nums[right] = nums[right], nums[random_index]
    pivot_index = partition(nums, left, right)
    quicksort_optimized(nums, left, pivot_index - 1)
    quicksort_optimized(nums, pivot_index + 1, right)

#X13 1985. Find the Kth Largest Integer in the Array
    # You are given an array of strings nums and an integer k. Each string in nums 
    # represents an integer without leading zeros.
    # Return the string that represents the kth largest integer in nums.
    # Note: Duplicate numbers should be counted distinctly. For example, 
    # if nums is ["1","2","2"], "2" is the first largest integer, "2" is the second-largest integer, and "1" is the third-largest integer.
def kth_largest_integer_min_heap(nums: List[int], k: int) -> int:
    min_heap = []
    heapq.heapify(min_heap)
    for num in nums:
        # Ensure the heap has at least 'k' integers.
        if len(min_heap) < k:
            heapq.heappush(min_heap, num)
        # If 'num' is greater than the smallest integer in the heap, pop
        # off this smallest integer from the heap and push in 'num'.
        elif num > min_heap[0]:
            heapq.heappop(min_heap)
            heapq.heappush(min_heap, num)
    return min_heap[0]

def kth_largest_integer_quickselect(nums: List[int], k: int) -> int:
    return quickselect(nums, 0, len(nums) - 1, k)

def quickselect(nums: List[int], left: int, right: int, k: int) -> None:
    n = len(nums)
    if left >= right:
        return nums[left]
    random_index = random.randint(left, right)
    nums[random_index], nums[right] = nums[right], nums[random_index]
    pivot_index = partition(nums, left, right)
    # If the pivot comes before 'n - k', the ('n - k')th smallest 
    # integer is somewhere to its right. Perform quickselect on the 
    # right part.
    if pivot_index < n - k:
        return quickselect(nums, pivot_index + 1, right, k)
    # If the pivot comes after 'n - k', the ('n - k')th smallest integer
    # is somewhere to its left. Perform quickselect on the left part.
    elif pivot_index > n - k:
        return quickselect(nums, left, pivot_index - 1, k)
    # If the pivot is at index 'n - k', it's the ('n - k')th smallest
    # integer.
    else:
        return nums[pivot_index]

def partition(nums: List[int], left: int, right: int) -> int:
    pivot = nums[right]
    lo = left
    for i in range(left, right):
        if nums[i] < pivot:
            nums[lo], nums[i] = nums[i], nums[lo]
            lo += 1
    nums[lo], nums[right] = nums[right], nums[lo]
    return lo

#X14 Dutch National Flag
    # Given an array of 0s, 1s, and 2s representing red, white, and blue, respectively, 
    # sort the array in place so that it resembles the Dutch national flag, with all reds (0s) coming first, followed by whites (1s), and finally blues (2s).
    # Example:
    # Input: nums = [0, 1, 2, 0, 1, 2, 0]
    # Output: [0, 0, 0, 1, 1, 2, 2]
def dutch_national_flag(nums: List[int]) -> None:
    i, left, right = 0, 0, len(nums) - 1
    while i <= right:
        # Swap 0s with the element at the left pointer.
        if nums[i] == 0:
            nums[i], nums[left] = nums[left], nums[i]
            left += 1
            i += 1
        # Swap 2s with the element at the right pointer.
        elif nums[i] == 2:
            nums[i], nums[right] = nums[right], nums[i]
            right -= 1
        else:
            i += 1