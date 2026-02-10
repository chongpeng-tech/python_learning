def count_long_subarray(A):
    '''
    Input:  A     | Python Tuple of positive integers
    Output: count | number of longest increasing subarrays of A
    '''
    count = 0
    ##################
    # YOUR CODE HERE #
    ##################
    current_len = 1
    max_len = 0
    
    for i in range(1, len(A)):
        if A[i] > A[i - 1]:
            current_len += 1
        else:
            if current_len > max_len:
                max_len = current_len
                current_len = 1
                count = 1
            elif current_len == max_len:
                count += 1
                current_len = 1
            else:
                current_len = 1
        
    if current_len > max_len:
        
        max_len = current_len
        current_len = 1
        count = 1
    elif current_len == max_len:
        count += 1
        current_len = 1
    else:
        current_len = 1            
    
    return count
