def findMaxSubArray(A):

    """
    Реализация алгоритма Кадане.
    Находит непрерывный подмассив в массиве, 
    содержащий хотя бы одно число, 
    который имеет наибольшую сумму.  
    """
    best_sum = float('-inf')
    best_start = best_end = 0
    current_sum = 0
    for current_end, x in enumerate(A):
        if current_sum <= 0:
            current_start = current_end
            current_sum = x
            
        else:
            current_sum += x

        if current_sum > best_sum:
            best_sum = current_sum
            best_start = current_start
            best_end = current_end + 1
    return A[best_start: best_end]