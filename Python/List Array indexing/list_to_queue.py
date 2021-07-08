'''
How to use effectively list as a replacement of queues to allow indexing

Pseudocode:
create a list
set a few parameters for the max size of the list

in a loop, add a few elements

result: to see if the 
'''


MAX_SIZE = 20
SEQUENCE_LEN = 5
q_list = []

for i in range(2*MAX_SIZE):
    # Should increase from 0 to max by sequences of 5 numbers
    number = i // SEQUENCE_LEN
    q_list.append(number)
    index = len(q_list) - 1
    if len(q_list) >= MAX_SIZE:
        while q_list[-1] == q_list[index]:
            index -= 1
        
        print(f'Max of list is {q_list[-1]}, min is {q_list[0]} and first lower value is {q_list[index]} at {index}')
        del q_list[0]
    print('size of q_list: ', len(q_list))