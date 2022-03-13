import threading
import time
import multiprocessing
'''
Goal is to separate a process on several thread to gain speed

https://realpython.com/python-concurrency/
'''
# Need to measure time of each process
NUM_PROCESSES = 50

# Without threading
# time
for i in range(NUM_PROCESSES):
    # instructions
    time.sleep(0.5)
# time first measrure

# with threading
# time start
for i in range(NUM_PROCESSES):