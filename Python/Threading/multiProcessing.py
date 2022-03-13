'''
Exemples sur comment utiliser le multi processing pour les processus qui sont davantage CPU bound

tutoriel corey schafer
https://www.youtube.com/watch?v=fKl2JW_qrso&ab_channel=CoreySchafer
'''

import time
import concurrent.futures
import multiprocessing

# start = time.perf_counter()

def do_something_arg(second):
    print(f'Sleeping {second} sec...')
    time.sleep(second)
    return 'Done Sleeping'


def do_something():
    print(f'Sleeping 1 sec...')
    time.sleep(1)
    print('Done Sleeping')


def main():
    ###########Without Multi processing ###############
    # do_something()
    # do_something()
    ###################################################

    ############ With multi processing ################
    # p1 = multiprocessing.Process(target=do_something)
    # p2 = multiprocessing.Process(target=do_something)

    # Processes do not start automatically, need to start them with .start() method
    # p1.start()
    # p2.start()

    # Processes will execute independently from main thread. Need the .join() methof
    # to hlod the main thread while others are executing and

    # Q: does it freezes all processes or just main threads?
    # Probablement juste le thread ou on a ajouté join(). Le voir comme des branches git?
    # p1.join()
    # p2.join()
    ###################################################
    # finish = time.perf_counter()

    # print(f'Finished in {round(finish - start, 2)} seconds')


    start = time.perf_counter()
    ############# With newer multiprocessing class ######################
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(do_something_arg, 1) for _ in range(10)]

        for f in concurrent.futures.as_completed(results):
            print(f.result())

    finish = time.perf_counter()
    print(f'Finished in {round(finish - start, 2)} seconds')
# Uniquement nécessaire sur Windows pour éviter de créer des appels récursifs
if __name__ == '__main__':
    main()



