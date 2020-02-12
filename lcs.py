from difflib import SequenceMatcher
import time
import numpy as np

Ecoli_1 = open('lib\\Ecoli_1.txt','r')
Ecoli_2 = open('lib\\Ecoli_2.txt','r')
Hepatitis_1 = open('lib\\Hepatitis_1.txt','r')
Hepatitis_2 = open('lib\\Hepatitis_2.txt','r')

def longestSubstring(a, b):
    # initialize SequenceMatcher object with
    # input string
    #print(a)
    #print(b)
    i, j, k = SequenceMatcher(autojunk=False, a=a, b=b).find_longest_match(0, len(a), 0, len(b))

    # find match of longest sub-string
    # output will be like Match(a=0, b=0, size=5)

    # print longest substring
    #print(a[i:i+k])
    return i,j,k

if __name__ == '__main__':
    now = time.time()
    print(longestSubstring(Hepatitis_1.read().replace('\n',''), Hepatitis_2.read().replace('\n','')))
    print(time.time()-now)