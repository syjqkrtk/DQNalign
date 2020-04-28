from difflib import SequenceMatcher
import time
import numpy as np

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
