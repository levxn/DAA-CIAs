import itertools
import numpy as np

def matrix(a, b, match_score=3, gap_cost=2):
    H = np.zeros((len(a) + 1, len(b) + 1), np.int)

    for i, j in itertools.product(range(1, H.shape[0]), range(1, H.shape[1])):
        match = H[i - 1, j - 1] + (match_score if a[i - 1] == b[j - 1] else - match_score)
        delete = H[i - 1, j] - gap_cost
        insert = H[i, j - 1] - gap_cost
        H[i, j] = max(match, delete, insert, 0)
    return H

def traceback(H, b, b_='', old_i=0):
    # flip H to get index of **last** occurrence of H.max() with np.argmax()
    H_flip = np.flip(np.flip(H, 0), 1)
    i_, j_ = np.unravel_index(H_flip.argmax(), H_flip.shape)
    i, j = np.subtract(H.shape, (i_ + 1, j_ + 1))  # (i, j) are **last** indexes of H.max()
    if H[i, j] == 0:
        return b_, j
    b_ = b[j - 1] + '-' + b_ if old_i - i > 1 else b[j - 1] + b_
    return traceback(H[0:i, 0:j], b, b_, i)

def smith_waterman(s1, s2, match_score=3, gap_cost=2):
    s1, s2 = s1.upper(), s2.upper()
    H = matrix(a, b, match_score, gap_cost)
    b_, pos = traceback(H, b)
    return pos, pos + len(b_)

 # prints correct scoring matrix from Wikipedia example
print(matrix('GGTTGACTA', 'TGTTACGG'))
print("\n")
a, b = 'ggttggaccttaccaa', 'ttggttaaccggcaca'
H = matrix(a, b)
print(traceback(H, b)) # ('gtt-ac', 1)
print("\n")
a, b = 'GGTTGGACCTTACCAA', 'TTGGTTAACCGGCACA'
start, end = smith_waterman(a, b)
print(a[start:end])     # GTTGAC