import numpy as np
from hashlib import sha256
from functools import lru_cache

# close homoglyphs in Ariel and monospace (default textbox and input fonts)
# chosen using https://www.irongeek.com/homoglyph-attack-generator.php
unicode_pairs = [('abcdefghijklmnopqrstuvwxyz', 'аbϲdеfɡhіϳklmnοрqrѕtuvwхуz'),
                 ('ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'ΑΒϹDΕFGΗΙЈΚLΜΝΟΡQRЅΤUVWΧΥΖ')]

char_dict = {}
for str1, str2 in unicode_pairs:
    # print(str1, str2)
    for char1, char2 in zip(str1, str2):
        if char1 != char2:
            # print(char1, char2)
            char_dict[char1] = char2
            
def selected(x, seed, prop=0.2):
    s = '%s secret %s' % (x, seed)
    hash = sha256(s.encode())
    seed = np.frombuffer(hash.digest(), dtype='uint32')
    np.random.seed(seed)
    
    return np.random.random() < prop
    
            
def replace_all(x, seed):
    s = '%s %d' % (x, seed)
    hash = sha256(s.encode())
    seed = np.frombuffer(hash.digest(), dtype='uint32')
    np.random.seed(seed)
    
    if np.random.randint(0,2) == 0:
        return x
    else:
        s = [ char_dict.get(c, c) for c in x ]
        s = ''.join(s)
        return s
    
@lru_cache(maxsize=2000000)
def sample_substitution(x, seed):
    s = '%s %d' % (x, seed)
    hash = sha256(s.encode())
    seed = np.frombuffer(hash.digest(), dtype='uint32')
    np.random.seed(seed)
    
    mask = np.random.randint(0, 2, size=(len(char_dict)))
    
    masked_dict = { i:j for (i, j), m in zip(char_dict.items(), mask) if m == 1 }
    substitute = ''.join([ masked_dict.get(c, c) for c in x ])
    return substitute

def sample_once(x, seed):
    s = '%d' % (seed)
    hash = sha256(s.encode())
    seed = np.frombuffer(hash.digest(), dtype='uint32')
    np.random.seed(seed)
    
    mask = np.random.randint(0, 2, size=(len(char_dict)))
    
    masked_dict = { i:j for (i, j), m in zip(char_dict.items(), mask) if m == 1 }
    substitute = ''.join([ masked_dict.get(c, c) for c in x ])
    return substitute