from collections import defaultdict
import random

def group_split(records,seed=42,fractions=(.7,.15,.15)):
    if len(fractions)!=3 or abs(sum(fractions)-1)>1e-8 or min(fractions)<=0: raise ValueError("invalid split fractions")
    groups=defaultdict(list)
    for r in records: groups[r['source_group']].append(r)
    keys=list(groups); random.Random(seed).shuffle(keys); n=len(keys)
    if n<3: raise ValueError("at least three independent source groups required")
    a=max(1,round(n*fractions[0])); b=max(1,round(n*fractions[1]))
    if a+b>=n: a=n-2; b=1
    return {name:[r for k in keys[i:j] for r in groups[k]] for name,(i,j) in zip(('train','validation','test'),((0,a),(a,a+b),(a+b,n)))}
