import pytest
from wakeword_lab.split import group_split

def test_group_disjoint():
 rows=[{'source_group':str(i),'path':str(i)} for i in range(30)]; s=group_split(rows,4); sets=[{r['source_group'] for r in s[k]} for k in s]
 assert all(not sets[i]&sets[j] for i in range(3) for j in range(i+1,3))
def test_min_groups():
 with pytest.raises(ValueError): group_split([{'source_group':'a'},{'source_group':'b'}])
