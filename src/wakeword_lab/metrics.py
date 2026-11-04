import numpy as np

def choose_threshold(scores,labels,target_fpr=.001):
    s=np.asarray(scores); y=np.asarray(labels).astype(bool); neg=s[~y]
    if not len(neg): raise ValueError("validation requires negatives")
    candidates=np.unique(np.r_[0.,neg,1.,1.000001])
    ok=[t for t in candidates if np.mean(neg>=t)<=target_fpr]
    return float(min(ok))

def classification_metrics(scores,labels,threshold):
    y=np.asarray(labels).astype(bool); p=np.asarray(scores)>=threshold
    tp=int(np.sum(p&y)); tn=int(np.sum(~p&~y)); fp=int(np.sum(p&~y)); fn=int(np.sum(~p&y))
    return {"accuracy":(tp+tn)/max(1,len(y)),"precision":tp/max(1,tp+fp),"recall":tp/max(1,tp+fn),"true_positives":tp,"true_negatives":tn,"false_positives":fp,"false_negatives":fn}

def false_activations_per_hour(event_times,duration_seconds,merge_gap=1.):
    if duration_seconds<=0: raise ValueError("duration must be positive")
    count=0; last=-float('inf')
    for t in sorted(map(float,event_times)):
        if t-last>=merge_gap: count+=1
        last=t
    return count*3600/duration_seconds
