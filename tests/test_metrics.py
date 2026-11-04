import numpy as np
import pytest
from wakeword_lab.metrics import choose_threshold, classification_metrics, false_activations_per_hour

def test_threshold_meets_fpr_and_keeps_lowest_cutoff():
    scores=np.array([.1,.2,.3,.9]); labels=np.array([0,0,0,1])
    t=choose_threshold(scores,labels,0)
    assert t > .3 and t < .301

def test_nonzero_fpr_threshold():
    scores=np.array([.1,.2,.3,.9]); labels=np.array([0,0,0,1])
    t=choose_threshold(scores,labels,1/3)
    assert np.isclose(t, np.float32(.3))

def test_metrics_counts():
    m=classification_metrics([.9,.1,.8,.2],[1,0,0,1],.5)
    assert m["true_positives"]==1 and m["false_positives"]==1

def test_event_merging():
    assert false_activations_per_hour([1,1.2,5],3600,1)==2

def test_duration_validation():
    with pytest.raises(ValueError): false_activations_per_hour([1],0)


def test_false_activation_merge_collapses_chained_burst():
    # Events less than the merge gap apart form one burst even if the burst lasts longer.
    assert false_activations_per_hour([1, 1.9, 2.8], 3600, 1.0) == 1


def test_zero_fpr_threshold_stays_above_float32_maximum():
    scores = np.array([1.0, 0.5], dtype=np.float32)
    labels = np.array([0, 1])
    threshold = choose_threshold(scores, labels, 0.0)
    assert np.float32(threshold) > np.float32(1.0)
    assert classification_metrics(scores, labels, threshold)["false_positives"] == 0


def test_classification_metrics_reject_non_finite_scores_or_threshold():
    import pytest
    with pytest.raises(ValueError):
        classification_metrics([float("nan")], [0], 0.5)
    with pytest.raises(ValueError):
        classification_metrics([0.2], [0], float("inf"))
