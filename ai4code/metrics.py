from bisect import bisect

import pytest
from scipy import stats

# implementation from https://www.kaggle.com/code/ryanholbrook/competition-metric-kendall-tau-correlation

def count_inversions(a):
    inversions = 0
    sorted_so_far = []
    for i, u in enumerate(a):  # O(N)
        j = bisect(sorted_so_far, u)  # O(log N)
        inversions += i - j
        sorted_so_far.insert(j, u)  # O(N)
    return inversions


def kendall_tau(ground_truth, predictions):
    total_inversions = 0  # total inversions in predicted ranks across all instances
    total_2max = 0  # maximum possible inversions across all instances
    for gt, pred in zip(ground_truth, predictions):
        ranks = [gt.index(x) for x in pred]  # rank predicted order in terms of ground truth
        total_inversions += count_inversions(ranks)
        n = len(gt)
        total_2max += n * (n - 1)
    return 1 - 4 * total_inversions / total_2max


def test_kendall_tau():
    x1 = [1, 2, 3, 4, 5]
    x2 = [1, 2, 5, 4, 3]

    tau, p_value = stats.kendalltau(x1, x2)

    tau2 = kendall_tau([x1], [x2])

    print(tau, tau2)
    assert tau == pytest.approx(tau2)


if __name__ == '__main__':
    test_kendall_tau()