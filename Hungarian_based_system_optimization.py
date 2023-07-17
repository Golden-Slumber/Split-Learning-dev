import functools
import sys
import numpy
from revised_system_optimization import modified_subcarrier_allocation_optimization

home_dir = './'
sys.path.append(home_dir)


def objective_calculation(tau_mat, h_mat, a_list, b_mat, indicator_mat):
    N, J = tau_mat.shape
    _, K, m = h_mat.shape
    stored_mat = numpy.zeros((N, J, K))
    square_stored_mat = numpy.zeros((N, J, K))
    for n in range(N):
        for j in range(J):
            for k in range(K):
                stored_mat[n, j, k] = b_mat[n, j] * numpy.sum(
                    numpy.dot(a_list[j].T, numpy.mat(h_mat[n, k, :]).reshape((m, 1))))
                square_stored_mat[n, j, k] = stored_mat[n, j, k] ** 2
    obj = 0
    for j in range(J):
        for k in range(K):
            if indicator_mat[j, k] == 1:
                obj += numpy.sum(numpy.multiply(tau_mat[:, j], square_stored_mat[:, j, k])) - 2 * numpy.sum(
                    numpy.multiply(tau_mat[:, j], stored_mat[:, j, k]))
    return obj


def weight_calculation(tau_mat, h_mat, a_list, b_mat):
    N, J = tau_mat.shape
    _, K, m = h_mat.shape
    weight_mat = numpy.zeros((J, K))
    stored_mat = numpy.zeros((N, J, K))
    square_stored_mat = numpy.zeros((N, J, K))
    for n in range(N):
        for j in range(J):
            for k in range(K):
                stored_mat[n, j, k] = b_mat[n, j] * numpy.sum(
                    numpy.dot(a_list[j].T, numpy.mat(h_mat[n, k, :]).reshape((m, 1))))
                square_stored_mat[n, j, k] = stored_mat[n, j, k] ** 2
    for j in range(J):
        for k in range(K):
            weight_mat[j, k] = numpy.sum(numpy.multiply(tau_mat[:, j], square_stored_mat[:, j, k])) - 2 * numpy.sum(
                numpy.multiply(tau_mat[:, j], stored_mat[:, j, k]))
    return weight_mat


def find_augment_path(idx, neuron_set, subcarrier_set, match_vec, slack, neuron_labels, subcarrier_labels, weight_mat):
    J, K = weight_mat.shape

    neuron_set.add(idx)
    for k in range(K):
        if k not in subcarrier_set:
            tmp = neuron_labels[idx] + subcarrier_labels[k] - weight_mat[idx, k]
            if tmp == 0:
                subcarrier_set.add(k)
                if match_vec[k] == -1:
                    match_vec[k] = idx
                    return True, neuron_set.copy(), subcarrier_set.copy(), match_vec.copy(), slack.copy()
                else:
                    flag, neuron_set, subcarrier_set, match_vec, slack = find_augment_path(int(match_vec[k]),
                                                                                           neuron_set.copy(),
                                                                                           subcarrier_set.copy(),
                                                                                           match_vec.copy(),
                                                                                           slack.copy(), neuron_labels,
                                                                                           subcarrier_labels,
                                                                                           weight_mat)
                    if flag:
                        match_vec[k] = idx
                        return True, neuron_set.copy(), subcarrier_set.copy(), match_vec.copy(), slack.copy()
            else:
                slack[k] = min(slack[k], tmp)
    return False, neuron_set.copy(), subcarrier_set.copy(), match_vec.copy(), slack.copy()


def Kuhn_Munkres_based_subcarrier_allocation(tau_mat, h_mat, a_list, b_mat):
    # calculate the weights between neurons and subcarriers
    weight_mat = weight_calculation(tau_mat, h_mat, a_list, b_mat)
    J, K = weight_mat.shape

    # turn the minimization problem into the maximization formula
    max_weight = numpy.max(weight_mat)
    for j in range(J):
        for k in range(K):
            weight_mat[j, k] = max_weight - weight_mat[j, k]

    # parameters
    neuron_labels = numpy.zeros(J)
    subcarrier_labels = numpy.zeros(K)
    neuron_set = set()
    subcarrier_set = set()
    slack = numpy.zeros(K)
    match_vec = numpy.zeros(K)

    # initialization
    for j in range(J):
        neuron_labels[j] = 0
        for k in range(K):
            subcarrier_labels[k] = 0
            match_vec[k] = -1
            neuron_labels[j] = max(neuron_labels[j], weight_mat[j, k])

    # allocation
    for idx in range(J):
        for k in range(K):
            slack[k] = 1e6

        while True:
            neuron_set.clear()
            subcarrier_set.clear()
            flag, neuron_set, subcarrier_set, match_vec, slack = find_augment_path(idx,
                                                                                   neuron_set.copy(),
                                                                                   subcarrier_set.copy(),
                                                                                   match_vec.copy(),
                                                                                   slack.copy(), neuron_labels,
                                                                                   subcarrier_labels,
                                                                                   weight_mat)
            if flag:
                break
            else:
                delta = 1e6
                for k in range(K):
                    if k not in subcarrier_set:
                        delta = min(delta, slack[k])
                for j in range(J):
                    if j in neuron_set:
                        neuron_labels[j] -= delta
                for k in range(K):
                    if k in subcarrier_set:
                        subcarrier_labels[k] += delta

    indicator_mat = numpy.zeros((J, K))
    for k in range(K):
        indicator_mat[int(match_vec[k]), k] = 1
    return indicator_mat


if __name__ == '__main__':
    N = 5
    J = 32
    K = 32
    m = 5
    sigma = 1
    P = 10
    tau_mat = numpy.zeros((N, J))
    h_mat = abs(numpy.random.randn(N, K, m))

    for n in range(N):
        for j in range(J):
            tau_mat[n, j] = numpy.random.uniform()

    b_mat = numpy.zeros((N, J))
    for n in range(N):
        for j in range(J):
            b_mat[n, j] = numpy.sqrt(P / J / tau_mat[n, j])
    a_list = list()
    for j in range(J):
        a_list.append(numpy.ones((m, 1)))

    indicator_mat = numpy.zeros((J, K))
    for j in range(J):
        indicator_mat[j, j] = 1
    print('fixed subcarrier allocation: ' + str(objective_calculation(tau_mat, h_mat, a_list, b_mat, indicator_mat)))

    indicator_mat = modified_subcarrier_allocation_optimization(tau_mat, h_mat, a_list, b_mat)
    print(indicator_mat)
    print('Lagrangian-based subcarrier allocation optimization: ' + str(
        objective_calculation(tau_mat, h_mat, a_list, b_mat, indicator_mat)))

    indicator_mat = Kuhn_Munkres_based_subcarrier_allocation(tau_mat, h_mat, a_list, b_mat)
    print(indicator_mat)
    print('Kuhn Munkres-based subcarrier allocation optimization: ' + str(
        objective_calculation(tau_mat, h_mat, a_list, b_mat, indicator_mat)))
