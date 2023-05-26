import functools
import sys
import numpy
from tqdm import tqdm

from revised_system_optimization import modified_subcarrier_allocation_optimization

home_dir = './'
sys.path.append(home_dir)


class Node(object):
    def __init__(self, idx, subcarrier_set, indicator_mat, lower_bound, upper_bound):
        super(Node, self).__init__()
        self.idx = idx
        self.subcarrier_set = subcarrier_set
        self.indicator_mat = indicator_mat
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def get_idx(self):
        return self.idx

    def get_subcarrier_set(self):
        return self.subcarrier_set.copy()

    def get_indicator_mat(self):
        return self.indicator_mat.copy()

    def get_lower_bound(self):
        return self.lower_bound

    def get_upper_bound(self):
        return self.upper_bound


def partial_subcarrier_optimization(idx, subcarrier_set, tau_mat, h_mat, stored_value_mat, stored_sqaure_value_mat,
                                    pre_indicator_mat,
                                    eta=None,
                                    max_iter=50):
    N, J = tau_mat.shape
    _, K, m = h_mat.shape
    mu_vec = numpy.zeros(J)
    lambda_vec = numpy.ones(K)
    indicator_mat = pre_indicator_mat
    i = 0
    subcarrier_list = list(subcarrier_set)
    for j in range(idx, J):
        for k in range(K):
            indicator_mat[j, k] = 1 if k == subcarrier_list[i] else 0
    if eta is None:
        # eta = 0.27
        eta = 0.01
    else:
        eta = eta

    value_vec = numpy.zeros(K)
    sub_value_vec = numpy.zeros(K)
    scale_value_vec = numpy.zeros(K)

    for it in range(max_iter):
        # multipler update
        tmp_vec = numpy.zeros(K)
        for j in range(idx, J):
            tmp_vec += indicator_mat[j]
        lambda_vec = lambda_vec + eta * (tmp_vec - 1)

        # indicator update
        vis_idx_list = list()
        for j in range(idx, J):
            for k in subcarrier_list:
                sub_value_vec[k] = lambda_vec[k] - stored_value_mat[j, k]
                value_vec[k] = lambda_vec[k] - stored_value_mat[j, k] + stored_sqaure_value_mat[j, k]

            inner_idx = numpy.argmin(value_vec)
            sub_value_list = list()
            for k in subcarrier_list:
                if k != inner_idx:
                    sub_value_list.append(sub_value_vec[k])
            sub_value_list.sort()
            if value_vec[inner_idx] < sub_value_list[0] and inner_idx not in vis_idx_list:
                vis_idx_list.append(inner_idx)
                for k in subcarrier_list:
                    indicator_mat[j, k] = 1 if k == inner_idx else 0
            else:
                left = numpy.max(value_vec) - numpy.max(stored_sqaure_value_mat[j])
                right = numpy.max(value_vec) + numpy.max(stored_sqaure_value_mat[j])
                for binary_it in range(20):
                    mu_vec[j] = (left + right) / 2
                    for k in subcarrier_list:
                        indicator_mat[j, k] = 0 if sub_value_vec[k] >= mu_vec[j] else (mu_vec[j] - sub_value_vec[k]) / \
                                                                                      stored_sqaure_value_mat[j, k]
                        tmp = numpy.sum(indicator_mat[j])
                        if abs(tmp - 1) < 1e-2:
                            break
                        elif tmp < 1:
                            left = mu_vec[j]
                        else:
                            right = mu_vec[j]
    return indicator_mat


def bound_calculation(tau_mat, h_mat, a_list, indicator_mat, b_mat):
    N, J = tau_mat.shape
    _, K, m = h_mat.shape
    obj = 0
    for n in range(N):
        for j in range(J):
            for k in range(K):
                h_vec = numpy.mat(h_mat[n, k, :]).reshape((m, 1))
                tmp = indicator_mat[j, k] * b_mat[n, j] * numpy.sum(numpy.dot(a_list[j].T, h_vec))
                obj += tau_mat[n, j] * (tmp ** 2 - 2 * tmp)
    return obj


def objective_calculation(tau_mat, h_mat, a_list, indicator_mat, b_mat):
    N, J = tau_mat.shape
    _, K, m = h_mat.shape
    obj = 0
    for n in range(N):
        for j in range(J):
            obj_k = 0
            for k in range(K):
                h_vec = numpy.mat(h_mat[n, k, :]).reshape((m, 1))
                obj_k += indicator_mat[j, k] * b_mat[n, j] * numpy.sum(numpy.dot(a_list[j].T, h_vec))
            obj += tau_mat[n, j] * (obj_k - 1) ** 2
    return obj


def round_indicator_mat(indicator_mat):
    J, K = indicator_mat.shape
    for j in range(J):
        idx = numpy.argmax(indicator_mat[j])
        for k in range(K):
            indicator_mat[j, k] = 1 if k == idx else 0
    return indicator_mat


def assignment_index(indicator_mat):
    J, K = indicator_mat.shape
    for j in range(J):
        if sum(indicator_mat[j]) == 0:
            return j
    return J


def round_indicator_mat_v2(idx, c_mat, subcarrier_set):
    J, K = c_mat.shape
    augmented_c_mat = numpy.zeros((K, K))
    for j in range(J):
        for k in range(K):
            augmented_c_mat[j, k] = c_mat[j, k]
    augmented_indicator_mat = numpy.zeros((K, K))
    for j in range(idx):
        for k in range(K):
            augmented_indicator_mat[j, k] = c_mat[j, k]
    visited_mat = numpy.zeros((K, K))

    while assignment_index(augmented_indicator_mat) < K:
        unassigned_j = assignment_index(augmented_indicator_mat)
        idx_list = numpy.argsort(augmented_c_mat[unassigned_j])
        idx = 0
        for k in range(K):
            if visited_mat[unassigned_j, idx_list[K - k - 1]] == 0 and idx_list[K - k - 1] in subcarrier_set:
                idx = idx_list[K - k - 1]
                visited_mat[unassigned_j, idx] = 1
                break

        if sum(augmented_indicator_mat[:, idx]) == 0:
            augmented_indicator_mat[unassigned_j, idx] = 1
        else:
            pre_j = 0
            for j in range(K):
                if augmented_indicator_mat[j, idx] == 1:
                    pre_j = j
                    break
            if augmented_c_mat[unassigned_j, idx] > augmented_c_mat[pre_j, idx]:
                augmented_indicator_mat[unassigned_j, idx] = 1
                augmented_indicator_mat[pre_j, idx] = 0

    indicator_mat = numpy.zeros((J, K))
    for j in range(J):
        for k in range(K):
            indicator_mat[j, k] = augmented_indicator_mat[j, k]
    return indicator_mat


def check_indicator_constraints(indicator_mat):
    J, K = indicator_mat.shape
    flag = True

    for j in range(J):
        if abs(numpy.sum(indicator_mat[j]) - 1) > 1e-2:
            flag = False

    for k in range(K):
        if numpy.sum(indicator_mat[:, k]) - 1 > 1e-2:
            flag = False

    return flag


def node_cmp(a, b):
    if a.get_lower_bound() < b.get_lower_bound():
        return -1
    elif a.get_lower_bound() > b.get_lower_bound():
        return 1
    else:
        return 0


def exhaustive_search(tau_mat, h_mat, a_list, b_mat):
    N, J = tau_mat.shape
    _, K, m = h_mat.shape
    indicator_mat = numpy.zeros((J, K))
    current_optimal = None
    bound = 1e6
    subcarrier_set = set()
    for k in range(K):
        subcarrier_set.add(k)
    node_list = list()
    node_list.append(Node(0, subcarrier_set, indicator_mat, 0, 0))
    while node_list:
        node = node_list.pop(0)
        idx = node.get_idx()
        if idx == J:
            indicator_mat = node.get_indicator_mat()
            # obj = bound_calculation(tau_mat, h_mat, a_list, indicator_mat, b_mat)
            obj = objective_calculation(tau_mat, h_mat, a_list, indicator_mat, b_mat)
            print(obj)
            if obj < bound:
                bound = obj
                current_optimal = indicator_mat
        else:
            subcarrier_set = node.get_subcarrier_set()
            for selected_k in list(subcarrier_set):
                indicator_mat = node.get_indicator_mat()
                for k in range(K):
                    indicator_mat[idx, k] = 1 if selected_k == k else 0
                tmp_subcarrier_set = subcarrier_set.copy()
                tmp_subcarrier_set.discard(selected_k)
                node_list.append(Node(idx + 1, tmp_subcarrier_set, indicator_mat, 0, 0))
    return current_optimal


def BnB_subcarrier_allocation(tau_mat, h_mat, a_list, b_mat, stored_value_mat, stored_sqaure_value_mat, max_iter=100,
                              tol=1e-3):
    N, J = tau_mat.shape
    _, K, m = h_mat.shape
    global_lower_bound = -1e6
    global_upper_bound = 1e6
    optimal_indicator_mat = numpy.zeros((J, K))
    for j in range(J):
        optimal_indicator_mat[j, j] = 1
    node_list = list()
    subcarrier_set = set()
    for k in range(K):
        subcarrier_set.add(k)
    indicator_mat = numpy.zeros((J, K))
    indicator_mat = partial_subcarrier_optimization(0, subcarrier_set, tau_mat, h_mat, stored_value_mat,
                                                    stored_sqaure_value_mat,
                                                    indicator_mat)
    # global_lower_bound = bound_calculation(tau_mat, h_mat, a_list, indicator_mat, b_mat)
    global_lower_bound = objective_calculation(tau_mat, h_mat, a_list, indicator_mat, b_mat)
    # discrete_indicator_mat = round_indicator_mat(indicator_mat.copy())
    # discrete_indicator_mat = round_indicator_mat_v2(0, indicator_mat.copy(), subcarrier_set)
    # if check_indicator_constraints(discrete_indicator_mat):
    #     global_upper_bound = bound_calculation(tau_mat, h_mat, a_list, discrete_indicator_mat, b_mat)
    # global_upper_bound = bound_calculation(tau_mat, h_mat, a_list, discrete_indicator_mat, b_mat)

    node_list.append(Node(0, subcarrier_set, indicator_mat, global_lower_bound, global_upper_bound))
    while global_upper_bound - global_lower_bound > tol:
        node = node_list.pop(0)
        idx = node.get_idx()
        subcarrier_set = node.get_subcarrier_set()

        # print('indicator mat', node.get_indicator_mat())
        print('global upper bound ' + str(global_upper_bound))
        print('global lower bound ' + str(global_lower_bound))
        print('idx ' + str(idx))
        print('subcarrier set: ', subcarrier_set)

        for selected_k in list(subcarrier_set):
            indicator_mat = node.get_indicator_mat()
            for k in range(K):
                indicator_mat[idx, k] = 1 if k == selected_k else 0

            tmp_subcarrier_set = subcarrier_set.copy()
            if idx < J - 1:
                tmp_subcarrier_set.discard(selected_k)
                indicator_mat = partial_subcarrier_optimization(idx + 1, tmp_subcarrier_set, tau_mat, h_mat,
                                                                stored_value_mat, stored_sqaure_value_mat,
                                                                indicator_mat)
            # lower_bound = bound_calculation(tau_mat, h_mat, a_list, indicator_mat, b_mat)
            lower_bound = objective_calculation(tau_mat, h_mat, a_list, indicator_mat, b_mat)
            if lower_bound < global_upper_bound:
                # global_lower_bound = lower_bound if lower_bound < global_lower_bound else global_lower_bound
                # discrete_indicator_mat = round_indicator_mat(indicator_mat.copy())
                discrete_indicator_mat = round_indicator_mat_v2(idx + 1, indicator_mat.copy(), tmp_subcarrier_set)
                upper_bound = global_upper_bound
                # if check_indicator_constraints(discrete_indicator_mat):
                #     upper_bound = bound_calculation(tau_mat, h_mat, a_list, discrete_indicator_mat, b_mat)
                #     if upper_bound < global_upper_bound:
                #         global_upper_bound = upper_bound
                #         optimal_indicator_mat = discrete_indicator_mat
                # upper_bound = bound_calculation(tau_mat, h_mat, a_list, discrete_indicator_mat, b_mat)
                upper_bound = objective_calculation(tau_mat, h_mat, a_list, discrete_indicator_mat, b_mat)
                if upper_bound < global_upper_bound:
                    global_upper_bound = upper_bound
                    optimal_indicator_mat = discrete_indicator_mat
                    # print('update optimal indicator')
                if idx < J - 1:
                    node_list.append(Node(idx + 1, tmp_subcarrier_set, indicator_mat, lower_bound, upper_bound))
        if node_list:
            node_list.sort(key=functools.cmp_to_key(node_cmp))
            global_lower_bound = node_list[0].get_lower_bound()
            for node in node_list:
                print(node.get_lower_bound())
        else:
            break
    return optimal_indicator_mat


if __name__ == '__main__':
    N = 5
    J = 5
    K = 5
    m = 5
    sigma = 1
    P = 10
    tau_mat = numpy.zeros((N, J))
    h_mat = abs(numpy.random.randn(N, K, m))

    for n in range(N):
        subcarrier_scale_list = numpy.zeros(K)
        step = int(K / 4)
        subcarrier_scale_list[0:step] = 0.1 * numpy.random.random_sample(step) + 0.1
        # subcarrier_scale_list[int(K / 4):] = 5 * numpy.random.random_sample(int(K / 2)) + 5
        # subcarrier_scale_list[int(K / 4):] = 1
        subcarrier_scale_list[step:2 * step] = 0.01 * numpy.random.random_sample(step) + 0.01
        subcarrier_scale_list[2 * step:3 * step] = 1 * numpy.random.random_sample(step) + 1
        subcarrier_scale_list[3 * step:] = 1
        subcarrier_scale_list = subcarrier_scale_list[numpy.random.permutation(K)]
        tmp_subcarrier_scale_list = subcarrier_scale_list[numpy.random.permutation(K)]
        for k in range(K):
            h_mat[n, k] = subcarrier_scale_list[k] * h_mat[n, k]

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
    print('fixed subcarrier allocation: ' + str(bound_calculation(tau_mat, h_mat, a_list, indicator_mat, b_mat)))

    indicator_mat = modified_subcarrier_allocation_optimization(tau_mat, h_mat, a_list, b_mat)
    print('subcarrier allocation optimization: ' + str(bound_calculation(tau_mat, h_mat, a_list, indicator_mat, b_mat)))

    alpha_mat = numpy.zeros((N, J, K))
    square_alpha_mat = numpy.zeros((N, J, K))
    for n in range(N):
        for j in range(J):
            for k in range(K):
                alpha_mat[n, j, k] = b_mat[n, j] * numpy.sum(
                    numpy.dot(a_list[j].T, numpy.mat(h_mat[n, k, :]).reshape((m, 1))))
                square_alpha_mat[n, j, k] = alpha_mat[n, j, k] ** 2
    stored_value_mat = numpy.zeros((J, K))
    stored_sqaure_value_mat = numpy.zeros((J, K))
    for j in range(J):
        for k in range(K):
            stored_value_mat[j, k] = 2 * numpy.sum(numpy.multiply(tau_mat[:, j], alpha_mat[:, j, k]))
            stored_sqaure_value_mat[j, k] = 2 * numpy.sum(numpy.multiply(tau_mat[:, j], square_alpha_mat[:, j, k]))
    indicator_mat = BnB_subcarrier_allocation(tau_mat, h_mat, a_list, b_mat, stored_value_mat, stored_sqaure_value_mat)
    print('BnB subcarrier allocation: ' + str(bound_calculation(tau_mat, h_mat, a_list, indicator_mat, b_mat)))

    indicator_mat = exhaustive_search(tau_mat, h_mat, a_list, b_mat)
    print('exhaustive search: ' + str(bound_calculation(tau_mat, h_mat, a_list, indicator_mat, b_mat)))
