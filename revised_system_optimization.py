import sys
import cmath

import cvxpy
import numpy
from tqdm import tqdm

home_dir = './'
sys.path.append(home_dir)


def objective_calculation(w_mat, h_mat, a_list, indicator_mat, b_mat, sigma):
    N, J = w_mat.shape
    _, K, m = h_mat.shape
    obj = 0
    for n in range(N):
        for j in range(J):
            for k in range(K):
                h_vec = numpy.mat(h_mat[n, k, :]).reshape((m, 1))
                tmp = indicator_mat[j, k] * b_mat[n, j] * numpy.sum(numpy.dot(a_list[j].T, h_vec))
                obj += w_mat[n, j] * (tmp ** 2 - 2 * tmp)
    for j in range(J):
        obj += sigma * numpy.linalg.norm(a_list[j]) ** 2

    return obj


def check_constraints(indicator_mat, w_mat, b_mat, P):
    J, K = indicator_mat.shape
    N, _ = w_mat.shape

    print('---check constraints---')

    # constraint 1
    for j in range(J):
        if abs(numpy.sum(indicator_mat[j]) - 1) < 1e-6:
            print('constraint 1 violated: index ' + str(j) + ' value: ' + str(numpy.sum(indicator_mat[j])))

    # constraint 2
    for k in range(K):
        if numpy.sum(indicator_mat[:, k]) - 1 > 1e-6:
            print('constraint 2 violated: index ' + str(k) + ' value: ' + str(numpy.sum(indicator_mat[:, k])))

    # constraint 3
    for n in range(N):
        transmit_power = 0
        for j in range(J):
            transmit_power += b_mat[n, j] ** 2 * w_mat[n, j]
        if transmit_power > P:
            print('constraint 3 violated: index ' + str(n) + '  value: ' + str(transmit_power))


def assignment_index(indicator_mat):
    J, K = indicator_mat.shape
    for j in range(J):
        if sum(indicator_mat[j]) == 0:
            return j
    return J


def round_indicator_mat(c_mat):
    J, K = c_mat.shape
    augmented_c_mat = numpy.zeros((K, K))
    for j in range(J):
        for k in range(K):
            augmented_c_mat[j, k] = c_mat[j, k]
    augmented_indicator_mat = numpy.zeros((K, K))
    visited_mat = numpy.zeros((K, K))

    while assignment_index(augmented_indicator_mat) < K:
        unassigned_j = assignment_index(augmented_indicator_mat)
        idx_list = numpy.argsort(augmented_c_mat[unassigned_j])
        idx = 0
        for k in range(K):
            if visited_mat[unassigned_j, idx_list[K - k - 1]] == 0:
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


def beamforming_optimization(w_mat, h_mat, indicator_mat, b_mat, sigma):
    N, J = w_mat.shape
    _, K, m = h_mat.shape
    a_list = list()
    alpha_vec = numpy.zeros((J, m, 1))
    alpha_mat = numpy.zeros(J)
    for j in range(J):
        tmp_vec = numpy.zeros((m, 1))
        tmp_value = 0
        for n in range(N):
            for k in range(K):
                h_vec = numpy.mat(h_mat[n, k]).reshape((m, 1))
                tmp_vec += w_mat[n, j] * indicator_mat[j, k] * b_mat[n, j] * h_vec
                tmp_value += w_mat[n, j] * numpy.linalg.norm(indicator_mat[j, k] * b_mat[n, j] * h_vec) ** 2
        alpha_vec[j] = tmp_vec
        alpha_mat[j] = tmp_value
    for j in range(J):
        a_list.append(alpha_vec[j] / (alpha_mat[j] + sigma))
    return a_list


def transmission_power_optimization(w_mat, h_mat, indicator_mat, a_list, P, max_iter=100):
    N, J = w_mat.shape
    _, K, m = h_mat.shape
    nu_vec = numpy.ones(N)
    b_mat = numpy.ones((N, J))
    alpha_mat = numpy.zeros((N, J, K))
    square_alpha_mat = numpy.zeros((N, J, K))
    for n in range(N):
        for j in range(J):
            b_mat[n, j] = numpy.sqrt(P / J / w_mat[n, j])
            for k in range(K):
                h_vec = numpy.mat(h_mat[n, k, :]).reshape((m, 1))
                alpha_mat[n, j, k] = indicator_mat[j, k] * numpy.sum(numpy.dot(a_list[j].T, h_vec))
                square_alpha_mat[n, j, k] = alpha_mat[n, j, k] ** 2

    pre_obj = 1e8
    eta = 0.01

    for it in range(max_iter):
        # multipler update
        for n in range(N):
            tmp = 0
            for j in range(J):
                tmp += w_mat[n, j] * b_mat[n, j] ** 2
            nu_vec[n] += eta * (tmp - P)
            nu_vec[n] = 0 if nu_vec[n] < 0 else nu_vec[n]

        # variable update
        for n in range(N):
            for j in range(J):
                b_mat[n, j] = (numpy.sum(alpha_mat[n, j])) / (
                        numpy.sum(square_alpha_mat[n, j]) + nu_vec[n])

        new_obj = objective_calculation(w_mat, h_mat, a_list, indicator_mat, b_mat, 1)
        if abs(new_obj - pre_obj) < 1e-3:
            break

    return b_mat


def subcarrier_allocation_optimization(w_mat, h_mat, a_list, b_mat, max_iter=100):
    N, J = w_mat.shape
    _, K, m = h_mat.shape
    indicator_mat = numpy.zeros((J, K))
    mu_vec = numpy.zeros(J)
    lambda_vec = numpy.ones(K)
    alpha_mat = numpy.zeros((N, J, K))
    square_alpha_mat = numpy.zeros((N, J, K))
    for n in range(N):
        for j in range(J):
            for k in range(K):
                alpha_mat[n, j, k] = b_mat[n, j] * numpy.sum(
                    numpy.dot(a_list[j].T, numpy.mat(h_mat[n, k, :]).reshape((m, 1))))
                square_alpha_mat[n, j, k] = alpha_mat[n, j, k] ** 2
    eta_list = [0.01, 0.005]

    for it in range(max_iter):
        # multipler update
        lambda_vec = lambda_vec + eta_list[1] * (numpy.sum(indicator_mat, axis=0) - 1)
        for k in range(K):
            lambda_vec[k] = 0 if lambda_vec[k] < 0 else lambda_vec[k]

        # indicator update
        for j in range(J):
            value_vec = numpy.zeros(K)
            sub_value_vec = numpy.zeros(K)
            scale_value_vec = numpy.zeros(K)
            for k in range(K):
                sub_value_vec[k] = lambda_vec[k] - 2 * numpy.sum(numpy.multiply(w_mat[:, j], alpha_mat[:, j, k]))
                value_vec[k] = lambda_vec[k] - 2 * numpy.sum(
                    numpy.multiply(w_mat[:, j], alpha_mat[:, j, k])) + 2 * numpy.sum(
                    numpy.multiply(w_mat[:, j], square_alpha_mat[:, j, k]))
                scale_value_vec[k] = 2 * numpy.sum(numpy.multiply(w_mat[:, j], square_alpha_mat[:, j, k]))

            idx = numpy.argmin(value_vec)
            sub_value_list = list()
            for k in range(K):
                if k != idx:
                    sub_value_list.append(
                        lambda_vec[k] - 2 * numpy.sum(numpy.multiply(w_mat[:, j], alpha_mat[:, j, k])))
            sub_value_list.sort()
            if value_vec[idx] < sub_value_list[0]:
                for k in range(K):
                    indicator_mat[j, k] = 1 if k == idx else 0
            else:
                left = numpy.max(value_vec) - numpy.max(scale_value_vec)
                right = numpy.max(value_vec) + numpy.max(scale_value_vec)
                for binary_it in range(100):
                    mu_vec[j] = (left + right) / 2
                    for k in range(K):
                        divider = 2 * numpy.sum(numpy.multiply(w_mat[:, j], square_alpha_mat[:, j, k]))
                        indicator_mat[j, k] = 0 if sub_value_vec[k] >= mu_vec[j] else (mu_vec[j] - sub_value_vec[
                            k]) / divider
                    if abs(numpy.sum(indicator_mat[j]) - 1) < 1e-2:
                        break
                    elif numpy.sum(indicator_mat[j]) < 1:
                        left = mu_vec[j]
                    else:
                        right = mu_vec[j]

    return indicator_mat


def alternating_optimization_framework(w_mat, h_mat, sigma, P, max_iter=50):
    N, J = w_mat.shape
    _, K, m = h_mat.shape
    indicator_mat = numpy.zeros((J, K))
    b_mat = numpy.zeros((N, J))
    for j in range(J):
        indicator_mat[j, j] = 1
    for n in range(N):
        for j in range(J):
            b_mat[n, j] = numpy.sqrt(P / J / w_mat[n, j])
    a_list = beamforming_optimization(w_mat, h_mat, indicator_mat, b_mat, sigma)
    pre_obj = 1e6

    for it in tqdm(range(max_iter)):
        # subcarrier allocation
        indicator_mat = subcarrier_allocation_optimization(w_mat, h_mat, a_list, b_mat)

        # transmission power optimization
        b_mat = transmission_power_optimization(w_mat, h_mat, indicator_mat, a_list, P)

        # beamforming optimization
        a_list = beamforming_optimization(w_mat, h_mat, indicator_mat, b_mat, sigma)

        new_obj = objective_calculation(w_mat, h_mat, a_list, indicator_mat, b_mat, sigma)
        print('iter ' + str(it) + ': objective: ' + str(new_obj))
        print(indicator_mat)
        check_constraints(indicator_mat, w_mat, b_mat, P)
        if abs(new_obj - pre_obj) < 1e-8:
            break
        pre_obj = new_obj

    indicator_mat = round_indicator_mat(indicator_mat)
    b_mat = transmission_power_optimization(w_mat, h_mat, indicator_mat, a_list, P)
    a_list = beamforming_optimization(w_mat, h_mat, indicator_mat, b_mat, sigma)
    return indicator_mat, b_mat, a_list