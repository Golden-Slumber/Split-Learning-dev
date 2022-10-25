import cmath

import cvxpy
import numpy
from tqdm import tqdm


def sample_spherical(dim):
    vec = numpy.random.randn(dim)
    vec /= numpy.linalg.norm(vec)
    return vec


def rand_c(pre_aa, num_candidates=100):
    candidates = list()

    k, d = pre_aa.shape
    eig_values, eig_vectors = numpy.linalg.eigh(pre_aa)
    dim = eig_values.shape[0]
    eig_val_mat = numpy.eye(dim)
    for i in range(dim):
        eig_val_mat[i][i] = numpy.sqrt(abs(eig_values[i]))
    mean = numpy.zeros(dim)
    cov = numpy.eye(dim)

    for i in range(num_candidates):
        sup_vec = numpy.random.multivariate_normal(mean, cov)
        candidate_a = numpy.dot(numpy.dot(eig_vectors, eig_val_mat), sup_vec)
        candidates.append(candidate_a.reshape(k, 1))
    return candidates


def find_best(candidates):
    best_candidate = candidates[0]
    for candidate in candidates:
        best_candidate = candidate if numpy.linalg.norm(candidate) < numpy.linalg.norm(
            best_candidate) else best_candidate
    return best_candidate


def subcarrier_allocation(alpha_mat):
    N, J, K = alpha_mat.shape

    alpha_lifting_list = list()
    for n in range(N):
        alpha_lifting_sublist = list()
        for j in range(J):
            alpha_vec = numpy.mat(alpha_mat[n, j, :]).reshape((K, 1))
            alpha_lifting_mat = numpy.dot(alpha_vec, alpha_vec.conj().T)
            alpha_lifting_sublist.append(alpha_lifting_mat)
        alpha_lifting_list.append(alpha_lifting_sublist)

    one_mat = numpy.ones((K, K))
    obj = cvxpy.Minimize(1)
    constraints = []

    sum_inverse_trace = 0
    C_list = list()
    for j in range(J):
        C_mat = cvxpy.Variable((K, K))
        C_list.append(C_mat)
        constraints = constraints + [C_mat >> 0, cvxpy.trace(one_mat @ C_mat) == 1]

        for n in range(N):
            sum_inverse_trace += cvxpy.inv_pos(cvxpy.trace(alpha_lifting_list[n][j] @ C_mat))
    constraints = constraints + [sum_inverse_trace <= 1]

    L_list = list()
    for k in range(K):
        current_l = numpy.zeros((K, 1))
        current_l[k, 0] = 1
        L_list.append(numpy.dot(current_l, current_l.T))
        sum_trace = 0
        for j in range(J):
            sum_trace += cvxpy.trace(L_list[k] @ C_list[j])
            constraints = constraints + [C_list[j][k, k] <= 1]
        constraints = constraints + [sum_trace <= 1]

    prob = cvxpy.Problem(obj, constraints)
    prob.solve(verbose=True)

    c_list = list()
    for j in range(len(C_list)):
        candidates = rand_c(C_list[j].value, num_candidates=1)
        print(candidates[0])


def subcarrier_allocation_v2(alpha_mat):
    N, J, K = alpha_mat.shape

    obj = cvxpy.Minimize(1)
    constraints = []
    C = cvxpy.Variable((J, K))

    for j in range(J):
        sum_versus_k = 0
        for k in range(K):
            sum_versus_k += C[j, k]
            constraints = constraints + [C[j, k] >= 0, C[j, k] <= 1]
        constraints = constraints + [sum_versus_k == 1]

    for k in range(K):
        sum_versus_j = 0
        for j in range(J):
            sum_versus_j += C[j, k]
        constraints = constraints + [sum_versus_j <= 1]

    for n in range(N):
        sum_power_constraint = 0
        for j in range(J):
            tmp = 0
            for k in range(K):
                tmp += alpha_mat[n, j, k] * C[j, k]
            sum_power_constraint += cvxpy.inv_pos(cvxpy.square(tmp))
            constraints = constraints + [tmp >= 0]
        constraints = constraints + [sum_power_constraint <= 1]

    prob = cvxpy.Problem(obj, constraints)
    prob.solve(verbose=True)

    print(C.value)


def test_indicator_mat(indicator_mat):
    J, K = indicator_mat.shape
    flag = True
    for j in range(J):
        if numpy.sum(indicator_mat[j]) != 1:
            flag = False
    return flag


def modified_subcarrier_allocation(alpha_mat, max_iter=1000):
    N, J, K = alpha_mat.shape
    indicator_mat = numpy.zeros((J, K))
    lambda_vec = numpy.ones(K)
    nu_vec = numpy.ones(N)
    eta_list = [0.002, 0.1, 0.0001]
    # for j in range(J):
    #     for k in range(K):
    #         indicator_mat[j, k] = 1 / K
    for j in range(J):
        indicator_mat[j, j] = 1
    pre_mat = numpy.ones((J, K))
    # print(alpha_mat)A
    # print(numpy.square(indicator_mat))
    # print(numpy.multiply(alpha_mat[0], numpy.square(indicator_mat)))
    # print(numpy.sum(numpy.multiply(alpha_mat[0], numpy.square(indicator_mat)), axis=(0,1)))

    for it in range(max_iter):
        # if numpy.linalg.norm(pre_mat - indicator_mat) < 1e-3:
        #     break

        # multipler update
        # mu_vec = mu_vec + eta_list[0] * (1 - numpy.sum(indicator_mat, axis=1))
        lambda_vec = lambda_vec + eta_list[1] * (numpy.sum(indicator_mat, axis=0) - 1)
        for k in range(K):
            lambda_vec[k] = 0 if lambda_vec[k] < 0 else lambda_vec[k]
        for n in range(N):
            nu_vec[n] = nu_vec[n] + eta_list[2] * (
                    numpy.sum(numpy.multiply(alpha_mat[n], numpy.square(indicator_mat))) - 1)
            nu_vec[n] = 0 if nu_vec[n] < 0 else nu_vec[n]
        pre_mat = indicator_mat.copy()

        # indicator update
        for j in range(J):
            value_list = numpy.zeros(K)
            for k in range(K):
                tmp = 2 * numpy.sum(numpy.multiply(nu_vec, alpha_mat[:, j, k]))
                value_list[k] = lambda_vec[k] + tmp
            idx = numpy.argmin(value_list)
            lambda_list = list()
            for k in range(K):
                if k != idx:
                    lambda_list.append(lambda_vec[k])
            lambda_list.sort()
            if value_list[idx] < lambda_list[0]:
                for k in range(K):
                    indicator_mat[j, k] = 1 if k == idx else 0
            else:
                mu_vec = numpy.zeros(J)
                for k in range(K):
                    tmp = 2 * numpy.sum(numpy.multiply(nu_vec, alpha_mat[:, j, k]))
                    indicator_mat[j, k] = 0 if lambda_vec[k] >= mu_vec[j] else (mu_vec[j] - lambda_vec[
                        k]) / tmp
                    for inner_it in range(max_iter):
                        if numpy.sum(indicator_mat[j]) >= 1:
                            break
                        mu_vec[j] += eta_list[0]
                        for k in range(K):
                            tmp = 2 * numpy.sum(numpy.multiply(nu_vec, alpha_mat[:, j, k]))
                            indicator_mat[j, k] = 0 if lambda_vec[k] >= mu_vec[j] else (mu_vec[j] - lambda_vec[
                                k]) / tmp

            # for k in range(K):
            #     tmp = 2 * numpy.sum(numpy.multiply(nu_vec, alpha_mat[:, j, k]))
            #     if lambda_vec[k] >= mu_vec[j]:
            #         indicator_mat[j, k] = 0
            #     elif lambda_vec[k] + tmp <= mu_vec[j]:
            #         indicator_mat[j, k] = 1
            #     else:
            #         indicator_mat[j, k] = (mu_vec[j] - lambda_vec[k]) / tmp

        # scale for constraint
        # print(indicator_mat)
        # for j in range(J):
        #     total = numpy.sum(indicator_mat[j])
        #     indicator_mat[j] = 0 if total ==0 else indicator_mat[j] / total

        # print('inner iter ' + str(it))
        # print(indicator_mat)

    return indicator_mat


def beamforming_optimization(h_mat, beta_mat, num_candidates=1000):
    N, J, K = beta_mat.shape
    m = h_mat.shape[2]
    # a_mat = numpy.ones((J, m))
    H_mat = numpy.ones((N, K, m, m))
    for n in range(N):
        for k in range(K):
            h_vec = numpy.mat(h_mat[n, k, :]).reshape((m, 1))
            H_mat[n, k] = numpy.dot(h_vec, h_vec.conj().T)

    sum_trace = 0
    A_list = list()
    constraints = []
    for j in range(J):
        A_mat = cvxpy.Variable((m, m))
        sum_trace = sum_trace + cvxpy.trace(A_mat)
        A_list.append(A_mat)
        constraints = constraints + [A_mat >> 0]

    obj = cvxpy.Minimize(sum_trace)
    for n in range(N):
        sum_inverse_trace = 0
        for j in range(J):
            for k in range(K):
                sum_inverse_trace = sum_inverse_trace + cvxpy.multiply(beta_mat[n, j, k], cvxpy.inv_pos(
                    cvxpy.trace(A_list[j] @ H_mat[n, k])))
        constraints = constraints + [sum_inverse_trace <= 1]

    prob = cvxpy.Problem(obj, constraints)
    prob.solve()

    obj = 1e10
    a_list = list()
    for num in range(num_candidates):
        tmp_a_list = list()
        for j in range(J):
            candidates_a = rand_c(A_list[j].value, num_candidates=1)
            tmp_a_list.append(candidates_a[0])
        tmp_a_list = scale_to_power_constraint(tmp_a_list, beta_mat, h_mat)
        if sum([numpy.linalg.norm(tmp_a_list[j]) ** 2 for j in range(J)]) < obj:
            obj = sum([numpy.linalg.norm(tmp_a_list[j]) ** 2 for j in range(J)])
            a_list = tmp_a_list
    return a_list


def scale_to_power_constraint(a_list, beta_mat, h_mat):
    N, J, K = beta_mat.shape
    m = h_mat.shape[2]
    scale_factor = 1
    for n in range(N):
        transmit_power = 0
        for j in range(J):
            for k in range(K):
                h_vec = numpy.mat(h_mat[n, k, :]).reshape((m, 1))
                transmit_power += beta_mat[n, j, k] / numpy.linalg.norm(numpy.dot(a_list[j].T, h_vec)) ** 2
        if transmit_power > 1:
            scale_factor = max(scale_factor, numpy.sqrt(transmit_power))
    for j in range(J):
        a_list[j] = scale_factor * a_list[j]
    return a_list


def beamforming_optimization_DC(h_mat, beta_mat, cache_a_list=None, max_iter=20):
    N, J, K = beta_mat.shape
    m = h_mat.shape[2]
    # a_mat = numpy.ones((J, m))
    H_mat = numpy.ones((N, K, m, m))
    for n in range(N):
        for k in range(K):
            h_vec = numpy.mat(h_mat[n, k, :]).reshape((m, 1))
            H_mat[n, k] = numpy.dot(h_vec, h_vec.conj().T)

    pre_A_list = list()
    for j in range(J):
        pre_A = numpy.random.rand(m, m)
        pre_A = numpy.dot(pre_A, pre_A.T)
        pre_A = numpy.add(pre_A, pre_A.T)
        pre_A_list.append(pre_A)
    if cache_a_list is not None:
        for j in range(J):
            pre_A_list[j] = numpy.dot(cache_a_list[j], cache_a_list[j].T)
    sum_trace = 0
    A_list = list()
    A_subgradient_list = list()

    constraints = []
    for j in range(J):
        A_mat = cvxpy.Variable((m, m))
        A_subgradient = cvxpy.Parameter((m, m))
        sum_trace = sum_trace + cvxpy.trace(A_mat) + cvxpy.trace(A_mat) - cvxpy.trace(A_subgradient.T @ A_mat)
        A_list.append(A_mat)
        A_subgradient_list.append(A_subgradient)
        constraints = constraints + [A_mat >> 0]

    obj = cvxpy.Minimize(sum_trace)
    for n in range(N):
        sum_inverse_trace = 0
        for j in range(J):
            for k in range(K):
                sum_inverse_trace = sum_inverse_trace + cvxpy.multiply(beta_mat[n, j, k], cvxpy.inv_pos(
                    cvxpy.trace(A_list[j] @ H_mat[n, k])))
        constraints = constraints + [sum_inverse_trace <= 1]

    prob = cvxpy.Problem(obj, constraints)

    for i in range(max_iter):
        for j in range(J):
            u, s, vh = numpy.linalg.svd(pre_A_list[j])
            um = u[:, 0]
            um = numpy.mat(um).T
            A_subgradient_list[j].value = numpy.dot(um, um.T)

        prob.solve()
        for j in range(J):
            if A_list[j].value is not None:
                pre_A_list[j] = A_list[j].value

        if abs(sum(numpy.trace(pre_A_list[j]) - numpy.linalg.norm(pre_A_list[j], ord=2) for j in range(J))) < 1e-5:
            break

    a_list = list()
    for j in range(J):
        eig_values, eig_vectors = numpy.linalg.eig(pre_A_list[j])
        idx = eig_values.argmax()
        a = eig_vectors[:, idx]
        a = numpy.multiply(cmath.sqrt(eig_values[idx]), numpy.matrix(a).T).reshape((m, 1))
        a_list.append(a)
    return scale_to_power_constraint(a_list, beta_mat, h_mat)


def alternating_optimization_framework(h_mat, w_mat, P, max_iter=20):
    N, J = w_mat.shape
    _, K, m = h_mat.shape
    a_list = list()
    c_mat = numpy.zeros((J, K))
    for j in range(J):
        a_list.append(numpy.ones((m, 1)))
        # for k in range(K):
        #     c_mat[j, k] = 1 / K
        c_mat[j, j] = 1
    pre_mat = numpy.ones((J, K))
    pre_obj = 1e6
    cache_a_list = None

    for it in tqdm(range(max_iter)):
        # if numpy.linalg.norm(pre_mat - c_mat) < 1e-3:
        #     break

        # beamforming optimization
        beta_mat = numpy.zeros((N, J, K))
        for n in range(N):
            for j in range(J):
                for k in range(K):
                    beta_mat[n, j, k] = w_mat[n, j] * c_mat[j, k] ** 2 / P
        a_list = beamforming_optimization_DC(h_mat, beta_mat, cache_a_list=cache_a_list)
        # a_list = beamforming_optimization(h_mat, beta_mat)
        # cache_a_list = a_list

        new_obj = sum([numpy.linalg.norm(a_list[j]) ** 2 for j in range(J)])
        print('iter ' + str(it) + ': objective: ' + str(new_obj))
        check_constraints(c_mat, a_list, w_mat, h_mat, P)
        if abs(new_obj - pre_obj) < 1e-8:
            break
        pre_obj = new_obj

        # subcarrier allocation
        alpha_mat = numpy.zeros((N, J, K))
        for n in range(N):
            for j in range(J):
                for k in range(K):
                    h_vec = numpy.mat(h_mat[n, k, :]).reshape((m, 1))
                    alpha_mat[n, j, k] = w_mat[n, j] / P / numpy.linalg.norm(numpy.dot(a_list[j].T, h_vec)) ** 2
        c_mat = modified_subcarrier_allocation(alpha_mat)
        check_subcarrier_allocation(c_mat, alpha_mat)
        # print('c_mat: ')
        # print(c_mat)

    # restricted_idx = numpy.argmax(c_mat, axis=0)
    # for k in range(K):
    #     for j in range(J):
    #         c_mat[j, k] = c_mat[j, k] if restricted_idx[k] == j else 0
    # restricted_idx = numpy.argmax(c_mat, axis=1)
    # for j in range(J):
    #     for k in range(K):
    #         c_mat[j, k] = 1 if restricted_idx[j] == k else 0
    indicator_mat = round_indicator_mat(c_mat)
    beta_mat = numpy.zeros((N, J, K))
    for n in range(N):
        for j in range(J):
            for k in range(K):
                beta_mat[n, j, k] = w_mat[n, j] * indicator_mat[j, k] ** 2 / P
    a_list = beamforming_optimization_DC(h_mat, beta_mat)
    return a_list, indicator_mat


def check_constraints(indicator_mat, a_list, w_mat, h_mat, P):
    J, K = indicator_mat.shape
    N, _, m = h_mat.shape

    # constraint 1
    for j in range(J):
        if numpy.sum(indicator_mat[j]) != 1:
            print('constraint 1 violated: index ' + str(j) + '  value: ' + str(numpy.sum(indicator_mat[j])))

    # constraint 2
    for k in range(K):
        if numpy.sum(indicator_mat[:, k]) > 1:
            print('constraint 2 violated: index ' + str(k) + '  value: ' + str(numpy.sum(indicator_mat[:, k])))

    # constraint 3
    for n in range(N):
        transmit_power = 0
        for j in range(J):
            for k in range(K):
                h_vec = numpy.mat(h_mat[n, k, :]).reshape((m, 1))
                transmit_power += indicator_mat[j, k] ** 2 * w_mat[n, j] / numpy.linalg.norm(
                    numpy.dot(a_list[j].T, h_vec)) ** 2
        if transmit_power > P:
            print('constraint 3 violated: index ' + str(n) + '  value: ' + str(transmit_power))


def check_subcarrier_allocation(indicator_mat, alpha_mat):
    N, J, K = alpha_mat.shape

    # subcarrier constraint 1
    for j in range(J):
        if numpy.sum(indicator_mat[j]) != 1:
            print('subcarrier allocation 1 violated: index ' + str(j) + '  value: ' + str(numpy.sum(indicator_mat[j])))

    # subcarrier constraint 2
    for k in range(K):
        if numpy.sum(indicator_mat[:, k]) > 1:
            print(
                'subcarrier allocation 2 violated: index ' + str(k) + '  value: ' + str(numpy.sum(indicator_mat[:, k])))

    # subcarrier constraint 3
    for n in range(N):
        transmit_power = 0
        for j in range(J):
            for k in range(K):
                transmit_power += alpha_mat[n, j, k] * indicator_mat[j, k] ** 2
        if transmit_power > 1:
            print('subcarrier allocation 3 violated: index ' + str(n) + '  value: ' + str(transmit_power))


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
            for j in range(J):
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


def test_alternating_optimization():
    N = 5
    J = 5
    K = 10
    m = 5
    P_list = [1, 10, 20]
    w_mat = numpy.zeros((N, J))
    h_mat = numpy.random.randn(N, K, m)
    for n in range(N):
        for j in range(J):
            w_mat[n, j] = numpy.random.uniform()
    for n in range(N):
        # scale_vec = numpy.random.randint(1, 10, size=J)
        # perm = numpy.random.permutation(J)
        # w_mat[n] = w_mat[n] * scale_vec[perm]
        scale_vec = numpy.random.randint(1, 10, size=K)
        perm = numpy.random.permutation(K)
        h_mat[n] = h_mat[n] * (scale_vec[perm].reshape((K, 1)))

    for t in range(5):
        for P in P_list:
            print(P)
            c_mat = numpy.zeros((J, K))
            for j in range(J):
                c_mat[j, j] = 1
            beta_mat = numpy.zeros((N, J, K))
            for n in range(N):
                for j in range(J):
                    for k in range(K):
                        beta_mat[n, j, k] = w_mat[n, j] * c_mat[j, k] ** 2 / P
            a_list = beamforming_optimization_DC(h_mat, beta_mat)
            print(c_mat)
            print(sum([numpy.linalg.norm(a_list[j]) ** 2 for j in range(J)]))
            check_constraints(c_mat, a_list, w_mat, h_mat, P)

            a_list, c_mat = alternating_optimization_framework(h_mat, w_mat, P)
            print(c_mat)
            print(sum([numpy.linalg.norm(a_list[j]) ** 2 for j in range(J)]))
            check_constraints(c_mat, a_list, w_mat, h_mat, P)


def alternating_optimization_v2(w_mat, h_mat, sigma, P, max_iter=100):
    N, J = w_mat.shape
    _, K, m = h_mat.shape
    a_list = list()
    c_mat = numpy.zeros((J, K))
    b_mat = numpy.zeros((N, J))
    for j in range(J):
        a_list.append(numpy.ones((m, 1)))
        c_mat[j, j] = 1
    pre_obj = 1e6

    for it in tqdm(range(max_iter)):
        # transmission power optimization
        b_mat = transmit_power_alternaing_optimization_v2(w_mat, h_mat, P, c_mat, a_list)

        # beamforming optimization
        a_list = beamforming_alternating_optimization_v2(w_mat, h_mat, sigma, c_mat, b_mat)

        new_obj = objective_calculation(a_list, c_mat, h_mat, b_mat, w_mat, sigma)
        print('iter ' + str(it) + ': objective: ' + str(new_obj))
        check_constraints_v2(c_mat, w_mat, b_mat, P)
        if abs(new_obj - pre_obj) < 1e-8:
            break
        pre_obj = new_obj

        # subcarrier allocation
        c_mat = subcarrier_allocation_optimization_v2(w_mat, h_mat, a_list, b_mat)

    indicator_mat = round_indicator_mat(c_mat)
    b_mat = transmit_power_alternaing_optimization_v2(w_mat, h_mat, P, indicator_mat, a_list)
    a_list = beamforming_alternating_optimization_v2(w_mat, h_mat, sigma, c_mat, b_mat)
    return indicator_mat, b_mat, a_list


def objective_calculation(a_list, c_mat, h_mat, b_mat, w_mat, sigma):
    N, J = w_mat.shape
    _, K, m = h_mat.shape
    obj = 0
    for n in range(N):
        for j in range(J):
            sum_h_vec = numpy.zeros((m, 1))
            for k in range(K):
                sum_h_vec += c_mat[j, k] * numpy.mat(h_mat[n, k, :]).reshape((m, 1))
            obj += w_mat[n, j] * (b_mat[n, j] * numpy.sum(numpy.dot(a_list[j].T, sum_h_vec)) - 1) ** 2

    for j in range(J):
        obj += sigma * numpy.linalg.norm(a_list[j]) ** 2
    return obj


def check_constraints_v2(indicator_mat, w_mat, b_mat, P):
    J, K = indicator_mat.shape
    N, _ = w_mat.shape

    # constraint 1
    for j in range(J):
        if numpy.sum(indicator_mat[j]) != 1:
            print('constraint 1 violated: index ' + str(j) + '  value: ' + str(numpy.sum(indicator_mat[j])))

    # constraint 2
    for k in range(K):
        if numpy.sum(indicator_mat[:, k]) > 1:
            print('constraint 2 violated: index ' + str(k) + '  value: ' + str(numpy.sum(indicator_mat[:, k])))

    # constraint 3
    for n in range(N):
        transmit_power = 0
        for j in range(J):
            transmit_power += b_mat[n, j] ** 2 * w_mat[n, j]
        if transmit_power > P:
            print('constraint 3 violated: index ' + str(n) + '  value: ' + str(transmit_power))


def transmit_power_alternaing_optimization_v2(w_mat, h_mat, P, indicator_mat, a_list, max_iter=200):
    N, J = w_mat.shape
    _, K, m = h_mat.shape
    alpha_mat = numpy.zeros((N, J, K))
    square_alpha_mat = numpy.zeros((N, J, K))
    for n in range(N):
        for j in range(J):
            for k in range(K):
                h_vec = numpy.mat(h_mat[n, k, :]).reshape((m, 1))
                alpha_mat[n, j, k] = indicator_mat[j, k] * numpy.sum(numpy.dot(a_list[j].T, h_vec))
                square_alpha_mat[n, j, k] = alpha_mat[n, j, k] ** 2

    eta = 0.01
    lambda_vec = numpy.ones(N)
    b_mat = numpy.zeros((N, J))
    for it in range(max_iter):
        # multipler update
        for n in range(N):
            lambda_vec[n] += eta * (numpy.sum(numpy.multiply(w_mat[n], b_mat[n])) - P)
            lambda_vec[n] = 0 if lambda_vec[n] < 0 else lambda_vec[n]
        # variable update
        for n in range(N):
            for j in range(J):
                b_mat[n, j] = (2 * numpy.sum(alpha_mat[n, j]) - lambda_vec[n]) / (2 * square_alpha_mat[n, j] ** 2)

    return b_mat


def beamforming_alternating_optimization_v2(w_mat, h_mat, sigma, indicator_mat, b_mat):
    N, J = w_mat.shape
    _, K, m = h_mat.shape
    a_list = list()
    alpha_mat = numpy.zeros((N, J, m))
    for n in range(N):
        for j in range(J):
            sum_h_vec = numpy.zeros(m)
            for k in range(K):
                sum_h_vec += indicator_mat[j, k] * h_mat[n, k, :] * b_mat[n, j]
            alpha_mat[n, j] = sum_h_vec

    for j in range(J):
        sum_tmp_vec = numpy.sum((m, 1))
        sum_tmp_value = 0
        for n in range(N):
            sum_tmp_vec += w_mat[n, j] * numpy.mat(alpha_mat[n, j]).reshape((m, 1))
            sum_tmp_value += w_mat[n, j] * numpy.linalg.norm(numpy.mat(alpha_mat[n, j]).reshape((m, 1))) ** 2
        tmp_a = sum_tmp_vec / (sigma + sum_tmp_value)
        a_list.append(tmp_a)

    return a_list


def subcarrier_allocation_optimization_v2(w_mat, h_mat, a_list, b_mat, max_iter=200):
    N, J = w_mat.shape
    _, K, m = h_mat.shape
    indicator_mat = numpy.zeros((J, K))
    alpha_mat = numpy.zeros((N, J, K))
    square_alpha_mat = numpy.zeros((N, J, K))
    for n in range(N):
        for j in range(J):
            for k in range(K):
                alpha_mat[n, j, k] = b_mat[n, j] * numpy.sum(
                    numpy.dot(a_list[j].T, numpy.mat(h_mat[n, k, :]).reshape((m, 1))))
                square_alpha_mat[n, j, k] = alpha_mat[n, j, k] ** 2
    lambda_vec = numpy.ones(K)
    # mu_vec = numpy.ones(J)
    eta_list = [0.01, 0.01]

    for it in range(max_iter):
        # multipler update
        lambda_vec = lambda_vec + eta_list[0] * (numpy.sum(indicator_mat, axis=0) - 1)
        for k in range(K):
            lambda_vec[k] = 0 if lambda_vec[k] < 0 else lambda_vec[k]

        # indicator update
        for j in range(J):
            value_vec = numpy.zeros(K)
            sub_value_vec = numpy.zeros(K)
            for k in range(K):
                sub_value_vec[k] = lambda_vec[k] - 2 * numpy.sum(numpy.multiply(w_mat[:, j], alpha_mat[:, j, k]))
                value_vec[k] = lambda_vec[k] - 2 * numpy.sum(
                    numpy.multiply(w_mat[:, j], alpha_mat[:, j, k])) + 2 * numpy.sum(
                    numpy.multiply(w_mat[:, j], square_alpha_mat[:, j, k]))

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
                mu_vec = numpy.zeros(J)
                for k in range(K):
                    for inner_it in range(max_iter):
                        divider = 2 * numpy.sum(numpy.multiply(w_mat[:, j], square_alpha_mat[:, j, k]))
                        indicator_mat[j, k] = 0 if sub_value_vec[k] >= mu_vec[j] else (mu_vec[j] - sub_value_vec[
                            k]) / divider
                        if numpy.sum(indicator_mat[j]) >= 1:
                            break
                        mu_vec[j] += eta_list[0]


def test_solver(H_list, k):
    V = cvxpy.Variable((k, k))
    obj = cvxpy.Minimize(cvxpy.trace(V))
    constraints = [V >> 0]
    sum_trace = 0
    for i in range(len(H_list)):
        sum_trace += cvxpy.inv_pos(cvxpy.trace(V @ H_list[i]))
    constraints = constraints + [sum_trace <= 1]

    prob = cvxpy.Problem(obj, constraints)
    prob.solve()


if __name__ == '__main__':
    # N = 5
    # J = 50
    # K = 100
    # m = 5
    # P = 100
    # a_list = list()
    # w_mat = numpy.zeros((N, J))
    # h_mat = numpy.random.randn(N, K, m)
    # for j in range(J):
    #     a_list.append(numpy.random.randn(m, 1))
    # for n in range(N):
    #     for j in range(J):
    #         w_mat[n, j] = numpy.random.uniform()
    #
    # alpha_mat = numpy.zeros((N, J, K))
    # for n in range(N):
    #     for j in range(J):
    #         for k in range(K):
    #             h_nk = numpy.mat(h_mat[n, k, :]).reshape((m, 1))
    #             alpha_mat[n, j, k] = numpy.sqrt(P / w_mat[n, j]) * numpy.linalg.norm(numpy.dot(a_list[j].T, h_nk))
    #
    # # subcarrier_allocation(alpha_mat)
    # subcarrier_allocation_v2(alpha_mat)

    # k = 5
    # H_list = list()
    # for i in range(10):
    #     H_list.append(numpy.random.randn(k, k))
    # test_solver(H_list, k)

    test_alternating_optimization()
