import cvxpy
import numpy


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


def modified_subcarrier_allocation(alpha_mat, max_iter=100):
    N, J, K = alpha_mat.shape
    indicator_mat = numpy.zeros((J, K))
    mu_vec = numpy.zeros(J)
    lambda_vec = numpy.zeros(K)
    nu_vec = numpy.zeros(N)
    eta = 0.005
    pre_mat = numpy.ones((J, K))

    for it in range(max_iter):
        if numpy.linalg.norm(pre_mat - indicator_mat) < 1e-3:
            break

        # multipler update
        mu_vec = mu_vec + eta * numpy.sum(indicator_mat, axis=1)
        lambda_vec = lambda_vec + eta * numpy.sum(indicator_mat, axis=0)
        for n in range(N):
            nu_vec[n] = nu_vec[n] + eta * numpy.sum(numpy.multiply(alpha_mat[n], numpy.square(indicator_mat)))

        pre_mat = indicator_mat.copy()
        # indicator update
        for j in range(J):
            for k in range(K):
                tmp = 2 * numpy.sum(numpy.multiply(nu_vec, alpha_mat[:, j, k]))
                if lambda_vec[k] >= mu_vec[j]:
                    indicator_mat[j, k] = 0
                elif lambda_vec[k] + tmp <= mu_vec[j]:
                    indicator_mat[j, k] = 1
                else:
                    indicator_mat[j, k] = (mu_vec[j] - lambda_vec[k]) / tmp

    return indicator_mat


def beamforming_optimization(h_mat, beta_mat):
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

    a_list = list()
    for j in range(J):
        candidates_a = rand_c(A_list[j].value, num_candidates=1)
        a_list.append(candidates_a)
    return a_list


def alternating_optimization_framework(h_mat, w_mat, P, max_iter=100):
    N, J = w_mat.shape
    _, K, m = h_mat.shape
    a_list = list()
    c_mat = numpy.zeros((J, K))
    for j in range(J):
        a_list.append(numpy.ones(m, 1))
        c_mat[j, j] = 1
    pre_mat = numpy.ones((J, K))

    for it in range(max_iter):
        if numpy.linalg.norm(pre_mat - c_mat) < 1e-3:
            break

        # beamforming optimization
        beta_mat = numpy.zeros((N, J, K))
        for n in range(N):
            for j in range(J):
                for k in range(K):
                    beta_mat[n, j, k] = c_mat[j, k] * w_mat[n, j] / P
        a_list = beamforming_optimization(h_mat, beta_mat)

        # subcarrier allocation
        alpha_mat = numpy.zeros((N, J, K))
        for n in range(N):
            for j in range(J):
                for k in range(K):
                    alpha_mat[n, j, k] = P * w_mat / numpy.linalg.norm(numpy.dot(a_list[j], h_mat[n, k])) ** 2
        c_mat = modified_subcarrier_allocation(alpha_mat)

    restricted_idx = numpy.argmax(c_mat, axis=0)
    for j in range(J):
        for k in range(K):
            c_mat[j, k] = 1 if restricted_idx[j] == k else 0
    beta_mat = numpy.zeros((N, J, K))
    for n in range(N):
        for j in range(J):
            for k in range(K):
                beta_mat[n, j, k] = c_mat[j, k] * w_mat[n, j] / P
    a_list = beamforming_optimization(h_mat, beta_mat)
    return a_list, c_mat


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
    N = 5
    J = 50
    K = 100
    m = 5
    P = 100
    a_list = list()
    w_mat = numpy.zeros((N, J))
    h_mat = numpy.random.randn(N, K, m)
    for j in range(J):
        a_list.append(numpy.random.randn(m, 1))
    for n in range(N):
        for j in range(J):
            w_mat[n, j] = numpy.random.uniform()

    alpha_mat = numpy.zeros((N, J, K))
    for n in range(N):
        for j in range(J):
            for k in range(K):
                h_nk = numpy.mat(h_mat[n, k, :]).reshape((m, 1))
                alpha_mat[n, j, k] = numpy.sqrt(P / w_mat[n, j]) * numpy.linalg.norm(numpy.dot(a_list[j].T, h_nk))

    # subcarrier_allocation(alpha_mat)
    subcarrier_allocation_v2(alpha_mat)

    # k = 5
    # H_list = list()
    # for i in range(10):
    #     H_list.append(numpy.random.randn(k, k))
    # test_solver(H_list, k)
