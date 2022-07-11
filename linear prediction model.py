import numpy
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numba
import sys

home_dir = '../'
sys.path.append(home_dir)


@numba.njit()
def stable_softmax(w_x):
    """
    accelerate the process of computing softmax function
    compute exp(xw) / sum_{i=1}^{C}{exp(xw[i])}
    """
    n, c = w_x.shape
    exps = numpy.exp(w_x - numpy.max(w_x))
    sum_exps = numpy.sum(exps.reshape(n, c, 1), axis=1)
    return numpy.divide(exps, sum_exps.reshape(n, 1))


@numba.njit()
def accelerate_obj(p, s, y_vec):
    """
    accelerate the process of computing objective function
    compute sum_{i=1}^{s}{ -log[ exp(xw[y_i]) / sum_{j=1}^{C}{exp(xw[j])} ] }
    """
    log_likelihood = 0
    for i in range(s):
        log_likelihood = log_likelihood - numpy.log(p[i, y_vec[i, 0]])
    return log_likelihood


@numba.njit()
def accelerate_hessian(p, x_mat, g_vectors, eye_mat, s, num_class, d):
    """
    accelerate the process of compute hessian
    compute H^-1 * g
    """
    p_vectors = numpy.zeros((num_class, d, 1))
    for i in range(num_class):
        p_i = p[i].reshape(s, 1)
        p_i = p_i - numpy.power(p_i, 2)
        pxx = numpy.dot(x_mat.T, numpy.multiply(x_mat, p_i))
        hessian = numpy.add(pxx / s, eye_mat)
        if numpy.linalg.det(hessian) == 0:
            hessian_inv = numpy.linalg.pinv(hessian)
        else:
            hessian_inv = numpy.linalg.inv(hessian)
        p_vectors[i] = numpy.dot(hessian_inv, g_vectors[i])
    return p_vectors


class CrossEntropySolver:
    def __init__(self, x_mat=None, y_vec=None, num_class=None, x_test=None, y_test=None):
        if (x_mat is not None) and (y_vec is not None):
            self.n, self.d = x_mat.shape
            self.x_mat = x_mat
            self.y_vec = y_vec
            self.num_class = num_class
            self.x_test = x_test
            self.y_test = y_test
            self.y_mat = numpy.zeros((self.n, self.num_class))
            for i in range(self.n):
                self.y_mat[i, self.y_vec[i, 0]] = 1

    def fit(self, x_mat, y_vec):
        self.n, self.d = x_mat.shape
        self.x_mat = x_mat
        self.y_vec = y_vec

    def softmax(self, w_x):
        """
        unstable version, overflow problem exists
        """
        exps = numpy.exp(w_x)
        return numpy.divide(exps, numpy.mat(numpy.sum(exps, axis=1)).reshape(self.n, 1))

    def predication(self, w_vec):
        w_vectors = numpy.split(w_vec, self.num_class, axis=0)
        w_x = []
        for i in range(self.num_class):
            w_x.append(numpy.dot(self.x_test, w_vectors[i].reshape(self.d, 1)))
        w_x = numpy.concatenate(w_x, axis=1)
        p = stable_softmax(w_x)
        pred = numpy.argmax(p, axis=1)

        cnt = 0
        tot = self.y_test.shape[0]
        for i in range(tot):
            if pred[i] == self.y_test[i, 0]:
                cnt += 1
        return cnt / tot

    def obj_fun(self, w_vec, *args):
        gamma = args[0]
        w_vectors = numpy.split(w_vec, self.num_class, axis=0)

        reg = 0
        w_x = []
        for i in range(self.num_class):
            w_x.append(numpy.dot(self.x_mat, w_vectors[i].reshape(self.d, 1)))
            reg += (gamma / 2) * (numpy.linalg.norm(w_vectors[i]) ** 2)
        w_x = numpy.concatenate(w_x, axis=1)

        p = stable_softmax(w_x)
        # log_likelihood = 0
        # for i in range(self.n):
        #     log_likelihood = log_likelihood - numpy.log(p[i, self.y_vec[i]])
        log_likelihood = accelerate_obj(p, self.n, self.y_vec)

        return log_likelihood / self.n + reg

    def grad(self, w_vec, *args):
        gamma = args[0]
        w_vectors = numpy.split(w_vec, self.num_class, axis=0)

        w_x = []
        for i in range(self.num_class):
            w_x.append(numpy.dot(self.x_mat, w_vectors[i].reshape(self.d, 1)))
        w_x = numpy.concatenate(w_x, axis=1)

        p = stable_softmax(w_x)
        grad = numpy.zeros((self.num_class * self.d, 1))
        for i in range(self.n):
            x = numpy.mat(self.x_mat[i]).reshape(self.d, 1)
            p_i = numpy.mat(p[i]).reshape(self.num_class, 1)
            p_i[self.y_vec[i], 0] = p_i[self.y_vec[i], 0] - 1
            grad = numpy.add(grad, numpy.kron(p_i, x))
        grad = grad / self.n
        # grad = accelerate_gradient(p.reshape(self.n, self.num_class, 1), self.x_mat.reshape(self.n, self.d, 1),
        #                            self.y_vec, self.n, self.num_class, self.d)
        return grad + numpy.multiply(gamma, w_vec)

    def gradient_descent(self, gamma, max_iter=50, tol=1e-15):
        """
        gradient descent solver for the minimization of cross entropy loss function
        """
        w_vec = numpy.random.randn(self.num_class * self.d, 1) * 0.01
        eta_list = 1 / (2 ** numpy.arange(0, 10))
        args = (gamma,)

        for t in range(max_iter):
            grad = self.grad(w_vec, args)
            grad_norm = numpy.linalg.norm(grad)
            obj = self.obj_fun(w_vec, *args)
            acc = self.predication(w_vec)
            print('Cross Entropy Solver: Iter ' + str(t) + ', L2 norm of gradient = ' + str(
                grad_norm) + ', objective value = ' + str(obj) + ', predication = ' + str(acc))
            if grad_norm < tol:
                print('The change of objective value is smaller than ' + str(tol))
                break

            eta = 0
            obj_val = self.obj_fun(w_vec, *args)
            if grad_norm > tol:
                pg = - 0.5 * numpy.sum(numpy.multiply(grad, grad))
                for eta in eta_list:
                    obj_val_new = self.obj_fun(w_vec - eta * grad, *args)
                    if obj_val_new < obj_val + eta * pg:
                        break
            else:
                eta = 0.5
            w_vec = w_vec - eta * grad
            print(w_vec)

        return w_vec

    def exact_newton(self, gamma, max_iter=50, tol=1e-15):
        """
        exact newton (compute exact newton update direction vector) solver for the minimization of cross entropy loss function
        """
        w_vec = numpy.random.randn(self.num_class * self.d, 1) * 0.01
        eta_list = 1 / (2 ** numpy.arange(0, 10))
        eye_mat = gamma * numpy.eye(self.d)
        args = (gamma,)

        for t in range(max_iter):
            # print('calculate grad:')
            grad = self.grad(w_vec, *args)
            grad_norm = numpy.linalg.norm(grad)
            obj = self.obj_fun(w_vec, *args)
            acc = self.predication(w_vec)
            print('Cross Entropy Solver: Iter ' + str(t) + ', L2 norm of gradient = ' + str(
                grad_norm) + ', objective value = ' + str(obj) + ', predication = ' + str(acc))
            if grad_norm < tol:
                print('The change of objective value is smaller than ' + str(tol))
                break

            w_vectors = numpy.split(w_vec, self.num_class, axis=0)
            w_x = []
            for i in range(self.num_class):
                w_x.append(numpy.dot(self.x_mat, w_vectors[i].reshape(self.d, 1)))
            w_x = numpy.concatenate(w_x, axis=1)

            g_vectors = numpy.array(numpy.split(grad, self.num_class, axis=0))
            p_vectors = numpy.zeros((self.num_class, self.d, 1))
            p = stable_softmax(w_x)
            for i in range(self.num_class):
                p_i = numpy.mat(p.T[i]).reshape(self.n, 1)
                p_i = p_i - numpy.power(p_i, 2)
                pxx = numpy.dot(self.x_mat.T, numpy.multiply(self.x_mat, p_i))
                hessian = numpy.add(pxx / self.n, eye_mat)
                if numpy.linalg.det(hessian) == 0:
                    hessian_inv = numpy.linalg.pinv(hessian)
                else:
                    hessian_inv = numpy.linalg.inv(hessian)
                p_vectors[i] = numpy.dot(hessian_inv, g_vectors[i])
            # p_vectors = accelerate_hessian(p.reshape(self.num_class, self.n, 1), self.x_mat, g_vectors, eye_mat, self.n,
            #                                self.num_class, self.d)
            p_vec = numpy.reshape(p_vectors, (self.num_class * self.d, 1))

            eta = 0
            obj_val = self.obj_fun(w_vec, *args)
            if grad_norm > tol:
                pg = - 0.5 * numpy.sum(numpy.multiply(p_vec, grad))
                for eta in eta_list:
                    obj_val_new = self.obj_fun(w_vec - eta * p_vec, *args)
                    if obj_val_new < obj_val + eta * pg:
                        break
            else:
                eta = 0
            # print(eta)
            w_vec = w_vec - eta * p_vec

        return w_vec


def normalization(x_train, x_test):
    """
    normalization of data
    """
    mean = numpy.mean(x_train)
    std_ev = numpy.sqrt(numpy.var(x_train))
    normalized_x_train = numpy.divide(numpy.subtract(x_train, mean), std_ev)
    mean = numpy.mean(x_test)
    std_ev = numpy.sqrt(numpy.var(x_test))
    normalized_x_test = numpy.divide(numpy.subtract(x_test, mean), std_ev)
    return normalized_x_train, normalized_x_test


def normal_inference(w_vec, num_class, dim, x_test, y_test):
    w_vectors = numpy.split(w_vec, num_class, axis=0)
    w_x = []
    for i in range(num_class):
        w_x.append(numpy.dot(x_test, w_vectors[i].reshape(dim, 1)))
    w_x = numpy.concatenate(w_x, axis=1)
    p = stable_softmax(w_x)
    pred = numpy.argmax(p, axis=1)

    cnt = 0
    tot = y_test.shape[0]
    for i in range(tot):
        if pred[i] == y_test[i, 0]:
            cnt += 1
    return cnt / tot


def split_inference(w_vec_by_devices, num_class, n_devices, x_test, y_test):
    w_x_list = []
    for i in range(num_class):
        w_x = numpy.zeros((x_test.shape[0], 1))
        for j in range(n_devices):
            w_x = numpy.add(w_x, numpy.dot(x_test[:, j * 49: (j + 1) * 49], w_vec_by_devices[j][i]))
        w_x_list.append(w_x)
    w_vec = numpy.concatenate(w_x_list, axis=1)
    p = stable_softmax(w_vec)
    pred = numpy.argmax(p, axis=1)

    cnt = 0
    tot = y_test.shape[0]
    for i in range(tot):
        if pred[i] == y_test[i, 0]:
            cnt += 1
    return cnt / tot


def aircomp_based_split_inference(w_vec_by_devices, num_class, n_devices, tau2, x_test, y_test):
    w_x_list = []
    for i in range(num_class):
        # H_square = 0.5 * numpy.random.exponential(0.5, [x_test.shape[0], 1])
        noise = numpy.random.normal(0, tau2, (x_test.shape[0], 1))
        w_x = numpy.zeros((x_test.shape[0], 1))
        for j in range(n_devices):
            transmit_signal = numpy.dot(x_test[:, j * 49: (j + 1) * 49], w_vec_by_devices[j][i])
            w_x = numpy.add(w_x, transmit_signal)
        received_signal = numpy.add(w_x, noise)
        w_x_list.append(received_signal)
    w_vec = numpy.concatenate(w_x_list, axis=1)
    p = stable_softmax(w_vec)
    pred = numpy.argmax(p, axis=1)

    cnt = 0
    tot = y_test.shape[0]
    for i in range(tot):
        if pred[i] == y_test[i, 0]:
            cnt += 1
    return cnt / tot


if __name__ == '__main__':
    print('PyCharm')
    training_data = datasets.FashionMNIST(root="../Resources", train=True, download=True, transform=ToTensor())
    test_data = datasets.FashionMNIST(root="../Resources", train=False, download=True, transform=ToTensor())
    # print(training_data)
    # print(test_data)

    train_data_loader = DataLoader(training_data, batch_size=60000)
    test_data_loader = DataLoader(test_data, batch_size=10000)
    train_X, train_y = next(iter(train_data_loader))
    train_X = train_X.numpy()
    train_y = train_y.numpy()
    test_X, test_y = next(iter(test_data_loader))
    test_X = test_X.numpy()
    test_y = test_y.numpy()
    print(train_X.shape)
    print(test_y.shape)

    train_X = train_X.reshape(60000, 28 * 28)
    train_y = numpy.array(train_y).reshape(60000, 1)
    test_X = test_X.reshape(10000, 28 * 28)
    test_y = numpy.array(test_y).reshape(10000, 1)

    train_X, test_X = normalization(train_X, test_X)

    num_class = numpy.max(train_y) + 1
    # solver = CrossEntropySolver(train_X, train_y, num_class, test_X, test_y)
    # gamma_list = [1e-8, 1e-6, 1e-4]
    # for gamma in gamma_list:
    #     w_opt = solver.exact_newton(gamma)
    #     out_file_name = home_dir + 'Resources/cross_entropy_solver_w_opt_gamma_' + str(gamma) + '.npz'
    #     numpy.savez(out_file_name, w_opt=w_opt)

    gamma = 1e-4
    n_devices = 16
    dim = 28 * 28
    out_file_name = home_dir + 'Resources/cross_entropy_solver_w_opt_gamma_' + str(gamma) + '.npz'
    npz_file = numpy.load(out_file_name, allow_pickle=True)
    w_opt = npz_file['w_opt']

    print('normal inference: ', normal_inference(w_opt, num_class, dim, test_X, test_y))

    # print(w_opt.shape)
    w_vectors = numpy.split(w_opt, num_class * n_devices, axis=0)
    # print(w_vectors)
    w_vectors_by_devices = []
    # for i in range(n_devices):
    #     w_vectors_by_devices.append([])
    for i in range(n_devices):
        w_list = []
        for j in range(num_class):
            w_list.append(w_vectors[j * n_devices + i])
        w_vectors_by_devices.append(w_list.copy())
    # x_test_by_devices = test_X.reshape((10000, 49, 16))

    print('split inference: ', split_inference(w_vectors_by_devices, num_class, n_devices, test_X, test_y))

    tau2 = 4
    print('aircomp-based split inference: ',
          aircomp_based_split_inference(w_vectors_by_devices, num_class, n_devices, tau2, test_X, test_y))
