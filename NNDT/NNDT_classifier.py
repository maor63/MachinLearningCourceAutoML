from functools import reduce
import numpy as np
import torch
from sklearn.base import BaseEstimator

class NNDT(BaseEstimator):
    def __init__(self, cut_count=4, lr=0.01, temprature=0.1, epochs=10):
        self.cut_count = cut_count
        self.lr = lr
        self.temprature = temprature
        self.epochs = epochs

    def fit(self, X, y):
        X_input = torch.from_numpy(X.astype(np.float32))
        y_input = torch.from_numpy(y)
        num_class = len(np.unique(y_input))

        num_cut = [1] * self.cut_count

        num_leaf = np.prod(np.array(num_cut) + 1)
        self.cut_points_list = [torch.rand([i], requires_grad=True) for i in num_cut]
        self.leaf_score = torch.rand([num_leaf, num_class], requires_grad=True)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.cut_points_list + [self.leaf_score], lr=self.lr)

        for i in range(self.epochs):
            self.optimizer.zero_grad()
            y_pred = self.nn_decision_tree(X_input, self.cut_points_list, self.leaf_score, temperature=self.temprature)
            loss = self.loss_function(y_pred, y_input)
            loss.backward()
            self.optimizer.step()
            # if i % 1 == 0:
            #     print(loss.detach().numpy())
        # print('error rate %.2f' % (1 - np.mean(np.argmax(y_pred.detach().numpy(), axis=1) == np.argmax(y, axis=1))))

    def predict_proba(self, X):
        X_input = torch.from_numpy(X.astype(np.float32))
        y_pred = self.nn_decision_tree(X_input, self.cut_points_list, self.leaf_score, temperature=self.temprature)
        return y_pred.detach().numpy()

    def predict(self, X):
        y_proba = self.predict_proba(X)
        return np.argmax(y_proba, axis=1)

    def torch_kron_prod(self, a, b):
        res = torch.einsum('ij,ik->ijk', [a, b])
        res = torch.reshape(res, [-1, np.prod(res.shape[1:])])
        return res

    def torch_bin(self, x, cut_points, temperature=0.1):
        # x is a N-by-1 matrix (column vector)
        # cut_points is a D-dim vector (D is the number of cut-points)
        # this function produces a N-by-(D+1) matrix, each row has only one element being one and the rest are all zeros
        D = cut_points.shape[0]
        W = torch.reshape(torch.linspace(1.0, D + 1.0, D + 1), [1, -1])
        cut_points, _ = torch.sort(cut_points)  # make sure cut_points is monotonically increasing
        b = torch.cumsum(torch.cat([torch.zeros([1]), -cut_points], 0), 0)
        h = torch.matmul(x, W) + b
        res = torch.exp(h - torch.max(h))
        res = res / torch.sum(res, dim=-1, keepdim=True)
        return h

    def nn_decision_tree(self, x, cut_points_list, leaf_score, temperature=0.1):
        # cut_points_list contains the cut_points for each dimension of feature
        leaf = reduce(self.torch_kron_prod,
                      map(lambda z: self.torch_bin(x[:, z[0]:z[0] + 1], z[1], temperature), enumerate(cut_points_list)))
        return torch.matmul(leaf, leaf_score)