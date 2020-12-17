import numpy as np
from numpy import linalg as la
import cvxopt
import cvxopt.solvers
import pandas as pd 
from sklearn.svm import SVC
import sys
import os 
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

""" 
sklearn API compliant bare-minimum implementation of 
support vector machines using cvxopt as QP solver with
kernel support.
"""
cvxopt.solvers.options['show_progress'] = False

class CVXOPT_KERNEL_SVM():
    def __init__(self, kernel="linear", C=None):
        self.kernel = kernel
        self.C = C

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Gram matrix
        if (self.kernel=="linear"):
            K = self.linear_kernel(X, X)
        elif (self.kernel=="rbf"):
            K = self.rbf_kernel(X, X)
        else:
            exit("Declare your own kernel within the SVM class!")

        # Convert the data to the format that is needed by cvxopt.
        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(-np.ones(n_samples))
        A = cvxopt.matrix(y.reshape(1, -1))
        b = cvxopt.matrix(0.0)
        G = cvxopt.matrix(np.vstack((np.diag(np.ones(n_samples) * -1), np.identity(n_samples))))
        h = cvxopt.matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lambdas are stored in a.
        a = np.squeeze(solution['x'])

        # Filter the support vectors that have nonzero lambdas.
        sv = a > 1e-5
        ind = np.arange(a.shape[0])[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]

        # Weight vector
        if (self.kernel=="linear"):
            self.w = np.sum(np.expand_dims(self.a, axis=1) * self.sv_y * self.sv, axis=0)
            self.b = (np.sum(self.sv_y) - np.sum(self.sv @ np.expand_dims(self.w, axis=1))) / a.shape[0]
        else:
            self.w = None
            self.b = (np.sum(self.sv_y) - np.sum(self.a * self.sv_y * K[ind,:][:, sv])) / a.shape[0]

    def project(self, X):
        if (self.w is not None):
            return np.dot(X, self.w) + self.b
        else:
            # y_predict = np.zeros(len(X))
            y_predict = np.sum(np.squeeze(np.multiply(self.a, self.sv_y.T).T) * self.rbf_kernel(self.sv, X).T, axis=1)
            # for i in np.arange(len(X)):
            #     s = np.squeeze(np.multiply(self.a, self.sv_y.T).T) * self.rbf_kernel(self.sv, X[i])
            #     y_predict[i] = np.sum(s)
            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))

    def score(self, X, y):
        preds = self.predict(X)
        labels = np.squeeze(y)
        assert preds.shape == labels.shape
        return np.mean(preds == labels)

    def linear_kernel(self, X_1, X_2):
        return X_1.dot(X_2.T)

    def rbf_kernel(self, X_1, X_2):
        gamma = 1.0 / (X_1.var()*X_1.shape[1])
        X_1_norm = la.norm(X_1, axis=-1)
        X_2_norm = la.norm(X_2, axis=-1)
        retval =  np.exp(-gamma * (X_1_norm[:, None] + X_2_norm[None, :] - 2 * np.dot(X_1, X_2.T)))
        return retval




def kernel_SVM(dataset: str , train_and_validate=True) -> None:
    C_LIST = np.asarray([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])

    mean_tr_error_list = []
    tr_error_std_list = []
    mean_cv_error_list = []
    cv_error_std_list = []

    df = pd.read_csv(dataset, header=None)
    X = np.asarray(df[df.columns[:-1]])
    y = np.asarray(df[df.columns[-1:]])
    y[y == 0] = -1
    X_main, X_holdout, y_main, y_holdout = train_test_split(X, y, 
                                    test_size=0.2, 
                                    stratify=y)

    skf = StratifiedKFold(n_splits=10, shuffle=True)


    """ FOR LINEAR  SVM """
    if train_and_validate:
        for C in C_LIST:
            linear_clf = CVXOPT_KERNEL_SVM(C=C, kernel='linear')
            fold_idx = 0
            tr_score_list = []
            cv_score_list = []
            for tr_idx, cv_idx in skf.split(X_main, y_main):
                print("====== FOLD NUMBER => ", fold_idx, " ======")
                X_tr, X_cv = X_main[tr_idx], X_main[cv_idx]
                y_tr, y_cv = y_main[tr_idx], y_main[cv_idx]
                linear_clf.fit(X_tr,y_tr)
                tr_score = linear_clf.score(X_tr, y_tr)
                cv_score = linear_clf.score(X_cv, y_cv)
                tr_score_list.append(tr_score)
                cv_score_list.append(cv_score)
                print("TR score is:", linear_clf.score(X_tr, y_tr))
                print("CV score is:", linear_clf.score(X_cv, y_cv))
                fold_idx += 1

            tr_error_list = 1 - np.asarray(tr_score_list)
            cv_error_list = 1 - np.asarray(cv_score_list)
            mean_tr_error_list.append(np.mean(tr_error_list))
            tr_error_std_list.append(np.std(tr_error_list))
            mean_cv_error_list.append(np.mean(cv_error_list))
            cv_error_std_list.append(np.std(cv_error_list))

        print("====== MEAN TRAINING ERRORS FOR LINEAR: \n", mean_tr_error_list)
        print("====== TRAINING ERRORS STD FOR LINEAR: \n", tr_error_std_list)
        print("====== MEAN CROSS VALIDATION ERRORS FOR LINEAR: \n", mean_cv_error_list)
        print("====== CROSS VALIDATION ERRORS STD FOR LINEAR: \n", cv_error_std_list)
    
    selected_C = 1e-3
    print("====== LINEAR TRAINING OVER THE WHOLE BATCH AND TESTING OVER THE HOLDOUT SET with optimal C= ", selected_C ," ======")
    lin_clf = CVXOPT_KERNEL_SVM(C=selected_C, kernel="linear")
    lin_clf.fit(X_main,y_main)
    print("====== LINEAR TRAINING ERROR OVER THE WHOLE BATCH IS ", 1-lin_clf.score(X_main, y_main)," =======")
    print("====== LINEAR TESTING ERROR OVER THE HOLDOUT BATCH IS ", 1-lin_clf.score(X_holdout, y_holdout)," =======")


    """ FOR RBF SVM """
    if train_and_validate:
        mean_tr_error_list = []
        tr_error_std_list = []
        mean_cv_error_list = []
        cv_error_std_list = []

        for C in C_LIST:
            rbf_clf = CVXOPT_KERNEL_SVM(C=C, kernel='rbf')
            fold_idx = 0
            tr_score_list = []
            cv_score_list = []
            for tr_idx, cv_idx in skf.split(X_main, y_main):
                print("====== FOLD NUMBER => ", fold_idx, " ======")
                X_tr, X_cv = X_main[tr_idx], X_main[cv_idx]
                y_tr, y_cv = y_main[tr_idx], y_main[cv_idx]
                rbf_clf.fit(X_tr,y_tr)
                tr_score = rbf_clf.score(X_tr, y_tr)
                cv_score = rbf_clf.score(X_cv, y_cv)
                tr_score_list.append(tr_score)
                cv_score_list.append(cv_score)
                print("TR score is:", rbf_clf.score(X_tr, y_tr))
                print("CV score is:", rbf_clf.score(X_cv, y_cv))
                fold_idx += 1
                
            tr_error_list = 1 - np.asarray(tr_score_list)
            cv_error_list = 1 - np.asarray(cv_score_list)
            mean_tr_error_list.append(np.mean(tr_error_list))
            tr_error_std_list.append(np.std(tr_error_list))
            mean_cv_error_list.append(np.mean(cv_error_list))
            cv_error_std_list.append(np.std(cv_error_list))

        print("====== MEAN TRAINING ERRORS FOR RBF: \n", mean_tr_error_list)
        print("====== TRAINING ERRORS STD FOR RBF: \n", tr_error_std_list)
        print("====== MEAN CROSS VALIDATION ERRORS FOR RBF: \n", mean_cv_error_list)
        print("====== CROSS VALIDATION ERRORS STD FOR RBF: \n", cv_error_std_list)

    selected_C = 1e-1
    print("====== RBF TRAINING OVER THE WHOLE BATCH AND TESTING OVER THE HOLDOUT SET with optimal C= ", selected_C ," ======")
    rbf_clf = CVXOPT_KERNEL_SVM(C=selected_C, kernel='rbf')
    rbf_clf.fit(X_main,y_main)
    print("====== RBF TRAINING ERROR OVER THE WHOLE BATCH IS ", 1-rbf_clf.score(X_main, y_main)," =======")
    print("====== RBF TESTING ERROR OVER THE HOLDOUT BATCH IS ", 1-rbf_clf.score(X_holdout, y_holdout)," =======")


if __name__ == "__main__":
    
    kernel_SVM(dataset="hw2_data_2020.csv", train_and_validate=False)

    exit()


