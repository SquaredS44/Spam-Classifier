import numpy as np
from sklearn.svm import SVC


def dataset3Params(x, y, xval, yval):
    err_row = 0
    error = np.zeros((64, 3))
    for c_test in np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]):
        for sigma_test in np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]):
            # Train SVM
            svm = SVC(kernel='rbf', C=c_test, gamma=sigma_test)
            svm.fit(x, y.flatten())
            # Compute the predictions on the cross validation set
            pred = svm.predict(xval)
            # Compute the error on the cross validation set
            error[err_row, 0] = np.mean(np.not_equal(pred,yval))
            error[err_row, 1] = c_test
            error[err_row, 2] = sigma_test
            err_row = err_row + 1
    index = np.argmin(error[:, 0])
    print(error)
    c = error[index, 1]
    sig = error[index, 2]
    return c, sig
