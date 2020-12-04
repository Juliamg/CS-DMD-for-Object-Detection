import numpy as np
import os
import imageio
import time
import sys
import cv2
import cvxpy as cvx
import matplotlib.pyplot as plt
import seaborn as sns

def src_algorithm(TrainSet, TestSet, num_classes, num_test_samples, sigma, thresh_certainty):
    print("SRC start")
    time.sleep(3)
    classes = np.unique(TrainSet['y']) # which classes have been acquainted
    identity = []
    failed_imgs = []

    for i in range(num_test_samples):
        y = TestSet['X'][:,i]
        D = TrainSet['X'] # Data matrix
        m, n = D.shape

        # do L1 optimization
        x = cvx.Variable(n)
        objective = cvx.Minimize(cvx.norm(x, 1))
        #constraints = [D@x == y]
        z = D @ x - y
        constraints = [cvx.norm(z, 2) <= sigma]
        prob = cvx.Problem(objective, constraints)
        result = prob.solve() # runs economy optimizer by default
        xp = np.array(x.value).squeeze()

        residuals = np.zeros((num_classes))

        # calculate residuals for each class
        for j in range(num_classes):
            idx = np.where(classes[j] == TrainSet['y'])
            last_index = np.size(idx) - 1
            residuals[j] = np.linalg.norm(y - TrainSet['X'][:,idx[0][0]:idx[0][last_index]+1].dot(xp[idx]))

        min_res = np.min(residuals)

        mean_res = np.mean(residuals)
        certainty = 1-min_res/mean_res

        label_index = np.argmin(residuals)

        #if classes[label_index] != TestSet['y'][i] or thresh_certainty > certainty:
        if thresh_certainty > certainty:
            failed_imgs.append(TestSet['files'][i])
            identity.append(None)
            print(f"INTRUDER WARNING - Face not recognized in file {TestSet['files'][i]}!")
        else:
            print("RECOGNIZED AS: ", classes[label_index], "TRUE: ", TestSet['y'][i])
            identity.append(classes[label_index])

        # graph = sns.barplot(x=classes, y=residuals)
        # plt.title('Residuals of each class')
        # graph.axhline(thresh_certainty, color='r', label='threshold')
        # graph.axhline(certainty, color='g', label='certainty')
        #
        # plt.show(block=False)
        # plt.legend()
        # plt.pause(5)
        # plt.close()

    ### Calculate accuracy ###
    correct_num = [i for i in range(len(identity)) if identity[i] == TestSet['y'][i]]
    rec_rate = len(correct_num)/num_test_samples * 100
    print(f"Predicted correctly: {len(correct_num)} out of {np.size(TestSet['y'])} with recognition rate: {rec_rate} %")

    return rec_rate, failed_imgs