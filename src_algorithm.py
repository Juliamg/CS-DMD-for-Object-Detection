import numpy as np
import os
import imageio
import sys
import cv2
import cvxpy as cvx
import matplotlib.pyplot as plt
import seaborn as sns

# TODO: add some validation threshold, plot the residuals, reject sample image if it crosses the validation threshold -> 0.4 for instance

def src_algorithm(TrainSet, TestSet, num_classes, num_test_samples, sigma):
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
            #print(idx, y.shape, TrainSet['X'][:,idx[0][0]:idx[0][last_index]+1].shape, xp[idx].shape)
            residuals[j] = np.linalg.norm(y - TrainSet['X'][:,idx[0][0]:idx[0][last_index]+1].dot(xp[idx]))

        label_index = np.argmin(residuals)
        identity.append(classes[label_index])

        print("PREDICTED: ", classes[label_index], "TRUE: ", TestSet['y'][i])
        if classes[label_index] != TestSet['y'][i]:
            failed_imgs.append(TestSet['files'][i])

        sns.barplot(x=classes, y=residuals)
        plt.title('Residuals of each class')

        plt.show(block=False)
        plt.pause(4)
        plt.close()

    ### Calculate accuracy ###
    correct_num = [i for i in range(len(identity)) if identity[i] == TestSet['y'][i]]
    accuracy = len(correct_num)/num_test_samples * 100
    print(f"Predicted correctly: {len(correct_num)} out of {np.size(TestSet['y'])} with an accuracy of: {accuracy} %")

    return accuracy, failed_imgs