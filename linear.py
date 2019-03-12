import csv
import numpy as np
from sklearn import linear_model

losses = ['squared_hinge', 'perceptron', 'squared_loss', 'epsilon_insensitive']
alphas = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]
lrs = ['optimal']
eta0s = [0.001]
penalties = ['l1', 'l2']

bows_test = np.load('bows5.npy')
bows = np.load('bows.npy')
targets = np.load('targets.npy')
y = targets[:40000]
delete = np.random.rand(y.shape[0])
del_arr = []
for idx, elem in enumerate(delete):
    if elem < 0.97 and y[idx] == 1:
        del_arr.append(idx)
bows_train = np.delete(bows, del_arr, 0)
y = np.delete(y, del_arr, 0)


with open('linear_bow_results.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['loss', 'alpha', 'lr', 'eta0', 'penalty', 'train_acc', 'train_neg_acc', 'all_acc', 'all_neg_acc', 'test_acc', 'test_neg_acc'])

    for loss in losses:
        for alpha in alphas:
            for lr in lrs:
                for eta0 in eta0s:
                    for penalty in penalties:
                        clf = linear_model.SGDClassifier(max_iter=3000, tol=1e-3, loss=loss, alpha=alpha, learning_rate=lr, penalty=penalty, eta0=eta0)
                        clf.fit(bows_train, y)


                        res = clf.predict(bows_train)
                        train_acc = np.sum(np.equal(res, y)) / res.shape[0]
                        train_neg_acc = np.sum(np.equal(res, y) * (np.equal(y, np.zeros(y.shape[0])))) / np.sum(np.equal(y, np.zeros(y.shape[0])))
                        # print('train accuracy: ', np.sum(np.equal(res, y)) / res.shape[0])
                        # print('train neg accuracy: ', np.sum(np.equal(res, y) * (np.equal(y, np.zeros(y.shape[0])))) / np.sum(np.equal(y, np.zeros(y.shape[0]))))

                        res = clf.predict(bows)
                        all_acc = np.sum(np.equal(res, targets[:40000])) / res.shape[0]
                        all_neg_acc = np.sum(np.equal(res, targets[:40000]) * (np.equal(targets[:40000], np.zeros(40000)))) / np.sum(np.equal(targets[:40000], np.zeros(40000)))
                        # print('all accuracy: ', np.sum(np.equal(res, targets[:40000])) / res.shape[0])
                        # print('all neg accuracy: ', np.sum(np.equal(res, targets[:40000]) * (np.equal(targets[:40000], np.zeros(40000)))) / np.sum(np.equal(targets[:40000], np.zeros(40000))))

                        res = clf.predict(bows_test)
                        test_acc = np.sum(np.equal(res, targets[40000:])) / res.shape[0]
                        test_neg_acc = np.sum(np.equal(res, targets[40000:]) * (np.equal(targets[40000:], np.zeros(10000)))) / np.sum(np.equal(targets[40000:], np.zeros(10000)))
                        # print('test accuracy: ', np.sum(np.equal(res, targets[40000:])) / res.shape[0])
                        # print('test neg accuracy: ', np.sum(np.equal(res, targets[40000:]) * (np.equal(targets[40000:], np.zeros(10000)))) / np.sum(np.equal(targets[40000:], np.zeros(10000))))


                        writer.writerow([loss, alpha, lr, eta0, penalty, train_acc, train_neg_acc, all_acc, all_neg_acc, test_acc, test_neg_acc])