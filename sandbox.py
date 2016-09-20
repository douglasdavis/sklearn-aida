#!/usr/bin/env python3

import ROOT
import numpy as np
from root_numpy import tree2array, root2array, rec2array
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import roc_curve, auc

def get_array(root_file):
    arr = tree2array(root_file.Get('AIDAflatTree'), branches=['met','njets','nbjets'], selection='elmu > 0')
    arr = rec2array(arr)
    w   = tree2array(root_file.Get('AIDAflatTree'), branches=['nomWeight'],    selection='elmu > 0')
    w   = rec2array(w)
    return arr, w

file1 = ROOT.TFile('410000.root')
file2 = ROOT.TFile('410016.root')
arr1, w1 = get_array(file1)
arr2, w2 = get_array(file2)

#arr1_train, arr2_train, w1_train, w2_train = arr1[:322593], arr2[:40180], w1[:322593], w2[:40180]
#arr1_test,  arr2_test,  w1_test,  w2_test  = arr1[322593:], arr2[40180:], w1[322593:], w2[40180:]

arr1_train, arr2_train, w1_train, w2_train = arr1[:7500], arr2[:7500], w1[:7500], w2[:7500]
arr1_test,  arr2_test,  w1_test,  w2_test  = arr1[7500:15000], arr2[7500:15000], w1[7500:15000], w2[7500:15000]

X_train = np.concatenate((arr1_train,arr2_train))
y_train = np.concatenate((np.ones(arr1_train.shape[0]), np.zeros(arr2_train.shape[0])))
w_train = np.concatenate((w1_train,w2_train))

X_test = np.concatenate((arr1_test,arr2_test))
y_test = np.concatenate((np.ones(arr1_test.shape[0]), np.zeros(arr2_test.shape[0])))
w_test = np.concatenate((w1_test,w2_test))




dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=0.05*len(X_train))
bdt = AdaBoostClassifier(dt, algorithm='SAMME', n_estimators=800, learning_rate=0.5)

bdt.fit(X_train, y_train, sample_weight=w_train)
y_predicted = bdt.predict(X_test)
print(classification_report(y_test, y_predicted, target_names=["Wt", "ttbar"]))
print("Area under ROC curve: %.4f"%(roc_auc_score(y_test, bdt.decision_function(X_test))))


decisions = bdt.decision_function(X_test)
# Compute ROC curve and area under the curve
fpr, tpr, thresholds = roc_curve(y_test, decisions)
roc_auc = auc(fpr, tpr)

# plot the roc curve
#plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)'%(roc_auc))
#plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
#plt.xlim([-0.05, 1.05])
#plt.ylim([-0.05, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic')
#plt.legend(loc="lower right")
#plt.grid()
#plt.show()

def compare_train_test(clf, X_train, y_train, X_test, y_test, bins=30):
    decisions = []
    for X,y in ((X_train, y_train), (X_test, y_test)):
        d1 = clf.decision_function(X[y>0.5]).ravel()
        d2 = clf.decision_function(X[y<0.5]).ravel()
        decisions += [d1, d2]
        
    low = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    low_high = (low,high)
    
    plt.hist(decisions[0],
             color='r', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', normed=True,
             label='S (train)')
    plt.hist(decisions[1],
             color='b', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', normed=True,
             label='B (train)')

    hist, bins = np.histogram(decisions[2],
                              bins=bins, range=low_high, normed=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale
    
    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='S (test)')
    
    hist, bins = np.histogram(decisions[3],
                              bins=bins, range=low_high, normed=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='B (test)')

    plt.xlabel("BDT output")
    plt.ylabel("Arbitrary units")
    plt.legend(loc='best')
    plt.show()

compare_train_test(bdt, X_train, y_train, X_test, y_test)
