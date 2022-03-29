import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


class Analysis:
    def __init__(self, model, X_test, y_test):
        self.y_test = y_test
        self.y_pred = model.predict(X_test)
    
    def plot_discriminator(self, npoints=500):
        thresholds = np.linspace(0, 1, npoints)
        hist, bins, patches = plt.hist(self.y_pred[self.y_test==1], bins='auto', density=True, histtype='step', label="ttH")
        plt.hist(self.y_pred[self.y_test==0], bins=bins, density=True, histtype="step", label=r"t\bart")
        plt.legend()
        plt.show()
    
    def plot_ROC(self, npoints=500):
        thresholds = np.linspace(0, 1, npoints)
        TP = np.array([((self.y_pred >= threshold) & (self.y_test==1)).sum() for threshold in thresholds])
        FP = np.array([((self.y_pred >= threshold) & (self.y_test==0)).sum() for threshold in thresholds])
        tpr = TP/np.sum(self.y_test)
        fpr = FP/(len(self.y_test)-np.sum(self.y_test))
        auc = roc_auc_score(self.y_test, self.y_pred)
        plt.plot(fpr, tpr, label=f'AUC={auc:.4f}')
        plt.plot(np.arange(0, 1.0001, 0.0001), np.arange(0, 1.0001, 0.0001), linestyle='dashed', linewidth=0.5, color='k', label="luck")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.show()
    
    
