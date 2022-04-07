import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


class Analysis:
    def __init__(self, model, X_test, y_test):
        self.y_test = y_test
        self.y_pred = model.predict(X_test).reshape(len(y_test))
    
    def plot_discriminator(self):
        hist, bins, patches = plt.hist(self.y_pred[self.y_test==1], bins=40, density=True, histtype='step', label="ttH")
        plt.hist(self.y_pred[self.y_test==0], bins=bins, density=True, histtype="step", label=r"$t\bart$")
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
    
    def significance(self, weights, lum=140e3, res=0.0001, plot=True, br=0):
        thresholds = np.arange(0, 1+res, res)
        ams_sigma0 = np.zeros(len(thresholds))
        ams_sigma5 = np.zeros(len(thresholds))
        ams_sigma10 = np.zeros(len(thresholds))
        ams_sigma20 = np.zeros(len(thresholds))

        sg = np.zeros(len(thresholds))
        bg = np.zeros(len(thresholds))
        
        for idx, threshold in enumerate(thresholds):
            sg[idx] = (lum * weights * 5 * ((self.y_pred >= threshold) & (self.y_test == 1)).astype(int)).sum()
            bg[idx] = (lum * weights * 5 * ((self.y_pred >= threshold) & (self.y_test == 0)).astype(int)).sum()
            # Min 10 signal events 
            if sg[idx] > 10: 
                ams_sigma0[idx] = asimov(sg[idx], bg[idx], 0, br)
                ams_sigma5[idx] = asimov(sg[idx], bg[idx], 0.05, br)
                ams_sigma10[idx] = asimov(sg[idx], bg[idx], 0.1, br)
                ams_sigma20[idx] = asimov(sg[idx], bg[idx], 0.2, br)
        if plot:
            plt.plot(thresholds, ams_sigma0, label=r"$\sigma=0\%$")
            plt.plot(thresholds, ams_sigma5, label=r"$\sigma=5\%$")
            plt.plot(thresholds, ams_sigma10, label=r"$\sigma=10\%$")
            plt.plot(thresholds, ams_sigma20, label=r"$\sigma=20\%$")
            plt.xlabel("Threshold")
            plt.ylabel("Significance")
            plt.legend()
            plt.show()
        return ams_sigma0, ams_sigma5, ams_sigma10, ams_sigma20
    
    def plot_cm(self, threshold):
        ...
    
        
    
def asimov(s, b, sigma, br=0):
    if sigma == 0:
        return np.sqrt(2 * ((s+b+br) * np.log(1+s/(b+br))-s))
    s_b = sigma * b
    return np.sqrt(2 * ( (s+b) * np.log( (s+b) * (b+s_b**2)/ (b**2 + (s+b) * s_b**2)) - (b/s_b)**2 * np.log(1 + (s_b**2 * s)/ (b * (b + s_b**2)))))
