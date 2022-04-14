import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from numpy import log, power, sqrt
import shap



class Evaluate:
    """
    Contains functions to evaluate performance of keras binary classifier
    """

    def __init__(self, model, X_test, y_test):
        """
        Sets y_test and y_pred values for classifier
        Assumes y_test is pandas dataframe with last column being onehot encoding of ttH (i.e. 1 for signal, 0 for background)
        """
        self.y_test = y_test
        self.y_pred = model.predict(X_test).reshape(len(y_test))
        self.X_test = X_test
        self.model = model

    def plot_discriminator(self, loc="best"):
        """
        Plot histogram of discriminator outputs for signal and backround
        """
        fig, ax = plt.subplots()
        hist, bins, patches = ax.hist(
            self.y_pred[self.y_test == 1],
            bins="auto",
            density=True,
            histtype="step",
            label="ttH",
        )
        ax.hist(
            self.y_pred[self.y_test == 0],
            bins=bins,
            density=True,
            histtype="step",
            label=r"$t\bart$",
        )
        ax.legend(loc=loc)
        ax.set_xlabel("Discriminator Output", fontsize=12)
        ax.set_ylabel("Frequency Density", fontsize=12)
        plt.show()
        return fig, ax

    def plot_ROC(self, npoints=500, log=False, loc="best"):
        """
        Plot ROC curve
        """
        thresholds = np.linspace(0, 1, npoints)
        TP = np.array(
            [
                ((self.y_pred >= threshold) & (self.y_test == 1)).sum()
                for threshold in thresholds
            ]
        )
        FP = np.array(
            [
                ((self.y_pred >= threshold) & (self.y_test == 0)).sum()
                for threshold in thresholds
            ]
        )
        tpr = TP / np.sum(self.y_test)
        fpr = FP / (len(self.y_test) - np.sum(self.y_test))
        auc = roc_auc_score(self.y_test, self.y_pred)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC={auc:.4f}")
        ax.plot(
            np.arange(0, 1.0001, 0.0001),
            np.arange(0, 1.0001, 0.0001),
            linestyle="dashed",
            linewidth=0.5,
            color="k",
            label="luck",
        )
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        if log:
            ax.set_xscale("log")
        ax.legend(loc=loc)
        plt.show()
        return fig, ax

    def significance(self, weights, lum=140e3, res=0.0001, plot=True, path=None):
        """
        Make significance plot at 0, 5 and 10% systematic bg uncertainties @ L = 140/fb
        """
        thresholds = np.arange(0, 1 + res, res)
        sg = np.zeros(len(thresholds))
        bg = np.zeros(len(thresholds))
        for idx, threshold in enumerate(thresholds):
            # Compute signal and background events for each cut on discriminator
            sg[idx] = (
                lum
                * weights
                * 5
                * (
                    (self.y_pred >= threshold) & (self.y_test == 1)
                ).astype(int)
            ).sum()
            bg[idx] = (
                lum
                * weights
                * 5
                * (
                    (self.y_pred >= threshold) & (self.y_test == 0)
                ).astype(int)
            ).sum()
        # Compute Asimov estimates and propagate statistical uncertainty
        Z_0, Z_5, Z_10, Z_20 = (
            Z(sg, bg, sig=0.0),
            Z(sg, bg, sig=0.05),
            Z(sg, bg, sig=0.1),
            Z(sg, bg, sig=0.2),
        )
        Z_err_0, Z_err_5, Z_err_10, Z_err_20 = (
            eZ(sg, bg, sig=0.0),
            eZ(sg, bg, sig=0.05),
            eZ(sg, bg, sig=0.1),
            eZ(sg, bg, sig=0.2),
        )
        if plot:
            fig, ax = plt.subplots()
            ax.plot(thresholds, Z_0, color="b", label=r"$\sigma_b=0\%$")
            ax.fill_between(
                thresholds, Z_0 - Z_err_0, Z_0 + Z_err_0, alpha=0.25, color="b"
            )
            ax.plot(thresholds, Z_5, color="r", label=r"$\sigma_b=5\%$")
            ax.fill_between(
                thresholds, Z_5 - Z_err_5, Z_5 + Z_err_5, alpha=0.25, color="r"
            )
            ax.plot(thresholds, Z_10, color="g", label=r"$\sigma_b=10\%$")
            ax.fill_between(
                thresholds, Z_10 - Z_err_10, Z_10 + Z_err_10, alpha=0.25, color="g"
            )
            ax.set_xlabel("Cut on the Discriminator", fontsize=12)
            ax.set_ylabel(r"$Z_A$", fontsize=12)
            ax.set_xticks(np.arange(0, 1.05, 0.05))
            ax.set_xticklabels(
                [
                    f"{j:.1f}" if i % 4 == 0 else ""
                    for i, j in enumerate(np.arange(0, 1.05, 0.05))
                ]
            )
            ax.legend(framealpha=0.0)
            ax.set_ylim(0)
            ax.grid(alpha=0.6)
            plt.show()
            if path is not None:
                if path[-1] != "/":
                    path += "/"
                fig.savefig(path + "significance.png", dpi=200)
            sigs = (0, 0.05, 0.1)
            Zs = (Z_0, Z_5, Z_10)
            Zerrs = (Z_err_0, Z_err_5, Z_err_10)
            for sig, Z_a, Zerr in zip(sigs, Zs, Zerrs):
                idx = np.argmax(Z_a[Z_a < np.inf]-Zerr[Zerr < np.inf])
                self.plot_cm(idx, thresholds[idx], sg, bg, sig=sig, path=path)
        return (Z_0, Z_5, Z_10, Z_20), (Z_err_0, Z_err_5, Z_err_10, Z_err_20)

    def plot_cm(self, idx, threshold, sg, bg, sig=0, path=None):
        """
        Plot confusion matrix at given threshold
        """
        TP = ((self.y_pred >= threshold) & (self.y_test == 1)).sum()
        FP = ((self.y_pred >= threshold) & (self.y_test == 0)).sum()
        tpr = TP / np.sum(self.y_test)
        fpr = FP / (len(self.y_test) - np.sum(self.y_test))
        conf_mat = [[1-fpr, fpr], [1-tpr, tpr]]
        events = {(0,0): round(np.sum(bg)-bg[idx]), (0,1): round(bg[idx]), (1, 0): round(np.sum(sg)-sg[idx]), (1,1): round(sg[idx])}
        mat = plt.matshow(conf_mat, cmap='blues', vmax=2)
        for (x, y), value in np.ndenumerate(conf_mat):
            plt.text(y, x, f"{value:.4f}% \n{events[(x,y)]:.2e}", va="center", ha="center", color="k", weight="bold")
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks([0,1], labels=['Background', 'Signal'])
        plt.yticks([0,1], labels=['Background', 'Signal'])
        cbar = plt.colorbar(mat)
        cbar.ax.set_ylabel(f"% events", rotation=270, labelpad=20)
        plt.suptitle(fr"$\sigma_b={100*sig}$%")
        if path is not None:
            if path[-1] != "/":
                path += "/" 
            plt.savefig(path+"cm_"+sig, dpi=200)
        plt.show()
        
    def shap(self, type, X_train, feature_names, n=25000):
        if type == "mlp":
            explainer = shap.GradientExplainer(self.model, X_train)
            shap_values = explainer.shap_values(shap.sample(self.X_test, n))
            shap.summary_plot(shap_values[0], features=shap.sample(self.X_test, n), feature_names=feature_names, plot_size=(10,7), show=False)
            plt.gcf().axes[-1].set_aspect(100)
            plt.gcf().axes[-1].set_box_aspect(100)
            plt.show()
    
    def noise_study(self):
        ...




def Z(s, b, sig=0):
    """
    Asimov median significance estimate
    - s, b are signal and background event counts
    - sig is relative systematic uncertainty on the background count (i.e. 0.1 = 10% systematic uncertainty)
    """
    if sig == 0:
        return np.sqrt(2 * ((s + b) * np.log(1 + s / (b)) - s))
    s_b = sig * b
    return np.sqrt(
        2
        * (
            (s + b) * np.log((s + b) * (b + s_b**2) / (b**2 + (s + b) * s_b**2))
            - (b / s_b) ** 2 * np.log(1 + (s_b**2 * s) / (b * (b + s_b**2)))
        )
    )


def eZ(s, b, es=None, eb=None, sig=None, eps=0.000001):
    """
    Propagate statistical error on signal and background counts to asimov estimate
    Taken from Adam Elwood github
    - s, b are signal and background event counts
    - es, eb are the absolute statistical errors in signal and background counts (i.e. sqrt(count) for poisson counting error)
    - sig is the relative systematic uncertainty on the background count (i.e. 0.1 = 10% systematic uncertainty)
    """
    if sig is None or sig == 0:
        sig = eps
    # 1 sigma poisson errors on signal and backgrounds counts
    if es == None:
        es = np.sqrt(s)
    if eb == None:
        eb = np.sqrt(b)
    return power(
        -(eb * eb)
        / (
            1.0
            / (sig * sig)
            * log(b / (b + (b * b) * (sig * sig)) * (sig * sig) * s + 1.0)
            - (b + s)
            * log(
                (b + s)
                * (b + (b * b) * (sig * sig))
                / ((b * b) + (b + s) * (b * b) * (sig * sig))
            )
        )
        * power(
            1.0
            / (b / (b + (b * b) * (sig * sig)) * (sig * sig) * s + 1.0)
            / (sig * sig)
            * (
                1.0 / (b + (b * b) * (sig * sig)) * (sig * sig) * s
                - b
                / power(b + (b * b) * (sig * sig), 2.0)
                * (sig * sig)
                * (2.0 * b * (sig * sig) + 1.0)
                * s
            )
            - (
                (b + s)
                * (2.0 * b * (sig * sig) + 1.0)
                / ((b * b) + (b + s) * (b * b) * (sig * sig))
                + (b + (b * b) * (sig * sig))
                / ((b * b) + (b + s) * (b * b) * (sig * sig))
                - (b + s)
                * (2.0 * (b + s) * b * (sig * sig) + 2.0 * b + (b * b) * (sig * sig))
                * (b + (b * b) * (sig * sig))
                / power((b * b) + (b + s) * (b * b) * (sig * sig), 2.0)
            )
            / (b + (b * b) * (sig * sig))
            * ((b * b) + (b + s) * (b * b) * (sig * sig))
            - log(
                (b + s)
                * (b + (b * b) * (sig * sig))
                / ((b * b) + (b + s) * (b * b) * (sig * sig))
            ),
            2.0,
        )
        / 2.0
        - 1.0
        / (
            1.0
            / (sig * sig)
            * log(b / (b + (b * b) * (sig * sig)) * (sig * sig) * s + 1.0)
            - (b + s)
            * log(
                (b + s)
                * (b + (b * b) * (sig * sig))
                / ((b * b) + (b + s) * (b * b) * (sig * sig))
            )
        )
        * power(
            log(
                (b + s)
                * (b + (b * b) * (sig * sig))
                / ((b * b) + (b + s) * (b * b) * (sig * sig))
            )
            + 1.0
            / (b + (b * b) * (sig * sig))
            * (
                (b + (b * b) * (sig * sig))
                / ((b * b) + (b + s) * (b * b) * (sig * sig))
                - (b + s)
                * (b * b)
                * (b + (b * b) * (sig * sig))
                * (sig * sig)
                / power((b * b) + (b + s) * (b * b) * (sig * sig), 2.0)
            )
            * ((b * b) + (b + s) * (b * b) * (sig * sig))
            - 1.0
            / (b / (b + (b * b) * (sig * sig)) * (sig * sig) * s + 1.0)
            * b
            / (b + (b * b) * (sig * sig)),
            2.0,
        )
        * (es * es)
        / 2.0,
        (1.0 / 2.0),
    )
