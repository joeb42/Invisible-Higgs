import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from numpy import log, power, sqrt


class Analysis:
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

    def plot_discriminator(self):
        hist, bins, patches = plt.hist(
            self.y_pred[self.y_test == 1],
            bins=40,
            density=True,
            histtype="step",
            label="ttH",
        )
        plt.hist(
            self.y_pred[self.y_test == 0],
            bins=bins,
            density=True,
            histtype="step",
            label=r"$t\bart$",
        )
        plt.legend()
        plt.show()

    def plot_ROC(self, npoints=500):
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
        plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
        plt.plot(
            np.arange(0, 1.0001, 0.0001),
            np.arange(0, 1.0001, 0.0001),
            linestyle="dashed",
            linewidth=0.5,
            color="k",
            label="luck",
        )
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.show()

    def significance(self, weights, lum=140e3, res=0.0001, plot=True, path=None):
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
                    (self.y_pred >= threshold) & (self.y_test.values[:, -1] == 1)
                ).astype(int)
            ).sum()
            bg[idx] = (
                lum
                * weights
                * 5
                * (
                    (self.y_pred >= threshold) & (self.y_test.values[:, -1] == 0)
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
            # plt.savefig("./models/trained_models/final_rnn/significance.png", dpi=200)
        return (Z_0, Z_5, Z_10, Z_20), (Z_err_0, Z_err_5, Z_err_10, Z_err_20)

        # thresholds = np.arange(0, 1+res, res)
        # ams_sigma0 = np.zeros(len(thresholds))
        # ams_sigma5 = np.zeros(len(thresholds))
        # ams_sigma10 = np.zeros(len(thresholds))
        # ams_sigma20 = np.zeros(len(thresholds))

        # sg = np.zeros(len(thresholds))
        # bg = np.zeros(len(thresholds))

        # for idx, threshold in enumerate(thresholds):
        #     sg[idx] = (lum * weights * 5 * ((self.y_pred >= threshold) & (self.y_test == 1)).astype(int)).sum()
        #     bg[idx] = (lum * weights * 5 * ((self.y_pred >= threshold) & (self.y_test == 0)).astype(int)).sum()
        #     # Min 10 signal events
        #     if sg[idx] > 10:
        #         ams_sigma0[idx] = asimov(sg[idx], bg[idx], 0, br)
        #         ams_sigma5[idx] = asimov(sg[idx], bg[idx], 0.05, br)
        #         ams_sigma10[idx] = asimov(sg[idx], bg[idx], 0.1, br)
        #         ams_sigma20[idx] = asimov(sg[idx], bg[idx], 0.2, br)
        # if plot:
        #     plt.plot(thresholds, ams_sigma0, label=r"$\sigma_b=0\%$")
        #     plt.plot(thresholds, ams_sigma5, label=r"$\sigma_b=5\%$")
        #     plt.plot(thresholds, ams_sigma10, label=r"$\sigma_b=10\%$")
        #     plt.plot(thresholds, ams_sigma20, label=r"$\sigma_b=20\%$")
        #     plt.xlabel("Threshold")
        #     plt.ylabel("Significance")
        #     plt.legend()
        #     plt.show()
        # return ams_sigma0, ams_sigma5, ams_sigma10, ams_sigma20

    def plot_cm(self, threshold):
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
