import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


class PreProcess:
    """
    Pre-processes data for mlp, RNN and CNN models
    - Strips useless columns
    - Scales appropriately
    - Additional preprocessing for image approach
    """

    def __init__(
        self,
        datasets=["ttH125", "TTToHadronic", "TTToSemiLeptonic", "TTTo2L2Nu"],
        seed=42,
        train_size=0.8,
        exclude=("BiasedDPhi",),
        path="/software/ys20884/ml_postproc/",
    ):
        path = "/software/ys20884/ml_postproc/"
        if datasets == "all":
            datasets = set(os.listdir(path)) - {"test"}
        data = pd.concat(
            [pd.read_hdf(path + dataset + "/df_ml_inputs.hd5") for dataset in datasets]
        )
        new_cols = {
            "QCD": "QCD",
            "Jets": "VJets",
            "WminusH125": "VH125",
            "WplusH125": "VH125",
            "ZH125": "VH125",
        }
        for col in new_cols:
            data["dataset"] = data.dataset.str.replace(
                r"(^.*" + col + r".*$)", new_cols[col]
            )
        self.y = pd.get_dummies(data["dataset"])
        # Drop unimportant cols
        self.MHT_phis = data["MHT_phi"]
        self.MHT_pts = data["MHT_pt"]
        cols = [
            "BiasedDPhi",
            "DiJet_mass",
            "HT",
            "InputMet_InputJet_mindPhi",
            "InputMet_pt",
            "MHT_pt",
            "MinChi",
            "MinOmegaHat",
            "MinOmegaTilde",
            "cleanedJet_area",
            "cleanedJet_btagDeepB",
            "cleanedJet_chHEF",
            "cleanedJet_eta",
            "cleanedJet_mass",
            "cleanedJet_neHEF",
            "cleanedJet_phi",
            "cleanedJet_pt",
            "ncleanedBJet",
            "ncleanedJet",
            "xs_weight",
        ]
        self.X = data.drop(
            [col for col in data.columns if col not in cols] + list(exclude),
            axis=1,
        )
        if not isinstance(seed, int):
            raise TypeError("Seed must be integer")
        self.seed = seed
        if not isinstance(train_size, float) or not 0 < train_size < 1:
            raise Exception("train size must be float between 0 and 1")
        self.train_size = train_size

    def labels(self):
        """
        Returns train test split of labels
        """
        y_train, y_test = train_test_split(
            self.y, train_size=self.train_size, random_state=self.seed, stratify=self.y
        )
        return y_train, y_test

    def event(self):
        """
        Returns scaled event level data in train test split for mlp/combined neural networks
        """
        X_train, X_test = train_test_split(
            self.X.select_dtypes(exclude=object).drop("xs_weight", axis=1),
            train_size=self.train_size,
            random_state=self.seed,
            stratify=self.y,
        )
        event_scaler = StandardScaler()
        X_train = event_scaler.fit_transform(X_train)
        X_test = event_scaler.transform(X_test)
        return X_train, X_test

    def sequential(self, exclude=[]):
        """
        Returns scaled object level data in train test split for RNN networks
        """

        inp_data = self.X.select_dtypes(object).drop(exclude, axis=1)
        max_jets = 14
        num_samples = len(inp_data)
        # num_cols = len(inp_data.columns) + 1
        cols = inp_data.columns
        num_cols = len(cols) + 1
        data = np.zeros(
            (num_samples, max_jets, num_cols)
        )  # Shape for RNN samples x sequence length x dimensionality
        # delta phi func for getting delta phi between jet and leading jet/ jet and MHT.
        dphi = np.vectorize(
            lambda x, y: np.nan
            if y == np.nan
            else 2 * np.pi - abs(x - y)
            if abs(x - y) > np.pi
            else abs(x - y)
        )
        for i in range(max_jets):
            for j, col in enumerate(cols):
                # Get delta phi wrt to leading jet
                if col == "cleanedJet_phi":
                    data[:, i, j] = inp_data.loc[:, col].map(
                        lambda x: np.nan
                        if len(x) <= i
                        else 2 * np.pi - abs(x[i] - x[0])
                        if abs(x[i] - x[0]) > np.pi
                        else abs(x[i] - x[0])
                    )
                else:
                    data[:, i, j] = inp_data.loc[:, col].map(
                        lambda x: x[i] if len(x) > i else np.nan
                    )  # scaler ignores nan
            # Delta phi wrt MHT_pt
            phis = inp_data.loc[:, "cleanedJet_phi"].map(
                lambda x: np.nan if len(x) <= i else x[i]
            )
            data[:, i, -1] = dphi(self.MHT_phis, phis)

            # if j == 6:
            #     data[:, i, j] = inp_data.iloc[:, j].map(
            #         lambda x: np.nan if len(x) <= i else 2*np.pi - abs(x[i] - x[0]) if abs(x[i] - x[0]) > np.pi else abs(x[i] - x[0])
            #     )
            # elif j == num_cols - 1:
            #     phis = inp_data.loc[:, "cleanedJet_phi"].map(lambda x: np.nan if len(x) <= i else x[i])
            #     data[:, i, j] = dphi(self.MHT_phis, phis)
            # else:
            #     data[:, i, j] = inp_data.iloc[:, j].map(
            #         lambda x: x[i] if len(x) > i else np.nan
            #     )  # scaler ignores nan
        X_train, X_test = train_test_split(
            data, train_size=self.train_size, random_state=self.seed, stratify=self.y
        )
        object_scaler = StandardScaler()
        nz_train, nz_test = np.any(X_train, -1), np.any(X_test, -1)
        X_train[nz_train] = object_scaler.fit_transform(X_train[nz_train])
        X_test[nz_test] = object_scaler.transform(X_test[nz_test])
        # Set nan to 0
        np.nan_to_num(X_train, copy=False, nan=0)
        np.nan_to_num(X_test, copy=False, nan=0)
        return X_train, X_test

    def image(self, eta_dim=40, phi_dim=40):
        """
        Returns train test split for jet image approach with pt and btag discriminator at each pixel
        """
        eta_res = 10 / eta_dim
        phi_res = (2 * np.pi) / phi_dim
        centre = int((eta_dim - 1) / 2), int((phi_dim - 1) / 2)
        cols = [
            "cleanedJet_eta",
            "cleanedJet_phi",
            "cleanedJet_pt",
            "cleanedJet_btagDeepB",
        ]
        jet_data = self.X.loc[:, cols]
        jet_data_arr = np.zeros((len(jet_data), 14, 4))
        for i in range(14):
            for j, k in enumerate(cols):
                jet_data_arr[:, i, j] = jet_data.loc[:, k].map(
                    lambda x: x[i] if len(x) > i else 0
                )
        # jet_data_arr[:, :, 0] *= np.sign(jet_data_arr[:, 0, 0] * jet_data_arr[:, :, 0])
        N = len(jet_data_arr)
        jet_images = np.zeros((N, eta_dim, phi_dim, 3), dtype=np.half)
        for jet in range(14):
            # if jet == 0:
            #     phis = 19 * np.ones(N).astype(int)
            # else:
            phis = np.minimum(
                np.mod(
                    jet_data_arr[:, jet, 1] - jet_data_arr[:, 0, 1] + np.pi,
                    2 * np.pi,
                )
                // phi_res,
                phi_dim - 1,
            ).astype(int)
            etas = np.minimum(
                (jet_data_arr[:, jet, 0] + 5) // eta_res,
                eta_dim - 1,
            ).astype(int)
            pts = jet_data_arr[:, jet, 2]
            btags = jet_data_arr[:, jet, 3]
            jet_images[range(N), etas, phis, 0] = pts
            jet_images[range(N), etas, phis, 1] = btags
        X_train, X_test = train_test_split(
            jet_images,
            train_size=self.train_size,
            random_state=self.seed,
            stratify=self.y,
        )
        return X_train, X_test

    def xs_weight(self):
        """
        Returns train test split for xs weights
        """
        return train_test_split(
            self.X["xs_weight"],
            train_size=self.train_size,
            random_state=self.seed,
            stratify=self.y,
        )
