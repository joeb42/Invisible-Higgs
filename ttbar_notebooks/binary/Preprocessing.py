import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

class PreProcess:
    def __init__(self, datasets, seed=42, train_size=0.8, exclude=[]):
        path = "/software/ys20884/ml_postproc/"
        if datasets == "all":
            datasets = set(os.listdir(path)) - {"test"}
        data = pd.concat([pd.read_hdf(path+dataset+"/df_ml_inputs.hd5") for dataset in datasets])
        new_cols = {"QCD":"QCD", "Jets":"VJets", "WminusH125": "VH125", "WplusH125":"VH125", "ZH125":"VH125"}
        for col in new_cols:
            data['dataset'] = data.dataset.str.replace(r'(^.*' + col + r'.*$)', new_cols[col])
        self.y = pd.get_dummies(data['dataset'])
        # Drop unimportant cols
        self.X = data.drop(['weight_nominal', 'entry', 'hashed_filename', 'MHT_phi', 'InputMet_phi', 'dataset', 'BiasedDPhi']+exclude, axis=1)
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
        y_train, y_test = train_test_split(self.y, train_size=self.train_size, random_state=self.seed, stratify=self.y)
        return y_train, y_test

    def event(self):
        """
        Returns scaled event level data in train test split for mlp/combined neural networks
        """
        X_train, X_test = train_test_split(self.X.select_dtypes(exclude=object), train_size=self.train_size, random_state=self.seed, stratify=self.y)
        event_scaler = StandardScaler()
        X_train = event_scaler.fit_transform(X_train)
        X_test = event_scaler.transform(X_test)
        return X_train, X_test
    
    def sequential(self):
        """
        Returns scaled object level data in train test split for RNN networks
        """

        inp_data = self.X.select_dtypes(object)
        max_jets = 14
        num_samples = len(inp_data)
        num_cols = len(inp_data.columns)
        data = np.zeros((num_samples, max_jets, num_cols)) # Shape for RNN samples x sequence length x dimensionality
        for i in range(max_jets):
            for j in range(num_cols):
                # Get delta phi wrt to leading jet
                if j == 6 and i > 0:
                    data[:, i, j] = inp_data.iloc[:, j].map(lambda x: abs(x[i] - x[0]) if len(x) > i else np.nan)
                else:
                    data[:, i, j] = inp_data.iloc[:, j].map(lambda x: x[i] if len(x) > i else np.nan) # scaler ignores nan
        X_train, X_test = train_test_split(data, train_size=self.train_size, random_state=self.seed, stratify=self.y)
        object_scaler = StandardScaler()
        nz_train, nz_test = np.any(X_train, -1), np.any(X_test, -1)
        X_train[nz_train] = object_scaler.fit_transform(X_train[nz_train])
        X_test[nz_test] = object_scaler.transform(X_test[nz_test])
        # Set nan to 0
        np.nan_to_num(X_train, copy=False, nan=0)
        np.nan_to_num(X_test, copy=False, nan=0)
        return X_train, X_test
    
    def image(self, eta_dim=40, phi_dim=40, pad=5):
        """
        Returns train test split for jet image approach with pt and btag discriminator at each pixel
        """
        eta_res = eta_dim // 10 
        phi_res = phi_dim // (2*np.pi)
        cols = ['cleanedJet_eta', 'cleanedJet_phi', 'cleanedJet_pt', 'cleanedJet_btagDeepB']
        jet_data = self.X.loc[:, cols]
        jet_data_arr = np.zeros((len(jet_data), 14, 4))
        for i in range(14):
                for j, k in enumerate(cols):
                    jet_data_arr[:, i, j] = jet_data.loc[:, k].map(lambda x: x[i] if len(x) > i else 0)
        N = len(jet_data_arr)
        jet_images = np.zeros((N, 40, 40+2*pad, 2), dtype=np.half)
        for jet in range(14):
            if jet == 0:
                etas = 19 * np.ones(N).astype(int)
                phis = 19 * np.ones(N).astype(int)
            else:
                etas = np.minimum((jet_data_arr[:,jet,0] - jet_data_arr[:,0,0] + 5) // eta_res, eta_dim).astype(int)
                phis = np.minimum(np.mod(jet_data_arr[:,jet,1] - jet_data_arr[:,0,1]+np.pi, 2*np.pi) // phi_res, phi_dim).astype(int) + 10 
            pts = jet_data_arr[:, jet, 2]
            btags = jet_data_arr[:, jet, 3]
            jet_images[range(N), etas, phis, 0] = pts
            jet_images[range(N), etas, phis, 1] = btags
        jet_images[:, :, :pad, :] = jet_images[:, :, -2*pad:-pad, :]
        jet_images[:, :, -pad:, :] = jet_images[:, :, pad:2*pad, :]
        X_train, X_test = train_test_split(jet_images, train_size=self.train_size, random_state=self.seed, stratify=self.y)
        return X_train, X_test
