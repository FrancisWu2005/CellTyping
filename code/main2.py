import os
import anndata
import os, sys
import numpy as np

import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import math
## import my written library
from utils import _utils

if __name__ == '__main__':
    ## load train_adata and test_adata
    # loads in dataset
    adata = anndata.read_h5ad("/Users/franciswu/compsci/SummerProject/data/Mouse_pFC.h5ad")
    bdata = anndata.read_h5ad("/Users/franciswu/compsci/SummerProject/data/Mouse_wholebrain_FC.h5ad")
    cdata = anndata.read_h5ad("/Users/franciswu/compsci/SummerProject/data/data/Mousecortex_protocals.h5ad")


    samples = []
    for i in adata.obs['Sample']:
        if i not in samples:
            samples.append(i)

    train_num = len(samples) - (math.ceil(len(samples)/10))
    counter = 0
    for sample in samples:
        if counter == 0:
            train_adata = adata[adata.obs['Sample'] == sample]
        elif(counter < train_num):
            train_adata =  anndata.AnnData.concatenate(*[train_adata,adata[adata.obs['Sample'] == sample] ],join="inner")
        elif(counter == train_num):
            test_adata = adata[adata.obs['Sample'] == sample]
        else:
            test_adata = anndata.AnnData.concatenate(*[test_adata,adata[adata.obs['Sample'] == sample] ],join="inner")

        counter += 1

    b_samples = []
    for i in bdata.obs['Sample']:
        if i not in b_samples:
            b_samples.append(i)

    b_train_num = len(b_samples) - (math.ceil(len(b_samples)/10))
    counter = 0
    for sample in b_samples:
        if counter == 0:
            b_train_adata = bdata[bdata.obs['ind'] == sample]
        elif(counter < b_train_num):
            b_train_adata =  anndata.AnnData.concatenate(*[b_train_adata,bdata[bdata.obs['ind'] == sample] ],join="inner")
        elif(counter == b_train_num):
            b_test_adata = bdata[bdata.obs['ind'] == sample]
        else:
            b_test_adata = anndata.AnnData.concatenate(*[b_test_adata,bdata[bdata.obs['ind'] == sample] ],join="inner")

        counter += 1

    c_train_adata = cdata[cdata["Experiment"] == "Cortex1"]
    c_train_adata = cdata[cdata["Experiment"] == "Cortex2"]

    common_genes = set(train_adata.var_names).intersection(set(test_adata.var_names))
    combined_train_adata = train_adata[:, list(common_genes)]
    test_adata = test_adata[:, list(common_genes)]
    train_adata = _utils._process_adata(train_adata, process_type='train')
    train_adata = _utils._select_feature(train_adata,
                                         fs_method='F-test',
                                         num_features=1000)  ## use F-test to select 1000 informative genes
    train_adata = _utils._scale_data(train_adata)  ## center-scale
    _utils._visualize_data(train_adata, output_dir=".",
                           prefix="traindata_vis")  ## visualize cell types with selected features on a low dimension (you might need to change some parameters to let them show all the cell labels)
    ## train an MLP model on it
    MLP_DIMS = _utils.MLP_DIMS  ## get MLP structure from _utils.py
    x_train = _utils._extract_adata(train_adata)
    enc = OneHotEncoder(handle_unknown='ignore')
    y_train = enc.fit_transform(train_adata.obs[[_utils.Celltype_COLUMN]]).toarray()
    mlp = _utils._init_MLP(x_train, y_train, dims=MLP_DIMS,
                           seed=_utils.RANDOM_SEED)
    mlp.compile()
    mlp.fit(x_train, y_train)
    mlp.model.save('./trained_MLP')  ## save the model so that you can load and play with it
    encoders = dict()
    for idx, cat in enumerate(enc.categories_[0]):
        encoders[idx] = cat
    # set(adata.obs['Sample'])
    ## preprocess the test data and predict cell types
    test_adata = _utils._process_adata(test_adata, process_type='test')
    test_adata = test_adata[:,
                 list(train_adata.var_names)]  ## extract out the features selected in the training dataset
    test_data_mat = _utils._extract_adata(test_adata)
    test_data_mat = (test_data_mat - np.array(train_adata.var['mean'])) / np.array(train_adata.var['std'])
    y_pred = tf.nn.softmax(mlp.model.predict(test_data_mat)).numpy()
    pred_celltypes = _utils._prob_to_label(y_pred, encoders)
    test_adata.obs[_utils.PredCelltype_COLUMN] = pred_celltypes

    ## let us evaluate the performance --> luckily you will have the accuracy over 99%
    from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score

    print("Overall Accuracy:",
          accuracy_score(test_adata.obs[_utils.Celltype_COLUMN], test_adata.obs[_utils.PredCelltype_COLUMN]))
    print("ARI:",
          adjusted_rand_score(test_adata.obs[_utils.Celltype_COLUMN], test_adata.obs[_utils.PredCelltype_COLUMN]))
    print("Macro F1:",
          f1_score(test_adata.obs[_utils.Celltype_COLUMN], test_adata.obs[_utils.PredCelltype_COLUMN], average='macro'))
    ## a lot more evaluation metrics can be found on sklearn.metrics and you can explore with them

