import anndata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

adata = anndata.read_h5ad("Mouse_wholebrain_FC.h5ad")
print(str(adata.obs['sampleID']))