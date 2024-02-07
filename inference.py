import argparse
import itertools

from src.pipeline import pipeline
from src.training_utils import training_utils

EXP_HPARAMS = {
    "params": (
        {},
    ),
    "seeds": (420,),
}
config = training_utils.get_config("MNIST")
for hparams_overwrite_list, seed in itertools.product(EXP_HPARAMS["params"], EXP_HPARAMS["seeds"]):
    hparams_str = ""
    for k, v in hparams_overwrite_list.items():
        config[k] = v
        hparams_str += str(k) + "-" + str(v) + "_"
    config["model_architecture"] = "bigbigan"
    config["hparams_str"] = hparams_str.strip("_")
    config["seed"] = seed

#LOAD one sample img from the dataset
# 1. Load the dataset
    


print("running inference")
pip = pipeline.BigBiGANInference.from_checkpoint(checkpoint_path="./data/MNIST/bigbigan/checkpoints/checkpoint_40.pth", data_path="./data", config=config)
sample_latent = pip.encode_sample(3)
# RUN PCA on the latent space and plot it
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

sample_latent_numpy = [vect.detach().cpu().numpy() for vect in sample_latent]
sample_latent_numpy = np.concatenate(sample_latent_numpy, axis=0)


pca = PCA(n_components=2)
pca.fit(sample_latent_numpy)
sample_latent_pca = pca.transform(sample_latent_numpy)
plt.scatter(sample_latent_pca[:,0], sample_latent_pca[:,1])
plt.show()