# CAEclust — Deep Autoencoder-Based Document Clustering

An unsupervised pipeline that clusters news articles from the
[Reuters newswire dataset](https://keras.io/api/datasets/reuters/) into topics
**without using their labels during training**, then measures the quality of the
discovered clusters against the true topics.

## Method

1. **Text → vectors.** Articles are converted into sequences of pretrained
   [GloVe](https://nlp.stanford.edu/projects/glove/) word vectors (100-d).
2. **Representation learning.** An **LSTM autoencoder** compresses each article into a
   compact fixed-length embedding by learning to reconstruct its own input.
3. **Affinity construction.** A **landmark-based** sparse affinity matrix is built from
   cosine similarities, keeping the computation scalable.
4. **Ensemble clustering.** Truncated SVD + K-Means produce the final cluster labels,
   evaluated with the **Normalized Mutual Information (NMI)** score.

The notebook also visualizes the training-loss curve and a 2-D t-SNE projection
comparing true labels against predicted clusters.

## Getting started

Open `CAEclust_document_clustering.ipynb` in
[Google Colab](https://colab.research.google.com/) (recommended) or a local Jupyter
environment, then run the cells top to bottom.

> A **GPU is recommended** — in Colab, go to *Runtime → Change runtime type → GPU*.
> The notebook downloads the GloVe embeddings (~822 MB) on first run.

### Dependencies

- `tensorflow`
- `scikit-learn`
- `scipy`
- `numpy`
- `matplotlib`

These are preinstalled in Colab. Locally you can install them with:

```bash
pip install tensorflow scikit-learn scipy numpy matplotlib
```

## Key parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `n_clusters` | `10` | Number of clusters to produce (Reuters has 46 topics — worth experimenting). |
| `n_landmarks` | `300` | Number of landmark points used to build the affinity matrix. |
| `encoding_dim` | `64` | Size of the compressed representation learned by the autoencoder. |

## Repository contents

- `CAEclust_document_clustering.ipynb` — the full, documented pipeline.
- `README.md` — this file.
