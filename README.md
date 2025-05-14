# Unsupervised Text Clustering using BERT Embeddings and Autoencoders on AG News Dataset

## 1\. Overview

This project implements an unsupervised text clustering pipeline to group news articles from the AG News dataset into their respective categories without using predefined labels. The core idea is to leverage powerful pre-trained BERT embeddings for text representation, reduce their dimensionality using an autoencoder neural network, and then apply K-Means clustering to the compressed embeddings. The project explores the effectiveness of this deep learning approach for discovering underlying structures in text data.

## 2\. Features

  * **Text Embedding**: Utilizes the 'bert-base-uncased' model to generate rich contextual embeddings (768 dimensions) for news articles.
  * **Dimensionality Reduction**: Employs a custom-built autoencoder to reduce the embedding dimension from 768 to 64, aiming to capture the most salient features.
  * **Unsupervised Clustering**: Applies the K-Means algorithm to group the dimension-reduced embeddings into 4 clusters (corresponding to the AG News categories).
  * **Evaluation**: Uses intrinsic clustering metrics (Silhouette Score, Davies-Bouldin Index) to assess cluster quality.
  * **Visualization**: Includes PCA and t-SNE for visualizing the formed clusters in 2D space.
  * **Dataset**: Uses a subset of the standard AG News dataset.

## 3\. Dataset

The project uses the **AG News dataset**, a widely used benchmark for text classification and clustering tasks.

  * **Source**: Loaded via the Hugging Face `datasets` library.
  * **Content**: News articles categorized into four classes:
    1.  World
    2.  Sports
    3.  Business
    4.  Sci/Tech
  * **Structure**: Each data point consists of `text` (the article content) and `label` (0-3 for the categories). The labels are *not* used during the clustering process but can be used for external validation of the results.
  * **Subsampling**: For this project, a subset of **5,000 articles** from the training set was used to expedite the embedding generation and model training processes, especially when running on CPU.

## 4\. Methodology

The pipeline consists of the following stages:

### 4.1. Text Embedding Generation

1.  **BERT Model**: The `bert-base-uncased` model from the Hugging Face `transformers` library is used.
2.  **Tokenization**: Texts are tokenized using the corresponding BERT tokenizer, with a maximum length of 128 tokens. Texts are padded or truncated as necessary.
3.  **Embedding Extraction**: The embedding for the `[CLS]` token from the last hidden state of BERT is used as the representative vector for each news article. This results in a 768-dimensional embedding for each text.
      * The BERT model is used in evaluation mode (`bert_model.eval()`) as it's only for feature extraction.

### 4.2. Autoencoder for Dimensionality Reduction

An autoencoder is trained to learn a compressed representation of the BERT embeddings.

  * **Architecture**:
      * **Encoder**:
          * Input Layer: 768 dimensions (BERT embedding)
          * Hidden Layer 1: Linear layer (768 $\\rightarrow$ 256 units) + ReLU activation
          * Bottleneck Layer (Latent Space): Linear layer (256 $\\rightarrow$ 64 units)
      * **Decoder**:
          * Input Layer: 64 dimensions (from bottleneck)
          * Hidden Layer 1: Linear layer (64 $\\rightarrow$ 256 units) + ReLU activation
          * Output Layer: Linear layer (256 $\\rightarrow$ 768 units) - Reconstructs the original embedding
  * **Training**:
      * Loss Function: Mean Squared Error (MSELoss) between the original and reconstructed embeddings.
      * Optimizer: Adam.
      * The encoder part of the trained autoencoder is then used to transform the 768-dim BERT embeddings into 64-dim latent representations.

### 4.3. K-Means Clustering

The K-Means algorithm is applied to the 64-dimensional latent embeddings obtained from the autoencoder.

  * **Number of Clusters (k)**: Set to 4, corresponding to the known number of categories in the AG News dataset.

## 5\. Neural Network Architecture Details

### 5.1. Block Diagram

*(Please refer to the SVG image previously generated or insert your diagram here if you have one.)*
A conceptual flow:
`Input (768-dim BERT Emb.)` $\\rightarrow$ `Encoder (Linear 768->256, ReLU, Linear 256->64)` $\\rightarrow$ `Latent Space (64-dim)` $\\rightarrow$ `Decoder (Linear 64->256, ReLU, Linear 256->768)` $\\rightarrow$ `Output (Reconstructed 768-dim Emb.)`

### 5.2. Model Parameter Counting

  * **Autoencoder**:
      * Encoder:
          * Layer 1 (768x256 weights + 256 biases) = 196,864 parameters
          * Layer 2 (256x64 weights + 64 biases) = 16,448 parameters
          * *Total Encoder Parameters: 213,312*
      * Decoder:
          * Layer 1 (64x256 weights + 256 biases) = 16,640 parameters
          * Layer 2 (256x768 weights + 768 biases) = 197,376 parameters
          * *Total Decoder Parameters: 214,016*
      * **Total Autoencoder Trainable Parameters: 427,328**
  * **BERT (`bert-base-uncased`)**: Contains approximately 110 million parameters. These were *not* trained in this project (used as a fixed feature extractor).

### 5.3. Regularization Techniques

  * **Batch Normalization**: Not explicitly used in the autoencoder architecture presented in the notebook.
  * **Dropout**: Not explicitly used in the autoencoder layers. BERT itself contains dropout layers, but these are typically inactive when the model is in `eval()` mode.
  * The primary form of regularization is the **bottleneck layer** in the autoencoder, which forces a compressed representation of the data.

## 6\. Setup and Installation

To run this project, you'll need Python 3 and the following libraries. You can install them using pip:

```bash
pip install torch torchvision torchaudio
pip install datasets transformers scikit-learn matplotlib seaborn jupyter notebook
```

**Environment:**
The notebook was run in a Google Colab environment (as indicated by the ipynb metadata). The device used was CPU (as per `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` and the output `Using device: cpu`). For significantly faster execution, especially BERT embedding generation and autoencoder training, a GPU environment is highly recommended.

## 7\. Usage / Running the Code

1.  **Download the Notebook**: Obtain the `Analysis_of_AG_News.ipynb` file.
2.  **Set up Environment**: Ensure all dependencies listed above are installed.
3.  **Open in Jupyter**: Launch Jupyter Notebook or JupyterLab and open the `.ipynb` file.
    ```bash
    jupyter notebook Analysis_of_AG_News.ipynb
    ```
    Alternatively, upload and run it in Google Colab.
4.  **Run Cells**: Execute the cells in the notebook sequentially. The first cell installs and imports necessary libraries. Subsequent cells set up the device, load the dataset, initialize the BERT tokenizer and model, define the AGNewsDataset class for embedding generation, create the autoencoder model, train the autoencoder, extract latent embeddings, perform K-Means clustering, and finally evaluate and visualize the clusters.

**Key Parameters in the Notebook:**

  * `sample_size = 5000`: Number of articles to use from the dataset.
  * `batch_size = 64`: Batch size for training the autoencoder.
  * `input_dim = 768`: Dimension of BERT embeddings.
  * `embedding_dim = 64`: Dimension of the autoencoder's bottleneck layer.
  * `learning_rate = 0.001`: Learning rate for the Adam optimizer.
  * `epochs = 20`: Number of epochs for training the autoencoder.
  * `n_clusters = 4`: Number of clusters for K-Means.

## 8\. Hyperparameter Optimization and Tuning

The hyperparameters were primarily set to common default values or reasonable choices for this task:

  * **Autoencoder Architecture**: The (768 $\\rightarrow$ 256 $\\rightarrow$ 64 $\\rightarrow$ 256 $\\rightarrow$ 768) structure was chosen. The bottleneck dimension of 64 is a significant reduction and was a primary point for potential tuning.
  * **Learning Rate, Epochs, Batch Size**: Values like 0.001 for learning rate, 20 epochs, and batch size 64 are standard starting points.
  * **Tuning Approach**: The notebook shows the autoencoder training loss decreasing over epochs, indicating learning. A more rigorous hyperparameter search (e.g., grid search over bottleneck dimensions, learning rates) could be performed if computational resources allow, using intrinsic clustering metrics as the optimization objective.

## 9\. Results and Evaluation

### 9.1. Intrinsic Clustering Metrics

The quality of the clusters formed by K-Means on the 64-dimensional autoencoder embeddings was assessed using:

  * **Silhouette Score**: 0.0643
  * **Davies-Bouldin Index**: 3.2711

These scores suggest that the clusters have some overlap and are not perfectly separated. A Silhouette Score closer to 1 and a Davies-Bouldin Index closer to 0 would indicate better clustering.

### 9.2. Determining Clustering Accuracy (Unsupervised Context)

In an unsupervised setting, "accuracy" is typically assessed by comparing the formed clusters to known ground-truth classes (if available for evaluation purposes, like in AG News). This involves:

  * **Mapping Clusters to True Labels**: Since K-Means assigns arbitrary cluster IDs (0, 1, 2, 3), these need to be mapped to the actual news categories (World, Sports, etc.). This can be done using:
      * **Majority Voting**: For each cluster, find the most frequent true label among its members and assign that label to the cluster.
      * **Hungarian Algorithm**: An optimal assignment algorithm that maximizes the number of correctly classified samples after clustering.
  * **Calculating Metrics**: Once mapped, standard classification metrics like Accuracy, Purity, Normalized Mutual Information (NMI), and Adjusted Rand Index (ARI) can be computed.

The `Analysis_of_AG_News.ipynb` notebook focuses on intrinsic metrics, which don't require this mapping.

### 9.3. Visualization

The notebook uses PCA and t-SNE to reduce the 64-dimensional embeddings to 2 dimensions for visualization. Scatter plots colored by the K-Means assigned cluster labels help in qualitatively assessing the cluster separation.

## 10\. Comparison with Existing Clustering Methods

This project explores a specific pipeline. For a comprehensive comparison:

**Baselines:**

  * K-Means directly on 768-dim BERT embeddings (to see the effect of the autoencoder).
  * K-Means on TF-IDF features.

**Other Unsupervised Methods:**

  * Hierarchical Clustering, DBSCAN.
  * Topic Modeling (e.g., LDA).
  * More advanced deep clustering methods (e.g., DEC, IDEC) that jointly learn representations and clusters.

The current intrinsic scores can be used as a reference when comparing against these other approaches on the same dataset subset.

## 11\. Limitations and Obstacles

  * **Computational Resources**: Training on CPU is slow. Using a GPU would significantly speed up BERT embedding generation and autoencoder training, allowing for experiments on the full dataset and more extensive hyperparameter tuning.
  * **Dataset Subsampling**: Using only 5,000 samples might limit the model's ability to learn generalizable features and could affect the overall clustering quality.
  * **Hyperparameter Sensitivity**: The performance of both the autoencoder and K-Means can be sensitive to hyperparameter choices. The current setup uses initial reasonable values; further tuning is likely to improve results.
  * **Evaluation Challenges**: Intrinsic metrics provide some insight but don't always correlate perfectly with how humans might perceive cluster quality or how well clusters align with external (true) categories.
  * **Sequential Approach**: The autoencoder is trained for reconstruction, not explicitly for creating a cluster-friendly latent space. Joint optimization techniques might yield better clustering.

## 12\. Future Work

  * Run experiments on the full AG News dataset using GPU resources.
  * Conduct a systematic hyperparameter search for the autoencoder (bottleneck dimension, layer sizes, learning rate, activation functions) and K-Means.
  * Experiment with different pooling strategies for BERT embeddings (e.g., mean pooling).
  * Incorporate regularization techniques like Dropout or Batch Normalization in the autoencoder.
  * Compare results with other clustering algorithms and dimensionality reduction techniques (e.g., PCA directly on BERT embeddings).
  * Explore end-to-end deep clustering models.
  * Perform external validation by mapping clusters to true labels and calculating metrics like NMI, ARI, and Purity.

## 13\. References

  * Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.
  * Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level convolutional networks for text classification. In Advances in neural information processing systems (pp. 649-657).
  * Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of machine learning research, 12(Oct), 2825-2830.
  * Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. In Advances in neural information processing systems (pp. 8026-8037).
  * Wolf, T., et al. (2019). HuggingFace's Transformers: State-of-the-art Natural Language Processing. arXiv preprint arXiv:1910.03771.
  * Lhoest, Q., et al. (2021). Datasets: A community library for natural language processing. arXiv preprint arXiv:2109.02846.

-----

```
```
