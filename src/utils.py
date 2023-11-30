from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import argparse
import pickle
import umap
import streamlit as st
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import joblib 

st.set_option('deprecation.showPyplotGlobalUse', False)
# def load_data():
#     # import data
#     ng20 = fetch_20newsgroups(subset='test')
#     corpus = ng20.data[:2000]
#     labels = ng20.target[:2000]
#     k = len(set(labels))
#     return corpus,labels,k
def load_data():

    loaded_data = joblib.load('data/ng20_data.joblib')

    # Retrieve the variables
    loaded_corpus = loaded_data['corpus']
    loaded_labels = loaded_data['labels']
    return loaded_corpus,loaded_labels
def load_embeddings(corpus) :
        
    # embedding
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(corpus)
    return embeddings

def plot_umap(embedding, labels, title='UMAP Projection of Data'):
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='viridis', s=10)
    plt.title(title)
    plt.colorbar()
    st.pyplot()





def plot_kmeans(embedding, kmeans_labels, title='UMAP Projection with K-means Clustering'):
    plt.scatter(embedding[:, 0], embedding[:, 1], c=kmeans_labels, cmap='viridis', s=10)
    plt.title(title)
    plt.colorbar()
    st.pyplot()


def train_umap_kmeans(corpus,labels, n_neighbors=5, min_dist=0.3, n_components=2, n_clusters=5):
    try:

        # embedding
        embeddings = load_embeddings(corpus)
        # Train UMAP
        # reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=20)
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.001)

        # Fit and transform the data
        embedding = reducer.fit_transform(embeddings)
        # Train KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(embedding)

        # Evaluate clustering results
        nmi_score = normalized_mutual_info_score(kmeans_labels, labels)
        ari_score = adjusted_rand_score(kmeans_labels, labels)
        method = 'umap'
        # Print results
        print(f'Method: {method}\nNMI: {nmi_score:.2f} \nARI: {ari_score:.2f}\n')
        
        # Save models to pickle files
        with open('models/reducers/umap_model.pkl', 'wb') as file:
            pickle.dump(reducer, file)

        with open('models/K-means/kmeans_model.pkl', 'wb') as file:
            pickle.dump(kmeans, file)
        return embedding,nmi_score,ari_score,kmeans_labels
    except Exception as e:
        print(f"Error saving models: {e}")
        return None, None,None,None
def train_tsne_kmeans(corpus,labels, n_neighbors=5, min_dist=0.3, n_components=2, n_clusters=5):
    try:

        # embedding
        embeddings = load_embeddings(corpus)
        # Train TSNE
        reducer = TSNE(n_components=2)
        embedding = reducer.fit_transform(embeddings)

        # Train KMeans
        kmeans = KMeans(n_clusters=20, random_state=42)
        kmeans_labels = kmeans.fit_predict(embedding)

        # Evaluate clustering results
        nmi_score = normalized_mutual_info_score(kmeans_labels, labels)
        ari_score = adjusted_rand_score(kmeans_labels, labels)
        method = 'TSNE'
        # Print results
        print(f'Method: {method}\nNMI: {nmi_score:.2f} \nARI: {ari_score:.2f}\n')
        
        # Save models to pickle files
        with open('models/reducers/tsne_model.pkl', 'wb') as file:
            pickle.dump(reducer, file)

        with open('models/K-means/Kmeans_Tsne.pkl', 'wb') as file:
            pickle.dump(kmeans, file)
        return embedding,nmi_score,ari_score,kmeans_labels

    except Exception as e:
        print(f"Error saving models: {e}")
        return None, None,None,None
def train_umap_kmeans(corpus,labels, n_neighbors=5, min_dist=0.3, n_components=2, n_clusters=5):
    try:
        embeddings = load_embeddings(corpus)
        # Train ACP
        reducer = PCA(n_components=20) 
        embedding = reducer.fit_transform(embeddings)

        # Train KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(embedding)

        # Evaluate clustering results
        nmi_score = normalized_mutual_info_score(kmeans_labels, labels)
        ari_score = adjusted_rand_score(kmeans_labels, labels)
        method = 'acp'
        # Print results
        print(f'Method: {method}\nNMI: {nmi_score:.2f} \nARI: {ari_score:.2f}\n')
        

        # Save models to pickle files
        with open('models/reducers/pca_model.pkl', 'wb') as file:
            pickle.dump(reducer, file)

        with open('models/K-means/Kmeans_pca.pkl', 'wb') as file:
            pickle.dump(kmeans, file)
        return embedding,nmi_score,ari_score,kmeans_labels


    except Exception as e:
        print(f"Error saving models: {e}")
        return None, None