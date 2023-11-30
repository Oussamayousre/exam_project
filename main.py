import argparse
import pickle
import umap
import streamlit as st

from sklearn.cluster import KMeans
import numpy as np
from src.utils import load_data,load_embeddings,plot_umap,plot_kmeans,train_umap_kmeans,train_tsne_kmeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
# Set Streamlit page configuration
st.set_page_config(
    page_title="test",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    /* Add your custom CSS styles here */
    body {
        background-color: #f0f0f0;
    }
    .prediction {
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
        text-align: center;
    }
    .image-container {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }
    .result-container {
        display: flex;
        justify-content: space-around;
        align-items: center;
        margin-top: 20px;
    }
    .styled-div {
        margin-top: 30px;
        margin-bottom: 30px;
        border: 0;
        border-top: 2px solid #eee;
    }
    .green-check {
        color: green;
        font-size: 24px;
        margin-right: 10px;
    }
    .red-alert {
        color: red;
        font-size: 24px;
        margin-right: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Update Streamlit app header and sidebar
# st.sidebar.image("images/logo.jpg")  # Replace with the path to your logo
st.markdown("# ")
st.sidebar.markdown("# Data Exploration and Clustering:")
st.sidebar.markdown(" This application enables you to reduce the dimensionality of your data using various techniques such as UMAP, t-SNE, and PCA, allowing you to explore the resulting clusters.")

# main function
def main(train_model,method,corpus,labels):
    st.title("Reducers and KMeans Training App")
    st.sidebar.header("Options")
    train_model = st.sidebar.checkbox("Train Models")
    reducer_type = st.sidebar.selectbox("Select Reducer Type", ["umap", "tsne", "pca"], index=0)
    if reducer_type == 'umap' : 
        if train_model:
            if st.button("Start Training"):

                st.header("Training Models")

                with st.spinner("Training models..."):
                    embedding,nmi_score,ari_score,kmeans_labels = train_umap_kmeans(corpus, labels, reducer_type)
                st.success("Models trained successfully!")

                st.header("Visualization")

                st.subheader(f"{reducer_type} Results")
                plot_umap(embedding, labels)

                st.subheader("KMeans Clustering")
                plot_kmeans(embedding,kmeans_labels)
                st.header(f'Method: {method}\nNMI: {nmi_score:.2f} \nARI: {ari_score:.2f}\n')
    if reducer_type == 'tsne' : 
        if train_model:
            if st.button("Start Training"):

                st.header("Training Models")

                with st.spinner("Training models..."):
                    embedding,nmi_score,ari_score ,kmeans_labels= train_tsne_kmeans(corpus, labels, reducer_type)
                st.success("Models trained successfully!")

                st.header("Visualization")

                st.subheader(f"{reducer_type} Results")
                plot_umap(embedding, labels)

                st.subheader("KMeans Clustering")
                plot_kmeans(embedding,kmeans_labels)
                st.header(f'Method: {method}\nNMI: {nmi_score:.2f} \nARI: {ari_score:.2f}\n')
    if reducer_type == 'pca' : 
        if train_model:
            if st.button("Start Training"):

                st.header("Training Models")

                with st.spinner("Training models..."):
                    embedding,nmi_score,ari_score ,kmeans_labels= train_umap_kmeans(corpus, labels, reducer_type)
                st.success("Models trained successfully!")

                st.header("Visualization")

                st.subheader("UMAP Results")
                plot_umap(embedding, labels)

                st.subheader("KMeans Clustering")
                plot_kmeans(embedding,kmeans_labels)
                st.header(f'Method: {method}\nNMI: {nmi_score:.2f} \nARI: {ari_score:.2f}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or load UMAP and KMeans models.")
    parser.add_argument("--train",type = int, help="Train models from scratch, 0 or 1")
    parser.add_argument("--model_type", type=str,default = 'umap', help="write the model type umap,acp or  tsne ")
    args = parser.parse_args()
    corpus,labels= load_data()
    main(args.train,args.model_type,corpus,labels)
