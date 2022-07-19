# from torch import cos_
from importlib_metadata import install
from send2trash import send2trash
import transformers
import streamlit as st

import pandas as pd
!pip install sentence-transformers
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.backend._utils import select_backend

@st.cache(allow_output_mutation=True)
def read_model():
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    model = select_backend(sentence_model)
    my_model = BERTopic.load('models/topic_model', embedding_model=model)
    return my_model

@st.cache
def read_data():
    shein_df = pd.read_csv('data/shein_data_with_topics.csv')
    cos_similarities_df = pd.read_parquet('data/cos_similarities.parquet')
    cos_similarities_df = cos_similarities_df.set_index('Unnamed: 0')
    return shein_df, cos_similarities_df

@st.cache(allow_output_mutation=True)
def vis_topics(model):
    return model.visualize_topics()

@st.cache(allow_output_mutation=True)
def vis_hierarchy(model):
    return model.visualize_hierarchy()

def retrieve_most_similar_products(cos_similarities_df, given_img, nb_closest_images=3):
    closest_imgs = cos_similarities_df.loc[given_img].sort_values(ascending=False)[1:nb_closest_images+1].index
    return closest_imgs
