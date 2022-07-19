from PIL import Image
import streamlit as st
import functions
import numpy as np
import os

st.set_page_config(page_title='Shein', page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)

my_model = functions.read_model()
shein_df, cos_similarities_df = functions.read_data()

st.write("# Recommender System - Shein")

st.sidebar.image('logo.png', width=200)
st.sidebar.write('## Menu options')
menu_options = st.sidebar.radio('', ['Visual similarity', 'Explore topics', 'Search for clothing'])

if menu_options == 'Visual similarity':
    how_many_similar_images = st.slider('Number of most similar items', 4, 24, 4, 4)

    st.button('Show another random image')
    all_pics = os.listdir('data/photos_without_duplicates')
    given_img = f'photos_without_duplicates/{all_pics[np.random.randint(0, len(all_pics))]}'
    
    col1, col2, col3, col4 = st.columns(4)
    result_imgs = functions.retrieve_most_similar_products(cos_similarities_df, given_img, how_many_similar_images)
    col1, col2, col3 = st.columns(3)
    col2.write('## Target item')
    col2.image(Image.open(f'data/{given_img}'))

    col2.write('## Recommended items')
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        for i in range(0, how_many_similar_images//4, 1):
            st.image(Image.open(f'data/{result_imgs[i]}'))
            st.write(' ')
    with col2:
        for i in range(how_many_similar_images//4, 2*how_many_similar_images//4, 1):
            st.image(Image.open(f'data/{result_imgs[i]}'))
            st.write(' ')
    with col3:
        for i in range(2*how_many_similar_images//4, 3*how_many_similar_images//4, 1):
            st.image(Image.open(f'data/{result_imgs[i]}'))
            st.write(' ')
    with col4:
        for i in range(3*how_many_similar_images//4, 4*how_many_similar_images//4, 1):
            st.image(Image.open(f'data/{result_imgs[i]}'))
            st.write(' ')

if menu_options == 'Explore topics':
    topic_count = my_model.get_topic_info()
    topic_count['Name'] = [i.replace('_', ' ') for i in topic_count['Name']]
    st.dataframe(topic_count)
    n_topics = st.slider('Number of topics in the plot', 4, 28, 12, 4)
    st.plotly_chart(my_model.visualize_barchart(top_n_topics=n_topics), use_container_width=True)
    st.plotly_chart(my_model.visualize_topics(), use_container_width=True)
    st.plotly_chart(my_model.visualize_heatmap(), use_container_width=True)

if menu_options == 'Search for clothing':

    with st.form("my_form"):
        search_term = st.text_input('Type a search term')
        topics, probs = my_model.find_topics(search_term)
        topic_name = my_model.topic_names[topics[0]]
        topic_name = topic_name.replace('_', ' ')

        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.write(f'With {probs[0].round(3)} probability, the search term `"{search_term}"` falls into `"{topic_name}"` topic.')
            selected_cloths = shein_df[shein_df.topic==topics[0]].reset_index(drop=True)
            how_many_cloths = st.slider('Number of random cloths from the topic', 4, 24, 8, 4)
            col1, col2, col3, col4 = st.columns(4)
            random_n = selected_cloths.iloc[np.random.choice(selected_cloths.shape[0], how_many_cloths, replace=False),]

            with col1:
                for i in range(0, how_many_cloths//4, 1):
                    st.write(f'[{random_n.iloc[i].title}]({random_n.iloc[i].link})')
                    # st.text_area('', random_n.iloc[i].photo_id, height=150, key='1')
                    st.image(Image.open(f'data/photos_without_duplicates/{random_n.iloc[i].photo_id}.png'))
                    st.write(' ')

            with col2:
                for i in range(how_many_cloths//4, 2*how_many_cloths//4, 1):
                    st.write(f'[{random_n.iloc[i].title}]({random_n.iloc[i].link})')
                    # st.text_area('', random_n.iloc[i].photo_id, height=150, key='2')
                    st.image(Image.open(f'data/photos_without_duplicates/{random_n.iloc[i].photo_id}.png'))
                    st.write(' ')

            with col3:
                for i in range(2*how_many_cloths//4, 3*how_many_cloths//4, 1):
                    st.write(f'[{random_n.iloc[i].title}]({random_n.iloc[i].link})')
                    # st.text_area('', random_n.iloc[i].photo_id, height=150, key='3')
                    st.image(Image.open(f'data/photos_without_duplicates/{random_n.iloc[i].photo_id}.png'))
                    st.write(' ')

            with col4:
                for i in range(3*how_many_cloths//4, 4*how_many_cloths//4, 1):
                    st.write(f'[{random_n.iloc[i].title}]({random_n.iloc[i].link})')
                    # st.text_area('', random_n.iloc[i].photo_id, height=150, key='4')
                    st.image(Image.open(f'data/photos_without_duplicates/{random_n.iloc[i].photo_id}.png'))
                    st.write(' ')

            with st.expander('Show all cloths in the selected topic'):
                st.dataframe(selected_cloths.drop('topic', axis=1))