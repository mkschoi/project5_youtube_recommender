import streamlit as st
import numpy as np
import pandas as pd
import pickle

from streamlit_player import st_player
import SessionState

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title('YouTube Video Recommender for Beginners in Stock Investing')

st.header('Introduction')

st.markdown("If you are one of those people who recently started investing in stocks, you might have heard your friends or so called 'experts' around you say 'buy XYZ, it will double in six months', or 'the EV industry is the next big thing, buy ABC to go along for the ride!' The truth is, investing is never that simple (if it was, everyone would do it and we would see a lot more millionaires and billionaires around us). Also, if you invest without getting properly educated on stock investing, and simply buy stocks based on some tips or casual observations without any analysis, you end up not knowing what to do when the time comes for some big decision making, like do you buy more, hold, or sell? The bottom line is: you need to do your own homework!")  
         
st.markdown("When you are buying a stock, you are not just trading a piece of paper. You are buying a piece of a company, a real _business_. And you are buying that because you believe it will grow revenues and generate a lot more cash flows than the market gives it credit for. Ultimately, a stock price reflects a company's _value_ in the long-term, and that value is determined based on how much cash flows its business will generate in the future, discounted all the way back to present.")
         
st.markdown("Okay, if this sounds all foreign to you, you have come to the right place. I worked in the investment industry for eight years, but there are way better resources on stock investing than I can provide. Books on investing are the obvious first place to go to, but many people nowadays don't want to grab a book and pore into hundreds of pages of text (sadly). So as an alternative, a lot more people seek educational videos, which are more approachable, interactive, and fun. And what other place to look for educational, free-for-all videos than YouTube? That is why I have decided to create a YouTube video recommender for those of you who want to get proper education on stock investing, including different types of investing styles, tools used in analyzing stocks, what makes a successful long-term investor, etc.")
         
st.markdown("You might ask, 'Why do I need your recommender? I can just go on YouTube and find the videos myself!' Well, the reason is simple. As a beginner in stock investing, you are unlikely to be able to distinguish good, educational videos from bad, misleading, promotional videos.  And there are A LOT of the latter on YouTube, based on my own experience. So without further ado, here is my recommender!")
         
st.header('Initial Video Recommender')

st.markdown("My recommender system is designed to generate a two-step recommendation: an initial set of video recommendations and a follow-up set of recommendations, both consisting of five videos. In the initial recommender, we let you choose your preferred topic, video duration, and upload date to narrow down the video candidates, and recommend the top five most relevant, high quality videos.  Below are brief definitions of the topic choices and why they are important to learn.")

st.markdown("**Valuation**: ")
st.markdown("**Competitive Moats**: ")
st.markdown("**Passive Investing**: ")
st.markdown("**Technology Stocks**: ")
st.markdown("**General**: ")

st.text("\n")
st.markdown("**Which topic would you like to learn about?**")

session_state = SessionState.get(search_button_init=False)

topic = st.selectbox('Select a topic:', options=['', 'Valuation', 'Competitive Moats', 'Passive Investing', 'Technology Stocks', 'General'],)

st.markdown("**How long would you like your videos to be (in minutes)?**")

duration = st.slider('Less than:', 0, 120, 30)

st.markdown("**How recent would you like your videos to be (in months since upload date)?**")

upload_date = st.slider('Less than:', 1, 60, 12)

search_button_init = st.button('Search for recommended videos', key=1)

df_videos_cleaned_v9 = pd.read_csv('df_videos_cleaned_v9.csv')

def initial_recommender(df, topic, duration, upload_date):
    '''
    Input: Final dataframe of video data, user input on topic, duration, and upload date
    Output: Top five recommendations
    '''

    ## Define a new variable to store the preferred videos. Copy the contents of df to filtered videos
    df_videos_filtered = df.copy()

    ## Return top five videos based on topic coefficient (how relevant the videos are to the user's topic)
    df_videos_filtered = df_videos_filtered[(df_videos_filtered['Topic']==topic) & 
                                      (df_videos_filtered['Duration']<duration) &
                                      (df_videos_filtered['Months Since Upload']<upload_date)]
    df_videos_filtered = df_videos_filtered.sort_values('Topic Coefficient', ascending=False)
    
    return df_videos_filtered[['Video_ID','Title']].head(), df_videos_filtered

def init_embedded_rec_videos(df):
    init_recs_video_ids = list(df['Video_ID'])
    init_recs_video_titles = list(df['Title'])
    init_recs_ids_titles = list(zip(init_recs_video_ids, init_recs_video_titles))
    
    for id, title in init_recs_ids_titles:
        st.subheader(title)
        st_player('www.youtube.com/watch?v=' + id)
        st.text("\n")
        
if search_button_init:
    session_state.search_button_init = True
        
if session_state.search_button_init:
    df_videos_recs_init = initial_recommender(df_videos_cleaned_v9, topic, duration, upload_date)[0]
    df_videos_filtered = initial_recommender(df_videos_cleaned_v9, topic, duration, upload_date)[1]
    init_embedded_rec_videos(df_videos_recs_init)
    
    st.header('Follow-up Video Recommender')

    st.markdown("For the follow-up video recommender, we let you pick your favorite video(s) from the initial recommender, and we will come up with five more recommended videos based on those favorite(s). If you didn't like any of the initial videos, you can search for more videos in the initial recommender again!")

    st.markdown("**Which video(s) did you like?**")
    
    titles = list(df_videos_recs_init['Title'])
    session_state.video_id_follow_up = st.multiselect('Select one or more:', options=['', titles[0], titles[1], titles[2], titles[3], titles[4]])
    
    def follow_up_recommender(title, df_videos_filtered):
        '''
        Input: Video ID of a user's liked video, and the dataframe of the filtered videos generated from the initial recommender
        Output: Top five follow-up recommendations using content-based recommender system
        '''
        ## Fit and transform the transcript into a document-term matrix
        word_list = [[word[0] for word in eval(doc)] for doc in df_videos_filtered['Transcript']]
        vec = TfidfVectorizer(tokenizer=lambda doc:doc, lowercase=False)
        matrix = vec.fit_transform(word_list).toarray()

        ## Generate a similarity matrix
        similarity_matrix = cosine_similarity(matrix, matrix)

        ## Create a series of indices for the videos (rows)  
        df_videos_filtered = df_videos_filtered.reset_index(drop=True)
        indices = pd.Series(df_videos_filtered['Title'])

        ## Get the index of the user's liked video
        idx = indices[indices == title].index[0]

        ## Create a series with the similarity scores in descending order and grab indices
        score_series = pd.Series(similarity_matrix[idx]).sort_values(ascending=False)
        similarity_indices = list(score_series.index)

        ## Drop videos that were in the original recommendation
        similarity_indices = [index for index in similarity_indices if index not in list(df_videos_filtered[:5].index)]
        top_5_indices = similarity_indices[:5]

        ## Populate a dataframe of the recommended videos
        df_videos_follow_up_recs = df_videos_filtered.iloc[top_5_indices]

        return df_videos_follow_up_recs[['Video_ID','Title']]
    
    def follow_up_embedded_rec_videos(df):
        follow_up_recs_video_ids = list(df['Video_ID'])
        follow_up_recs_video_titles = list(df['Title'])
        follow_up_recs_ids_titles = list(zip(follow_up_recs_video_ids, follow_up_recs_video_titles))
    
        for id, title in follow_up_recs_ids_titles:
            st.subheader(title)
            st_player('www.youtube.com/watch?v=' + id)
            st.text("\n")
    
    if session_state.video_id_follow_up:
        session_state.search_button_follow_up = st.button('Search for recommended videos', key=2)
        if session_state.search_button_follow_up:
            df_videos_recs_follow_up = follow_up_recommender(session_state.video_id_follow_up[0], df_videos_filtered)
            follow_up_embedded_rec_videos(df_videos_recs_follow_up)