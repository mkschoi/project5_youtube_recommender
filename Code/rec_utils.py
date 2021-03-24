import numpy as np
import pandas as pd

from datetime import date
from IPython.display import clear_output

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def initial_recommender(df):
    '''
    Input: Final dataframe of video data, user input on topic, duration, and upload date
    Output: Top five recommendations
    '''
    ## Take user inputs
    print('Which topics would you like to learn about?')
    topic = input()
    
    print('\n')
    print('What is the upper limit of the duration of videos (in minutes)?')
    duration = int(input())
    
    print('\n')
    print('How recent do you want the videos to be (in months since upload date)?')
    upload_date = int(input())
    today = date.today()
    df['Months Since Upload'] = df['Upload Date'].apply(lambda x:((today - x).days)*12/365)

    clear_output()
    
    ## Define a new variable to store the preferred videos. Copy the contents of df to filtered videos
    df_videos_filtered = df.copy()

    ## Return top five videos based on topic coefficient (how relevant the videos are to the user's topic)
    df_videos_filtered = df_videos_filtered[(df_videos_filtered['Topic']==topic) & 
                                      (df_videos_filtered['Duration']<duration) &
                                      (df_videos_filtered['Months Since Upload']<upload_date)]
    df_videos_filtered = df_videos_filtered.sort_values('Topic Coefficient', ascending=False)
    
    return df_videos_filtered.head(), df_videos_filtered

def follow_up_recommender(video_id, df_videos_filtered, vectorizer, similarity_metric):
    '''
    Input: Video ID of a user's liked video, and the dataframe of the filtered videos generated from the initial recommender
    Output: Top five follow-up recommendations using content-based recommender system
    '''
    ## Fit and transform the transcript into a document-term matrix
    word_list = [[word[0] for word in doc] for doc in df_videos_filtered['Transcript']]
    vec = vectorizer(tokenizer=lambda doc:doc, lowercase=False)
    matrix = vec.fit_transform(word_list).toarray()
    
    ## Generate a similarity matrix
    similarity_matrix = similarity_metric(matrix, matrix)
    
    ## Create a series of indices for the videos (rows)  
    df_videos_filtered = df_videos_filtered.reset_index(drop=True)
    indices = pd.Series(df_videos_filtered['Video_ID'])
    
    ## Get the index of the user's liked video
    idx = indices[indices == video_id].index[0]
    
    ## Create a series with the similarity scores in descending order and grab indices
    score_series = pd.Series(similarity_matrix[idx]).sort_values(ascending=False)
    similarity_indices = list(score_series.index)
    
    ## Drop videos that were in the original recommendation
    similarity_indices = [index for index in similarity_indices if index not in list(df_videos_filtered[:5].index)]
    top_5_indices = similarity_indices[:5]
    
    ## Populate a dataframe of the recommended videos
    df_videos_follow_up_recs = df_videos_filtered.iloc[top_5_indices]
    
    return df_videos_follow_up_recs