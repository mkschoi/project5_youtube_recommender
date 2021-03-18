import streamlit as st
import numpy as np
import pandas as pd
import pickle

from PIL import Image
from streamlit_player import st_player
import SessionState

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Toc:

    def __init__(self):
        self._items = []
        self._placeholder = None

    def title(self, text):
        self._markdown(text, "h1")
        
    def header(self, text):
        self._markdown(text, "h2", " " * 2)

    def placeholder(self, sidebar=True):
        self._placeholder = st.sidebar.empty() if sidebar else st.empty()

    def generate(self):
        if self._placeholder:
            self._placeholder.markdown("\n".join(self._items), unsafe_allow_html=True)
    
    def _markdown(self, text, level, space=""):
        key = "".join(filter(str.isalnum, text)).lower()

        st.markdown(f"<{level} id='{key}'>{text}</{level}>", unsafe_allow_html=True)
        self._items.append(f"{space}* <a href='#{key}'>{text}</a>")

st.title('YouTube Video Recommender for Beginners in Stock Investing')

st.subheader('_"The stock market is filled with individuals who know the price of everything, but the value of nothing."_ — Phillip Fisher')
st.text("\n")

image = Image.open('title_image.jpeg')
st.image(image)
        
toc = Toc()

st.sidebar.title('Table of Contents')

toc.placeholder()

toc.header("Why Use My Recommender?")

st.markdown("If you are one of those people who recently started investing in stocks, you might have heard your friends or so-called 'experts' around you say 'buy XYZ, it will double in six months!', or 'the EV industry is the next big thing, buy ABC to go along for the ride!' The truth is, investing is never that simple (if it was, we would have a lot more millionaire friends). Also, if you invest without getting properly educated, and simply buy stocks based on some tips or casual observations without any analysis, you end up not knowing what to do when the time comes for making a big decision, like should you buy more, hold, or sell when a stock is down 30%? The bottom line is: you need to do your own homework!")  
         
st.markdown("When you are buying a stock, you are not just trading a piece of paper. You are buying a piece of a company, a real _business_. And you are buying that because you believe it will grow revenues and generate a lot more cash flows than the market gives it credit for. Ultimately, a stock price reflects a company's _value_ in the long-term, and that value is determined based on how much cash flow its business will generate in the future, discounted all the way back to present.")
         
st.markdown("Okay, if this sounds all foreign to you, you have come to the right place. I worked in the investment industry for eight years, but there are way better resources on stock investing than I can provide. Books on investing are the obvious first place to go to, but many people nowadays don't want to grab a book and pore into hundreds of pages of text (sadly). So as an alternative, a lot more people seek educational videos, which are more approachable, interactive, and fun. And what other place to look for content-rich, free-for-all videos than YouTube? That is why I have created a YouTube video recommender for those of you who want to get proper education on stock investing, including different investing styles, tools used in analyzing stocks, what makes a successful long-term investor, etc.")
         
st.markdown("You might ask, 'Why do I need your recommender? I can just go on YouTube and find the videos myself!' Well, the reason is simple. As a beginner in stock investing, you are unlikely to be able to distinguish good, educational videos from bad, promotional, and often misleading videos.  And there are A LOT of the latter on YouTube, based on my own experience. So without further ado, let's start your investment journey!")

toc.header("Topics to Explore")

st.markdown("The videos are categorized into 12 different topics on investing. Below are the brief descriptions of the topics and why they are important to learn as a beginner. You don't have to go from the top to bottom, but I have ordered them in a logical sequence so that you learn the basics of different investment styles before delving into the tools used in analyzing stocks.")

st.markdown("**Value Investing**: An investment strategy that involves picking stocks that appear to be trading for less than their intrinsic value. In the traditional sense, a value investor cares more about the value of a company's assets on the balance sheet and its current earnings power than its future growth. Benjamin Graham (the father of value investing) first institutionalized the concept in the 1930s, and has since become a pillar for other styles of investing. Naturally, a beginner looking to become a proper investor would want to understand the concept of value investing as his/her first stepping stone.")
st.markdown("**Growth Investing vs. Value Investing**: Growth investing is an investment style/strategy that is focused on seeking companies that offer rapid revenue and earnings growth into the future, with asset values and the current earnings power taking a backseat. However, many investors view growth investing to be part of value investing, because growth is simply a component of value. It is important to learn why the distinction could get fuzzy.")
st.markdown("**Long-term Investing**: This is not a separate investment style, but a philosophy that applies to all types of investing. It is crucial to understand why investing over the long-term stacks the odds in your favor simply due to the wonders of compounding returns, and every investor's behavioral bias and misconception in predicting short-term stock price movements (even some of the well-known investors are prone to this).")
st.markdown("**Dividend Investing**: A strategy of buying stocks that pay periodic dividends in order to receive a regular income from your investments. A company has discretion over whether to pay a dividend out of its profit, and it is considered _after_ making all the internal investments needed to grow its business. Therefore, dividend stocks are typically found among stable, mature companies/industries generating ample excess cash flow (e.g., Consumer Staples, Utilities, and Telecom)")
st.markdown("**Passive Investing**: Broadly refers to an investment strategy that tracks a benchmark index or portfolio (e.g., S&P 500, Russell 3000). A passive investor has a buy-and-hold mentality and limits the amount of buying and selling within his/her portfolio, making this a very cost-effictive way to invest. In fact it is due to the low fees associated with passive investing, along with the underperformance of active funds in recent years that are pushing many stock investors (both institutional and retail) towards the former.")
st.markdown("**Fundamental vs. Technical Analysis**: Two major schools of thought or approaches to making money in the stock market. Fundamental analysis involves studying a company's business operations, financial statements, and competitive/macro landscape in order to derive its intrinsic value (all the investment styles mentioned above require at least some fundamental analysis). Technical analysis is different in that _traders_ (not investors) attempt to identify opportunities by studying a stock's price/volume charts and finding patterns and trends in the short-term price movements. Technical analysis can be an ancillary component of an investor's stock analysis, but it should never be the main driver behind his/her investment thesis.")
st.markdown("**Economic Moats**: An economic moat is a distinct, sustainable advantage a company has over its competitors which allows it to protect its market share and profitability (like moats around a castle to protect against an invasion). It is often an advantage that is difficult to mimic or duplicate (brand identity, patents, network effects). Economic moats are one of the most important qualities that an investor should look for in a company when performing fundamental analysis with a long-term mindset.")
st.markdown("**Valuation**: Valuation is an analytical process of determining the fair value of an asset or a company. There are many ways to valuing a company depending on which industry it is in, whether it is generating positive or negative cash flow, etc (many of the YouTube videos here delve into these different approaches). If finding economic moats is the ultimate goal of a qualitative analysis, valuation is the final quantitative output you want to get to compare to the company's current stock price, and determine if it is buy, hold, or a sell.")
st.markdown("**Valuation (Case Studies)**: Valuation methods are hard to grasp with just theories and simple examples. Conducting case studies and walking through each step of a valuation process is helpful in cementing the logic behind how an intrinsic value is determined. A properly educated investor should have the skills to correctly use different valuation methods under varying situations, and as a result, have the conviction to make big investment decisions")
st.markdown("**Technology Stocks**: Over the past decade, the performance of the stock market has mostly been driven by the so-called 'mega-cap tech stocks' or 'FAANGs'. It is important to study why these companys have outperformed for so long, what their differentiating qualities (or moats) are, and whether it is sustainable.")
st.markdown("**Electric Vehicle Stocks**: One industry that has caught the investment world's attention over the last few years, is the electric vehicle industry, led by Tesla. While there are no doubts as to the fact that much of the meteoric rise in Tesla's stock price is euphoria-driven, it is still important to understand why the industry has gained so much popularity among not only retail but institutional investors, what the long-term trends are, and how one should value a company in this hyper-growth but cash flow-shy sector.")
st.markdown("**General**: Investment topics that don't neatly fall under the above 11 categories are included here.")

toc.header('Initial Video Recommender')

st.markdown("My recommender system is designed to generate a two-step recommendation: an initial set of video recommendations and a follow-up set of recommendations, both consisting of five videos. In the initial recommender, we let you choose your preferred topic, video duration, and upload date to narrow down the video candidates, and recommend the top five most relevant, high quality videos.") 
st.text("\n")

df_videos_cleaned_v10 = pd.read_csv('df_videos_cleaned_v10.csv')

st.markdown("**Which topic would you like to learn about?**")

session_state = SessionState.get(search_button_init=False)

topic_list = list(df_videos_cleaned_v10['Topic'].value_counts().index)
topic_list_ordered = [topic_list[1], topic_list[11], topic_list[8], topic_list[9], topic_list[6], topic_list[4], topic_list[5], topic_list[2], topic_list[10], topic_list[3], topic_list[7], topic_list[0]]
topic_list_ordered.insert(0,'')

topic = st.selectbox('Select a topic:', options=topic_list_ordered)

st.markdown("**How long would you like your videos to be (in minutes)?**")

duration = st.slider('Less than:', 0, 120, 30)

st.markdown("**How recent would you like your videos to be (in months since upload date)?**")

upload_date = st.slider('Less than:', 1, 60, 12)

search_button_init = st.button('Search for recommended videos', key=1)

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
    df_videos_recs_init = initial_recommender(df_videos_cleaned_v10, topic, duration, upload_date)[0]
    df_videos_filtered = initial_recommender(df_videos_cleaned_v10, topic, duration, upload_date)[1]
    init_embedded_rec_videos(df_videos_recs_init)
    
    toc.header('Follow-up Video Recommender')

    st.markdown("For the follow-up video recommender, we let you pick your favorite video(s) from the initial recommender, and we will come up with five more recommended videos based on those favorite(s). If you didn't like any of the above videos, you can search for more videos in the initial recommender again!")

    st.markdown("**Which video(s) did you like?**")
    
    titles = list(df_videos_recs_init['Title'])
    session_state.video_id_follow_up = st.multiselect('Select one or more:', options=['', titles[0], titles[1], titles[2], titles[3], titles[4]])
    
    def follow_up_recommender(liked_video_list, df_videos_filtered):
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

        ## Create a series of titles for the videos (rows)  
        df_videos_filtered = df_videos_filtered.reset_index(drop=True)
        indices = pd.Series(df_videos_filtered['Title'])

        ## Get the indices of the user's liked video(s)
        idx_list = []
        for liked_video in liked_video_list:
            idx_list.append(indices[indices == liked_video].index[0])

        ## Create a dataframe of the similarity matrix, but only showing columns for the liked videos
        scores_list = []
        for idx in idx_list:
            scores_list.append(similarity_matrix[idx])
            
        scores_df = pd.DataFrame(scores_list).T
        scores_df.columns = idx_list
            
        ## Drop videos that were in the original recommendation
        scores_df.drop([0,1,2,3,4], inplace=True)
        
        ## Calculate the mean cosine similarity score for each video    
        mean_score_series = scores_df.mean(axis='columns').sort_values(ascending=False)
        
        ## Get the indices of the five highest scores
        similarity_indices = list(mean_score_series.index)
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
            df_videos_recs_follow_up = follow_up_recommender(session_state.video_id_follow_up, df_videos_filtered)
            follow_up_embedded_rec_videos(df_videos_recs_follow_up)

toc.generate()

