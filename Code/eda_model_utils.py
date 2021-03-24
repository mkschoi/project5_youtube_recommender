import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt

from wordcloud import WordCloud

from sklearn.manifold import TSNE
from gensim.models import word2vec

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

df_videos_cleaned_v9 = pickle.load(open('../Data/df_videos_cleaned_v9.pickle', 'rb'))
X_tfidf = pickle.load(open('../Data/X_tfidf.pickle', 'rb'))

def wordcloud(df, text_column):
    '''
    Input: Cleaned dataframe 
    Output: a Wordcloud viz showing most frequently represented words
    '''
    ## Take the words out of the (word, POS) tuple, count-vectorize, and fit-transform into a matrix
    if text_column == 'Transcript':
        word_list = [[word[0] for word in doc] for doc in df['Transcript']]
    elif text_column == 'Title':
        word_list = [[word for word in doc] for doc in df['Title']]
        
    vec = CountVectorizer(tokenizer=lambda doc:doc, lowercase=False, min_df=2, max_df=0.3)
    matrix = vec.fit_transform(word_list).toarray()
    
    ## A list of words sorted by word frequency in the text
    sum_words = matrix.sum(axis=0)
    words_freq = [(word, sum_words[idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x:x[1], reverse=True)[2:]
    
    wordcloud = WordCloud(width=400, height=330, max_words=150,colormap="Dark2")
    
    wordcloud.generate_from_frequencies(dict(words_freq))

    plt.figure(figsize=(12,10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    
def word2vec_plot(df, text_column, min_count, window):
    '''
    Input: Cleaned dataframe 
    Output: A t-SNE plot showing clusters based on Word2Vec embeddings
    '''
    ## Take the words out of the (word, POS) tuple in each transcript and put them into a list
    if text_column == 'Transcript':
        word_list = [[word[0] for word in doc] for doc in df['Transcript']]
    elif text_column == 'Title':
        word_list = [[word for word in doc] for doc in df['Title']]
    
    ## Initialize a Word2Vec model and set parameters
    model = word2vec.Word2Vec(word_list, min_count=min_count, window=window, ns_exponent = -10)
    
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    ## Perform dimensionality reduction using TSNE
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    ## Create a plot using matplotlib
    plt.figure(figsize=(16, 10)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
        
    plt.tight_layout()
    plt.savefig('../Charts/word2vec_tsne_plot', dpi=600)
    
def word2vec_most_similar(df, text_column, min_count, window, word):
    '''
    Input: Cleaned dataframe 
    Output: 10 most similar words based on Word2Vec embeddings
    '''
    if text_column == 'Transcript':
        word_list = [[text[0] for text in doc] for doc in df['Transcript']]
    elif text_column == 'Title':
        word_list = [[text for text in doc] for doc in df['Title']]
        
    model = word2vec.Word2Vec(word_list, min_count=min_count, window=window, ns_exponent = -10)
    
    return model.most_similar(word)

def document_term_matrix_df(df, text_column, vectorizer):
    '''
    Input: Cleaned dataframe 
    Output: Document-term matrix in a dataframe
    '''
    ## Take the words out of the (word, POS) tuple, vectorize, and fit-transform into a matrix
    if text_column == 'Transcript':
        word_list = [[word[0] for word in doc] for doc in df['Transcript']]
    elif text_column == 'Title':
        word_list = [[word for word in doc] for doc in df['Title']]
        
    vec = vectorizer(tokenizer=lambda doc:doc, lowercase=False)
    matrix = vec.fit_transform(word_list).toarray()
    df_matrix = pd.DataFrame(matrix, columns=vec.get_feature_names())
        
    return df_matrix

def bad_videos(df, df_matrix_bad):
    '''
    Input: Cleaned dataframe and a document-term matrix for the bad videos
    Output: Dataframe of info on the bad videos
    '''
    indices = list(df_matrix_bad.index)
    df_bad = df.loc[indices]
    
    return df_bad

def sentiment_analysis(df, text_column):
    '''
    Input: Cleaned dataframe 
    Output: Dataframe with a new column showing sentiment scores
    '''
    df_copy = df.copy()
    
    def sentiment_score(text):
        ## combine the word tokens into a sentence
        if text_column == 'Transcript':
            sentence = ' '.join([word[0] for word in text])
        elif text_column == 'Title':
            sentence = ' '.join(text)
        
        ## Create a SentimentIntensityAnalyzer object 
        sid_obj = SentimentIntensityAnalyzer()
        
        ## Polarity_scores method of SentimentIntensityAnalyzer object gives a sentiment dictionary, which contains pos, neg, neu, and compound scores. 
        sentiment_dict = sid_obj.polarity_scores(sentence) 
        
        return sentiment_dict['compound']
    
    df_copy['Sentiment Score'] = df_copy[text_column].apply(sentiment_score)
            
    return df_copy

def document_term_matrix(df, vectorizer):
    '''
    Input: Cleaned dataframe (after removing custom stopwords) and type of vectorizer
    Output: Document-term matrix
    '''
    ## Take the words out of the (word, POS) tuple, vectorize, and fit-transform into a matrix
    word_list = [[word[0] for word in doc] for doc in df['Transcript']]
    vec = vectorizer(tokenizer=lambda doc:doc, lowercase=False, min_df=2, max_df=0.3)
    matrix = vec.fit_transform(word_list).toarray()
        
    return matrix, vec.get_feature_names()

def topic_model(matrix, model, num_topics, num_words):
    '''
    Input: Document-term matrix, type of topic model, number of topics, and number of words is each topic
    Output: a list of lists containing topic words
    '''
    ## Creates an instance of an NMF or LDA model
    if model == NMF:
        model = model(num_topics)
    elif model == LatentDirichletAllocation:
        model = model(n_components=num_topics)
        
    ## Fit_transform (matrix factorization for NMF) the doc_word matrix to get doc_topic and topic_word matrices
    doc_topic = model.fit_transform(matrix)
    topic_word = model.components_
    
    ## Retrieves the top words in each topic
    words = document_term_matrix(df_videos_cleaned_v9, CountVectorizer)[1]
    t_model = topic_word.argsort(axis=1)[:, -1:-(num_words+1):-1]
    top_topic_words = [[words[i] for i in topic] for topic in t_model]
        
    return top_topic_words, doc_topic

def corpus_of_adjectives(df):
    '''
    Input: Cleaned dataframe (after removing custom stopwords) 
    Output: Dataframe with only adjectives in the transcript corpus
    '''
    def adjectives(cleaned_text):
        
        preprocessed_text_adj = [(word.lower(), pos) for word, pos in cleaned_text 
                                    if pos=='ADJ'] 
        
        return preprocessed_text_adj
    
    df['Transcript'] = df['Transcript'].apply(adjectives)
            
    return df

def topic_assignment(df):
    '''
    Input: Cleaned dataframe (after removing custom stopwords)
    Output: Dataframe with topic and topic coefficient added
    '''
    ## Takes the highest coefficient for each video (row) in the doc_topic matrix, and puts them into a list 
    doc_topic = topic_model(X_tfidf, NMF, 12, 7)[1]
    topic_coeff = [round(np.max(coeffs),3) for coeffs in doc_topic]
    topic = list(doc_topic.argmax(axis=1))
    
    ## Map topic indices to topic names
    topic_keys = {0:'Value Investing', 1:'Valuation', 2:'Economic Moats', 3:'Passive Investing', 
                  4:'Valuation (Case Studies)', 5:'Technology Stocks', 6:'General', 7:'Value Investing', 8:'Fundamental vs. Technical Analysis', 
                  9:'Electric Vehicle Stocks', 10:'Value Investing', 11:'Dividend Investing'}
    
    topic_name = [topic_keys.get(topic_index,'') for topic_index in topic]
    
    ## Add the Topic and Topic Coefficient columns
    df['Topic'] = topic_name
    df['Topic Coefficient'] = topic_coeff
    
    return df