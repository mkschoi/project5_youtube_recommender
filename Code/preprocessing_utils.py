import pandas as pd
import numpy as np
import glob
from ast import literal_eval
from datetime import datetime

import string
import re
from collections import defaultdict
import truecase

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag

import spacy
nlp = spacy.load('en_core_web_sm')


def merge_video_dataframes(df_videos_raw_str):
    '''
    Input: Name of dataframes containing video data in a string format
    Output: Merged dataframe
    '''
    df_merged = eval(df_videos_raw_str + '_1')
    for i in range(2,16):
        df_merged = pd.concat([df_merged, eval(df_videos_raw_str + '_' + str(i))])
    return df_merged

def read_and_merge(path):
    '''
    Input: Path of directory where all the csv files are located
    Output: Merged dataframe of all csv files in the directory
    '''
    ## Finds all the csv files in a directory and put them into a list
    df_list = []
    for file in glob.glob(path):
        df = pd.read_csv(str(file))
        df_list.append(df)
        
    ## Merges all dataframes in the csv files
    df_merged = df_list[0]
    for df in df_list[1:]:
        df_merged = pd.concat([df_merged, df])
        
    return df_merged

def df_preprocessing_pipeline(df):
    '''
    Input: Datafrome to be preprocessed
    Output: Preprocessed dataframe
    '''
    ## Drop rows with NaNs in the 'Transcript' column
    df = df[df['Transcript'].notna()]
    
    ## Drop duplicate videos
    df = df.drop_duplicates(subset=['Video_ID'])
    
    ## Drop videos with duration < 1 minute
    df = df[(df['Duration'].str.len() >=5) | (df['Duration'].str[-4]!='0')]
    
    return df

def df_preprocessing_pipeline2(df):
    '''
    Input: Datafrome to be preprocessed
    Output: Preprocessed dataframe
    ''' 
    ## Drop videos without clear upload date
    rows_to_drop = df[(df['Upload Date'].str.contains('hours ago')==True)].index
    df = df.drop(rows_to_drop).reset_index(drop=True)
    
    ## Change upload date string to datetime format
    df['Upload Date'] = df['Upload Date'].apply(lambda x: x.replace(',',''))
    df['Upload Date'] = df['Upload Date'].apply(lambda x: x.replace('Premiered ',''))
    df['Upload Date'] = df['Upload Date'].apply(lambda x: x.replace('Streamed live on ',''))
    
    df['Upload Date'] = df['Upload Date'].apply(lambda x: datetime.strptime(x, '%b %d %Y').date())
    
    ## Change duration string to minutes format
    
    def convert_to_minutes(duration):
        if len(duration) >= 7:
            time = (int(duration.split(':')[0])*60) + (int(duration.split(':')[1])) + (int(duration.split(':')[2])/60)
        else:
            time = (int(duration.split(':')[0])) + (int(duration.split(':')[1])/60)
            
        return time
    
    df['Duration'] = df['Duration'].apply(convert_to_minutes)

    return df

def grab_transcript_text(df):
    '''
    Input: Dataframe with raw values in the Transcript column
    Output: Dataframe with only text in the Transcript column, in a string format
    '''
    ## Create a function for grabbing the text values and putting into a single string
    def transcript_text_string(transcript_dict_list):
        text_list = []
        for dict_text in literal_eval(transcript_dict_list):
            text = dict_text['text']
            text_list.append(text)
            
        text_string = ' '.join(text_list)
            
        return text_string
    
    df['Transcript'] = df['Transcript'].apply(transcript_text_string)
            
    return df   

def text_preprocessing_pipeline_1(df):
    '''
    Input: Dataframe with the raw transcript text in a string format
    Output: Dataframe with line breaks, punctuations, and numbers removed from the transcript text
    '''
    ## Create a function for applying the text preprocessing pipeline
    def initial_preprocessing(raw_text):
        ## Remove line breaks
        preprocessed_text_1 = raw_text.replace('\n', ' ')
        
        ## Remove punctuations, except for apostrophes
        preprocessed_text_2 = preprocessed_text_1.translate(str.maketrans('', '', string.punctuation.replace("'", "")))
        
        ## Remove numbers
        preprocessed_text_3 = re.sub('\w*\d\w*', '', preprocessed_text_2)
        
        return preprocessed_text_3
    
    df['Transcript'] = df['Transcript'].apply(initial_preprocessing)
            
    return df   

def text_preprocessing_pipeline_2(df):
    '''
    Input: Dataframe after applying text_processing_pipeline_1
    Output: Dataframe with the transcript further preprocessed - tokenization, stopwords removal, lemmatization
    '''
    ## Create a function for applying the text preprocessing pipeline
    def second_preprocessing(preprocessed_text_3):
        ## Remove stopwords
        stopwords = spacy.lang.en.stop_words.STOP_WORDS
        
        preprocessed_text_4 = nlp(preprocessed_text_3)
        preprocessed_text_5 = [word.text for word in preprocessed_text_4 
                                    if str(word).lower() not in stopwords and word.text!= ' ']
        
        ## Lemmatization
        lemmatizer = WordNetLemmatizer()
        tag_map = defaultdict(lambda : wordnet.NOUN)
        tag_map['V'] = wordnet.VERB 
        tag_map['J'] = wordnet.ADJ
        tag_map['R'] = wordnet.ADV

        preprocessed_text_6 = [lemmatizer.lemmatize(word.lower(), tag_map[tag[0]]) 
                                    for word, tag in pos_tag(preprocessed_text_5)]
        
        return preprocessed_text_6
    
    df['Transcript'] = df['Transcript'].apply(second_preprocessing)
            
    return df   

def text_preprocessing_pipeline_3(df):
    '''
    Input: Dataframe after applying text_preprocessing_pipeline_2
    Output: Dataframe with the transcript further preprocessed - truecasing, part of speech tagging
    '''
    ## Create a function for applying the text preprocessing pipeline
    def third_preprocessing(preprocessed_text_6):
        ## Remove random letters
        preprocessed_text_7 = ' '.join([word for word in preprocessed_text_6 if len(word)>1])
        
        ## Truecasing
        preprocessed_text_8 = truecase.get_true_case(preprocessed_text_7)
        
        ## Part of speech tagging
        preprocessed_text_9 = nlp(preprocessed_text_8)
        preprocessed_text_10 = [(word.text, word.pos_) for word in preprocessed_text_9]
        
        return preprocessed_text_10
    
    df['Transcript'] = df['Transcript'].apply(third_preprocessing)
            
    return df  

def remove_custom_stopwords(df):
    '''
    Input: Cleaned dataframe
    Output: Dataframe with custom stopwords removed
    '''
    def final_preprocessing(cleaned_text):  
        nlp.Defaults.stop_words |= {'uh','yeah','man','um','oh','guy','maybe','bye','hey', 'sort'}
        stopwords = nlp.Defaults.stop_words
        
        preprocessed_text_12 = [(word.lower(), pos) for word, pos in cleaned_text 
                                    if word.lower() not in stopwords] 
        
        return preprocessed_text_12
    
    df['Transcript'] = df['Transcript'].apply(final_preprocessing)
            
    return df  

def title_preprocessing(df):
    '''
    Input: Cleaned dataframe 
    Output: Tokenized titles in the dataframe 
    '''
    df_copy = df.copy()
    
    def preprocessing_pipeline(raw_text):
        ## Remove punctuations, except for apostrophes
        preprocessed_text_1 = raw_text.translate(str.maketrans('', '', string.punctuation.replace("'", "")))
        
        ## Remove numbers
        preprocessed_text_2 = re.sub('\w*\d\w*', '', preprocessed_text_1)
        
        ## Remove stopwords
        stopwords = spacy.lang.en.stop_words.STOP_WORDS
        
        preprocessed_text_3 = nlp(preprocessed_text_2)
        preprocessed_text_4 = [word.text for word in preprocessed_text_3 
                                    if str(word).lower() not in stopwords and word.text!= ' ']
        
        ## Lemmatization
        lemmatizer = WordNetLemmatizer()
        tag_map = defaultdict(lambda : wordnet.NOUN)
        tag_map['V'] = wordnet.VERB 
        tag_map['J'] = wordnet.ADJ
        tag_map['R'] = wordnet.ADV

        preprocessed_text_5 = [lemmatizer.lemmatize(word.lower(), tag_map[tag[0]]) 
                                    for word, tag in pos_tag(preprocessed_text_4)]
        
        preprocessed_text_6 = [word for word in preprocessed_text_5 if len(word)>1]
        
        return preprocessed_text_6
    
    df_copy['Title'] = df_copy['Title'].apply(preprocessing_pipeline)
            
    return df_copy   

def remove_pos(df):
    '''
    Input: Dataframe with POS tagged in the transcripts
    Output: Dataframe without POS tagging
    '''
    ## Create a function for grabbing words from word, pos tuples
    def grab_word(transcript_text):
        ## Remove line breaks
        words = [word for word, pos in transcript_text]
        
        return words
    
    df['Transcript'] = df['Transcript'].apply(grab_word)
            
    return df  