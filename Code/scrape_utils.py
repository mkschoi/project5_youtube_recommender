import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By 
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC

import os, time, datetime
from youtube_transcript_api import YouTubeTranscriptApi


def get_video_link(search_word, num_scrolls):
    '''
    Input: search query, number of scrolls 
    Output: links for the video results
    '''
    
    ## Uses an automated Chrome browser to do a search on YouTube.com
    chromedriver = "/Applications/chromedriver" ## path to the chromedriver executable
    os.environ["webdriver.chrome.driver"] = chromedriver
    
    query = search_word
    youtube_search = "https://www.youtube.com/results?search_query="
    youtube_query = youtube_search + query.replace(' ', '+')
    
    driver = webdriver.Chrome(chromedriver)
    driver.get(youtube_query)
    
    ## Scrolls through the video results page
    for i in range(num_scrolls):
        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
        time.sleep(1.5)
    
    ## Grabs the URLs of the videos and put them into a list 
    user_data = driver.find_elements_by_xpath('//*[@id="video-title"]')
    links = [link for link in [i.get_attribute('href') for i in user_data] if link]
    
    driver.quit()
    
    return links   

def video_page_scraper(list_links):
    '''
    Input: a list of links 
    Output: video data scraped into a dataframe, each row corresponding to a video
    '''
    chromedriver = "/Applications/chromedriver" ## path to the chromedriver executable
    os.environ["webdriver.chrome.driver"] = chromedriver
    
    driver = webdriver.Chrome(chromedriver)
    wait = WebDriverWait(driver, 10)
    
    ## Create a dataframe containing video data
    df = pd.DataFrame(columns = ['Video ID', 'Title', 'Upload Date', 'Duration (Minutes)', 'Views', 'Number of Likes', 'Description', 'Transcript'])
    
    ## Scrape relevant video data
    for link in list_links:
        driver.get(link)
        time.sleep(2)
        
        ## Video ID
        v_id = wait.until(lambda browser: browser.find_elements_by_xpath("//ytd-watch-flexy[@class='style-scope ytd-page-manager hide-skeleton']")[0].get_attribute('video-id'))
        
        ## Video Title
        v_title = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,"h1.title yt-formatted-string"))).text
        
        ## Date
        v_date = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,"div#date yt-formatted-string"))).text
        
        ## Duration
        v_duration = driver.find_elements_by_xpath("//span[@class='ytp-time-duration']")[0].text
        
        ## Views
        v_views =  wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,"div#count span"))).text
        
        ## Number of likes
        v_likes =  wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,"div#top-level-buttons yt-formatted-string"))).text
        
        ## Video Description
        v_description =  wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,"div#description yt-formatted-string"))).text
        
        ## Transcripts
        try:
            v_transcript = YouTubeTranscriptApi.get_transcript(v_id)
        except:
            v_transcript = np.NaN
        
        df.loc[len(df)] = [v_id, v_title, v_date, v_duration, v_views, v_likes, v_description, v_transcript]

    driver.quit()
    
    return df
