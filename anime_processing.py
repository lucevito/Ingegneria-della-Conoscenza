import pandas as pd
import numpy as np
import re


import warnings
warnings.filterwarnings('ignore')

# function to clean the text from impurities
def text_cleaning(text):
    text = re.sub(r'&quot;', '', text)
    text = re.sub(r'.hack//', '', text)
    text = re.sub(r'&#039;', '', text)
    text = re.sub(r'A&#039;s', '', text)
    text = re.sub(r'I&#039;', 'I\'', text)
    text = re.sub(r'&amp;', 'and', text)
    return text

# function to extract genres
def genre_extraction(anime_data):
    return anime_data['genre'].str.split(',').astype(str)

# function to extract an anime index
def anime_index_extraction(anime_data):
    return pd.Series(anime_data.index, index=anime_data['name']).drop_duplicates()

# function to better transform and organize data
def anime_data_processing(anime_data):
    anime_data['name'] = anime_data['name'].apply(text_cleaning)
    anime_data["rating"].replace({-1: np.nan}, inplace=True)
    anime_data = anime_data.dropna(axis = 0, how ='any') 
    anime_data['genre'] = anime_data['genre'].fillna('')
    anime_data['genre'] = anime_data['genre'].astype('str')
    anime_data['genre'] = anime_data['genre'].str.split(', ')
    genre_columns_temp=anime_data.genre.apply(pd.Series).stack().str.get_dummies().sum(level=0)
    anime_data = anime_data.drop(['genre'], axis=1)
    anime_data = pd.concat([anime_data,genre_columns_temp],axis=1)
    del genre_columns_temp
    return anime_data

# function to merge the data between the list of anime and their votes
def anime_pivot_processing(anime_data, rating_data):
    anime_dataclone = anime_data[['anime_id', 'name']].copy()
    anime_fulldata=pd.merge(anime_dataclone,rating_data,on='anime_id')
    del anime_dataclone
    del rating_data
    anime_fulldata["rating"].replace({-1: np.nan}, inplace=True)
    anime_fulldata = anime_fulldata.dropna(axis = 0, how ='any') 
    counts = anime_fulldata['user_id'].value_counts()
    anime_fulldata = anime_fulldata[anime_fulldata['user_id'].isin(counts[counts >= 500].index)]
    anime_pivot=anime_fulldata.pivot_table(index='name',columns='user_id',values='rating').fillna(0)
    del anime_fulldata
    return anime_pivot
