import requests
import pandas as pd
import numpy as np
from time import sleep
from random import randint
import json
from bs4 import BeautifulSoup
import re
from IPython.display import clear_output
from newspaper import Article

class Subreddit():
    # Set static url
    url = 'https://api.pushshift.io/reddit/search/submission'
    
    def __init__(self, name):
        """
        Initialize a new instance of `name`.
        Private method.
        """
        # Set instance name
        self.subreddit = name
        
    def __req_json(self, size):
        """
        Sends a request through the pushshift api to request for the latest `size, default=500` posts in the subreddit.
        Saves the json.file in the subreddit's data folder.
        Private method.
        """
        # Set start count
        count = 0
        
        # Set start parameters
        self.__params = {
            'subreddit': self.subreddit,
            'size':100}     
        
        # Create new list for all data
        self.all_data = []
        
        # Loop while count is less than size as we can only scrape 100 posts at a time.
        while count <= size:
            
            # Request for posts from the subreddit
            self.res = requests.get(Subreddit.url, self.__params)

            # Checks if status code is 200 (OK) before continuing
            if self.res.status_code == 200:
                self.data = self.res.json()['data']
                # Iterate through each post in the data and add all non-removed items to the all_data list
                for item in self.data:
                    if re.search(r'removed_by_category', str(item)):
                        continue
                    else:
                        self.all_data.append(item)

                # Update the parameters to find posts before the earliest post in this loop
                self.__params = {
                    'subreddit': self.subreddit,
                    'size':100,
                    'before': self.data[-1]['created_utc']-1} 

                # Update count
                count = len(self.all_data)

                # Sleep
                sleep(randint(5,10))

                # Clear cell output
                clear_output(wait=True)

                # Print current progress
                print(f'Current progress: {count}/{size}.')
                
            else:
                sleep(randint(15,30))
                pass

        # Save the all_data list as a json file backup
        with open(f'../data/{self.subreddit}/{self.subreddit}.json', 'w') as json_file:
            json.dump(self.all_data, json_file)        
        
    def __dataframe(self):
        """
        Converts the json data to a Pandas DataFrame.
        Saves the DataFrame in the subreddit's data folder.
        Private method.
        """        
        # Convert the json data to a DataFrame
        self.df = pd.DataFrame(self.all_data)
        
        # Save the DataFrame as a csv backup
        self.df.to_csv(path_or_buf=f'../data/{self.subreddit}/{self.subreddit}.csv', index = False)
        
    def __scope_dataframe(self):
        """
        Filters the DataFrame to only relevant data..
        Saves the filtered DataFrame in the subreddit's data folder.
        Private method.
        """ 
        # Filters DataFrame only to relevant columns and saves it as a backup csv
        self.df_scope = self.df[['subreddit','title','url']]
        self.df_scope.to_csv(path_or_buf=f'../data/{self.subreddit}/{self.subreddit}_scope.csv', index = False)
        
    def scrape(self, num=500):
        """
        Public method for user to scrape the latest `num` posts from reddit and save them as a filtered DataFrame containing only relevant data.
        `num` only accepts `int` in increments of 100 and the mtehod will scrape at least `num`. Default `num=500`.
        """
        # Call the methods in order
        self.__req_json(num)
        self.__dataframe()
        self.__scope_dataframe()

    def df_from_json(self):
        """
        Public method for user to create the filtered and unfiltered DataFrames from the saved .json file.
        """
        # Loads the json as an unfiltered DataFrame
        self.df = pd.read_json(f'../data/{self.subreddit}/{self.subreddit}.json')
        
        # Creates the filtered DataFrame from the unfiltered DataFrame
        self.df_scope = self.df[['subreddit','title','url']]
        
        
    def df_from_csv(self, scope='full'):
        """
        Public method for user to create the DataFrame from the saved .csv file.
        `scope` accepts 'unfiltered', 'filtered', and 'full'.
        If the whole DataFrame is required, use `scope='unfiltered'`, 'scope='filtered' for the filtered DataFrame, and default `scope='full'` for the filtered DataFrame which includes article text.
        """
        # Set whether DataFrame should be filtered
        self.scope = scope
        
        # Loads filtered or unfiltered .csv to create DataFrame according to the scope specified
        if self.scope == 'filtered':
            self.df_scope = pd.read_csv(f'../data/{self.subreddit}/{self.subreddit}_scope.csv')
        elif self.scope == 'unfiltered':
            self.df = pd.read_csv(f'../data/{self.subreddit}/{self.subreddit}.csv')
        else:
            self.full_df = pd.read_csv(f'../data/{self.subreddit}/{self.subreddit}_full.csv')
        
    def news_text_pull(self):
        """
        Public method for user to pull the news text from its respective link to save in a DataFrame.
        This uses the Newspaper3 python library.
        `https://github.com/codelucas/newspaper`
        """
        # Create empty list of news texts
        self.article_texts = []
        
        # Start counter
        count = 0
        
        # Create list of urls to iterate from self.df_scope
        self.urls = list(self.df_scope['url'])
        # Iterate through each link to scrape the article text using Newspaper3
        for link in self.urls:
            # Puts article text in article_texts if no errors
            try:
                url = link
                article = Article(url)
                article.download()
                article.parse()
                self.article_texts.append(article.text)
                
            # Appends np.nan if there is an error
            except:
                self.article_texts.append(np.nan)
                
            # Increase counter
            count += 1
               
            # Clear cell output
            clear_output(wait=True) 
                
            # Print current progress
            print(f'Current progress: {count}/{len(self.urls)}.')
        
        # Create a DataFrame from the list and merge it into a full DataFrame.
        self.full_df = self.df_scope.copy()
        self.full_df['article_text'] = self.article_texts
        self.full_df.to_csv(path_or_buf=f'../data/{self.subreddit}/{self.subreddit}_full.csv', index = False)

    def remove_duplicates(self, column='title'):
        """
        Public method to remove duplicates from the DataFrames.
        This will remove duplicates while keeping the first entry and resetting the index.
        Input: `column` to search for duplicated values from. Default `column='title`.
        Output: DataFrames with no duplicates
        """
        # Try to remove duplicates from the available DataFrames. Passes if DataFrame is not set (e.g. when loading from .csv)
        try:
            self.full_df.drop_duplicates(subset=column, keep='first', inplace=True, ignore_index=True)
        except:
            pass
        try:
            self.df_scope.drop_duplicates(subset=column, keep='first', inplace=True, ignore_index=True)            
        except:
            pass
        try:
            self.df.drop_duplicates(subset=column, keep='first', inplace=True, ignore_index=True)            
        except:
            pass
        
    def sort_length(self, column='article_text'):
        """
        Public method to sort full_df by `column` length by creating a new column `length` containing the number of characters in the column. 
        This is to check that we do not have articles with erroneous non-null text.
        Additionally, a column for number of words `num_words` will be created.
        Accepts 'title' or 'article_text' as inputs. Default = 'article_text'
        Only to be used with full_df.
        """
        self.length_column_name = column+'_length'
        self.num_column_name = column+'_num_words'
        try:
            # Count the number of non-whitespace words to create a 'num_words' column
            self.full_df[self.num_column_name] = self.full_df[column].apply(lambda x: len(re.findall("\\S+", x)))
            # Count the number of characters in the article
            self.full_df[self.length_column_name] = self.full_df[column].str.len()
            self.full_df.sort_values(self.length_column_name, ascending=True, inplace=True)
            
        # Print an error if full_df is not loaded
        except:
            print(f'Error: {self.full_df} not loaded.')
            
    def doge(self):
        """
        Bonus
        """
        print("""                    ▄              ▄
                  ▌▒█           ▄▀▒▌
                  ▌▒▒█        ▄▀▒▒▒▐
                 ▐▄▀▒▒▀▀▀▀▄▄▄▀▒▒▒▒▒▐
               ▄▄▀▒░▒▒▒▒▒▒▒▒▒█▒▒▄█▒▐
             ▄▀▒▒▒░░░▒▒▒░░░▒▒▒▀██▀▒▌
            ▐▒▒▒▄▄▒▒▒▒░░░▒▒▒▒▒▒▒▀▄▒▒▌
            ▌░░▌█▀▒▒▒▒▒▄▀█▄▒▒▒▒▒▒▒█▒▐
           ▐░░░▒▒▒▒▒▒▒▒▌██▀▒▒░░░▒▒▒▀▄▌
           ▌░▒▄██▄▒▒▒▒▒▒▒▒▒░░░░░░▒▒▒▒▌
          ▌▒▀▐▄█▄█▌▄░▀▒▒░░░░░░░░░░▒▒▒▐
          ▐▒▒▐▀▐▀▒░▄▄▒▄▒▒▒▒▒▒░▒░▒░▒▒▒▒▌
          ▐▒▒▒▀▀▄▄▒▒▒▄▒▒▒▒▒▒▒▒░▒░▒░▒▒▐
           ▌▒▒▒▒▒▒▀▀▀▒▒▒▒▒▒░▒░▒░▒░▒▒▒▌
           ▐▒▒▒▒▒▒▒▒▒▒▒▒▒▒░▒░▒░▒▒▄▒▒▐
            ▀▄▒▒▒▒▒▒▒▒▒▒▒░▒░▒░▒▄▒▒▒▒▌
              ▀▄▒▒▒▒▒▒▒▒▒▒▄▄▄▀▒▒▒▒▄▀
                ▀▄▄▄▄▄▄▀▀▀▒▒▒▒▒▄▄▀
                   ▒▒▒▒▒▒▒▒▒▒▀▀ """)