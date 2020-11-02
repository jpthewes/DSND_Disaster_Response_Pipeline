import sys
import pandas as pd
import sqlite3

from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    ''' Loads data from hardcoded csv's into a pandas dataframe.'''
    # get data and merge to one df
    messages = pd.read_csv('../data/disaster_messages.csv')
    categories = pd.read_csv('../data/disaster_categories.csv')
    df = messages.merge(categories, on='id')

    return df


def clean_data(df):
    ''' Cleans given dataframe.'''
    # Split categories into separate category columns
    categories = pd.DataFrame(df.categories.str.split(';', expand=True))
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
    
    # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # Replace categories column in df with new category columns
    df = df.drop(columns="categories")
    df = pd.concat([df, categories], axis = 1)

    # remove duplicates
    df = df.drop_duplicates()

    return df

def save_data(df, database_filename):
    ''' Saves given dataframe into a sqlite database.'''
    # save to sql database
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages_categories', engine, index=False) 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()