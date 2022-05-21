import sys
import pandas as pd
import numpy as np
import sqlite3

def load_data(messages_filepath, categories_filepath):
    """Load data of 2 csv files

    Args:
    messages_filepath: file path of first file
    categories_filepath: file path of second file

    Returns:
    1 merged DataFrame from 2 files
    """
    messages=pd.read_csv(messages_filepath)
    categories=pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='left',on='id')
    return df


def clean_data(df):
    """Clean df dataset

    Args:  
    df: the DataFrame merged from messages and categories
    Returns:
    Cleaned DataFrame
    """
    # Split categories into separate category columns.
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";",expand=True)
    # Use the first row of categories dataframe to create column names
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x: x[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    # Convert category values to numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    # Replace categories column in df with new category columns.
    df.drop(columns='categories',inplace=True)
    df = pd.concat([df,categories],axis=1)
    # Remove duplicates
    df.drop_duplicates(keep='first',inplace=True)
    return df


def save_data(df, database_filename):
    """Save the clean dataset into an sqlite database.
    Args:  
    df: the cleaned DataFrame
    database_filename: the filepath of the database to save the cleaned data
    """  
    from sqlalchemy import create_engine
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('df_cleaned', engine, index=False,if_exists='replace')


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