import sys
from sqlalchemy import create_engine
import pandas as pd

# load data into pandas dataframe
def load_data(messages_filepath, categories_filepath):
        """
    Loads messages and categories from CSV files and returns them as pandas dataframes
    
    Input:
        messages_filepath: CSV with messages
        categories_filepath: CSV of categories
    Returns:
        df: Merged Csvs without preprocessing
    """
        
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge dataframes
    df = pd.merge(messages, categories, how='left', left_on='id', right_on='id').drop(labels=["id"], axis=1)

    return df


def clean_data(df):
     """
    Clear received dataframe from load_data
    
    Input:
        df: "load_data" dataframe
    Returns:
        df: Cleaned data frame.
    """
    # split categories by delimiter ;
    categories = df.categories.str.split(";", expand=True)

    # select the first row of the categories dataframe
    row = categories.loc[0]
    
    catNames = [i.split("-")[0] for i in row]
        
    # rename the columns
    categories.columns = catNames

    # Clean each value to leave only a numeric value
    for column in catNames:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split("-")[1])
 
    # Change columns from strings to numerical values
    categories = categories.astype(int)

    # drop the old categories column
    df = df.drop(labels=["categories"], axis=1)

    # Change columns from strings to numerical values
    categories = categories.astype(int)

    # join df with the new categories df
    df = df.join(categories)

    # drop duplicates
    df = df.drop_duplicates()
    
    return df
    

def save_data(df, database_filename):
     """
    Saves dataframe into SQLite database with a desired filename.
    
    Input:
        df: Clean data frame from `clean_data`
        database_filename: database filename
    Returns:
        None
    """
        
    engine = create_engine('sqlite:///'+database_filename)
    print(df)
    df.to_sql('msgs', engine, index=False)  


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