import json
import pandas as pd
import ast
"""
    Flatten all category lists from the `cleaned_key_categories` column of a DataFrame.

    This function extracts the `cleaned_key_categories` column (which contains lists of categories),
    merges them into a single list, and returns the result. The output list may contain duplicates.

    Args:
        df (pd.DataFrame): A DataFrame containing a column named `cleaned_key_categories`.
                          Each entry in this column should be a list of categories.

    Returns:
        list: A flat list containing all categories from the column, including duplicates.
    """
def flatten_category_lists(df):

    all_categories = df["cleaned_key_categories"].explode().tolist()

    return all_categories






"""
    Generate and save a merged list of all categories as a JSON file.

    This function uses `flatten_category_lists` to collect all categories from the 
    `cleaned_key_categories` column, then saves the combined list (with duplicates preserved) 
    into a JSON file.

    Args:
        df (pd.DataFrame): A DataFrame containing a column named `cleaned_key_categories`.
        output_path (str, optional): The path for saving the JSON file. 
                                     Defaults to "all_categories_list.json".

    Returns:
        None
    """
def export_all_categories():
    all_categories_joined_list=flatten_category_lists(df_jobs)
    path = 'all_categories_lists.json'
    with open(path,"w") as f:
        json.dump(all_categories_joined_list,f)
    return path




"""
    Normalize the categories in a single row using a mapping DataFrame.

    Each row contains multiple categories (from a DataFrame). The mapper DataFrame
    provides the mapping of original category titles to their normalized form, 
    based on cluster assignments.

    Args:
        row (pd.Series): A single row of the main DataFrame, expected to contain 
                         a category list column (e.g., `cleaned_key_categories`).
        mapper (pd.DataFrame): A DataFrame with three required columns:
            - 'title' (str): The raw category title.
            - 'cluster' (int/str): The cluster identifier the title belongs to.
            - 'normalized_title' (str): The canonical category name for the cluster.

    Returns:
        list: A list of normalized categories for the given row.
    """
def normalize_category_list(row,normalized_mapper):
    normalized_category=[]
    for category in row['cleaned_key_categories']:
        mask = normalized_mapper['title'] == category
        general_category=normalized_mapper.loc[mask, 'normalized_title'].iloc[0]
        normalized_category.append(general_category)
    return normalized_category








"""
    Apply category normalization to an entire DataFrame.

    This function applies `normalize_category_list` across all rows of the input DataFrame,
    creating a new column `general_category_list` that contains normalized category names.

    Args:
        df (pd.DataFrame): The input DataFrame, which should contain a column of raw category lists.
        mapper (pd.DataFrame): A mapping DataFrame with columns 'title', 'cluster',
                               and 'normalized_title' for category normalization.

    Returns:
        pd.DataFrame: A modified DataFrame with an added column `general_category_list`.
    """
def normalized_all(df,normalized_mapper):

    df['general_category_list']= df.apply(lambda row :normalize_category_list(row,normalized_mapper),axis=1)
    return df




if __name__=="__main__":
    df_jobs=pd.read_csv('cleaned_job_db.csv')
    df_jobs["cleaned_key_categories"]=df_jobs["cleaned_key_categories"].apply(lambda s: list(ast.literal_eval(s)))

    normalized_mapper=pd.read_csv('normalizing_mapper.csv')
    new_df=normalized_all(df_jobs,normalized_mapper)
    print(new_df['general_category_list'])
    new_df.to_csv('final_normalized_job_db.csv', index=False)
