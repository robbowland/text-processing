import nltk
import numpy as np
import pandas as pd
from typing import Set

# None of these functions ended up having a positive effect
# so they're not used in the final model
class FeatureExtractor:
    """
    Utility class containing feature extraction functionality.
    """
    
    @staticmethod
    def get_parts_of_speech(dataframe: pd.DataFrame, 
                            col_name: str, pos_tags: Set[str], 
                            as_string: bool = False) -> pd.Series:
        """
        Extract specified parts of speech from a dataframe.
    
        Parameters
        ----------
        dataframe : pd.DataFrame
            The data frame to containing the column from which the features
            will be extracted.
        col_name : str
            The name of the column containinf the text that features will be 
            extracted from.
            Column should contain pre-tokenised text string that has been split
            into a list.
        pos_tags : Set[str]
           A set of strings corresponding the the nltk pos tags of the 
           parts of speech to be extracted.
           Tag list can be found here:
           https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
        as_string: bool, optional
            Whether or not to join the extracted features lists into strings.
            False by default.
        
        Returns
        -------
        extracted_features : pd.Series
            A series containg the extracted features from each row of 
            the original dataframe, as either a list or string.

        """
        extracted_features = pd.Series(np.vectorize(
            lambda text: [word for (word, pos) in nltk.pos_tag(text) 
                          if  pos in pos_tags], 
            otypes=[list]) (dataframe[col_name]))
        
        if as_string:
            return extracted_features.str.join(' ')
        
        return extracted_features
    
    @staticmethod
    def make_ngrams(dataframe: pd.DataFrame, 
                    col_name: str, min_n: int, max_n: int) -> pd.Series:
        """
        Make ngrams from of within specified length range from contents of
        dataframe column. ngrams are given in string form.
        e.g. For the tokenised phrase:
            
            ['this', 'is', 'good']
            
            and min_n = 1, max_n = 2 the output will be
            
            ['this', 'is', 'good', 'this is', 'is good']
            
            where elements 0-2 are 1grams and elements 3-4 are 2grams.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The data frame to containing the column from which the features
            will be extracted.
        col_name : str
            The name of the column containinf the text that features will be 
            extracted from.
            Column should contain pre-tokenised text string that has been split
            into a list.
        min_n : int
            The shortest 'n' for ngrams that will be made.
        max_n : int
            The longest 'n' for ngrams that will be made. max_n >= min_n.

        Returns
        -------
        ngrams : pd.Series
            List containing ngrams in specified range for each
            row content of the dataframe column, as a pandas series.

        """
        ngrams = pd.Series(np.vectorize(
                    lambda phrase: 
                        [' '.join(item) for item in nltk.everygrams(phrase, 
                                                                    min_len=min_n, 
                                                                    max_len=max_n)],
                        otypes=[list])(dataframe[col_name]))
        
        return ngrams


