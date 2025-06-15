import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union

class NaiveBayes:
    """
    Class continaing functionality for Multinomial Naive Bayes training and 
    classification for the primary purpose of text sentiment analysis.
    """
    
    def __init__(self) -> None:
        """
        Initialise a new instance of the NaiveBayes class.

        Returns
        -------
        None.
        
        """
        self.class_counts = None
        self.vocabulary = None
        self.class_word_counts = {}
        self.priors = None
        self.likelihoods = None 
    
    # =============================| TRAINING |=============================== 
    def train(self, dataset: pd.DataFrame, 
              text_col: str, class_col: str, smooth: bool = True) -> None:
        """
        Train the Naive Bayes classifer using training data. Compute priors
        and likelihoods that make up the classification model.

        Parameters
        ----------
        dataset : pd.DataFrame
            The training dataset.
        text_col : str
            The name of the column of the dataset that contains the text to be
            used for training. Column should contain a list of strings, 
            where each stringis a feature.
        class_col : str
            The name of the column of the dataset that contains the class 
            labels for the text that will be used for training.
        smooth : bool, optional
            Whether or not Laplace Smoothing will be applied during the 
            training stage. The default is True.

        Returns
        -------
        None
            Populates class attributes with relevant data needed for 
            classification.
            
        """
        # Compute dataset meta-data
        self.class_counts = self._get_class_counts(dataset, class_col)
        self.vocabulary = self._get_dataset_vocabulary(dataset, text_col)
        # Train model
        self.priors = self._calculate_priors(dataset, class_col)
        self.likelihoods = self._calculate_likelihoods(dataset, text_col, 
                                                       class_col, smooth)
    
    def _get_class_counts(self, dataset: pd.DataFrame, 
                          class_col: str) -> Dict:
        """
        Extract the unique classes from the dataset a long with a count
        of their number of occurences.
        Intended for internal use only.
        
        Parameters
        ----------
        dataset : pd.DataFrame
            The dataset to extract the information from.
        class_col : str
            The name of the column in the dataset where the classes are
            stored.

        Returns
        -------
        Dict
           A Dictionary with (Class, Class_Count) Key, Value pairs.
           
        """
        return dataset[class_col].value_counts().to_dict()
    
    def _get_dataset_vocabulary(self, dataset: pd.DataFrame, 
                                text_col: str) -> int:
        """
        Get the vocabulary of the dataset - the number of distinct features.
        This is equivalent to |V| in the Laplace Smoothing technique.
        Intended for internal use only.

        Parameters
        ----------
        dataset : pd.DataFrame
            The dataset to extract the information from.
        text_col : str
            The name of the column in the dataset where the text is
            stored.

        Returns
        -------
        int
            An Integer representing the number of distinct features present in 
            the dataset.
            
        """
        return dataset[text_col].explode().nunique()
    
    def _calculate_priors(self, dataset: pd.DataFrame, 
                          class_col: str) -> Dict:
        """
        Calculate the prior probabilities for each class present in the 
        dataset.
        Intended for internal use only.
        
        Parameters
        ----------
        dataset : pd.DataFrame
            The dataset to extract the information from.
        class_col : str
            The name of the column in the dataset where the classes are
            stored.

        Returns
        -------
        dict
           A Dictionary with (Class, Class_Prior) Key, Value pairs.
           
        """
        if self.class_counts is None:
            self.class_counts = self._get_class_counts(dataset, class_col)
        return { 
            class_name: val/sum(self.class_counts.values()) 
            for (class_name, val) in self.class_counts.items() 
            }

    def _calculate_likelihoods(self, dataset: pd.DataFrame, 
                               text_col: str, class_col: str, 
                               smooth: bool = True) -> Dict:
        """
        Calculate the likelihood values for each feature in each class.
        Intended for internal use only.

        Parameters
        ----------
        dataset : pd.DataFrame
            The dataset to extract the information from.
        text_col : str
             The name of the column in the dataset where the text is
            stored.
        class_col : str
            The name of the column in the dataset where the classes are
            stored.
        smooth: bool, optional
            Whether or not Laplace Smoothing should be applied when 
            calculating the likelihoods. The default is True.
        
        Returns
        -------
        dict
            A Dictionary containing the likelihoods for each feature for each
            class. Dictionary has the following form:
                
                { 
                Class_Name: 
                    {  Feature : Feature_Likelihood  }
                }
                    
        """
        return {
            class_name: self._calculate_class_likelihoods(dataset, text_col, 
                                                     class_col, class_name,
                                                     smooth) 
            for class_name in self.class_counts.keys()
            }
    
    def _calculate_class_likelihoods(self, dataset: pd.DataFrame, 
                                     text_col: str, class_col: str, 
                                     class_name: Any, 
                                     smooth: bool = True) -> Dict:
        """
        Calculate the likelihoods for each word in a given class.
        Additionally, populate class fields for the total number of 
        non-distinct features for each class.
        Intended for internal use only.

        Parameters
        ----------
        dataset : pd.DataFrame
            The dataset to extract the information from.
        text_col : str
             The name of the column in the dataset where the text is
            stored.
        class_col : str
            The name of the column in the dataset where the classes are
            stored.
        class_name : Any
            The name of class for which the liklihoods will be calculated
        smooth : bool, optional
            Whether or not Laplace Smoothing should be used.
            The default is True.

        Returns
        -------
        dict
            A Dictionary containing the likelihoods for each feature in a 
            given class. Dictionary is of the form:
                
                { Feature : Feature_Likelihood }

        """
        # Get all phrases labelled with given class
        class_text = dataset.loc[dataset[class_col] == class_name][text_col]
        # Calculate total number of words in class (not unique)
        self.class_word_counts[class_name] = class_text.explode().count()
        # Calculate the counts of each unique word in the class
        unique_word_counts = class_text.explode().value_counts().to_dict()
        # Check that vocabulary has been calculated previously
        if self.vocabulary is None:
            self.vocabulary = self._get_dataset_vocabulary(dataset, text_col)
        
        # With Laplace Smoothing Applied
        if smooth:            
            return {
                key: (val + 1)/(self.class_word_counts[class_name] + self.vocabulary)
                for (key, val) in unique_word_counts.items()
                }
        
        # Without Laplace Smoothing Applied
        return {
            key: val/self.class_word_counts[class_name]
            for (key, val) in unique_word_counts.items()
            }
    # ===============================| END |==================================    
    
    # =============================| TESTING |================================ 
    def classify(self, dataset: pd.DataFrame, 
              text_col: str,
              output_col: str,
              smoothed: bool = True,
              return_all: bool = False) -> None:
        """
        Classify all text entries in the dataset using the trained NaiveBayes
        model. The 'train' function must be called prior to this function.

        Parameters
        ----------
        dataset : pd.DataFrame
            The dataset to perform classification on.
        text_col : str
            The name of the column containing the text to be classified.
            Column should contain a list of strings, where each string
            is a feature.
        smoothed : bool, optional
            Whether or not Laplace Smoothing was used in the training stage.
            The default is True.
        return_all : bool, optional
            Whether or not to return data on every class with their 
            probability. Will be sorted in reverse order, with the highest
            probability class label first. Default is False.
            
        Returns
        -------
        None.

        """
        dataset[output_col] = np.vectorize(
            self._classify_text, otypes=[list])(
                dataset[text_col], smoothed, return_all)
    
    
    def _classify_text(self, text: List[str], smoothed: bool = True,
                       return_all: bool = False) -> Union[Any, List]:
        """
        Classify text (in the form of a list of features) using Naive Bayes 
        model from training stage.
        Intended for internal use only.

        Parameters
        ----------
        text : List[str]
            The text to be classified, in the form of a list of strings, where 
            each element is a feature.
        smoothed : bool, optional
            Whether or not Laplace Smoothing was used in the training stage.
            The default is True.
        return_all : bool, optional
            Whether or not to return data on every class with their 
            probability. Will be sorted in reverse order, with the highest
            probability class label first. Default is False.

        Returns
        -------
        Union[Any, List]
            If return_all is True, will return a List containing Lists for 
            each class label and their determined probability. Otherwise
            will return the class label of the highest probability 
            classification.

        """
        class_probs = [[class_name, 0] for class_name in self.class_counts.keys()] 
        for class_pair in class_probs:
            class_name = class_pair[0]
            # Init probability with prior
            class_prob = self.priors[class_name]
            class_likelihoods = self.likelihoods[class_name] 
            for word in text:
                if word in class_likelihoods.keys():
                    class_prob *= class_likelihoods[word]
                # Account for missing values with smoothing 
                elif word not in class_likelihoods.keys() and smoothed:
                    class_prob *= 1/(self.class_word_counts[class_name] 
                                     + self.vocabulary)
                # Account for missing values without smoothing
                else: 
                    class_prob = 0
                    break  # since a single missing value will result in 0
                    
            class_pair[1] = class_prob
        
        if return_all:
            return sorted(class_probs, key=lambda x: x[1], reverse=True)
        
        return max(class_probs, key=lambda x: x[1])[0]
    # ===============================| END |==================================  



# TEST MANTLE
if __name__ == '__main__':
    # Verify implementation using data from lectures.
    TRAIN_DATA = pd.DataFrame(
        {
        'text_col':
            [
                ['great', 'excellent', 'renowned'],
                ['fantastic', 'good', 'amazing', '!!!'],
                ['lovely', 'amazing', 'bad'],
                ['bad', 'great', 'poor', 'unimaginative'],
                ['original'],
                ['great', 'bad'],
                ['bad'],
            ],
        'class_col':
            ['positive', 'positive', 'positive', 
             'negative', 'negative', 'negative', 'negative']
        })
    TEST_DATA = pd.DataFrame(
        {
        'text_col': 
            [
                ['fantastic', 'great', 'lovely'],
                ['great', 'great', 'great'],
                ['boring', 'annoying', 'unimaginative']
            ]
        })
        
    naive_bayes = NaiveBayes()
    naive_bayes.train(TRAIN_DATA, 'text_col', 'class_col', smooth=False)
    naive_bayes.classify(TEST_DATA, 'text_col', 'pred', smoothed=False, 
                         return_all=True)
    