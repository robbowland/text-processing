import os
from enum import Enum
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from typing import Dict
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from naivebayes import NaiveBayes
from textpreprocessor import TextPreprocessor
from config import CONTRACTION_MAP, STOP_WORDS, PUNCTUATION, POSITIVE, NEUTRAL, NEGATIVE, CONNECTIVES, NEGATORS
from featureextractor import FeatureExtractor as fe
import nltk

# nltk.download('averaged_perceptron_tagger')
# nltk.download('opinion_lexicon')

class Sentiment(Enum):
    """
    Enum containing values for 5-value sentiment scale.
    """
    NEGATIVE = 0
    SOMEWHAT_NEGATIVE = 1
    NEUTRAL = 2
    SOMEWHAT_POSITIVE = 3
    POSITIVE = 4
    
class ReducedSentiment(Enum):
    """
    Enum containing values for reduced 3-value sentiment scale.
    """
    NEGATIVE = 0
    NEUTRAL = 1
    POSITIVE = 2


#  Mapping for 5-value sentiment scale to 3-value sentiment scale
SENTIMENT_REDUCTION_MAP = {
    Sentiment.NEGATIVE.value: ReducedSentiment.NEGATIVE.value,
    Sentiment.SOMEWHAT_NEGATIVE.value: ReducedSentiment.NEGATIVE.value,
    Sentiment.NEUTRAL.value: ReducedSentiment.NEUTRAL.value,
    Sentiment.SOMEWHAT_POSITIVE.value: ReducedSentiment.POSITIVE.value,
    Sentiment.POSITIVE.value: ReducedSentiment.POSITIVE.value
    }


def reduce_sentiment_scale(data: pd.DataFrame, 
                           col_name: str, 
                           reduction_map: Dict,
                           output_col: str = None):
    """
    Reduce an n-value sentiment scale to an m-value sentiment scale using
    a specified reduction map.
    Operations are performed in place on the specified dataframe.

    Parameters
    ----------
    data: pd.DataFrame
        Pandas DataFrame containing the sentiment values to be reduced.
    col_name : str
        The name of the column in the DataFrame that contains the sentiment
        values to be reduced.
    reduction_map : Dict
        A Dictionary serving as a map from the old valued to new values.
        (Key, Value) pairs should be of the following form:
            (old sentiment value, new sentiment value)
    output_col : str, optional
        The name of a column that the reduced sentiment values will be placed 
        in in the pass in dataset. If not provided, the sentiment value column 
        specified by col_name, will be replaced.

    Returns
    -------
    None.
    """
    if output_col:
         data[output_col] = data[col_name].replace(reduction_map)
    else:
        data[col_name].replace(reduction_map, inplace=True)
       

# TEST MANTLE
if __name__ == '__main__':
    
    # Set up Dataset Paths
    DATA_PATH = r'.{sep}moviereviews{sep}'.format(sep=os.path.sep)
    PREDICTIONS_OUTPUT_DATAPATH = r'.{sep}predictions{sep}'.format(sep=os.path.sep)
    # Load datasets for use with ALL WORDS model
    TRAIN_DATA_ALL_WORDS = pd.read_csv(DATA_PATH + "train.tsv", sep='\t')
    DEV_DATA_ALL_WORDS = pd.read_csv(DATA_PATH + "dev.tsv", sep='\t')
    # Load datasets for use with SELECT WORDS model
    TRAIN_DATA_SELECT = pd.read_csv(DATA_PATH + "train.tsv", sep='\t')
    DEV_DATA_SELECT = pd.read_csv(DATA_PATH + "dev.tsv", sep='\t')
    TEST_DATA_SELECT = pd.read_csv(DATA_PATH + "test.tsv", sep='\t')
    
    # COLUMN NAME CONSTANTS
    TEXT_COL_NAME = 'Phrase'
    FIVE_SENTIMENT_COL_NAME = 'Sentiment'
    THREE_SENTIMENT_COL_NAME = 'Sentiment (3 Value Scale)'
    FIVE_SENT_PRED_COL_NAME = 'Predicted Sentiment (5 Value Scale)'
    THREE_SENT_PRED_COL_NAME = 'Predicted Sentiment (3 Value Scale)'
    
    # ADD REDUCED SENTIMENT COLUMNS TO DATA
    # Reduce TRAINING Data Sentiment Values
    reduce_sentiment_scale(TRAIN_DATA_ALL_WORDS, 
                           FIVE_SENTIMENT_COL_NAME, 
                           reduction_map=SENTIMENT_REDUCTION_MAP, 
                           output_col=THREE_SENTIMENT_COL_NAME)
    reduce_sentiment_scale(TRAIN_DATA_SELECT, 
                           FIVE_SENTIMENT_COL_NAME, 
                           reduction_map=SENTIMENT_REDUCTION_MAP, 
                           output_col=THREE_SENTIMENT_COL_NAME)
    
    # Reduce DEVELOPMENT Data Sentiment Values
    reduce_sentiment_scale(DEV_DATA_ALL_WORDS, 
                           FIVE_SENTIMENT_COL_NAME, 
                           reduction_map=SENTIMENT_REDUCTION_MAP,
                           output_col=THREE_SENTIMENT_COL_NAME)
    reduce_sentiment_scale(DEV_DATA_SELECT, 
                           FIVE_SENTIMENT_COL_NAME, 
                           reduction_map=SENTIMENT_REDUCTION_MAP,
                           output_col=THREE_SENTIMENT_COL_NAME)
    
    # RUNNING CONFIGURATION
    # Whether or not to run the dev test runs
    RUN_DEV_TESTS = True
    # Whether or not to run on unknown test data
    RUN_UNKNOWN_DATA = True
    
    # NAIVE BAYES SMOOTHING
    # Whether or not to use Laplace Smoothing with Naive Bayes
    USE_SMOOTHING = True
    
    # INSTANTIATE TEXT PREPROCESSOR
    tp = TextPreprocessor(contraction_map=CONTRACTION_MAP, 
                          stop_words=STOP_WORDS,
                          negators=NEGATORS,
                          connectives=CONNECTIVES,
                          punctuation=PUNCTUATION, 
                          opinion_lexicons=[POSITIVE, NEUTRAL, NEGATIVE],
                          stemmer=SnowballStemmer('english'), 
                          lemmatizer=WordNetLemmatizer())
    
    # TEXT PREPROCESSING SEQUENCES
    # Edit these to alter the preprocessing applied to either stage.
    ALL_WORDS_PREPROCESSING = ['lc', 'tl']
    SELECT_WORDS_PREPROCESSING = ['lc', 'ec', 'rs', 'rn', 'ns', 'n', 'rp', 'tl']
   
    
    
    # Training + Running the Models
    # ==========================| ALL WORDS |=================================
    # ----- TEXT PREPROCESSING -----------------------------------------------
    # Normalise text via lower casing, but do not remove any tokens.
    # TRAIN DATA
    tp.preprocess(TRAIN_DATA_ALL_WORDS, TEXT_COL_NAME, ALL_WORDS_PREPROCESSING)
    # DEV DATA
    tp.preprocess(DEV_DATA_ALL_WORDS, TEXT_COL_NAME, ALL_WORDS_PREPROCESSING)
    # ------------------------------------------------------------------------
    
    # ----- TRAINING ---------------------------------------------------------
    # TRAIN both Naive Bayes models using all features as words and
    # FIVE SENTIMENT VALUE
    naive_bayes_five_value_all_word = NaiveBayes()
    naive_bayes_five_value_all_word.train(TRAIN_DATA_ALL_WORDS, 
                                          TEXT_COL_NAME, 
                                          FIVE_SENTIMENT_COL_NAME, 
                                          smooth=USE_SMOOTHING)
    # THREE SENTIMENT VALUE
    naive_bayes_three_value_all_word = NaiveBayes()
    naive_bayes_three_value_all_word.train(TRAIN_DATA_ALL_WORDS, 
                                           TEXT_COL_NAME,
                                           THREE_SENTIMENT_COL_NAME, 
                                           smooth=USE_SMOOTHING)
    # ------------------------------------------------------------------------
    
    # ----- DEV TESTING ------------------------------------------------------
    if RUN_DEV_TESTS:
        # FIVE SENTIMENT VALUE
        naive_bayes_five_value_all_word.classify(DEV_DATA_ALL_WORDS, 
                                                TEXT_COL_NAME,
                                                 FIVE_SENT_PRED_COL_NAME,
                                                 smoothed=USE_SMOOTHING)
        # CONFUSION MATRIX
        five_value_all_word_cm = confusion_matrix(
            DEV_DATA_ALL_WORDS[FIVE_SENTIMENT_COL_NAME], 
            DEV_DATA_ALL_WORDS[FIVE_SENT_PRED_COL_NAME].astype(int),    
            normalize='true')
        plt.figure(figsize = (10,7))
        sn.heatmap(five_value_all_word_cm, annot=True, cmap='Blues', fmt='.2%').set_title('5-Value Sentiment Scale All Words')
        print(
            'ALL WORDS 5 Value Sentiment Scale Prediction Metrics:\nUsing Preprocessing Sequence:  ' + str(ALL_WORDS_PREPROCESSING) + "\n"+ "-"*50 + "\n",
            classification_report(
            DEV_DATA_ALL_WORDS[FIVE_SENTIMENT_COL_NAME],
            DEV_DATA_ALL_WORDS[FIVE_SENT_PRED_COL_NAME].astype(int),
            digits=3),
            "-"*50 + "\n"
            )
        
        # THREE SENTIMENT VALUE
        naive_bayes_three_value_all_word.classify(DEV_DATA_ALL_WORDS, 
                                                  TEXT_COL_NAME,
                                                  THREE_SENT_PRED_COL_NAME,
                                                  smoothed=USE_SMOOTHING)
        three_value_all_word_cm = confusion_matrix(
            DEV_DATA_ALL_WORDS[THREE_SENTIMENT_COL_NAME], 
            DEV_DATA_ALL_WORDS[THREE_SENT_PRED_COL_NAME].astype(int),
            normalize='true')
        # CONFUSION MATRIX
        plt.figure(figsize = (10,7))
        sn.heatmap(three_value_all_word_cm, annot=True, cmap='Blues', fmt='.2%').set_title('3-Value Sentiment Scale All Words')
        print(
            'ALL WORDS 3 Value Sentiment Scale Prediction Metrics:\nUsing Preprocessing Sequence:  ' + str(ALL_WORDS_PREPROCESSING) + "\n"+ "-"*50 + "\n",
            classification_report(
            DEV_DATA_ALL_WORDS[THREE_SENTIMENT_COL_NAME],
            DEV_DATA_ALL_WORDS[THREE_SENT_PRED_COL_NAME].astype(int),
            digits=3),
            "-"*50 + "\n"
            )
    # ------------------------------------------------------------------------
    # =============================| END |====================================
    
    
    
    # ==========================| SELECTED WORDS |============================
    # ----- TEXT PREPROCESSING -----------------------------------------------
    # TRAIN DATA
    tp.preprocess(TRAIN_DATA_SELECT, TEXT_COL_NAME, SELECT_WORDS_PREPROCESSING)
    # DEV DATA
    tp.preprocess(DEV_DATA_SELECT, TEXT_COL_NAME, SELECT_WORDS_PREPROCESSING)
    # TEST DATA
    tp.preprocess(TEST_DATA_SELECT, TEXT_COL_NAME, SELECT_WORDS_PREPROCESSING)
    # ------------------------------------------------------------------------
    
    # ----- TRAINING ---------------------------------------------------------
    # FIVE SENTIMENT VALUE
    naive_bayes_five_value_select = NaiveBayes()
    naive_bayes_five_value_select.train(TRAIN_DATA_SELECT,
                                        TEXT_COL_NAME, 
                                        FIVE_SENTIMENT_COL_NAME, 
                                        smooth=USE_SMOOTHING)
    # THREE SENTIMENT VALUE
    naive_bayes_three_value_select = NaiveBayes()
    naive_bayes_three_value_select.train(TRAIN_DATA_SELECT, 
                                         TEXT_COL_NAME, 
                                         THREE_SENTIMENT_COL_NAME, 
                                         smooth=USE_SMOOTHING)
    # ------------------------------------------------------------------------
    
    # ----- DEV TESTING ------------------------------------------------------
    if RUN_DEV_TESTS:
        # TEST both Naive Bayes on DEV_DATA using all features as words an plot
        # confusion matrices for each.
        # FIVE SENTIMENT VALUE
        naive_bayes_five_value_select.classify(DEV_DATA_SELECT, 
                                                 TEXT_COL_NAME, 
                                                 FIVE_SENT_PRED_COL_NAME,
                                                 smoothed=USE_SMOOTHING)
        # CONFUSION MATRIX
        five_value_select_cm = confusion_matrix(
            DEV_DATA_SELECT[FIVE_SENTIMENT_COL_NAME], 
            DEV_DATA_SELECT[FIVE_SENT_PRED_COL_NAME].astype(int),    
            normalize='true')
        plt.figure(figsize = (10,7))
        sn.heatmap(five_value_select_cm, annot=True, cmap='Blues', fmt='.2%').set_title('5-Value Sentiment Scale Select Words')
        print(
            'SELECT FEATURES 5 Value Sentiment Scale Prediction Metrics:\nUsing Preprocessing Sequence:  ' + str(SELECT_WORDS_PREPROCESSING) + "\n"+ "-"*50 + "\n",
            classification_report(
            DEV_DATA_SELECT[FIVE_SENTIMENT_COL_NAME],
            DEV_DATA_SELECT[FIVE_SENT_PRED_COL_NAME].astype(int),
            digits=3),
            "-"*50 + "\n"
            )
        
        # THREE SENTIMENT VALUE
        naive_bayes_three_value_select.classify(DEV_DATA_SELECT, 
                                                  TEXT_COL_NAME, 
                                                  THREE_SENT_PRED_COL_NAME,
                                                  smoothed=USE_SMOOTHING)
        three_value_select_cm = confusion_matrix(
            DEV_DATA_SELECT[THREE_SENTIMENT_COL_NAME], 
            DEV_DATA_SELECT[THREE_SENT_PRED_COL_NAME].astype(int),
            normalize='true')
        # CONFUSION MATRIX
        plt.figure(figsize = (10,7))
        sn.heatmap(three_value_select_cm, annot=True, cmap='Blues', fmt='.2%').set_title('3-Value Sentiment Scale Select Words')
        print(
            'SELECT FEATURES 3 Value Sentiment Scale Prediction Metrics:\nUsing Preprocessing Sequence:  ' + str(SELECT_WORDS_PREPROCESSING) + "\n"+ "-"*50 + "\n",
            classification_report(
            DEV_DATA_SELECT[THREE_SENTIMENT_COL_NAME],
            DEV_DATA_SELECT[THREE_SENT_PRED_COL_NAME].astype(int),
            digits=3),
            "-"*50 + "\n"
            )
        
        # OUTPUT RESULTS
        # 3 VALUE
        DEV_DATA_SELECT.to_csv(PREDICTIONS_OUTPUT_DATAPATH + 'dev_predictions_3classes_Rob_BOWLAND.tsv', sep='\t', columns=['SentenceId', THREE_SENT_PRED_COL_NAME], header=['SentenceId', 'Sentiment'], index=False)
        # 5 VALUE
        DEV_DATA_SELECT.to_csv(PREDICTIONS_OUTPUT_DATAPATH + 'dev_predictions_5classes_Rob_BOWLAND.tsv', sep='\t', columns=['SentenceId', FIVE_SENT_PRED_COL_NAME], header=['SentenceId', 'Sentiment'], index=False)
    # ------------------------------------------------------------------------
    
    # ----- UNKNOWN TEST DATA RUN --------------------------------------------
    if RUN_UNKNOWN_DATA:
        # FIVE SENTIMENT VALUE
        naive_bayes_five_value_select.classify(TEST_DATA_SELECT, 
                                                 TEXT_COL_NAME, 
                                                 FIVE_SENT_PRED_COL_NAME,
                                                 smoothed=USE_SMOOTHING)
    
        # THREE SENTIMENT VALUE
        naive_bayes_three_value_select.classify(TEST_DATA_SELECT, 
                                                  TEXT_COL_NAME, 
                                                  THREE_SENT_PRED_COL_NAME,
                                                  smoothed=USE_SMOOTHING)
    
        # OUTPUT RESULTS
        # 3 VALUE
        TEST_DATA_SELECT.to_csv(PREDICTIONS_OUTPUT_DATAPATH + 'test_predictions_3classes_Rob_BOWLAND.tsv', sep='\t', columns=['SentenceId', THREE_SENT_PRED_COL_NAME], header=['SentenceId', 'Sentiment'], index=False)
        # 5 VALUE
        TEST_DATA_SELECT.to_csv(PREDICTIONS_OUTPUT_DATAPATH + 'test_predictions_5classes_Rob_BOWLAND.tsv', sep='\t', columns=['SentenceId', FIVE_SENT_PRED_COL_NAME], header=['SentenceId', 'Sentiment'], index=False)
    # ------------------------------------------------------------------------
    # =============================| END |====================================
    
    # Example for applying feature extraction techniques, these were removed
    # from the above code because they did not improve performance and
    # made the code hard to read.
    # 
    # 1. Convert to text list prior to part of speech extraction
    # tp.preprocess(TRAIN_DATA_SELECT, TEXT_COL_NAME, ['tl'])
    # 2. Extract parts of speech as string so further preprocessing can be performed.
    # TRAIN_DATA_SELECT[TEXT_COL_NAME]= fe.get_parts_of_speech(TRAIN_DATA_SELECT, TEXT_COL_NAME, {'JJ'}, as_string=True) 
    # 3. Perfrom standard preprocessing sequence
    # tp.preprocess(TRAIN_DATA_SELECT, TEXT_COL_NAME, SELECT_WORDS_PREPROCESSING)
    # 4. Make ngrams
    # TRAIN_DATA_SELECT[TEXT_COL_NAME]= fe.make_ngrams(TRAIN_DATA_SELECT, TEXT_COL_NAME, 1, 2)
    