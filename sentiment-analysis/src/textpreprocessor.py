import pandas as pd
import numpy as np
from typing import Dict, List
import regex as re
import nltk
from nltk.corpus import wordnet



class TextPreprocessor:
    """
    Class containing functionanilty for performing various preprocessing
    steps on a dataset containing text.
    """
    
    def __init__(self, contraction_map: Dict, 
                 connectives: set,
                 negators: set, 
                 stop_words: set, 
                 punctuation: str, 
                 opinion_lexicons: List[set],
                 stemmer = None, 
                 lemmatizer = None) -> None:
        """
        Initialise TextPreprocessor.

        Parameters
        ----------
        contraction_map : Dict
           Dictionary containing representing contraction map that will 
           be used for contraction expansion.
           Of the form:
               (Key: Value) = (Contraction: Expansion)
               e.g. ("n't": "not)
        stop_words : set
            Set of stop_words that will be removed via stop_word removal.
        punctuation : str
            String containing all characters that will be considered as 
            punctuation when conducting punctuation removal.
        opinion_lexicons : List[set]
            List of sets where each set contains the words that are a part of 
            that opinion lexicon, based on sentiment.
            Should be in the order [positive, neurtal, negative].
        stemmer : nltk stemmer, optional
            An initialised stemmer from the nltk package that will be used
            for stemming. Default is None.
        lemmatizer : nltk stemmer, optional
            An initialised lemmatizer from the nltk package that will be used
            for lemmatizing. Default is None.

        Returns
        -------
        None.

        """
        self.contraction_map = contraction_map
        self.stop_words = stop_words
        self.punc = punctuation
        self.negators = negators
        self.connectives = connectives
        self.opinion_lexicons = opinion_lexicons
        self.stemmer = stemmer
        self.lemmatizer = lemmatizer
        self.techniques = {
                # To be used on STRINGS
                'lc': self._make_lower,  # Lower Case
                'ec': self._expand_contractions, # Expand Contractions
                'rp': self._remove_punctuation, # Remove Punctuation
                'rn': self._remove_numbers, # Remove Numbers
                'n': self._apply_negation, # Negate 
                'ns': self._normalise_sentiment, # Normalise from Opinon Lexicon
                'tl': self._make_list, # To List
                'rs': self._remove_stopwords, # Remove Stopwords
                # To be used on LIST
                's': self._apply_stemming, # Stem
                'l': self._apply_lematization, # Lematize
            }
    
    def preprocess(self, dataframe: pd.DataFrame, col_name: str, use: List[str]) -> None:
        """
        Preprocess a dataframe column using specified techniques.
        Modifies the dataframe in place.
        
        Parameters
        ----------
        dataframe : pd.DataFrame
            The data frame to be modified.
        col_name : str
            The name of the column to which preprocessing will be applied.
        use : List[str]
            A list of preprocessing technique keys in order they should be
            applied to the column.
            
            Keys are:
            
            --- TO BE USED ON STRING COLUMN CONTENTS ---
            
            'lc' - make contents lower case
            'ec' - expand contractions using contraction map
            'rp' - remove punctuation specified on class instantiation
            'rn' - remove numbers
            'rs' - remove words from the column that are present in the 
                   stoplist that was specified on instantiation
            'n'  - negate words between 'not' and a punctuation mark
                   i.e. prepend 'NOT_' to these words
            'ns' - add normalised sentiment words based on opinion lexicons
                   e.g. add one instance of POSITIVE for every word present 
                        in the text that's present in the postitive lexicon
            'tl' - split string column on spaces to make column conents a list
            
            --- TO BE USED ON LIST COLUMN CONTENTS (i.e. after 'tl')---
            
            'l' - use lemmatizer to lemmatise words, returns column contents 
                  as a string so that other techniques can easily be used after 
                  it.
            's' - use stemmer to stem words, returns column contents 
                  as a string so that other techniques can easily be used after 
                  it.

        Returns
        -------
        None.

        """
        for technique in use:
            self.techniques[technique](dataframe, col_name)
    
    def _make_lower(self, dataframe: pd.DataFrame, col_name: str) -> None:
        """
        Make a specified dataframe column lower case.
        Modifies the dataframe in place.
        Intended for internal use only.
        
        Parameters
        ----------
        dataframe : pd.DataFrame
            The data frame to be modified.
        col_name : str
            The name of the column to be made lower case.
            Column should contain pre-tokenised text in string form.
            
        Returns
        -------
        None.

        """
        dataframe[col_name] = dataframe[col_name].str.lower()
    
    def _expand_contractions(self, dataframe: pd.DataFrame, 
                             col_name: str) -> None:
        """
        Expand contractions in a given DataFrame Column using the specified
        contraction_map on class instantiation.
        Modifies the dataframe in place.
        Intended for internal use only.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The data frame to be modified.
        col_name : str
            The name of the column to be made for which contractions will 
            be expanded.
            Column should contain pre-tokenised text in string form.

        Returns
        -------
        None.

        """
        dataframe[col_name] = dataframe[col_name].replace(self.contraction_map,
                                                          regex=True)
    
    def _remove_punctuation(self, dataframe: pd.DataFrame, col_name: str) -> None:
        """
        Remove punctuation from the text in the column as per the punctuation 
        that was specified on class instantiation.
        Modifies the dataframe in place.
        Intended for internal use only.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The data frame to be modified.
        col_name : str
            The name of the column to be made from which punctuation will 
            be removed.
            Column should contain pre-tokenised text in string form.

        Returns
        -------
        None.

        """
        dataframe[col_name] = np.vectorize(
            lambda phrase: 
                phrase.translate(
                    str.maketrans('', '', self.punc)),
                otypes=[str])(dataframe[col_name])
    
    def _remove_numbers(self, dataframe: pd.DataFrame, col_name: str) -> None:
        """
        Remove numbers from the text in the column.
        Modifies the dataframe in place.
        Intended for internal use only.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The data frame to be modified.
        col_name : str
            The name of the column to be made from which numbers will 
            be removed.
            Column should contain pre-tokenised text in string form.

        Returns
        -------
        None.

        """
        dataframe[col_name] = np.vectorize(
            lambda phrase: 
                phrase.translate(
                    str.maketrans('', '', "0123456789")),
                otypes=[str])(dataframe[col_name])
            
    def _remove_stopwords(self, dataframe: pd.DataFrame, col_name: str) -> None:
        """
        Remove stop words from the text in the column based on the specified
        stop word list specified on class instantiation.
        Modifies the dataframe in place.
        Intended for internal use only.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The data frame to be modified.
        col_name : str
            The name of the column to be made from which stop words will 
            be removed.
            Column should contain pre-tokenised text in string form.

        Returns
        -------
        None.

        """
        dataframe[col_name] = np.vectorize(
            lambda phrase: ' '.join([word for word in phrase.split(' ') 
                                     if word not in self.stop_words]), 
            otypes=[str])(dataframe[col_name])
    
    # This could probably be improved even further so that it will only 
    # negate certain words based on POS tagging or similar, but I don't 
    # have the time to explore this.
    def _apply_negation(self, dataframe: pd.DataFrame, col_name: str) -> None:
        """
        Apply negation to text in a dataframe column.
        Apply negation to all words in a clause by prepending 'NOT_' to them.
        Where a clause is defined to be all words occuring between the 
        specified negators (specified on classed instatiation) and 
        connectives (specified on classed instatiation) or a punctuation/ end 
        of line symbol in a tokenised string.
        Then remove all instances of the negators.
        Modifies the dataframe in place.
        Intended for internal use only.
        
        e.g. Assuming 'not' is a specified negator and 'and' is 
             a specified connective the phrase:
            
            'this is not looking good and i do not like it .'
            
            would become:
                
            'this is NOT_looking NOT_good and i do NOT_like NOT_it .'

        Parameters
        ----------
        dataframe : pd.DataFrame
            The data frame to be modified.
        col_name : str
            The name of the column to which negation will be applied.
            Column should contain pre-tokenised text in string form.

        Returns
        -------
        None.

        """
        dataframe[col_name] = np.vectorize(
            lambda phrase: self._negate(phrase),
            otypes=[str])(dataframe[col_name])
        dataframe[col_name] = dataframe[col_name].str.replace(r'\b({})\b'.format('|'.join(self.negators)), '', regex=True)

    def _negate(self, phrase: str) -> str:
        """
        Apply negation to all words in a clause by prepending 'NOT_' to them.
        Where a clause is defined to be all words occuring between the 
        specified negators (specified on classed instatiation) and 
        connectives (specified on classed instatiation) or a punctuation/ end 
        of line symbol in a tokenised string.
        Then remove all instances of the negators.
        Intended for internal use only.
        
        e.g. Assuming 'not' is a specified negator and 'and' is 
             a specified connective the phrase:
            
            'this is not looking good and i do not like it .'
            
            would become:
                
            'this is NOT_looking NOT_good and i do NOT_like NOT_it .'

        Parameters
        ----------
        phrase : str
            The str to which negation wil be applied.
            Column should contain pre-tokenised text in string form.

        Returns
        -------
        str
            The negated string.

        """
        # Adapted from: https://stackoverflow.com/questions/23384351/how-to-add-tags-to-negated-words-in-strings-that-follow-not-no-and-never?rq=1
        return re.sub(r'(?:{})[\w\s]+([^\w\s]|{})'.format(
            '|'.join(self.negators),'|'.join(self.connectives)), 
           		lambda match: re.sub(r'(\s+)(\w+)', 
                                  r'\1NOT_\2',
                                  match.group(0)), 
        			phrase,
                    flags=re.IGNORECASE)
    
    # This is likely not the most effecient way of doing this, having a count
    # or similar would likely be better, by my naive bayes implementation 
    # is not set up to handle that directly, so for the sake of exploration, 
    # this implementation will have to suffice.
    # I also tried this where it replaced the words in the lexicons rather 
    # than making addition, but this did not have a positive effect.
    def _normalise_sentiment(self, dataframe: pd.DataFrame, 
                             col_name: str) -> None:
        """
        Add normalised sentiment terms to the existing text based on the 
        words in the specified opinion lexicons. 
        Modifies the dataframe in place.
        Intended for internal use only.
        
        For example, take the phrase:
            
            'there are some good and some bad things'

        Assuming 'good' is in the positive opinion lexicon and 'bad' is in the
        negative opinion lexicon, this will become:
            
             'there are some good and some bad things POSITIVE NEGATIVE'
            
        The objective here is to add additional features that may help with 
        classification, without removing existing features.
        
        Parameters
        ----------
        dataframe : pd.DataFrame
            The data frame to be modified.
        col_name : str
            The name of the column to which sentiment normalisation will be 
            applied.
            Column should contain pre-tokenised text in string form.

        Returns
        -------
        None.

        """
        self._normalise_positive(dataframe, col_name)
        self._normalise_negative(dataframe, col_name)
        self._normalise_neutral(dataframe, col_name)
    
    def _normalise_positive(self, dataframe: pd.DataFrame, 
                            col_name: str) -> None:
        """
        Add normalised sentiment terms to the existing text based on the 
        words in the specified POSITVE opinion lexicon. 
        Modifies the dataframe in place.
        Intended for internal use only.
        
        Parameters
        ----------
        dataframe : pd.DataFrame
            The data frame to be modified.
        col_name : str
            The name of the column to which sentiment normalisation will be 
            applied.
            Column should contain pre-tokenised text in string form.

        Returns
        -------
        None.

        """
        dataframe[col_name] = np.vectorize(
            lambda phrase: 
                phrase + ' ' + ' '.join(
                    ['POSITIVE' for word in phrase.split(' ') 
                     if word in self.opinion_lexicons[0]]), 
            otypes=[str])(dataframe[col_name])    
        
    def _normalise_neutral(self, dataframe: pd.DataFrame, 
                           col_name: str) -> None:
        """
        Add normalised sentiment terms to the existing text based on the 
        words in the specified NEUTRAL opinion lexicon.
        Modifies the dataframe in place.
        Intended for internal use only.
        
        Parameters
        ----------
        dataframe : pd.DataFrame
            The data frame to be modified.
        col_name : str
            The name of the column to which sentiment normalisation will be 
            applied.
            Column should contain pre-tokenised text in string form.

        Returns
        -------
        None.

        """
        dataframe[col_name] = np.vectorize(
        lambda phrase: 
            phrase + ' ' + ' '.join(
                ['NEUTRAL' for word in phrase.split(' ') 
                 if word in self.opinion_lexicons[1]]), 
        otypes=[str])(dataframe[col_name])  
        
    
    def _normalise_negative(self, dataframe: pd.DataFrame, 
                            col_name: str) -> None:
        """
        Add normalised sentiment terms to the existing text based on the 
        words in the specified NEGATIVE opinion lexicon.
        Modifies the dataframe in place.
        Intended for internal use only.
        
        Parameters
        ----------
        dataframe : pd.DataFrame
            The data frame to be modified.
        col_name : str
            The name of the column to which sentiment normalisation will be 
            applied.
            Column should contain pre-tokenised text in string form.

        Returns
        -------
        None.

        """
        dataframe[col_name] = np.vectorize(
            lambda phrase: 
                phrase + ' ' + ' '.join(
                    ['NEGATIVE' for word in phrase.split(' ') 
                     if word in self.opinion_lexicons[2]]), 
            otypes=[str])(dataframe[col_name])  
        
    def _make_list(self, dataframe: pd.DataFrame, col_name: str) -> None:
        """
        Split a specified DataFrame column, splitting space separated elements 
        to form a list.
        Modifies the dataframe in place.
        Intended for internal use only.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The data frame to be modified.
        col_name : str
            The name of the column to be split.
            Column should contain pre-tokenised text in string form.

        Returns
        -------
        None.

        """
        dataframe[col_name] = np.vectorize(
            lambda phrase: list(filter(None, phrase.split(' '))),
            otypes=[list])(dataframe[col_name])
    
    # Note that for these two functions I've made them return strings for 
    # ease of preprocessing setp chaining for experimentation. Having them
    # return lists, since they take in lists would be more logical but would 
    # then need another function to turn lists back into strings which would,
    # only be used on these. These didn't give positive results so I didn't bother.
    
    # This slows things down quite a lot, likely because of the tag conversion.
    # Doesn't seem to have an overall positive effect on performance.
    def _apply_lematization(self, dataframe: pd.DataFrame, col_name: str) -> None:
        """
        Apply lemmatization to the the specified dataframe column using the
        lemmatizer specified on class instantiation.
        Note that although this function requires text to be in list form for 
        input, it will output the stemmed text in string form. This is for ease
        of chaining with other preprocessing steps, converting to a list,
        before or aftet can be easily achieved using to to list preprocessor.
        Modifies the dataframe in place.
        Intended for internal use only.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The data frame to be modified.
        col_name : str
            The name of the column to be split.
            Column should contain pre-tokenised text string that has been split
            into a list.
            
        Raises
        -------
        Exception
            If a Lemmatizer has not be specified for the class.

        Returns
        -------
        None.

        """
        if self.lemmatizer is None:
            raise Exception("A Lemmatizer has not been specified in the TextPreprocessor. Please specifiy a Lemmatizer to use.")
        lemmatizer = self.lemmatizer
        dataframe[col_name] =  np.vectorize(
        lambda phrase: ' '.join([lemmatizer.lemmatize(word, self._convert_tag(tag)) for (word,tag) in nltk.pos_tag(phrase)]), 
            otypes=[str])(dataframe[col_name])

    def _convert_tag(self, tag: str) -> str:
        """
        Convert a part of speech tag to a tag format that can be 
        recognised by the nltk WordNetLemmatizer.

        Parameters
        ----------
        tag : str
            The nltk pos tag.

        Returns
        -------
        str
            The converted tag.

        """
        tag_map = {
             'J': wordnet.ADJ,
             'V': wordnet.VERB,
             'N': wordnet.NOUN,
             'R': wordnet.ADV
             }
        if tag[0] in tag_map:
            return tag_map[tag[0]]
        else:
            return 'a'
        
    def _apply_stemming(self, dataframe: pd.DataFrame, col_name: str) -> None:
        """
        Apply stemming to the the specified dataframe column using the
        stemmer specified on class instantiation.
        Note that although this function requires text to be in list form for 
        input, it will output the stemmed text in string form. This is for ease
        of chaining with other preprocessing steps, converting to a list,
        before or aftet can be easily achieved using to to list preprocessor.
        Modifies the dataframe in place.
        Intended for internal use only.
        
        Parameters
        ----------
        dataframe : pd.DataFrame
            The data frame to be modified.
        col_name : str
            The name of the column to be split.
            Column should contain pre-tokenised text string that has been split
            into a list.
            
        Raises
        -------
        Exception
            If a Stemmer has not be specified for the class.

        Returns
        -------
        None.
        
        """
        if self.stemmer is None:
            raise Exception("A Stemmer has not been specified in the TextPreprocessor. Please specifiy a Stemmer to use.")
        stemmer = self.stemmer
        dataframe[col_name] =  np.vectorize(
        lambda phrase: ' '.join([stemmer.stem(word) for word in phrase]), 
            otypes=[str])(dataframe[col_name])
        
        
        
        
        
        