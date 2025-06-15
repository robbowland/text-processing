# Sentiment Analysis
This was the final assignment for the module. The aim of this assignment was to implement a machine learning model based on Naive Bayes for a sentiment analysis task using the Rotten Tomatoes movie review dataset. Obstacles like sentence negation, sarcasm, terseness, language ambiguity, and many others make this task very challenging.

## Key Files & Folders
The `[src]` folder contains the files for conducting sentiment analysis using a Naive Bayes classifier, as well as configuration and additional functionality modules for text preprocessing and feature extraction.

**`main.py`** - main file for running the classification experiments

**`config.py`** - configuration file for things like stopword lists

**`naivebayes.py`** - naive bayes implementation

**`textpreprocessor.py`** - text preprocessor implementation

**`featureextractor.py`** - feature extractor implementation

**`[moviereviews]`** - folder containing movie reviews for classification 

**`[predictions]`** - folder containing tsv files containing predicted classes

## How to Use

**Running:**
The following command line arguments can be used to run classification using the configured model. Anaconda is recommended to avoid needing to install multiple dependencies.

    cd src             
    conda activate     (activate an anaconda environment)
    python main.py     (run the file)

Note: You may be required to download certain certain things using `nltk.download` prompts that can be followed should appear in the command line.

**Output:**
On running a series of performance reports will be printed to the command line indicating the performance of each of the configured models as well as the preprocessing steps that were used on them, based on classification of development data.
e.g.

    SELECT FEATURES 3 Value Sentiment Scale Prediction Metrics:
    Using Preprocessing Sequence:  ['lc', 'ec', 'rs', 'rn', 'ns', 'n', 'rp', 'tl']
    --------------------------------------------------
                   precision    recall  f1-score   support
    
               0      0.675     0.793     0.729       386
               1      0.431     0.122     0.190       181
               2      0.726     0.831     0.775       433
    
        accuracy                          0.688      1000
       macro avg      0.611     0.582     0.565      1000
    weighted avg      0.653     0.688     0.651      1000
     --------------------------------------------------

Additionally, output files will be produced in the **`[predictions]`** for the development data classifcation as well as the test data classification. Confusion matrices will also be displayed for each model.

**Reconfiguring:**
Most configuration is performed in `main.py`, its various sections are commented to provide guidance. `config.py` contains additional configuration for items that are used for preprocessing.
Examples for applying feature extraction techniques can be found at the bottom of `main.py`.

## Mark Breakdown

| Section                                | Mark        | Comments                                                     |
| -------------------------------------- | ----------- | ------------------------------------------------------------ |
| *Implementation & Code Style* | ***14/15*** | - Functions are commented appropiately Code is modular.<br />- A classification report is used but a function for specifically calculating the accuracy is not implemented (**-1 mark**). |
 *Report* | ***9.5/10*** | - Full description of implementation Rich set of features explored Good analysis and conclusions.<br /> - Only 2/4 Confusion matrices are presented (**-0.5 marks**). |

***Overall:    23.5/25   |   94%***
