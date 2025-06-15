from nltk.corpus import opinion_lexicon


# ==========================| OPINION LEXICONS |==============================
# Used for 'sentiment normalisation' preprocessing.
# Positive and Negative lexicons are extended to include the negation of 
# each other, for use with negation preprocessing.
POSITIVE = set(opinion_lexicon.positive()).union({"NOT_" + word for word in set(opinion_lexicon.negative())})
NEGATIVE = set(opinion_lexicon.negative()).union({"NOT_" + word for word in set(opinion_lexicon.positive())})
NEUTRAL = {'ok', 'fine', 'passable', 'alright', 'okay', 'neutral', 'average'}
# ============================================================================


# ==========================| PUNCTUATION SET |===============================
# Contains punctuation characters that are deemed as meaningless, for 
# use in punctuation removal preprocessing.
PUNCTUATION = "\"#$%&'()*+-/:;<=>@[\\]^_`{|}~"
# NOTE:
# Punctuation Marks that add value: ! ? , .
# Comma perhaps indicates longer review length and therefore more engagement
# Fullstop perhaps contrasts to reviews ending with ? or ! 
# i.e. sentence ending adds value
# ============================================================================

# ================================| CONNECTIVES |=============================
# Set of words defined as connectives
CONNECTIVES = {
    "and",
    "so",
    "but",
    "because"
    }
# ============================================================================

# ================================| NEGATORS |=============================
# Set of words that are defined to act as negators
NEGATORS = {
    "not"
    }
# ============================================================================

# ============================| CONTRACTION MAP |=============================
# Maps contracted tokens to expanded form, for use with expanding contractions,
# preprocessing.
CONTRACTION_MAP = {
    "n't": "not",
    "'t": "not",
    "'re": "are",
    "'s": "is",
    "'ll": "will",
    "'ve": "have",
    "'m": "am",
    "'d": "would"
    }
# ============================================================================


# ===============================| STOP WORDS |===============================
# Set of words deemed to provide no information. Built manually through 
# experimentation. 
# NOTE:
# A Custom stop list is useful as removal of typical stoplist words can lead
#  to decreases in classification accuracy.
STOP_WORDS = {
    "t",
    "s",
    "ll",
    "ve",
    "ca",
    "rrb",
    "rlb",
    "lrb",
    "llb",
    "is",
    "am",
    "will",
    "have",
    "us",
    "we",
    "audience",
    "viewer",
    "however",
    "anyone",
    "anybody",
    "review",
    "reviewer",
    "actor",
    "actress",
    "director",
    "cinema",
    "acting",
    "act",
    "place",
    "your",
    "for",
    "story",
    "she",
    "he",
    "him",
    "himself",
    "herslef",
    "his",
    "hers",
    "yourself",
    "self",
    "the",
    "ourselves",
    "themselves"
    "them",
    "their",
    "theirs",
    "they",
    "itself", 
    "myself",
    "my",
    "who",
    "whom",
    "whoever",
    "this",
    "what",
    "which",
    "these",
    "those",
    "be",
    "being",
    "have",
    "having",
    "at",
    "as",
    "on",
    "is",
    "by",
    "do",
    "see",
    "seen",
    "where",
    "when",
    "can",
    "will",
    "would",
    "off",
    "those",
    "start",
    "finish",
    "end",
    "begin",
    'beginning',
    "before",
    "middle",
    "than",
    "hand",
    "other",
    "however",
    "thought",
    "same",
    "film",
    "movie",
    "films",
    "movies",
    "far",
    "thing",
    "around",
    "go",
    "goes",
    "to",
    "it",
    "that",
    "of",
    "you",
    "with",
    "go",
    "view",
    "goes",
    "do",
    "to",
    "its",
    "there",
    # 0.431, 0.670 ['lc', 'ec', 'rp', 'rs', 'tl']
    "new",
    "old",
    "newer",
    "younger",
    "older",
    "recent",
    # 0.432, 0.672 ['lc', 'ec', 'rp', 'rs', 'tl']
    "make",
    "from",
    "work"
    # 0.432, 0.673 ['lc', 'ec', 'rp', 'rs', 'tl']
    }
# ============================================================================
    
 
