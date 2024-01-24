from ...utils.llama_utils import softmax
from ...utils.string_modification import count_syllables
from textblob import TextBlob
import spacy
import numpy as np
from collections import Counter
from scipy import stats
import pandas as pd

def flesch_reading_ease(article):
    words = article.split()
    num_words = len(words)
    num_sentences = article.count('. ') + article.count('! ') + article.count('? ')
    num_syllables = 0
    for word in words:
        if word:  # Ensure the word is not empty
            syllables = count_syllables(word)
            num_syllables += syllables
    if num_words > 0 and num_sentences > 0:
        flesch_score = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words) # Formula for Flesch-Kincaid reading ease
        return flesch_score
    else:
        return 0
    
def shannon_entropy(article):
    prob = [float(article.count(c)) / len(article) for c in dict.fromkeys(list(article))]
    entropy = -sum([p * np.log2(p) for p in prob])
    return entropy

def special_chars_count(article):
    special_chars = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '_', '+', '=', '{', '}', '[', ']', '|', '\\', ':', ';', '"', "'", '<', '>', ',', '.', '?']
    return sum([article.count(char) for char in special_chars])

def sentiment_consistency(article):
    sentences = article.split('. ')
    sentiments = [TextBlob(sentence).sentiment.polarity for sentence in sentences]
    variance = np.var(sentiments)
    return variance

def named_entity_analysis(article, nlp=spacy.load("en_core_web_sm")):
    doc = nlp(article)
    entities = set([ent.text for ent in doc.ents])
    return len(entities)

def repetition_patterns(article, n=3):
    tokens = article.split()
    ngrams = zip(*[tokens[i:] for i in range(n)])
    freq = Counter(ngrams)
    # Count n-grams that appear more than once
    repetitions = sum(1 for item in freq.values() if item > 1)
    return repetitions

def count_transition_words(article):
    transition_words = ['however', 'furthermore', 'therefore', 'consequently', 'meanwhile', 'nonetheless', 'moreover', 'likewise', 'instead', 'nevertheless', 'otherwise', 'similarly', 'accordingly', 'subsequently', 'hence', 'thus', 'still', 'then', 'yet', 'accordingly', 'additionally', 'alternatively', 'besides', 'comparatively', 'conversely', 'finally', 'further', 'furthermore', 'hence', 'however', 'indeed', 'instead', 'likewise', 'meanwhile', 'moreover', 'nevertheless', 'next', 'nonetheless', 'otherwise', 'similarly', 'still', 'subsequently', 'then', 'therefore', 'thus', 'whereas', 'while', 'yet'] 
    count = sum(article.count(word) for word in transition_words)
    return count

def ks_test(baseline_scores, article_scores):
    _, p_value = stats.ks_2samp(baseline_scores, article_scores)
    #return p_value
    return -1 if p_value < 0.05 else 1

def ad_test_normal_dist(article_scores):
    result = stats.anderson(article_scores, dist='norm')
    for i in range(len(result.critical_values)):
        sl, cv = result.significance_level[i], result.critical_values[i]
        return -1 if result.statistic > cv else 1

def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Sort the data
    x = np.sort(data)
    # Calculate ECDF values
    y = np.arange(1, len(x) + 1) / len(x)
    return lambda t: np.interp(t, x, y, left=0, right=1)


def ad_test_two_sample(baseline_scores, article_scores):

    combined_sample = np.concatenate([baseline_scores, article_scores])

    # empirical cumulative distribution functions (ECDF)
    ecdf1 = ecdf(baseline_scores)
    ecdf2 = ecdf(article_scores)

    # Compute the Anderson-Darling statistic
    n1 = len(baseline_scores)
    n2 = len(article_scores)
    n = n1 * n2 / (n1 + n2)
    
    # Calculate ECDF values for the combined sample
    ecdf1_values = ecdf1(combined_sample)
    ecdf2_values = ecdf2(combined_sample)
    
    ad_statistic = n * np.sum((ecdf1_values - ecdf2_values)**2)

    # Compare the statistic to critical values or return it
    # Note: Critical values for the two-sample test are not standard and would normally require simulation or a lookup table
    return -1 if ad_statistic > 0.5 else 1

def t_test(baseline_scores, article_scores):
    _, p_value = stats.ttest_ind(baseline_scores, article_scores, equal_var=True)
    #return p_value
    return -1 if p_value < 0.05 else 1