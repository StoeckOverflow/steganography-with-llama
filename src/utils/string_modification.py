import string
import numpy as np

def clean(text):
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = text.replace("\r", " ")
    text = text.replace("-", " ")
    return text

def count_syllables(word):
    if not word or not word.strip(): 
        return 0

    vowels = 'aeiouy'
    num_syllables = 0
    word = word.lower().strip(".:;?!")
    
    if len(word) > 0 and word[0] in vowels:
        num_syllables += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            num_syllables += 1
    if word.endswith('e'):
        num_syllables -= 1
    if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
        num_syllables += 1
    if num_syllables == 0:
        num_syllables = 1
    return num_syllables

def get_length(text):
       return len(text)

def get_entropy(text):
    entropy = 0
    for c in text:
        p = text.count(c) / len(text)
        if p > 0:
            entropy += -p * np.log2(p)
    return entropy

def get_alpha(text):
    return sum(c.isalpha() for c in text)

def get_punct(text):
    return sum(c in string.punctuation for c in text)