# Libraries import
import spacy
from collections import Counter
from nltk import pos_tag
from nltk.data import load
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from textblob import TextBlob
from pycorenlp import StanfordCoreNLP
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import en_core_web_sm
from nltk.tokenize.treebank import TreebankWordDetokenizer as Detok
import pickle
import numpy as np


# count-word feature extraction
def get_CountVector3(all_data, train_data, test_data):
    # Vectorizer
    count_vect = CountVectorizer()
    count_vect = count_vect.fit(all_data)
    x_train_data = count_vect.transform(train_data)
    x_test_data = count_vect.transform(test_data)
    return x_train_data, x_test_data

# count-word feature extraction using all dataset
def get_CountVector1(all_data):
    count_vect = CountVectorizer()
    count_vect = count_vect.fit(all_data)
    return count_vect.transform(all_data)


# removing Stop-words NLTK
def remove_NLTK_stop3(all_data, train_data, test_data):
    sw = stopwords.words('english')
    deto = Detok()

    all_cleaned = list()
    train_cleaned = list()
    test_cleaned = list()
    # word tokenization in all data
    for article in all_data:
        word_tokens = word_tokenize(article)
        all_cleaned.append(deto.detokenize([w for w in word_tokens if not w in sw]))
    # word tokenization in train data
    for article in train_data:
        word_tokens = word_tokenize(article)
        train_cleaned.append(deto.detokenize([w for w in word_tokens if not w in sw]))
    # word tokenization in test data
    for article in test_data:
        word_tokens = word_tokenize(article)
        test_cleaned.append(deto.detokenize([w for w in word_tokens if not w in sw]))
    return all_cleaned, train_cleaned, test_cleaned

# removing Stop-words NLTK
def remove_NLTK_stop1(all_data):
    sw = stopwords.words('english')
    deto = Detok()

    all_cleaned = list()

    for article in all_data:
        word_tokens = word_tokenize(article)
        all_cleaned.append(deto.detokenize([w for w in word_tokens if not w in sw]))

    return all_cleaned


# removing Stop-words SpaCy using all, train, test data
def remove_spaCy_stop3(all_data, train_data, test_data):
    sw = spacy.lang.en.stop_words.STOP_WORDS
    deto = Detok()
    all_cleaned = list()
    train_cleaned = list()
    test_cleaned = list()

    for article in all_data:
        word_tokens = word_tokenize(article)
        all_cleaned.append(deto.detokenize([w for w in word_tokens if not w in sw]))

    for article in train_data:
        word_tokens = word_tokenize(article)
        train_cleaned.append(deto.detokenize([w for w in word_tokens if not w in sw]))

    for article in test_data:
        word_tokens = word_tokenize(article)
        test_cleaned.append(deto.detokenize([w for w in word_tokens if not w in sw]))

    return all_cleaned, train_cleaned, test_cleaned

# removing Stop-words SpaCy using all data
def remove_spaCy_stop1(all_data):
    spacy_nlp = en_core_web_sm.load()
    sw = spacy.lang.en.stop_words.STOP_WORDS
    deto = Detok()

    all_cleaned = list()

    for article in all_data:
        word_tokens = word_tokenize(article)
        all_cleaned.append(deto.detokenize([w for w in word_tokens if not w in sw]))
    return all_cleaned


# Stop-word Counter NLTK
def get_CountVector_NLTK_Stop3(all_data, train_data, test_data):
    sw = stopwords.words('english')
    # Vectorizer
    count_vect = CountVectorizer(stop_words=sw)
    count_vect = count_vect.fit(all_data)
    x_train_data = count_vect.transform(train_data)
    x_test_data = count_vect.transform(test_data)
    return x_train_data, x_test_data


# Stop-word Counter SpaCy
def get_CountVector_spaCy_Stop3(all_data, train_data, test_data):
    spacy_nlp = en_core_web_sm.load()
    spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

    count_vect = CountVectorizer(stop_words=spacy_stopwords)
    count_vect = count_vect.fit(all_data)
    x_train_data = count_vect.transform(train_data)
    x_test_data = count_vect.transform(test_data)
    return x_train_data, x_test_data


# count-ngram feature extraction
def get_CountVector_Ngram3(all_data, train_data, test_data):
    count_vect = CountVectorizer(ngram_range=(2,3))
    count_vect = count_vect.fit(all_data)
    with open("Vectorizer_cv.pickle", 'wb') as dump_var:
        pickle.dump(count_vect, dump_var)
    x_train_data = count_vect.transform(train_data)
    x_test_data = count_vect.transform(test_data)
    return x_train_data, x_test_data

# count-ngram feature extraction using all data
def get_CountVector_Ngram1(all_data):
    count_vect = CountVectorizer(ngram_range=(2,3))
    count_vect = count_vect.fit(all_data)
    return count_vect.transform(all_data)


# TFIDF-word feature extraction
def get_TFIDF_Word3(all_data, train_data, test_data):
    # Vectorizer
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(all_data)
    x_train_data = tfidf_vect.transform(train_data)
    x_test_data = tfidf_vect.transform(test_data)
    return x_train_data, x_test_data

# TFIDF-word feature extraction using all data
def get_TFIDF_Word1(all_data):
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(all_data)
    return tfidf_vect.transform(all_data)


# TFIDF-ngram feature extraction
def get_TFIDF_NGram3(all_data, train_data, test_data):
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
    tfidf_vect.fit(all_data)
    x_train_data = tfidf_vect.transform(train_data)
    x_test_data = tfidf_vect.transform(test_data)
    return x_train_data, x_test_data


# TFIDF-ngram feature extraction
def get_TFIDF_NGram1(all_data):
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}',
    ngram_range=(2,3), max_features=5000)
    tfidf_vect.fit(all_data)
    return tfidf_vect.transform(all_data)


# VADER feature extraction
def get_VADER_score(data_list):
    # Vectorizer
    analyser = SentimentIntensityAnalyzer()
    ret_list = list()
    for data in data_list:
        ret_list.append(list(analyser.polarity_scores(data).values()))
    return ret_list

# normalization of VADER score
def make_VADER_score_non_neg(article_list):
    ret_list = list()
    for article_vals in article_list:
        ret_list.append([x+1 for x in article_vals])
    return ret_list


# Part-of-speech NLTK using all data
def get_PoS_NLTK(all_data):
    # Turn all_data into PoS
    all_pos = list()
    for article in all_data:
        all_pos.append(pos_tag(word_tokenize(article)))

# Create a counter for all_pos
    all_pos_counter = list()
    for article in all_pos:
        all_pos_counter.append(Counter( tag for word, tag in article))

    all_pos_count = list()
    tagdict = load('help/tagsets/upenn_tagset.pickle')
    # Count up each PoS and giving a value of 0 to those that do not occur
    for counter in all_pos_counter:
        temp = list()
        for key in tagdict:
            temp.append(counter[key])
        all_pos_count.append(temp)

    return all_pos_count


# Part-of-speech coreNLP
def get_PoS_coreNLP(all_data):
    # Turn all_data into PoS
    # set up of a coreNLP live server
    nlp = StanfordCoreNLP('http://localhost:9000')
    all_pos = list()
    for article in all_data:
        result = nlp.annotate(str(article),
                            properties={
                                'annotators': 'pos',
                                'outputFormat': 'json',
                                'timeout': 1000,
                            })

        pos = list()
        all_pos_counter = list()
        for i in range(0, len(result["sentences"])):
            for word in result["sentences"][i]["tokens"]:
                temp = (word["word"], word["pos"])
                pos.append(temp)
                # " ".join(pos)
        all_pos.append(pos)

    all_pos_counter = list()
    for article in all_pos:
        all_pos_counter.append(Counter(tag for word, tag in article))
    tagdict = load('help/tagsets/upenn_tagset.pickle')
    # Count up each PoS and giving a value of 0 to those that do not occur
    all_pos_count = list()
    for counter in all_pos_counter:
        temp = list()
        for key in tagdict:
            temp.append(counter[key])
        all_pos_count.append(temp)

    return all_pos_count


# Part-of-speech SpaCy
def get_PoS_spaCy(all_data):
    tagdict = load('help/tagsets/upenn_tagset.pickle')
    nlp = en_core_web_sm.load()

    all_list = list()

    # get pos
    for article in all_data:
        nlpa = nlp(article)
        all_list.append(Counter([(X.pos) for X in nlpa]))

    all_list_counts = list()

    for counter in all_list:
        temp = list()
        for pos in tagdict:
            temp.append(counter[pos])
        all_list_counts.append(temp)

    return all_list_counts


# Named Entity Recognition SpaCy
def get_ER_spaCy(all_data):
    named_entity_list = ("PERSON", "NORP", "FAC", "ORG", "GPE", "LOC",
                         "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE",
                         "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY",
                         "ORDINAL", "CARDINAL")
    nlp = en_core_web_sm.load()

    all_list = list()
    nlpas = list()
    # get entites
    for article in all_data:
        nlpa = nlp(article)
        all_list.append(Counter([(X.label_) for X in nlpa.ents]))

    all_list_counts = list()
    # Count-vectorize the entity tags
    for counter in all_list:
        temp = list()
        for entity in named_entity_list:
            temp.append(counter[entity])
        all_list_counts.append(temp)
    return all_list_counts


# Named Entity Recognition Specific Entities SpaCy
def get_SPECIFIC_spaCy(all_data):
    nlp = en_core_web_sm.load()
    all_list = list()
    all_list_per = list()
    all_list_org = list()
    all_list_gpe = list()
    all_list_date = list()
    all_list_percent = list()
    # get entities
    for article in all_data:
        nlpa = nlp(str(article))
        all_list.append(Counter([(X.label_) for X in nlpa.ents]))
    # Count-Vectorize the specific tags
    for counter in all_list:
        print(counter)
        temp = list()
        temp.append(counter["PERSON"])
        all_list_per.append(temp)
        temp.clear()
        temp.append(counter["ORG"])
        all_list_org.append(temp)
        temp.clear()
        temp.append(counter["GPE"])
        all_list_gpe.append(temp)
        temp.clear()
        temp.append(counter["DATE"])
        all_list_date.append(temp)
        temp.clear()
        temp.append(counter["PERCENT"])
        all_list_percent.append(temp)
    return all_list_per, all_list_org, all_list_gpe, all_list_date, all_list_percent


