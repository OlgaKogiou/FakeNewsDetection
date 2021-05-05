"""
Author: Olga Kogiou
MACHINE LEARNING SURVEY
A simple train-test model using:
dataset: Fake and Real news datasets from Kaggle
classifiers: Random Forest Classifier, Naive Bayes Classifier, Support Vector machine

The datasets used consists of 3 columns: text, date, subject. When concatenating the two datasets in one pandas
dataframe we add a new column: label to distinguish fake and real news.

We only need the text column for this analysis. Our goal is to find the best characteristic out of all and to compare
how useful NER is.

We first clean all articles from characters that are useless. Then we extract all characteristics. Only by
using the characteristics can we train the classifiers because our classifiers cannot process plain text. Therefore,
we extract and compare the classification accuracy that every characteristic has to each classifier and then use the
pickling technique to store the train data.

Confusion matrices of a characteristic can be analysed as an example of the way the algorithms work.

Python's sklearn is used to provide us with the fixed algorithms so do not build them from scratch.
"""

from FeatureExtraction import *
from DataFunctions import *
import pickle


# train-test wrapper function
def basic_tests(train_data, train_labels, validate_data, validate_labels):
    print("Random Forest Classifier:")
    clf_1 = train_random_forest(train_data, train_labels, 50)
    test_classifier(clf_1, validate_data, validate_labels)
    print("Naive Bayes Classifier:")
    clf_2 = train_NB(train_data, train_labels)
    test_classifier(clf_2, validate_data, validate_labels)
    print("Support Vector machine:")
    clf_3 = train_SVC(train_data, train_labels, 'linear', 1.0)
    test_classifier(clf_3, validate_data, validate_labels)
    return clf_1, clf_2, clf_3


def count_ngram_only(total_raw_data, raw_train_data, train_labels, raw_validate_data, validate_labels):
    print("======================")
    print("== Count_Ngram Only ==")
    print("======================")

    train_data, validate_data = get_CountVector_Ngram3(total_raw_data, raw_train_data, raw_validate_data)
    clf_1, clf_2, clf_3 = basic_tests(train_data, train_labels, validate_data, validate_labels)
    pickling(clf_1, "RandomForest.pickle")
    pickling(clf_2, "NaiveBayes.pickle")
    pickling(clf_3, "SupportVectorMachine.pickle")


def count_word_only(total_raw_data, raw_train_data, train_labels, raw_validate_data, validate_labels):
    print("======================")
    print("== Count_Word Only ==")
    print("======================")
    train_data, validate_data = get_CountVector3(total_raw_data, raw_train_data, raw_validate_data)
    basic_tests(train_data, train_labels, validate_data, validate_labels)


def tfidf_word_only(total_raw_data, raw_train_data, train_labels, raw_validate_data, validate_labels):
    print("======================")
    print("== TFIDF_Ngram Only ==")
    print("======================")
    train_data, validate_data = get_TFIDF_Word3(total_raw_data, raw_train_data, raw_validate_data)
    basic_tests(train_data, train_labels, validate_data, validate_labels)


def tfidf_ngram_only(total_raw_data, raw_train_data, train_labels, raw_validate_data, validate_labels):
    print("======================")
    print("== TFIDF_Ngram Only ==")
    print("======================")
    train_data, validate_data = get_TFIDF_NGram3(total_raw_data, raw_train_data, raw_validate_data)
    basic_tests(train_data, train_labels, validate_data, validate_labels)


def spaCy_removed_Count(total_raw_data, labels):
    print("===========================")
    print("== spaCy_removed + Count ==")
    print("===========================")

    raw_data_stop = remove_spaCy_stop1(total_raw_data)
    raw_data = get_CountVector_Ngram1(raw_data_stop)
    train_data, validate_data, train_labels, validate_labels = split_data(raw_data, labels)
    basic_tests(train_data, train_labels, validate_data, validate_labels)


def NLTK_removed_Count(total_raw_data, labels):
    print("===========================")
    print("== NLTK_removed + Count ==")
    print("===========================")

    raw_data_stop = remove_NLTK_stop1(total_raw_data)
    raw_data = get_CountVector_Ngram1(raw_data_stop)
    train_data, validate_data, train_labels, validate_labels = split_data(raw_data, labels)
    basic_tests(train_data, train_labels, validate_data, validate_labels)


def VADER_only(raw_train_data, train_labels, raw_validate_data, validate_labels):
    print("===========================")
    print("====    VADER_only     ====")
    print("===========================")
    train_data = make_VADER_score_non_neg(get_VADER_score(raw_train_data))
    validate_data = make_VADER_score_non_neg(get_VADER_score(raw_validate_data))
    basic_tests(train_data, train_labels, validate_data, validate_labels)


def ER_only(raw_train_data, train_labels, raw_validate_data, validate_labels):
    print("===========================")
    print("======    ER_only     =====")
    print("===========================")
    train_data = get_ER_spaCy(raw_train_data)
    validate_data = get_ER_spaCy(raw_validate_data)
    basic_tests(train_data, train_labels, validate_data, validate_labels)


def PoS_only(raw_train_data, train_labels, raw_validate_data, validate_labels):
    print("===========================")
    print("=====    PoS_only     =====")
    print("===========================")
    print("SpaCy")
    train_data = get_PoS_spaCy(raw_train_data)
    validate_data = get_PoS_spaCy(raw_validate_data)
    basic_tests(train_data, train_labels, validate_data, validate_labels)
    print("NLTK")
    train_data = get_PoS_NLTK(raw_train_data)
    validate_data = get_PoS_NLTK(raw_validate_data)
    basic_tests(train_data, train_labels, validate_data, validate_labels)


def PoS_coreNLP_only(raw_train_data, train_labels, raw_validate_data, validate_labels):
    print("===========================")
    print("======PoS_coreNLP_only=====")
    print("===========================")

    train_data = get_PoS_coreNLP(raw_train_data)
    validate_data = get_PoS_coreNLP(raw_validate_data)
    basic_tests(train_data, train_labels, validate_data, validate_labels)


def ER_SPECIFIC_only(raw_train_data, train_labels, raw_validate_data, validate_labels):

    train_data_person, train_data_org, train_data_gpe, train_data_date, train_data_percent = get_SPECIFIC_spaCy(
        raw_train_data)
    validate_data_person, validate_data_org, validate_data_gpe, validate_data_date, validate_data_percent = \
        get_SPECIFIC_spaCy(raw_validate_data)
    print("===========================")
    print("======ER_PERSON_only=====")
    print("===========================")
    basic_tests(train_data_person, train_labels, validate_data_person, validate_labels)
    print("===========================")
    print("======ER_ORG_only=====")
    print("===========================")
    basic_tests(train_data_org, train_labels, validate_data_org, validate_labels)
    print("===========================")
    print("======ER_GPE_only=====")
    print("===========================")
    basic_tests(train_data_gpe, train_labels, validate_data_gpe, validate_labels)
    print("===========================")
    print("======ER_DATE_only=====")
    print("===========================")
    basic_tests(train_data_date, train_labels, validate_data_date, validate_labels)
    print("===========================")
    print("======ER_PERCENT_only=====")
    print("===========================")
    basic_tests(train_data_percent, train_labels, validate_data_percent, validate_labels)


# pickling function for train data
def pickling(clf, name_file):
    with open(name_file, 'wb') as dump_var:
        pickle.dump(clf, dump_var)

    # Load the Pickle fule in the memory
    pickle_in = open(name_file, 'rb')
    pickle_clf = pickle.load(pickle_in)


def main():
    fake, true = read_datasets()
    data = data_cleaning(fake, true)
    # Using smaller dataset for pickling so that our app is quicker
    new_data = data.iloc[:10000, :]

    raw_train_data, raw_validate_data, train_labels, validate_labels = split_data(new_data['text'], new_data['label'])

    # 1st Feature
    count_ngram_only(new_data['text'], raw_train_data, train_labels, raw_validate_data, validate_labels)
    raw_train_data, raw_validate_data, train_labels, validate_labels = split_data(data['text'], data['label'])
    # 2nd Feature
    count_word_only(data['text'], raw_train_data, train_labels, raw_validate_data, validate_labels)
    # 3rd Feature
    NLTK_removed_Count(data['text'], data['label'])
    # 4th Feature
    spaCy_removed_Count(data['text'], data['label'])
    # 5th Feature
    VADER_only(raw_train_data, train_labels, raw_validate_data, validate_labels)
    # 6th Feature
    tfidf_ngram_only(data['text'], raw_train_data, train_labels, raw_validate_data, validate_labels)
    # 7th Feature
    tfidf_word_only(data['text'], raw_train_data, train_labels, raw_validate_data, validate_labels)

    raw_train_data, raw_validate_data, train_labels, validate_labels = train_test_split(data['text'], data['label'],
                                                                                        test_size=0.3, random_state=42,
                                                                                        shuffle='True')
    # 8th Feature
    PoS_only(raw_train_data, train_labels, raw_validate_data, validate_labels)
    # 9th Feature
    PoS_coreNLP_only(raw_train_data, train_labels, raw_validate_data, validate_labels)
    # 10th Feature
    ER_only(raw_train_data, train_labels, raw_validate_data, validate_labels)
    # 11th Feature
    ER_only(raw_train_data, train_labels, raw_validate_data, validate_labels)


if __name__ == '__main__':
    main()
