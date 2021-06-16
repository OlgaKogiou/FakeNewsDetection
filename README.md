# FakeNewsDetection

## For Fake.csv: download form here https://drive.google.com/file/d/1Zmjp3gWLm6P2yLYtK0J25WFwlY92YZQm/view?usp=sharing and put in the same folder as the rest of the files

## For True.csv: https://drive.google.com/file/d/1XX7QaAx84yZe0-iQcX51qDvCjNDy34Bc/view?usp=sharing download form here  and put in the same folder as the rest of the files

### Author: Olga Kogiou
### MACHINE LEARNING SURVEY
A simple train-test model using:
dataset: Fake and Real news datasets from Kaggle
classifiers: Random Forest Classifier, Naive Bayes Classifier, Support Vector machine
The datasets used consists of 3 columns: text, date, subject. When concatenating the two datasets in one pandas dataframe we add a new column: label to distinguish fake and real news.
We only need the text column for this analysis. Our goal is to find the best characteristic out of all and to compare how useful NER is.

We first clean all articles from characters that are useless. Then we extract all characteristics. Only by using the characteristics can we train the classifiers because our classifiers cannot process plain text. Therefore, we extract and compare the classification accuracy that every characteristic has to each classifier and then use the pickling technique to store the train data.

### Confusion matrices of a characteristic can be analysed as an example of the way the algorithms work.
### Python's sklearn is used to provide us with the fixed algorithms so do not build them from scratch.

We extract 14 featues in total. Libraries used: Spacy, NLTK, VADER, sklearn, coreNLP
Our 3 goals are: 
- To find the best feature for data representation
- To find which of the libraries: NLTK, coreNLP, SpaCy is the best for POS tagging and NER
- To extarct .pickle files of models of the best feature.
