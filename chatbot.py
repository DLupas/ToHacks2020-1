import nltk
nltk.download('punkt')
from nltk import word_tokenize,sent_tokenize
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tflearn
import tensorflow as tf
import random
import json

with open ("Tensorflow Cases.json") as file:
    data = json.load(file)

possibleWords = []
finalLabels = []

for response in data['Responses']:
    finalLabels.append(response['tag'])
    #gather words 
    for pattern in response['patterns']:
        words = nltk.word_tokenize(pattern)
        possibleWords.extend(words)

possibleWords = [stemmer.stem(word.lower()) for word in possibleWords] #reduces the words
possibleWords = sorted(list(set(possibleWords))) #remove duplicates
finalLabels = sorted(finalLabels)

print(possibleWords, finalLabels)

