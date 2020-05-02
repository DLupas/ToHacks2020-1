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
wordsByLabel = [] #same as possible words but grouped by label
labelsWithDuplicates = []

for response in data['Responses']:
    finalLabels.append(response['tag'])
    #gather words 
    for pattern in response['patterns']:
        words = nltk.word_tokenize(pattern)
        words = [stemmer.stem(word.lower()) for word in words if word not in [",", ".", "?"]] #reduces the words
        possibleWords.extend(words)
        wordsByLabel.append(words)
        labelsWithDuplicates.append(response['tag'])

possibleWords = sorted(list(set(possibleWords))) #remove duplicates
finalLabels = sorted(finalLabels)

trainingSets = []
expectedOutput = []

for pos, wordsInLabel in enumerate(wordsByLabel):
    bagOfWords = []

    for word in possibleWords:
        doesAppear = 1 if word in wordsInLabel else 0 #creates a list with a 1 if the word appears out of the total list
        bagOfWords.append(doesAppear)

    trainingSets.append(bagOfWords)

    outputRow = [0] * len(finalLabels) #There will only be a 1 per list, others are 0
    correctTag = labelsWithDuplicates[pos] #sets which label corresponds to each set of words
    outputRow[finalLabels.index(correctTag)] = 1
    expectedOutput.append(outputRow)

trainingSets = numpy.array(trainingSets)
expectedOutput = numpy.array(expectedOutput)

#actual tensorflow now

tf.reset_default_graph()

net = tflearn.input_data(shape=[None, len(trainingSets[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(expectedOutput[0]))
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(trainingSets, expectedOutput, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")