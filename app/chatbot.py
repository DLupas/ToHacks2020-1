import numpy
import random
import json
import pickle
import nltk
import tensorflow as tf
import tflearn
from nltk import word_tokenize,sent_tokenize
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()



def calcBag(sentence, possibleWords):
    bag = [0] * len(possibleWords)

    sentence = nltk.word_tokenize(sentence)
    sentence = [stemmer.stem(word.lower()) for word in sentence if word not in [",", ".", "?"]] #reduces the words

    for word in sentence:
        for pos, wordFromList in enumerate(possibleWords):
            if wordFromList == word:
                bag[pos] = 1
    
    return numpy.array(bag)

def returnPhrase(userInput):
    with open ("Cases.json") as file:
        data = json.load(file)

    try:
        with open("Models/modeldata.pickle", "rb") as savedFile:
            possibleWords, finalLabels, trainingSets, expectedOutput = pickle.load(savedFile)
    except:
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

        with open("Models/modeldata.pickle", "wb") as savedFile:
            pickle.dump((possibleWords, finalLabels, trainingSets, expectedOutput), savedFile)

    #actual tensorflow now

    tf.reset_default_graph()

    net = tflearn.input_data(shape=[None, len(trainingSets[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(expectedOutput[0]), activation="softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net)

    try:
        model.load("Models/model.tflearn")
    except:
        model.fit(trainingSets, expectedOutput, n_epoch=1000, batch_size=8, show_metric=True)
        model.save("Models/model.tflearn")

    likelyhoods = model.predict([calcBag(userInput, possibleWords)])
    labelIndex = numpy.argmax(likelyhoods)
    label = finalLabels[labelIndex]

    responses = []
    for section in data['Responses']:
        if section['tag'] == label:
            responses = section['responses']
    return random.choice(responses)

'''
while True:
    userInput = input("You: ")
    likelyhoods = model.predict([calcBag(userInput, possibleWords)])
    labelIndex = numpy.argmax(likelyhoods)
    label = finalLabels[labelIndex]

    responses = []
    for section in data['Responses']:
        if section['tag'] == label:
            responses = section['responses']

    print(random.choice(responses))
'''