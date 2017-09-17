import os
import glob
import numpy
import random
import re
import nltk
from nltk.stem.lancaster import LancasterStemmer

class NaiveBayesClassifier:
    # There are three types of Naive Bayes Classifier Model 
    # MULTINOMINAL - Represents a frequency of word in the document (vector of integer values 1 to N)
    # BERNOULLI - Represents absence/presence of word in the document (vector of binary values 0 or 1)
    # This class use BERNOULLI to represent each document at this point,
    # but it can be extended to work with MULTINOMINAL

    # Note : This script needs placed in the same folder as the data set folder, say Articles. 
    # Inside Articles, its expected to have multiple folder each represents a particular class Ci.
    # Each Ci folder will have set of text files which will be modeled into vector of size of M+1.
    # Where M is the size of Vocabulary and M+1th slot is the name of the actual class. 
    # Apart from the data set, the current folder should also have the file with list of stop words.

    # Text classification using categorical attribute. It doesn't work with continuous valued attributes.  
    
    GAUSSIAN, MULTINOMINAL, BERNOULLI = 1, 2, 3
    
    def __init__(self, classifierType, splitRatio, rootFolder, classFolders, classLabels, stopWordsFile, labelExpansion):
        print("Naive Bayes Classifier is initiated")
        if classifierType == NaiveBayesClassifier.BERNOULLI :
            # STEP 1 : Read whole dataset and split it to training data and test data
            D = self.readData(rootFolder, classFolders)
            '''for i in range(len(D)):
                print("{} documents read from {}".format(len(D[i]),classFolders[i]))
            print("Spliting D into training and test data with split of {} %".format(splitRatio*100))'''
            
            trainingData, testData = self.splitData(splitRatio, D)
            trainingDataSize = testDataSize = 0
            for i in range(len(trainingData)):
                #print("Training Set : {} documents randomly chosen from {}".format(len(trainingData[i]),classFolders[i]))
                trainingDataSize += len(trainingData[i])
            for i in range(len(testData)):
                #print("Tesing Set : Remaining {} documents from {}".format(len(testData[i]),classFolders[i]))
                testDataSize += len(testData[i])

            # STEP 2 : Construct a Vocabulary with stop words removed from training data
            vocabulary = self.constructVocubulary(trainingData, stopWordsFile)
            #print(vocabulary)
            #print("Size of vocabulary = {}".format(len(vocabulary)))

            # STEP 3 : Convert training data into Bernoulli feature vectors 
            training_bernoulliFeatureVectorMatrix = self.convertToFeatureVectorMatrix(trainingData, vocabulary, trainingDataSize, classLabels)
            #print(training_bernoulliFeatureVectorMatrix[0])
            print("Pre-processing steps done!")
            print("Making predictions and computing accuracy... ")
            print("Wait for about a minute to see the final output!!")

            # STEP 4 : Construct the parameters used by the classifier from training data
            class_conditional_probability = self.computeWordLikelihoods(training_bernoulliFeatureVectorMatrix, classLabels)
            prior_probability = self.computeWordPriorProbability(training_bernoulliFeatureVectorMatrix, classLabels)
            # Note : Evidence P(X) can be ignored since it's a constant

            # STEP 5 : Form feature matrix for testing data and classify the label for each article in the testing data
            # Print the results (Actual and classified labels and accuracy for each class)
            testing_bernoulliFeatureVectorMatrix = self.convertToFeatureVectorMatrix(testData, vocabulary, testDataSize, classLabels, mode='test')
            print(len(testing_bernoulliFeatureVectorMatrix))
            self.makePredictions(testing_bernoulliFeatureVectorMatrix, training_bernoulliFeatureVectorMatrix, class_conditional_probability, prior_probability, labelExpansion)
            
        else :
            print("Other classifier types are not supported as of now!! Please try BERNOULLI:3 !!")

    def readData(self, rootFolder, classFolders):
        # Temporay storage for holding data from each file
        # represented as list of numpy arrays, each represents set of words in single document 
        dsfile_data_temp = []

        # Holds the data collected from all the files from all class folders 
        # represented as list of list of numpy arrays.. Each class folder is represented as an numpy array
        D = []

        # Read everything inside the directory 'rootFolder/classFolders/*.txt' (uncompressed dataset)
        for i in range(len(classFolders)):
            filepath = rootFolder+"/"+classFolders[i]+"/*.txt"
            path_txt_all = os.path.join(os.path.dirname(__file__), filepath)
            for path_txt in glob.glob(path_txt_all):
                #print("reading "+path_txt)
                with open(path_txt,errors='ignore') as f:
                    dsfile_data_temp.append(numpy.array([w.lower() for l in f for w in l.split()]))
            #print("appending {} items to D".format(len(dsfile_data_temp)))
            D.append(numpy.array(dsfile_data_temp))
            # Initialize it to process the documents from next class folder
            dsfile_data_temp = []
        return D
    
    def splitData(self, splitRatio, D):        
        DTrain = []
        DTest = []
        Dtemp = []
        for i in range(len(D)):
            # Randomly chose documents indices
            indices = numpy.random.choice(numpy.arange(D[i].shape[0]), int(D[i].shape[0]*splitRatio), replace=False)
            # Add all the random documents to the training set
            for j in indices:
                Dtemp.append(D[i][j])
            DTrain.append(Dtemp)
            Dtemp = []
            # Add the remaining documents to the test data set
            DTest.append(numpy.delete(D[i],indices,0))
        return DTrain,DTest

    def constructVocubulary(self, trainingData, stopWordsFile):
        #print("Constructing the vocabulary from training data... ")
        vocabulary = {}
        V = []
        filepath = os.path.join(os.path.dirname(__file__),stopWordsFile)
        stopWords = []
        with open(filepath) as f:
            for line in f:
                stopWords.append(line.strip('\n'))
        #print(stopWords)
        ls = LancasterStemmer()
        #ps = PorterStemmer()
        # Remove stop words and junk characters
        for td in trainingData:
            for arr in td:
                # Frequency of word should be increased only once per document
                # though it occurs multiple times in a document. To achieve this, 
                # convert a list of words to set of unique words so that we can avoid
                # counting the same word multiple times per document
                unique = set(arr)
                for element in unique:
                    # Apply stemming and lematization before counting
                    # To consider only yield and its count rather than
                    # taking counting yield, yielded, yielding and yields. 
                    w = ls.stem(element)
                    if w not in stopWords and re.match("^[a-zA-Z_-]*$", w):
                        #print("ignoring stopword {}".format(element))
                        if w in vocabulary:
                            vocabulary[w] += 1
                        else: 
                            vocabulary[w] = 1
        # Remove the words that occur only once
        for key in vocabulary:
            if vocabulary[key] > 2:
                V.append(key)
        # Sort the vocabulary in alphabetical order
        V.sort()
        return V
    
    def convertToFeatureVectorMatrix(self, trainingData, vocabulary, trainingDataSize, classLabels, mode='train'):
        #print("Converting the training data into bernoulli feature vectors... ")
        # Each row represents a feature vector for particular document in training data set
        M = len(vocabulary)
        #print("Size = {}".format(M))
        featureVectorMatrix = []
        
        # Initialize M+1 slots to 0 (for training data) and only (M slots for tesing data)
        for i in range(trainingDataSize):
            if(mode == 'train'):
                featureVectorMatrix.append([0]*(M+1))
            else:
                featureVectorMatrix.append([0]*(M))

        counter = 0
        ls = LancasterStemmer()
        for i in range(len(trainingData)):
            td = trainingData[i]
            classLabel = classLabels[i]
            for row in range(len(td)):
                for word in td[row]:
                    try:
                        # If the word is in vocabulary, change the value to 1 at correponding slot
                        j = vocabulary.index(ls.stem(word))
                        featureVectorMatrix[counter][j] = 1
                    except ValueError:
                        # Word is not there. Don't do anything. All slots have 0 by default
                        pass
                # Put actual class label in M+1th slot (only for training data)
                if(mode == 'train'):
                    featureVectorMatrix[counter][M] = classLabel
                counter+=1
        return featureVectorMatrix

    # Compute P(X|Ci) for each article where X = {x1,x2,...xM} for each class Ci
    def computeWordLikelihoods(self, bernoulliFeatureVectorMatrix, classLabels):
        wordlikelihoods = {}
        for cl in classLabels:
            wordlikelihoods[cl] = [0]*(len(bernoulliFeatureVectorMatrix[0])-1)
        # Count the number of occurrences of each word (counting number of 1s in each col)
        for row in bernoulliFeatureVectorMatrix:
            classLabel = row[-1]
            for i in range(len(row)):
                if row[i] == 1:
                    wordlikelihoods[classLabel][i] += 1
        # Divide each class's individual word sample by total number of samples for that class
        # We have 3 classes with 150 samples in each
        totalCount = len(bernoulliFeatureVectorMatrix)/len(classLabels)
        for cl in wordlikelihoods:
            wordlikelihoods[cl] = [count/totalCount for count in wordlikelihoods[cl]]
            #print("{} : {}".format(cl, wordlikelihoods[cl][:12]))
        return wordlikelihoods
    
    # Compute P(Ci) for each class Ci, in our case 150/450 for all classes
    def computeWordPriorProbability(self, bernoulliFeatureVectorMatrix, classLabels):
        prior_probability = {}
        classSampleCount = len(bernoulliFeatureVectorMatrix)/3
        for cl in classLabels:
            prior_probability[cl] = classSampleCount/len(bernoulliFeatureVectorMatrix)
            #print("{} : {}".format(cl, prior_probability[cl]))
        return prior_probability
        
    def makePredictions(self, testing_bernoulliFeatureVectorMatrix, training_bernoulliFeatureVectorMatrix, class_conditional_probability, prior_probability, labelExpansion):
        predictions = []

        # Number of correct predictions for each class
        correct_predictions = {}
        for cl in labelExpansion:
            correct_predictions[cl] = 0

        # 450/3 = 150 
        total_predictions_per_class = len(testing_bernoulliFeatureVectorMatrix)/len(prior_probability)
        print("TP : "+str(total_predictions_per_class))

        for index in range(len(testing_bernoulliFeatureVectorMatrix)):
            row = testing_bernoulliFeatureVectorMatrix[index]
            actualClassLabel = training_bernoulliFeatureVectorMatrix[index][-1]
            prediction = None
            maxValue = None
            # Predict a class for current row/article
            for key in class_conditional_probability:
                condtional_probability = class_conditional_probability[key]
                prior_probability_value = prior_probability[key]
                posterior_probability = 1
                for i in range(len(row)):
                    if row[i]==1:
                        val = condtional_probability[i]
                        if(val!=0):
                            posterior_probability *= val
                    else:
                        val = (1-condtional_probability[i])
                        if(val!=0):
                            posterior_probability *= val
                posterior_probability *= prior_probability_value
                # Classify based on whichever has the maximum value 
                if maxValue==None or posterior_probability > maxValue:
                    maxValue = posterior_probability
                    prediction = key
            # Add the prediction for this article to the list of predictions
            predictions.append(prediction)
            print("-----------------------------------------")
            print("Actual Class : {}".format(labelExpansion[actualClassLabel]))
            print("Classified Class : {}".format(labelExpansion[prediction]))
            print("-----------------------------------------")
            if(prediction == actualClassLabel):
                correct_predictions[actualClassLabel] += 1
        
        print(correct_predictions)
        # Print the accuracy for each class
        for cl in labelExpansion:
            print('Accuracy for {} = {}%'.format(labelExpansion[cl], (correct_predictions[cl]/total_predictions_per_class)*100))

# Invoke the alogirthm with input details

classifierType = NaiveBayesClassifier.BERNOULLI # BERNOULLI
splitRatio = 0.5 # 50%
rootFolder = "articles" # The name of the folder 
classFolders = ['arxiv','jdm','plos'] # The name of all sub folders
classLabels = ["A", "J", "P"] # Corresponding class label for each sub folder
stopWordsFile = "stoplist.txt" # The name of stop words file in the current directory
labelExpansion = {
    'A' : 'ARXIV',
    'J' : 'JDM',
    'P' : 'PLOS'
}
bernoulliNB = NaiveBayesClassifier(classifierType,splitRatio,rootFolder,classFolders,classLabels,stopWordsFile,labelExpansion)