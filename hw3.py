__author__ = 'penghao'


from numpy import *
import numpy as np
from math import *
import csv
import sys
from collections import Counter

def cal_reviewlistlabel(filename):
    #read csv files
    f = open(filename)
    csv_f = csv.reader(f)
    review_list =[]
    review_label = []
    for row in csv_f:
        if row[0].split(' ') not in review_list and row[0].split(' ') != ',':
            review_list.append(row[0].split(' '))
        if row[1] is'+':    #if the label is '+' then it returns to one, else it returns to zero
            review_label.append(1)
        else :
            review_label.append(-1)
    return review_list, review_label

def calListuniWord(review_list):
    list_word=[]
    all_word=[]
    al_w=set()
    for i in range(review_list.__len__()):
        for column in review_list[i]:
            if column != ',' and column != '.' and column !=''and column !='!'and column !='...'and column !='(' and column !=')'and column !='"'and column !='--'and column !='?' \
                    and column !=';'and column !=':': #Get rid of the '.' and other unrelative symbol
                all_word.append(column)
                al_w.add(column)  #all_w is a set(), it will save the unique words

    list_word=list(al_w)
    #Delete the most common words and words that appears less than five time
    c= Counter(all_word).most_common(list_word.__len__())
    for elements in c:
        removestr = elements
        if removestr[1]<5 or removestr[1]>300: #The most words commmon words are "a, the, and" , whcih we don't need
            list_word.remove(removestr[0])
    return list_word

def calListbiWord(review_list): #This function create the bigram list of words
    list_word=[]
    all_word=[]
    all_word_copy=[]
    list_biword=[]
    al_w=set()
    for i in range(review_list.__len__()):
        for column in review_list[i]:
            if column != ',' and column != '.' and column !=''and column !='!'and column !='...'and column !='(' and column !=')'and column !='"'and column !='--'and column !='?' \
                    and column !=';'and column !=':' and review_list[i].index(column) < review_list[i].__len__()-1:
                all_word.append(column+ ' ' + review_list[i][review_list[i].index(column)+1])
                al_w.add(column+ ' ' + review_list[i][review_list[i].index(column)+1])

    list_word=list(al_w)

    #Delete the most common words and words that appears less than four time
    c= Counter(all_word).most_common(list_word.__len__())
    for elements in c:
        removestr = elements

        if  removestr[1]<4 or removestr[1]>100: #removestr[0] is the word removestr[1] is the count
            list_word.remove(removestr[0])

    return list_word

def cal_feature_array(review_list,list_word): #calculate feature array of unigram
    feature_array = np.zeros((review_list.__len__(),list_word.__len__()))
    #feature_array contain all x, x is features of every row
    for idx in range(review_list.__len__()):
        x = [0]*list_word.__len__()
        for words in review_list[idx]:
            if words in list_word:
                index =list_word.index(words)
                x[index] = 1  #if the words in the word list, then change the value in that position to 1
                feature_array[idx] = np.array(x,dtype=float)
    return feature_array

def cal_bifeature_array(review_list,list_word): # similar as cal_feature_array, this function calculate feature array of  bigram
    feature_array = np.zeros((review_list.__len__(),list_word.__len__()))
    #feature_array contain all x, x is features of every row
    for idx in range(review_list.__len__()):
        x = [0]*list_word.__len__()
        for words in review_list[idx]:
            if review_list[idx].index(words) < review_list[idx].__len__()-1:
                biwords = words + ' ' + review_list[idx][review_list[idx].index(words)+1]
                if biwords in list_word:
                    index =list_word.index(biwords)
                    x[index] = 1
                    feature_array[idx] = np.array(x,dtype=float)
    return feature_array

def cal_both_feature_array(review_list,list_word):# similar as cal_feature_array, this function calculate feature array of both of unigram and bigram
    feature_array = np.zeros((review_list.__len__(),list_word.__len__()))
    #feature_array contain all x, x is features of every row
    for idx in range(review_list.__len__()):
        x = [0]*list_word.__len__()
        for words in review_list[idx]:
            if review_list[idx].index(words) < review_list[idx].__len__()-1:
                biwords = words + ' ' + review_list[idx][review_list[idx].index(words)+1]
                if biwords in list_word :
                    index =list_word.index(biwords)
                    x[index] = 1
            if words in list_word:
                index =list_word.index(words)
                x[index] = 1
            feature_array[idx] = np.array(x,dtype=float)
    return feature_array

def perceptron(maxIter,review_list,review_label,list_word,feature_array): #Perceptron algorithm
    w = np.zeros(list_word.__len__())
    maxIter = maxIter
    b = 0

    for iteration in range(maxIter):
        w_copy = w
        random_idx = np.random.permutation(review_list.__len__())
        for row_idx in random_idx:
            y = predict_one_p(w,feature_array[row_idx]) # it is the same as y = np.dot(w,feature_array[row_idx])
            if y>0:
                predict_label = 1
            else:
                predict_label = -1
            review_l = review_label[row_idx]
            if review_l != predict_label:
                w = w + review_l* feature_array[row_idx] # predict label are not ture, then do the calculation.
        if np.array_equal(w_copy,w):

            print("converge")
            print(iteration)
            break

    return w


def winnow(maxIter,review_list,review_label,list_word,feature_array):  #winnow algorithm
    w = np.ones(list_word.__len__())
    maxIter = maxIter
    feature_list = []

    for iteration in range(maxIter):
        w_copy =w
        random_idx = np.random.permutation(review_list.__len__())
        for row_idx in random_idx:
            y = np.dot(w,feature_array[row_idx])

            if y>=w.__len__():
                predict_label = 1
            else:
                predict_label = 0
            review_l = review_label[row_idx]
            if predict_label == 0 and review_l == 1:
                #feature_list = feature_array[row_idx].tolist()
                indices = [i for i, x in enumerate(feature_array[row_idx]) if x == 1]
                w[indices] = w[indices]*2.0
            if predict_label == 1 and review_l == -1:
                feature_list = feature_array[row_idx].tolist
                indices = [i for i, x in enumerate(feature_array[row_idx]) if x == 1]
                w[indices] = w[indices]/2.0
        if np.array_equal(w_copy,w):
            print(iteration)
            break
    return w
#print(winnow(10))

def calPrecsion_p(filename,list_word,w,condition,type): #calculate precision
    vreviewlist,vreviewlabel = cal_reviewlistlabel(filename)
    if condition == 1: #1 means unigram
        vfeatureArray = cal_feature_array(vreviewlist,list_word)
    if condition == 2: #2 means bigram
        vfeatureArray = cal_bifeature_array(vreviewlist,list_word)
    if condition == 3: #3 means both
        vfeatureArray = cal_both_feature_array(vreviewlist,list_word)
    b=0
    turePositiveCount=0.0
    falsePositiveCount = 0.0
    if type == 1: #perceptron is type 1
        for idx in range(vreviewlist.__len__()):
            y = np.dot(w,vfeatureArray[idx]) + b
            if y>0:
                predict_label = 1
            else:
                predict_label = -1
            review_l = vreviewlabel[idx]
            if predict_label == 1:
                if review_l == 1:
                    turePositiveCount = turePositiveCount + 1
                if review_l == -1:
                    falsePositiveCount = falsePositiveCount +1
        precision = turePositiveCount/(turePositiveCount+falsePositiveCount) # calculate the precision
    if type == 2:
        for idx in range(vreviewlist.__len__()):
            y = np.dot(w,vfeatureArray[idx])

            if y>w.__len__():
                predict_label = 1
            else:
                predict_label = 0
            review_l = review_label[idx]
            if predict_label == 1:
                if review_l == 1:
                    turePositiveCount = turePositiveCount + 1
                if review_l == -1:
                    falsePositiveCount = falsePositiveCount +1
        precision = turePositiveCount/(turePositiveCount+falsePositiveCount) # calculate the precision

    return(precision)

def calRecall_p(filename,list_word,w,condition,type): #calculate recall
    vreviewlist,vreviewlabel = cal_reviewlistlabel(filename) # generate a new review list that contain different words
    if condition == 1: #1 means unigram
        vfeatureArray = cal_feature_array(vreviewlist,list_word)
    if condition == 2: #2 means bigram
        vfeatureArray = cal_bifeature_array(vreviewlist,list_word)
    if condition == 3: #3 means both
        vfeatureArray = cal_both_feature_array(vreviewlist,list_word)
    b=0
    turePositiveCount=0.0
    falseNegativeCount = 0.0
    if type == 1: #perceptron is type 1
        for idx in range(vreviewlist.__len__()):
            y = np.dot(w,vfeatureArray[idx]) + b
            if y>0:
                predict_label = 1
            else:
                predict_label = -1
            review_l = vreviewlabel[idx]
            if review_l == 1: # if the true table is 1
                if predict_label == 1:
                    turePositiveCount = turePositiveCount + 1
                if predict_label == -1:
                    falseNegativeCount = falseNegativeCount +1
        recall = turePositiveCount/(turePositiveCount+falseNegativeCount) # calculate the precision
    if type == 2: #winnow is type 2
        for idx in range(vreviewlist.__len__()):
            y = np.dot(w,vfeatureArray[idx])

            if y>w.__len__():
                predict_label = 1
            else:
                predict_label = 0
            review_l = review_label[idx]
            if review_l == 1: # if the true table is 1
                if predict_label == 1:
                    turePositiveCount = turePositiveCount + 1
                if predict_label == 0:
                    falseNegativeCount = falseNegativeCount +1
        recall = turePositiveCount/(turePositiveCount+falseNegativeCount) # calculate the precision

    return(recall)

def calFscore(precision,recall):
    return 2*(precision*recall)/(precision+recall)

def predict_one_p(w,input_snippet ): #it calculate the y, then in the perceptron, it define the sign.
    #input_snippet is the feature_array like :(0,0,0,0,1,0......)
    b = 0
    y = np.dot(w,input_snippet) + b #for perceptron
    return y

def calTrainError_p(filename,list_word,w,condition,type): #calculate accuracy
    vreviewlist,vreviewlabel = cal_reviewlistlabel(filename)
    if condition == 1: #1 means unigram
        vfeatureArray = cal_feature_array(vreviewlist,list_word)
    if condition == 2: #2 means bigram
        vfeatureArray = cal_bifeature_array(vreviewlist,list_word)
    if condition == 3: #3 means both
        vfeatureArray = cal_both_feature_array(vreviewlist,list_word)
    b=0
    count=0.0
    if type == 1: #perceptron is type 1
        for idx in range(vreviewlist.__len__()):
            y = predict_one_p(w,vfeatureArray[idx])
            if y>0:
                predict_label = 1
            else:
                predict_label = -1
            review_l = vreviewlabel[idx]
            if review_l != predict_label:
                count=count+1
    if type == 2:
        for idx in range(vreviewlist.__len__()):
            y = np.dot(w,vfeatureArray[idx])

            if y>=w.__len__():
                predict_label = 1
            else:
                predict_label = 0
            review_l = review_label[idx]

            if review_l ==-1 and predict_label ==1:
                count=count+1
            if review_l ==1 and predict_label ==0:
                count=count+1
    return((vreviewlist.__len__()-count)/vreviewlist.__len__())

def calaprf(w,filename,list_word,condition,type):

    print(calTrainError_p(filename,list_word,w,condition,type))
    precision_uni_train =calPrecsion_p(filename,list_word,w,condition,type)
    print(precision_uni_train)
    recall_uni_train = calRecall_p(filename,list_word,w,condition,type)
    print(recall_uni_train)
    fscore_uni_train = calFscore(precision_uni_train,recall_uni_train)
    print(fscore_uni_train)

review_list,review_label = cal_reviewlistlabel('train.csv')
list_uniWord = calListuniWord(review_list)
list_biWord = calListbiWord(review_list)
list_bothWord = []
list_bothWord = list_uniWord + list_biWord
# the first parameter is maxIter
# When run the program, it will print the w and all accuracy, precision, recall, and fscore of each file
w = perceptron(10,review_list,review_label,list_uniWord,cal_feature_array(review_list,list_uniWord)) #cal_feature_array is for unigram only
'''
print(w)
calaprf(w,'train.csv',list_uniWord,1,1) # first is condition, condition 1 is for unigram. type 1 is perceptron
print("validation")
calaprf(w,'validation.csv',list_uniWord,1,1)
print ("test")
calaprf(w,'test.csv',list_uniWord,1,1)


w = perceptron(40,review_list,review_label,list_biWord,cal_bifeature_array(review_list,list_biWord))
print("bigram")
print(w)
calaprf(w,'train.csv',list_biWord,2,1) # first one is condition, condition 2 is for bigram. type 1 is perceptron
print("validation")
calaprf(w,'validation.csv',list_biWord,2,1)
print ("test")
calaprf(w,'test.csv',list_biWord,2,1)

w = perceptron(30,review_list,review_label,list_bothWord,cal_both_feature_array(review_list,list_bothWord))
print("both")
print(w)
calaprf(w,'train.csv',list_bothWord,3,1) # first one is condition, condition 3 is for both unigram and bigram. type 1 is perceptron
print("validation")
calaprf(w,'validation.csv',list_bothWord,3,1)
print ("test")
calaprf(w,'test.csv',list_bothWord,3,1)


w_winnow = winnow(10,review_list,review_label,list_uniWord,cal_feature_array(review_list,list_uniWord)) #cal_feature_array is for unigram only


calaprf(w_winnow,'train.csv',list_uniWord,1,2) # first is condition, condition 1 is for unigram. type 2 is winnow
print("validation")
calaprf(w_winnow,'validation.csv',list_uniWord,1,2)
print ("test")
calaprf(w_winnow,'test.csv',list_uniWord,1,2)
'''