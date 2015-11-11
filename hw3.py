__author__ = 'penghao'


from numpy import *
import numpy as np
from math import *
import csv
import sys
from collections import Counter

def GD(maxIterations,review_list,review_label,list_word,regularization,stepSize,lmbd,featureSet):
    #regularization has two value, 1 is l1, 2 is l2
    if featureSet ==1: #unigram
        feature_array = cal_feature_array(review_list,list_uniWord)
    if featureSet ==2:#bigram
        feature_array = cal_bifeature_array(review_list,list_biWord)
    if featureSet ==3:#both
        feature_array = cal_both_feature_array(review_list,list_bothWord)
    w = np.zeros(list_word.__len__())
    maxIter = maxIterations
    b=0
    o_old = obj(w,lmbd,review_label,list_word,feature_array) # We want to minimize this number
    for iteration in range(maxIter):
        w_copy =w
        random_idx = np.random.permutation(review_list.__len__())
        g= np.zeros(list_word.__len__())
        g_bias = 0
        for row_idx in range(review_label.__len__()):

            y= np.dot(w,feature_array[row_idx])+b
            review_l = review_label[row_idx]
            y1=review_l
            if y*y1 <=1:
                g=g + y1* feature_array[row_idx]
                g_bias = g_bias + y1
        if regularization ==2:
            g = g - lmbd*w
        if regularization ==1:
            g= g - lmbd*(np.sign(w))
        w = w + stepSize*g
        b = b + stepSize*g_bias

        o=obj(w,lmbd,review_label,list_word,feature_array)
        if np.absolute(o-o_old)<0.001: # If it changes a little, then converge
            print(iteration)
            print("Converge")
            return w,b
        o_old=o

    #print(o)
    return w,b

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

def obj(w,lmbd,review_label,list_word,feature_array):
    w=np.array(w)
    loss =0
    for row_idx in range(review_label.__len__()):
        y= review_label[row_idx]
        x=np.array(feature_array[row_idx])
        loss = loss + max(0,1-y*np.dot(w,x))

    return lmbd*0.5*vectornorm(w)**2 + loss

def calPrecsion_p(filename,list_word,w,condition,type,bias): #calculate precision
    vreviewlist,vreviewlabel = cal_reviewlistlabel(filename)
    if condition == 1: #1 means unigram
        vfeatureArray = cal_feature_array(vreviewlist,list_word)
    if condition == 2: #2 means bigram
        vfeatureArray = cal_bifeature_array(vreviewlist,list_word)
    if condition == 3: #3 means both
        vfeatureArray = cal_both_feature_array(vreviewlist,list_word)
    b=bias
    turePositiveCount=0.0
    falsePositiveCount = 0.0
    if type == 1:
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


    return(precision)

def calRecall_p(filename,list_word,w,condition,type,bias): #calculate recall
    vreviewlist,vreviewlabel = cal_reviewlistlabel(filename) # generate a new review list that contain different words
    if condition == 1: #1 means unigram
        vfeatureArray = cal_feature_array(vreviewlist,list_word)
    if condition == 2: #2 means bigram
        vfeatureArray = cal_bifeature_array(vreviewlist,list_word)
    if condition == 3: #3 means both
        vfeatureArray = cal_both_feature_array(vreviewlist,list_word)
    b=bias
    turePositiveCount=0.0
    falseNegativeCount = 0.0
    if type == 1:
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
    return(recall)

def calFscore(precision,recall):
    return 2*(precision*recall)/(precision+recall)

def predict_one_p(w,input_snippet ):
    #input_snippet is the feature_array like :(0,0,0,0,1,0......)
    b = 0
    y = np.dot(w,input_snippet) + b
    return y

def calTrainError_p(filename,list_word,w,condition,type,bias): #calculate accuracy
    vreviewlist,vreviewlabel = cal_reviewlistlabel(filename)
    if condition == 1: #1 means unigram
        vfeatureArray = cal_feature_array(vreviewlist,list_word)
    if condition == 2: #2 means bigram
        vfeatureArray = cal_bifeature_array(vreviewlist,list_word)
    if condition == 3: #3 means both
        vfeatureArray = cal_both_feature_array(vreviewlist,list_word)
    b=bias
    count=0.0
    if type == 1:
        for idx in range(vreviewlist.__len__()):
            y = predict_one_p(w,vfeatureArray[idx]) + b
            if y>0:
                predict_label = 1
            else:
                predict_label = -1
            review_l = vreviewlabel[idx]
            if review_l != predict_label:
                count=count+1

    return((vreviewlist.__len__()-count)/vreviewlist.__len__())

def calaprf(w,filename,list_word,condition,type,bias):
    print("1-TrainError")
    print(calTrainError_p(filename,list_word,w,condition,type,bias))
    precision_uni_train =calPrecsion_p(filename,list_word,w,condition,type,bias)
    print("Precision")
    print(precision_uni_train)
    recall_uni_train = calRecall_p(filename,list_word,w,condition,type,bias)
    print("Recall")
    print(recall_uni_train)
    fscore_uni_train = calFscore(precision_uni_train,recall_uni_train)
    print("F-score")
    print(fscore_uni_train)

def vectornorm(w):
    norm =0
    for one in w:
        norm = np.absolute(one)**2
    return norm**(1/2)

review_list,review_label = cal_reviewlistlabel('train.csv')
list_uniWord = calListuniWord(review_list)
list_biWord = calListbiWord(review_list)
list_bothWord = []
list_bothWord = list_uniWord + list_biWord

def main():


    # the first parameter is maxIter
    # When run the program, it will print the w and all accuracy, precision, recall, and fscore of each file

    print("Unigram -l1 The maxiter is 750, it is an example")
    w,bias= GD(750,review_list,review_label,list_uniWord,1,0.0005,0.001,1) #l1
    print(w)
    print(bias)
    print("train")
    calaprf(w,'train.csv',list_uniWord,1,1,bias)
    print("validation")
    calaprf(w,'validation.csv',list_uniWord,1,1,bias)
    print("test")
    calaprf(w,'test.csv',list_uniWord,1,1,bias)
    '''
    print("ul2")

    w,bias= GD(375,review_list,review_label,list_uniWord,2,0.0005,0.001,1)
    print(w)
    print(bias)
    print("train")
    calaprf(w,'train.csv',list_uniWord,1,1,bias)
    print("v")
    calaprf(w,'validation.csv',list_uniWord,1,1,bias)
    print("test")
    calaprf(w,'test.csv',list_uniWord,1,1,bias)
    print("bi")
    w,bias= GD(125,review_list,review_label,list_biWord,1,0.0005,0.001,2)
    print(w)
    print(bias)
    print("train")
    calaprf(w,'train.csv',list_biWord,2,1,bias)
    print("v")
    calaprf(w,'validation.csv',list_biWord,2,1,bias)
    print("test")
    calaprf(w,'test.csv',list_biWord,2,1,bias)
    w,bias= GD(125,review_list,review_label,list_biWord,2,0.0005,0.001,2)
    print(w)
    print(bias)
    print("train")
    calaprf(w,'train.csv',list_biWord,2,1,bias)
    print("v")
    calaprf(w,'validation.csv',list_biWord,2,1,bias)
    print("test")
    calaprf(w,'test.csv',list_biWord,2,1,bias)
    print("both")
    w,bias= GD(175,review_list,review_label,list_bothWord,1,0.0005,0.001,3)
    print(w)
    print(bias)
    print("train")
    calaprf(w,'train.csv',list_bothWord,3,1,bias)
    print("v")
    calaprf(w,'validation.csv',list_bothWord,3,1,bias)
    print("test")
    calaprf(w,'test.csv',list_bothWord,3,1,bias)
    w,bias= GD(175,review_list,review_label,list_bothWord,2,0.0005,0.001,3)
    print(w)
    print(bias)

    print("train")
    calaprf(w,'train.csv',list_bothWord,3,1,bias)
    print("v")
    calaprf(w,'validation.csv',list_bothWord,3,1,bias)
    print("test")
    calaprf(w,'test.csv',list_bothWord,3,1,bias)


'''
if __name__ == '__main__':
    main()