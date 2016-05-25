#Title: machine_learning.py
#Author: Ling Hong Ren
#Description: algorithms to use for machine learning purpose.
#   Includes algorithms to be used once when a user starts a search
#   and algorithms that needs to be called everytime a user has an input
#Usage: import test_ml


import json
import numpy as np
from sklearn.cluster import KMeans
from pymongo import MongoClient
from sklearn.externals import joblib
from sklearn.neighbors import NearestNeighbors

#-----------------READ ME-------------------
##call when a user starts using the platform
##  1. initialize some values
##      initialize_user_search ()
##  2. call to return a string of a list of styleid for centroids
##      get_centroids ()
##
##must be called everytime to get a list of nearest neighbors based on user's input
##which is a list of list. Missing values will be representated as '-1'.
##This is the inputs in this method
##return an unique list of styleid for K nearest neighbors
##    get_result(inputs)


#sample input for testing
#inputs = [[2, -1, -1, -1, 100],[6, 3, 56, -1, -1]]

client = MongoClient("mongodb://ipicar:CharizardBrizan123@ds011933.mlab.com:11933/ipicartest")
db = client.ipicartest

#load the already saved model
cluster = joblib.load('cluster_model.pkl')

#6 clusters in total
maxlist = []    #stores the max num of each column for normalization of modified instances
mean_median = 0 # mean and median string from database

# constant for KNN
K = 20
#id lists have styleid corresponding to values in cluster lists
cluster0_id = []
cluster1_id = []
cluster2_id = []
cluster3_id = []
cluster4_id = []
cluster5_id = []

neigh0 = NearestNeighbors(n_neighbors = K)
neigh1 = NearestNeighbors(n_neighbors = K)
neigh2 = NearestNeighbors(n_neighbors = K)
neigh3 = NearestNeighbors(n_neighbors = K)
neigh4 = NearestNeighbors(n_neighbors = K)
neigh5 = NearestNeighbors(n_neighbors = K)

#find 6 KNN lists
#must have previously initialized 12 lists like 12 lines right above it
def create_6_KNN_lists_ml():
    cursor_data = db.cars_new.find()
    cluster0 = []
    cluster1 = []
    cluster2 = []
    cluster3 = []
    cluster4 = []
    cluster5 = []
    for doc in cursor_data:
        if doc['label'] == '0':
            cluster0.append([doc['perf'], doc['reliability'], doc['build'], doc['overall'], doc['price']])
            cluster0_id.append(doc['styleid'])
        elif doc['label'] == '1':
            cluster1.append([doc['perf'], doc['reliability'], doc['build'], doc['overall'], doc['price']])
            cluster1_id.append(doc['styleid'])
        elif doc['label'] == '2':
            cluster2.append([doc['perf'], doc['reliability'], doc['build'], doc['overall'], doc['price']])
            cluster2_id.append(doc['styleid'])
        elif doc['label'] == '3':
            cluster3.append([doc['perf'], doc['reliability'], doc['build'], doc['overall'], doc['price']])
            cluster3_id.append(doc['styleid'])
        elif doc['label'] == '4':
            cluster4.append([doc['perf'], doc['reliability'], doc['build'], doc['overall'], doc['price']])
            cluster4_id.append(doc['styleid'])
        elif doc['label'] == '5':
            cluster5.append([doc['perf'], doc['reliability'], doc['build'], doc['overall'], doc['price']])
            cluster5_id.append(doc['styleid'])

    #building neighbors model
    global neigh0
    global neigh1
    global neigh2
    global neigh3
    global neigh4
    global neigh5
    neigh0.fit(cluster0)
    neigh1.fit(cluster1)
    neigh2.fit(cluster2)
    neigh3.fit(cluster3)
    neigh4.fit(cluster4)
    neigh5.fit(cluster5)
    

#return a string of centroids' styleid list
#get it from replace value
#it is in the third line
def get_centroids ():
    centroidpointer = db.replacevalue.find()
    count = 0
    for doc in centroidpointer:
        if count == 2:
            return doc['centroid']
        count +=1

#read max list for normalization, mean and median to replace missing values from
#replacevalue collection
#mean_median is in the first line
#maxlist is in the second line
def get_maxlist_mean_median ():
    temp = db.replacevalue.find()
    count = 0
    for doc in temp:
        if count == 0:
            global mean_median
            mean_median = doc
            count +=1
        elif count ==1:
            global maxlist
            maxlist = doc['max']
            break
        
#modified upser inputs, fill, in the missing value and normalize all    
def modified_user_inputs (inputs):
    user_set = inputs
    #fill in missing value
    for item in user_set:
        if item[0] < 0:
            item[0] = float(mean_median['perf'])
        if item[1] < 0:
            item[1] = float(mean_median['reliability'])
        if item[2] < 0:
            item[2] = float(mean_median['build'])
        if item[3] < 0:
            item[3] = float(mean_median['overall'])
        if item[4] < 0:
            item[4] = float(mean_median['price'])
    #print user_set

    #normalize
    np_user_set = np.array(user_set)
    #print maxlist
    normalized_user_set = np_user_set /maxlist
    #print normalized_user_set
    
    return normalized_user_set

#find the list of cluster labels the inputs list belong
def find_cluster (modified_inputs):
    md_labels = cluster.predict(modified_inputs)
    return md_labels

#return a list of list of styleid of knn
#take lists of modifed_inputs, and md_labels
def find_knn (modifed_inputs, md_labels):
    result_list_id = []
    for i in range (0, len(modifed_inputs)):
        #find out
        indice =[]
        if md_labels[i] == 0:
            dist, indice = neigh0.kneighbors(modifed_inputs[i].reshape(1, -1))
            result_list_id.append (get_styleid(indice, 0))
        
        elif md_labels[i] == 1:
            print modifed_inputs[i]
            dist, indice = neigh1.kneighbors(modifed_inputs[i].reshape(1, -1))
            result_list_id.append (get_styleid(indice, 1))
        
        elif md_labels[i] == 2:
            dist, indice = neigh2.kneighbors(modifed_inputs[i].reshape(1, -1))
            result_list_id.append (get_styleid(indice, 2))
            
        elif md_labels[i] == 3:
            dist, indice = neigh3.kneighbors(modifed_inputs[i].reshape(1, -1))
            result_list_id.append (get_styleid(indice, 3))
        
        elif md_labels[i] == 4:
            dist, indice = neigh4.kneighbors(modifed_inputs[i].reshape(1, -1))
            result_list_id.append (get_styleid(indice, 4))
        
        elif md_labels[i] == 5:
            dist, indice = neigh5.kneighbors(modifed_inputs[i].reshape(1, -1))
            result_list_id.append (get_styleid(indice, 5))
            
    return result_list_id

#return a list of styleid of based on the input_indice
def get_styleid (input_indice, label):
    styleid_list = []
    indice = input_indice.tolist()
    if label == 0:
        for index in indice:
            for i in index:
                styleid_list.append(int(cluster0_id[i]))

    elif label == 1:
        for index in indice:
            for i in index:
                styleid_list.append(int(cluster1_id[i]))

    elif label == 2:
        for index in indice:
            for i in index:
                styleid_list.append(int(cluster2_id[i]))
    
    elif label == 3:
        for index in indice:
            for i in index:
                styleid_list.append(int(cluster3_id[i]))

    elif label ==4:
        for index in indice:
            for i in index:
                styleid_list.append(int(cluster4_id[i]))

    elif label ==5:
        for index in indice:
            for i in index:
                styleid_list.append(int(cluster5_id[i]))

    return styleid_list


#get the result to UI
def get_result(inputs):
    output_list = []
    normalized_set = modified_user_inputs(inputs)
    result_labels = find_cluster(normalized_set)
    result_list_of_list = find_knn ( normalized_set, result_labels)

    return get_output_list (result_list_of_list)

#interleave results captured in the list of list
#take out the repeated items in the list
#the output is a unique list
def get_output_list(prelist):
    output = []
    #each prelist[x] has same length
    for i in range(0, len(prelist[0])):
        for j in range(0, len(prelist)):
            output.append(prelist[j][i])

    have_seen = set()
    unique_output =[]
    for i in output:
        if i not in have_seen:
            unique_output.append(i)
            have_seen.add(i)
    
    return unique_output


#call when a user starts using the platform
def initialize_user_search ()
    create_6_KNN_lists_ml()
    get_maxlist_mean_median()


##print get_result(inputs)

