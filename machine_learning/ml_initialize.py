#Title: ml_initialize.py
#Author: Ling Hong Ren
#Description: algorithms to build machine learning models only once. Including saving
#   needed information in Mongodb for easy retrival in different files
#Usage: import test_initialize


from pymongo import MongoClient
import json
import numpy as np
import sklearn.preprocessing as preprocessing
from sklearn.cluster import MiniBatchKMeans
from sklearn.externals import joblib
import random
from sklearn.neighbors import NearestNeighbors

#-----------------READ ME-------------------
##should run in the initial stage
##run this file only once and when the database is updated
##
##collection names:
##cars -- original data
##cars_new -- normalized value with label and styleid
##replacevalue -- mean, median, max (for normalization)

##Use this file in the following way:
##
##   prestage()
##
##will create the clustering model, read from mongodb, and save needed information


price_weight = 3
num_cluster = 6
client = MongoClient("mongodb://ipicar:CharizardBrizan123@ds011933.mlab.com:11933/ipicartest")
db = client.ipicartest
instances = []  #all features including styleid
ml_instances = []   #only feature values for machine learning purpose

class find_all_mean_median:
    #without missing values
    non_missing_list = {'perf': [],
                        'reliability':[], 
                        'build': [], 
                        'overall': [],
                        'price':[]}
    #final saved results for replacement
    replace_missing = {'perf': 0, #mean
                       'reliability': 0, #mean
                       'build': 0,  #mean
                       'overall': 0, #mean
                       'price': 0} #median
    
    def __init__(self, data):
        self.data = data

    #look for mean and mode
    def all_lists (self):
        for each in self.data:
            if each[1] > -1:
                self.non_missing_list['perf'].append(float(each[1]))

            if each[2] > -1:
                self.non_missing_list['reliability'].append(float(each[2]))

            if each[3] > -1:
                self.non_missing_list['build'].append(float(each[3]))

            if each[4] > -1:
                self.non_missing_list['overall'].append(float(each[4]))

            if each[5] > -1:
                self.non_missing_list['price'].append(float(each[5]))

                
    #find all the means and median
    #median for all categories other than price, which uses mean
    def mean_4_cat(self):
        self.all_lists()
        for eachlist in self.non_missing_list:
            if eachlist == 'price':
                self.replace_missing[eachlist] = round(np.mean(self.non_missing_list[eachlist]), 2)
                #print self.replace_missing[eachlist]
            elif eachlist != 'styleid':
                self.replace_missing[eachlist] = round(np.median(np.array(self.non_missing_list[eachlist])),2)
                #print self.replace_missing[eachlist]

    #save mean and median into mongodb in replacevalue cars
    def save_result(self):
        db.replacevalue.drop()
        db.replacevalue.insert_one(
            {
                "perf": str(self.replace_missing['perf']),
                "reliability": str(self.replace_missing['reliability']),
                "build": str(self.replace_missing['build']),
                "overall": str(self.replace_missing['overall']),
                "price": str(self.replace_missing['price'])
                }
            )
        #print self.replace_missing
        

#document parsing to create instances list
def read_data ():
    cursor = db.cars.find()
    for document in cursor:
        line_id = document["styleid"]
        if document["perfscore"] =="":
            perf = -1
        else:
            perf = float (document["perfscore"])

        if document["reliabilityscore"] == "":
            reliability = -1
        else:
            reliability = float (document["reliabilityscore"])

        if document ["buildscore"] == "":
            build = -1
        else:
            build = float (document["buildscore"])

        if document ["overall"] == "":
            overall = -1
        else:
            overall = float (document["overall"])
            
        if document["price"] == "":
            price = -1
        else:
            price = document["price"]
        
        instances.append([line_id, perf, reliability, build, overall, price])



#replace all missing values in set with mean and median from mean_median class
def replace_missing_value(mean_median):
    total_size = len(instances)
    for i in range(0, total_size):
        #print instances[i]
        if instances[i][1] < 0:
            instances[i][1] = mean_median.replace_missing['perf']
            #print instances[i][0]
        if instances[i][2] <0:
            instances[i][2] = mean_median.replace_missing['reliability']
            #print instances[i][1]
        if instances[i][3] <0:
            instances[i][3] = mean_median.replace_missing['build']
            #print instances[i][2]
        if instances[i][4] <0:
            instances[i][4] = mean_median.replace_missing['overall']
            #print instances[i][3]
        if instances[i][5] <0:
            instances[i][5] = mean_median.replace_missing['price']
            #print instances[i][4]
        #print instances[i]

#retrieve data from cars_new for machine learning purpose in order
#meaning does not have style id
def machine_learning_data ():
    for line in instances:
        #print line
        ml_instances.append([float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5])])
    print "first ", len(ml_instances)
        
        
#call every time there is a change to database
def initialize ():
    read_data()
    ##prepare the dataset for normalization
    mean_median = find_all_mean_median(instances)
    mean_median.mean_4_cat()
    mean_median.save_result()
    replace_missing_value(mean_median)

#normalize the machine learning dataset
def machine_learning():
    machine_learning_data()

    npml = np.array(ml_instances)
    normalized_instances = npml / npml.max(axis=0)
    print npml.max(axis=0)
    db.replacevalue.insert_one (
        {'max': npml.max(axis=0).tolist()}
    )
##    normalized_instances = preprocessing.normalize(ml_instances, norm = 'l2')
    norm_size = len(normalized_instances)
    
    ## weighted the price by 2
    for i in range(0, norm_size):
        normalized_instances[i][4] *= price_weight
        
#    print normalized_instances
    return normalized_instances


    
#build clustering model and save the model
def build_clustering_model():
    print 'machine learning'
    data_set = machine_learning()
    print len(data_set)
    print 'cluster model'
    cluster = MiniBatchKMeans (n_clusters=num_cluster, init='k-means++', max_iter=100, batch_size=100, \
                               verbose=0, compute_labels=True, random_state=None, tol=0.0, \
                               max_no_improvement=10, init_size=None, n_init=3, reassignment_ratio=0.01)

    print 'fitting'
    cluster.fit(data_set)
    labels = cluster.labels_

    test_function(cluster)
    db.cars_new.drop()

    for i in range(0, len (labels)):
        save_newset(instances[i], data_set[i], labels[i])
    print 'save the model'
    joblib.dump(cluster, 'cluster_model.pkl')

    

#save the normalzied set with styleid and label into cars_new
def save_newset(originalset, normalizedset, label):
#    print originalset[0], originalset[1], normalizedset[0],normalizedset[1], normalizedset[2], normalizedset[3], normalizedset[4],label
    db.cars_new.insert_one(
        {
        "styleid": str(originalset[0]),
        "perf": str(normalizedset[0]),
        "reliability": str(normalizedset[1]),
        "build": str(normalizedset[2]),
        "overall": str(normalizedset[3]),
        "price": str(normalizedset[4]),
        "label": str(label)
        }
    )

#for clustering testing
def test_function (cluster):
    print cluster.cluster_centers_
    print cluster.labels_
    labels0 = cluster.labels_
    print len(cluster.labels_)

    labels = labels0.tolist()
    print labels.count(0), " ", float(labels.count(0))/len(labels)*100
    print labels.count(1), " ", float(labels.count(1))/len(labels)*100
    print labels.count(2), " ", float(labels.count(2))/len(labels)*100
    print labels.count(3), " ", float(labels.count(3))/len(labels)*100
    print labels.count(4), " ", float(labels.count(4))/len(labels)*100
    print labels.count(5), " ", float(labels.count(5))/len(labels)*100



cluster0 = []
cluster0_id = []
cluster1 = []
cluster1_id = []
cluster2 = []
cluster2_id = []
cluster3 = []
cluster3_id = []
cluster4 = []
cluster4_id = []
cluster5 = []
cluster5_id = []

#find 6 KNN lists
#must have previously initialized 12 lists like 12 lines right above it
def create_6_KNN_lists():
    cursor_data = db.cars_new.find()
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

#find styleid of the instances that are closest to centroids
#save them into database replacevalue
def find_centroids():

    cluster = joblib.load('cluster_model.pkl')
    centroids = cluster.cluster_centers_

    centroid_list=[]
    
    ini_neigh0 = NearestNeighbors(n_neighbors = 1)
    ini_neigh0.fit(cluster0)
    dist, indice = ini_neigh0.kneighbors(centroids[0].reshape(1,-1))
    centroid_list.append(cluster0_id[indice[0][0]])

    ini_neigh1 = NearestNeighbors(n_neighbors = 1)
    ini_neigh1.fit(cluster1)
    dist, indice = ini_neigh1.kneighbors(centroids[1].reshape(1,-1))
    centroid_list.append(cluster1_id[indice[0][0]])

    ini_neigh2 = NearestNeighbors(n_neighbors = 1)
    ini_neigh2.fit(cluster2)
    dist, indice = ini_neigh2.kneighbors(centroids[2].reshape(1,-1))
    centroid_list.append(cluster2_id[indice[0][0]])

    ini_neigh3 = NearestNeighbors(n_neighbors = 1)
    ini_neigh3.fit(cluster3)
    dist, indice = ini_neigh3.kneighbors(centroids[3].reshape(1,-1))
    centroid_list.append(cluster3_id[indice[0][0]])

    ini_neigh4 = NearestNeighbors(n_neighbors = 1)
    ini_neigh4.fit(cluster4)
    dist, indice = ini_neigh4.kneighbors(centroids[4].reshape(1,-1))
    centroid_list.append(cluster4_id[indice[0][0]])

    ini_neigh5 = NearestNeighbors(n_neighbors = 1)
    ini_neigh5.fit(cluster5)
    dist, indice = ini_neigh5.kneighbors(centroids[5].reshape(1,-1))
    centroid_list.append(cluster5_id[indice[0][0]])

    db.replacevalue.insert_one (
        {'centroid': centroid_list}
    )

#call this function once
#update only when the database updates
def prestage():
    initialize()
    #could be called with calling initialize() right before
    build_clustering_model()
    create_6_KNN_lists()
    find_centroids()

#prestage()

