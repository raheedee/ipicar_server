import os, csv, numpy as np, json, random
from flask import Flask, render_template, request, redirect, url_for, json, Response, jsonify
from flask.ext.pymongo import PyMongo
from sklearn.cluster import KMeans
from pymongo import MongoClient
from sklearn.externals import joblib
from sklearn.neighbors import NearestNeighbors
import sklearn.preprocessing as preprocessing
from sklearn.cluster import MiniBatchKMeans

app = Flask(__name__)
app.config['MONGO_DBNAME'] = 'ipicar'
app.config['MONGO_URI'] = 'mongodb://ipicar:CharizardBrizan123@ds011933.mlab.com:11933/ipicartest'

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'this_should_be_configured')

mongo = PyMongo(app)

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


price_weight = 3
num_cluster = 6

instances = []  #all features including styleid
ml_instances = []   #only feature values for machine learning purpose


@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
  return response


###
# Routing for your application.
###

@app.route('/')
def home():
    """Render website's home page."""
    return 'Hello World!'

@app.route('/find')
def find():
    styles = [101418219,200419090,101413943,200476884,200726808]
    responseCars = []
    car = mongo.db.cars
    for style in styles:
        foundCar = car.find_one({'styleid' : style})
        foundCar['_id'] = ''
        #set imporant missing values to -1
        if not foundCar['perfscore']:
            foundCar['perfscore'] = -1
        if not foundCar['buildscore']:
            foundCar['buildscore'] = -1
        if not foundCar['reliabilityscore']:
            foundCar['reliabilityscore'] = -1
        if not foundCar['overall']:
            foundCar['overall'] = -1
        if not foundCar['price']:
            foundCar['price'] = -1

        responseCars.append(foundCar)

    # return str(responseCars)
    resp = jsonify(result=responseCars)
    resp.status_code = 200
    resp.headers['Link'] = 'https://ipicarapp.herokuapp.com/'

    return resp

@app.route('/about/')
def about():
    """Render the website's about page."""
    return render_template('about.html')


###
# The functions below should be applicable to all Flask apps.
###

@app.route('/<file_name>.txt')
def send_text_file(file_name):
    """Send your static text file."""
    file_dot_text = file_name + '.txt'
    return app.send_static_file(file_dot_text)


@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=600'
    return response


@app.errorhandler(404)
def page_not_found(error):
    """Custom 404 page."""
    return render_template('404.html'), 404

#Initialize users
@app.route('/GetCars', methods = ['GET'])
def api_hello():

	responseCars = []
	car = mongo.db.cars
	styles = get_centroids()
	
	for style in styles:
		foundCar = car.find_one({'styleid' : int(style)})
		foundCar['_id'] = ''
		#set imporant missing values to -1
		if not foundCar['perfscore']:
		    foundCar['perfscore'] = -1
		if not foundCar['buildscore']:
		    foundCar['buildscore'] = -1
		if not foundCar['reliabilityscore']:
		    foundCar['reliabilityscore'] = -1
		if not foundCar['overall']:
		    foundCar['overall'] = -1
		if not foundCar['price']:
		    foundCar['price'] = -1
		responseCars.append(foundCar)
	
	resp = jsonify(result=responseCars)
	resp.status_code = 200
	resp.headers['Link'] = 'https://ipicarapp.herokuapp.com/'
	initialize_user_search()

	return resp

@app.route('/ReloadCars', methods = ['GET', 'POST'])
def api_reload():
    post = request.get_json()
    styles = [101418219,200419090,101413943,200476884,200726808,101418219,200419090,200727569,200745568]
    responseCars = []
    car = mongo.db.cars
    for style in styles:
        foundCar = car.find_one({'styleid' : style})
        foundCar['_id'] = ''
        #set imporant missing values to -1
        if not foundCar['perfscore']:
            foundCar['perfscore'] = -1
        if not foundCar['buildscore']:
            foundCar['buildscore'] = -1
        if not foundCar['reliabilityscore']:
            foundCar['reliabilityscore'] = -1
        if not foundCar['overall']:
            foundCar['overall'] = -1
        if not foundCar['price']:
            foundCar['price'] = -1

        responseCars.append(foundCar)

    # return str(responseCars)
    resp = jsonify(result=responseCars)
    resp.status_code = 200
    resp.headers['Link'] = 'https://ipicarapp.herokuapp.com/test'

    return resp

# Save each std to MongoDB
# Adjusts the attribute values using standard deviation based on the comment value inputted by the user
# Returns resultLists - a list of lists containing the updated (or original) values for the following attributes:
# In order: perfscore, buildscore, reliabilityscore, overall, price
@app.route('/adjust', methods = ['GET','post'])
def adjustAttributeValues():
    post = request.get_json()
    resp = jsonify(result=post)
    resp.status_code = 200
    resp.headers['Link'] = 'https://ipicarapp.herokuapp.com/'
    #return resp
    machine_results = []
    #cars = []


    cars = post


    std_perfscore = 0
    std_buildscore = 0
    std_reliabilityscore = 0
    std_price = 0

    # Missin rating values coded as negative one value (via UI)
    MISSING_VALUE = -1

    # The min and max rating scores for an attribute (i.e: perfscore)
    MIN_SCORE = 0
    MAX_SCORE = 10
 #   print('adjustAttributeValues() called.')

    std_perfscore = float(2.621213577)
    std_buildscore = float(2.734989743)
    std_reliabilityscore = float(2.784363134)
    std_price = float(20348.98747)

    # The dictionary of cars (will contain each row in the database)
    #cars = {}

    # resultLists is a list of lists where each list contains the adjusted values for each attribute
    resultLists = []

    

    for car in cars:

        key = car['styleid']

        if key in cars:
            # implement duplicate row handling here
            pass



        perfscore_key = float(car['perfscore'])
        buildscore_key = float(car['buildscore'])
        reliabilityscore_key = float(car['reliabilityscore'])
        price_key = float(car['price'])
        #Check the comment value for each car to control how to adjust the attribute values

        #(dislike, cannotAfford)
        if car['comment'] == 0:
         #   print ("(dislike, cannotAfford)")


            if MIN_SCORE <= perfscore_key < MAX_SCORE:
                perfscore_key += std_perfscore/2
                if perfscore_key > MAX_SCORE:
                    perfscore_key = MAX_SCORE
            elif perfscore_key == MISSING_VALUE:
                pass

            if MIN_SCORE <= buildscore_key < MAX_SCORE:
                buildscore_key += std_perfscore/2
                if buildscore_key > MAX_SCORE:
                    buildscore_key = MAX_SCORE
            elif buildscore_key == MISSING_VALUE:
                pass

            if MIN_SCORE <= reliabilityscore_key < MAX_SCORE:
                reliabilityscore_key += std_perfscore/2
                if reliabilityscore_key > MAX_SCORE:
                    reliabilityscore_key = MAX_SCORE
            elif reliabilityscore_key == MISSING_VALUE:
                pass

            
            if perfscore_key != MISSING_VALUE and buildscore_key != MISSING_VALUE and reliabilityscore_key != MISSING_VALUE:
            	overall_key = round(((perfscore_key + buildscore_key + reliabilityscore_key) / 3), 1)
            elif perfscore_key != MISSING_VALUE and buildscore_key != MISSING_VALUE and reliabilityscore_key == MISSING_VALUE:
            	overall_key = round(((perfscore_key + buildscore_key) / 2), 1)
            elif perfscore_key != MISSING_VALUE and buildscore_key == MISSING_VALUE and reliabilityscore_key != MISSING_VALUE:
            	overall_key = round(((perfscore_key + reliabilityscore_key) / 2), 1)
            elif perfscore_key != MISSING_VALUE and buildscore_key == MISSING_VALUE and reliabilityscore_key == MISSING_VALUE:
            	overall_key = round(perfscore_key, 1)
            elif perfscore_key == MISSING_VALUE and buildscore_key != MISSING_VALUE and reliabilityscore_key != MISSING_VALUE:
            	overall_key = round(((buildscore_key + reliabilityscore_key) / 2), 1)
            elif perfscore_key == MISSING_VALUE and buildscore_key != MISSING_VALUE and reliabilityscore_key == MISSING_VALUE:
            	overall_key = round(buildscore_key, 1)
            elif perfscore_key == MISSING_VALUE and buildscore_key == MISSING_VALUE and reliabilityscore_key != MISSING_VALUE:
            	overall_key = round(reliabilityscore_key, 1)
            elif perfscore_key == MISSING_VALUE and buildscore_key == MISSING_VALUE and reliabilityscore_key == MISSING_VALUE:
            	overall_key = MISSING_VALUE

            # decrease price value
            if price_key == MISSING_VALUE:
            	price_key = MISSING_VALUE
            else:
            	price_key -= std_price/2

            # Add list of adjusted values to resultsList
            adjustedValues0 = [perfscore_key, reliabilityscore_key, buildscore_key, overall_key, price_key]
            resultLists.append(adjustedValues0)

        # (dislike, canAfford)
        if car['comment'] == 1:
          #  print ("(dislike, canAfford)")

            if MIN_SCORE <= perfscore_key < MAX_SCORE:
                perfscore_key += std_perfscore/2
                if perfscore_key > MAX_SCORE:
                    perfscore_key = MAX_SCORE
            elif perfscore_key == MISSING_VALUE:
                pass

            if MIN_SCORE <= buildscore_key < MAX_SCORE:
                buildscore_key += std_perfscore/2
                if buildscore_key > MAX_SCORE:
                    buildscore_key = MAX_SCORE
            elif buildscore_key == MISSING_VALUE:
                pass

            if MIN_SCORE <= reliabilityscore_key < MAX_SCORE:
                reliabilityscore_key += std_perfscore/2
                if reliabilityscore_key > MAX_SCORE:
                    reliabilityscore_key = MAX_SCORE
            elif reliabilityscore_key == MISSING_VALUE:
                pass

            if perfscore_key != MISSING_VALUE and buildscore_key != MISSING_VALUE and reliabilityscore_key != MISSING_VALUE:
            	overall_key = round(((perfscore_key + buildscore_key + reliabilityscore_key) / 3), 1)
            elif perfscore_key != MISSING_VALUE and buildscore_key != MISSING_VALUE and reliabilityscore_key == MISSING_VALUE:
            	overall_key = round(((perfscore_key + buildscore_key) / 2), 1)
            elif perfscore_key != MISSING_VALUE and buildscore_key == MISSING_VALUE and reliabilityscore_key != MISSING_VALUE:
            	overall_key = round(((perfscore_key + reliabilityscore_key) / 2), 1)
            elif perfscore_key != MISSING_VALUE and buildscore_key == MISSING_VALUE and reliabilityscore_key == MISSING_VALUE:
            	overall_key = round(perfscore_key, 1)
            elif perfscore_key == MISSING_VALUE and buildscore_key != MISSING_VALUE and reliabilityscore_key != MISSING_VALUE:
            	overall_key = round(((buildscore_key + reliabilityscore_key) / 2), 1)
            elif perfscore_key == MISSING_VALUE and buildscore_key != MISSING_VALUE and reliabilityscore_key == MISSING_VALUE:
            	overall_key = round(buildscore_key, 1)
            elif perfscore_key == MISSING_VALUE and buildscore_key == MISSING_VALUE and reliabilityscore_key != MISSING_VALUE:
            	overall_key = round(reliabilityscore_key, 1)
            elif perfscore_key == MISSING_VALUE and buildscore_key == MISSING_VALUE and reliabilityscore_key == MISSING_VALUE:
            	overall_key = MISSING_VALUE

            # leave price

            #Add list of adjusted values to resultsList
            adjustedValues1 = [perfscore_key, reliabilityscore_key, buildscore_key, overall_key, price_key]
            resultLists.append(adjustedValues1)


        # (like, cannotAfford)
        if car['comment'] == 2:
           # print ("(like, cannotAfford)")
            
            # leave attribute values
            if perfscore_key != MISSING_VALUE and buildscore_key != MISSING_VALUE and reliabilityscore_key != MISSING_VALUE:
            	overall_key = round(((perfscore_key + buildscore_key + reliabilityscore_key) / 3), 1)
            elif perfscore_key != MISSING_VALUE and buildscore_key != MISSING_VALUE and reliabilityscore_key == MISSING_VALUE:
            	overall_key = round(((perfscore_key + buildscore_key) / 2), 1)
            elif perfscore_key != MISSING_VALUE and buildscore_key == MISSING_VALUE and reliabilityscore_key != MISSING_VALUE:
            	overall_key = round(((perfscore_key + reliabilityscore_key) / 2), 1)
            elif perfscore_key != MISSING_VALUE and buildscore_key == MISSING_VALUE and reliabilityscore_key == MISSING_VALUE:
            	overall_key = round(perfscore_key, 1)
            elif perfscore_key == MISSING_VALUE and buildscore_key != MISSING_VALUE and reliabilityscore_key != MISSING_VALUE:
            	overall_key = round(((buildscore_key + reliabilityscore_key) / 2), 1)
            elif perfscore_key == MISSING_VALUE and buildscore_key != MISSING_VALUE and reliabilityscore_key == MISSING_VALUE:
            	overall_key = round(buildscore_key, 1)
            elif perfscore_key == MISSING_VALUE and buildscore_key == MISSING_VALUE and reliabilityscore_key != MISSING_VALUE:
            	overall_key = round(reliabilityscore_key, 1)
            elif perfscore_key == MISSING_VALUE and buildscore_key == MISSING_VALUE and reliabilityscore_key == MISSING_VALUE:
            	overall_key = MISSING_VALUE

            # decrease price value
            if price_key == MISSING_VALUE:
            	price_key = MISSING_VALUE
            else:
            	price_key -= std_price/2

            # Add list of adjusted values to resultsList
            adjustedValues2 = [perfscore_key, reliabilityscore_key, buildscore_key, overall_key, price_key]
            resultLists.append(adjustedValues2)

        # (like, canAfford)
        if car['comment'] == 3:
            #print ("(like, canAfford)")

           # leave attribute values
            if perfscore_key != MISSING_VALUE and buildscore_key != MISSING_VALUE and reliabilityscore_key != MISSING_VALUE:
            	overall_key = round(((perfscore_key + buildscore_key + reliabilityscore_key) / 3), 1)
            elif perfscore_key != MISSING_VALUE and buildscore_key != MISSING_VALUE and reliabilityscore_key == MISSING_VALUE:
            	overall_key = round(((perfscore_key + buildscore_key) / 2), 1)
            elif perfscore_key != MISSING_VALUE and buildscore_key == MISSING_VALUE and reliabilityscore_key != MISSING_VALUE:
            	overall_key = round(((perfscore_key + reliabilityscore_key) / 2), 1)
            elif perfscore_key != MISSING_VALUE and buildscore_key == MISSING_VALUE and reliabilityscore_key == MISSING_VALUE:
            	overall_key = round(perfscore_key, 1)
            elif perfscore_key == MISSING_VALUE and buildscore_key != MISSING_VALUE and reliabilityscore_key != MISSING_VALUE:
            	overall_key = round(((buildscore_key + reliabilityscore_key) / 2), 1)
            elif perfscore_key == MISSING_VALUE and buildscore_key != MISSING_VALUE and reliabilityscore_key == MISSING_VALUE:
            	overall_key = round(buildscore_key, 1)
            elif perfscore_key == MISSING_VALUE and buildscore_key == MISSING_VALUE and reliabilityscore_key != MISSING_VALUE:
            	overall_key = round(reliabilityscore_key, 1)
            elif perfscore_key == MISSING_VALUE and buildscore_key == MISSING_VALUE and reliabilityscore_key == MISSING_VALUE:
            	overall_key = MISSING_VALUE
           

            # leave price

            # Add list of adjusted values to resultsList
            adjustedValues3 = [perfscore_key, reliabilityscore_key, buildscore_key, overall_key, price_key]
            resultLists.append(adjustedValues3)

            

    machine_results = get_result(resultLists)
    responseCars = []
    car = mongo.db.cars
    for style in machine_results:
        foundCar = car.find_one({'styleid' : style})
        foundCar['_id'] = ''
        #set imporant missing values to -1
        if not foundCar['perfscore']:
            foundCar['perfscore'] = -1
        if not foundCar['buildscore']:
            foundCar['buildscore'] = -1
        if not foundCar['reliabilityscore']:
            foundCar['reliabilityscore'] = -1
        if not foundCar['overall']:
            foundCar['overall'] = -1
        if not foundCar['price']:
            foundCar['price'] = -1
        responseCars.append(foundCar)

    print (resultLists)
    resp = jsonify(result=responseCars)
    resp.status_code = 200
    resp.headers['Link'] = 'https://ipicarapp.herokuapp.com/test'

    return resp


@app.route('/prestage', methods = ['GET'])
def prestaging():
	create_6_KNN_lists()
	find_centroids()
	create_6_KNN_lists_ml()
	return 'hello'

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

    db.cars_new.drop()

    for i in range(0, len (labels)):
    	print 'saving'
        save_newset(instances[i], data_set[i], labels[i])
    print 'save the model'
    joblib.dump(cluster, 'cluster_model.pkl')
    test_function(cluster)
    

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
def find_centroids():

    cluster = joblib.load('cluster_model.pkl')
    centroids = cluster.cluster_centers_
    print len(cluster0)
    centroid_list=[]
    print 'here1'
    ini_neigh0 = NearestNeighbors(n_neighbors = 1)
    print 'here2'
    ini_neigh0.fit(cluster0)
    print 'here3'
    dist, indice = ini_neigh0.kneighbors(centroids[0].reshape(1,-1))
    print 'here4'
    centroid_list.append(cluster0_id[indice[0][0]])
    
    ini_neigh1 = NearestNeighbors(n_neighbors = 1)
    ini_neigh1.fit(cluster1)
    dist, indice = ini_neigh1.kneighbors(centroids[1].reshape(1,-1))
    centroid_list.append(cluster1_id[indice[0][0]])
    print 'here3'
    ini_neigh2 = NearestNeighbors(n_neighbors = 1)
    ini_neigh2.fit(cluster2)
    dist, indice = ini_neigh2.kneighbors(centroids[2].reshape(1,-1))
    centroid_list.append(cluster2_id[indice[0][0]])
    print 'here4'
    ini_neigh3 = NearestNeighbors(n_neighbors = 1)
    ini_neigh3.fit(cluster3)
    dist, indice = ini_neigh3.kneighbors(centroids[3].reshape(1,-1))
    centroid_list.append(cluster3_id[indice[0][0]])
    print 'here5'
    ini_neigh4 = NearestNeighbors(n_neighbors = 1)
    ini_neigh4.fit(cluster4)
    dist, indice = ini_neigh4.kneighbors(centroids[4].reshape(1,-1))
    centroid_list.append(cluster4_id[indice[0][0]])
    print'here6'
    ini_neigh5 = NearestNeighbors(n_neighbors = 1)
    ini_neigh5.fit(cluster5)
    dist, indice = ini_neigh5.kneighbors(centroids[5].reshape(1,-1))
    centroid_list.append(cluster5_id[indice[0][0]])
    print 'here7'
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




#load the already saved model
cluster = joblib.load('cluster_model.pkl')

#6 clusters in total
maxlist = []    #stores the max num of each column for normalization of modified instances
mean_median = 0 # mean and median string from database

# constant for KNN
K = 10
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
    

#return a list of centroids' styleid
def get_centroids ():
    centroidpointer = db.replacevalue.find()
    count = 0
    for doc in centroidpointer:
        if count == 2:
        	#cen = doc['centroid'].tolist()
	        return doc['centroid']
        count +=1


def get_maxlist_mean_median ():
    temp = db.replacevalue.find()
    count = 0
    for doc in temp:
        if count == 0:
            global mean_median
            mean_median = doc
            print (mean_median)
            count +=1
        elif count ==1:
            global maxlist
            maxlist = doc['max']
            break
#modified upser inputs, fill, in the missing value and normalize all    
def modified_user_inputs (inputs):
	print (mean_median)
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
def initialize_user_search():
    create_6_KNN_lists_ml()
    get_maxlist_mean_median()


if __name__ == '__main__':
    #app.debug = True
    app.run(host='0.0.0.0')
