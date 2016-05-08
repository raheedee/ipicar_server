"""
Flask Documentation:     http://flask.pocoo.org/docs/
Jinja2 Documentation:    http://jinja.pocoo.org/2/documentation/
Werkzeug Documentation:  http://werkzeug.pocoo.org/documentation/

This file creates your application.
"""

import os
from flask import Flask, render_template, request, redirect, url_for, json, Response, jsonify
from flask.ext.cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'this_should_be_configured')


###
# Routing for your application.
###

@app.route('/')
def home():
    """Render website's home page."""
    return 'Hello World!'
    #return render_template('home.html')


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


<<<<<<< HEAD
@app.route('/GetCars', methods = ['GET'])
@cross_origin(origins='*')
=======
@app.route('/hello', methods = ['GET'])
>>>>>>> parent of e2c6ea0... updated json
def api_hello():
    list = [
        {'car': 'S7', 'make':'Audi', 'year' : '2013', 'val': 2, 'picture' : 'http://media.ed.edmunds-media.com/audi/s7/2013/oem/2013_audi_s7_sedan_prestige_fq_oem_6_98.jpg'},
        {'car': '428i', 'make' : 'BMW', 'year' : '2014', 'val': 2, 'picture' : 'https://media.ed.edmunds-media.com/bmw/4-series/2014/oem/2014_bmw_4-series_convertible_428i_fq_oem_2_98.jpg'}
    ]
    
    resp = jsonify(result=list)
    resp.status_code = 200
    resp.headers['Link'] = 'https://ipicarapp.herokuapp.com/'

    return resp


if __name__ == '__main__':
    app.run(debug=True)
