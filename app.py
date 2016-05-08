"""
Flask Documentation:     http://flask.pocoo.org/docs/
Jinja2 Documentation:    http://jinja.pocoo.org/2/documentation/
Werkzeug Documentation:  http://werkzeug.pocoo.org/documentation/

This file creates your application.
"""

import os
from flask import Flask, render_template, request, redirect, url_for, json, Response, jsonify

app = Flask(__name__)

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


@app.route('/hello', methods = ['GET'])
def api_hello():
    data = {"gender":"male","name":{"title":"mr","first":"joseph","last":"richardson"},"location":{"street":"7571 edwards rd","city":"hervey bay","state":"new south wales","postcode":1866},"email":"joseph.richardson@example.com","login":{"username":"whitemouse984","password":"damien","salt":"OAZh0lg5","md5":"9cf72bc757e2f2cdc1aa1a8cbb94c5ef","sha1":"80f0e65c4a85ed6116800ded9efed720915e8448","sha256":"0054243b023c077470e1c9925363d49b2992eb02cefaf37ab23d57a525bc50a6"},"registered":1240101185,"dob":1142157398,"phone":"03-7795-7788","cell":"0408-984-842","id":{"name":"TFN","value":"128066967"},"picture":{"large":"https://randomuser.me/api/portraits/men/94.jpg","medium":"https://randomuser.me/api/portraits/med/men/94.jpg","thumbnail":"https://randomuser.me/api/portraits/thumb/men/94.jpg"},"nat":"AU"}
    js = json.dumps(data)

    resp = jsonify(data)
    resp.status_code = 200
    resp.headers['Link'] = 'https://ipicarapp.herokuapp.com/'

    return resp


if __name__ == '__main__':
    app.run(debug=True)
