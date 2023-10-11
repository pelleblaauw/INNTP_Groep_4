import os
from flask import Flask, render_template, send_from_directory, request

app = Flask(__name__)

app.static_folder = 'static'
app.static_url_path = "/static"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result')
def result():
    query_string = request.query_string
    return render_template('result.html')

@app.route('/favicon.ico')
def fav():
    return send_from_directory(os.path.join(app.root_path, 'static'),'favicon.ico')

if __name__ == '__main__':
    app.run()