from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os


from process_img import *

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/your-image', methods=['GET', 'POST'])
def your_image():

    f = request.files['file']
    file_name = secure_filename(f.filename)

    cwd = os.getcwd()
    f.save(cwd + '/static/' + file_name)

    process_image(cwd + '/static/' + file_name, request.form['size'])

    if request.method == 'POST':
        return redirect(url_for('static', filename='final.png'))
    else:
        return redirect(url_for('home'))

    

