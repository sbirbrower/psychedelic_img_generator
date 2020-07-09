from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

from process_img import *

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/your-image', methods=['GET', 'POST'])
def your_image():

    # print(request.files['file'])
    f = request.files['file']
    file_name = secure_filename(f.filename)
    f.save('/Users/sydney/Files/projects/static/' + file_name)

    process_image('/Users/sydney/Files/projects/static/' + file_name, request.form['size'])

    if request.method == 'POST':
        return redirect(url_for('static', filename='final.png'))
        #return render_template('your_image.html', var=request.files['file'])
    else:
        return redirect(url_for('home'))

    # return redirect(url_for('home'))

    

