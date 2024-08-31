import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '/Users/keyur/dev/flask/cellcano/uploads'
ALLOWED_EXTENSIONS = {'html','h5ad'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.errorhandler(413)
def too_large(e):
    return "File is too large", 413

@app.route("/cellcano")
def run_cellcano():
    return render_template("cellcano.html")

@app.route("/links")
def run_links():
    return render_template("links.html")

@app.route('/upload', methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        #flash('file requested: '+file.filename)
        #flash('request URL'+request.url)
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash('File type not supported, upload an h5ad file.')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            #flash('UPLOAD_FOLDER: '+app.config['UPLOAD_FOLDER'])
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            

            # now launch the NN from here, which should dump results in a file.

            # read results from the output file and set prediction_value to that...
            prediction_value = '0.99'
            return render_template("upload.html", prediction = prediction_value)

    return render_template("upload.html", prediction='No results/no upload')

if __name__=="__main__":
    app.run(debug=True) 