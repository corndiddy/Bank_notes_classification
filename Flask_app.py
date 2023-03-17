from flask import Flask, render_template, request, url_for
import numpy as np
import pickle

app = Flask(__name__)

with open('bankscaler','rb') as file:
    sc = pickle.load(file)

with open('bankmodel', 'rb') as file:
    clf = pickle.load(file)

@app.route('/') # / means home page
def home(): # whenever server runs ..home function will run
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    a1 = float(request.form['a1'])
    a2 = float(request.form['a2'])
    a3 = float(request.form['a3'])
    a4 = float(request.form['a4'])
    x=np.array([[a1,a2,a3,a4]])
    x_sc=sc.transform(x) # scaling
    final=clf.predict(x_sc) # predictiom
    final=final[0]
    if final == 1:
        z= 'FAKE'   # for fake notes
        return render_template('predict.html',result=z)
    else:            # for real notes
        z= 'REAL'
        return render_template('predict.html',result=z)

if __name__=='__main__':
    app.run(debug=True)
