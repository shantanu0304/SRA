'''
Day 1 Work
'''
from flask import Flask, render_template,request
import os
#import numpy as np
import pandas as pd
app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
    '''
    models=['Linear Regression','Logistic Regression','Polynomial Regression','Stepwise Regression','Ridge Regression','Lasso Regression','ElasticNet Regression']
    return render_template('base.html',models=models)
    '''
@app.route('/recommender')
def rec():
    return render_template('recommend.html')
@app.route('/Linear_Regression')
def linear():
    return render_template('Linear_Regression.html')
def rec():
    return render_template('recommend.html')
@app.route('/Dataset')
def dataset():
    return render_template('data.html')
@app.route('/algorithm/<name>')
def dynamic(name):
    return render_template('dynamic.html',name=name)
@app.route('/success',methods=['POST'])
def success():
    # ML code to be written here
    # add below
    if(request.method=='POST'):
        data=request.files["dataset"]
        data.save(data.filename)
        print(str(data.filename))
        h= str(data.filename)
        ans=os.path.join(os.path.dirname(__file__),h)
        print(ans)
        df=pd.read_csv(ans)
        #dataset = pd.DataFrame(df.data)
        #print(dataset.head())
        return render_template("success.html")


if __name__=='__main__':
    app.run(debug=True)
