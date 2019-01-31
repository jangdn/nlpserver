import os
from flask import Flask, request, jsonify, send_from_directory, send_file, render_template, Response
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS
import werkzeug
#sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from bidaf_flask import execute_bidaf

import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import math

app = Flask(__name__, static_url_path='')
cors = CORS(app)
api = Api(app)

amount = 5000

temprate = 0



class tsne(Resource):
    def get(self, id):
        getid = id
        if(getid == 'c') :
            print("start tsnec")
            data = np.load('./data/0_trainData.npy')
            contextSet = []
            context_cSet = np.array([[0]*200])
            for a in range(0, 100):
                contextSet= contextSet+data[a]['context']
                context_cSet = np.concatenate((context_cSet, data[a]['save_data']['context_c'][0]))
            context_cSet = np.delete(context_cSet, 0, 0)
            model = TSNE(n_iter=2000)
            transformed = model.fit_transform(context_cSet[:amount])
            xs = transformed[:,0]
            ys = transformed[:,1]
            return {'context': contextSet[:amount], "transform": transformed.tolist(), 'xs': (xs.tolist()), 'ys': (ys.tolist())}

        elif(getid == 'cc') :
            print("start tsnecc")
            data = np.load('./data/0_trainData.npy')
            contextSet = []
            char_cSet = np.array([[0]*100])
            for a in range(0, 100):
                contextSet= contextSet+data[a]['context']
                char_cSet = np.concatenate((char_cSet, data[a]['save_data']['c_char'][0]))
            char_cSet = np.delete(char_cSet, 0, 0)
            model = TSNE(n_iter=2000)
            transformed = model.fit_transform(char_cSet[:amount])
            xs = transformed[:,0]
            ys = transformed[:,1]
            return {'context': contextSet[:amount], "transform": transformed.tolist(), 'xs': (xs.tolist()), 'ys': (ys.tolist())}
        
        elif(getid == 'cq') :
            print("start tsnecq")
            data = np.load('./data/0_trainData.npy')
            questionSet = []
            char_qSet = np.array([[0]*100])
            for a in range(0, 100):
                questionSet= questionSet+data[a]['question']
                char_qSet = np.concatenate((char_qSet, data[a]['save_data']['q_char'][0]))
            char_qSet = np.delete(char_qSet, 0, 0)
            model = TSNE(n_iter=2000)
            transformed = model.fit_transform(char_qSet[:amount])
            xs = transformed[:,0]
            ys = transformed[:,1]
            return {'question': questionSet[:amount], "transform": transformed.tolist(), 'xs': (xs.tolist()), 'ys': (ys.tolist())}

        elif(getid == 'hc') :
            print("start tsnehc")
            data = np.load('./data/0_trainData.npy')
            contextSet = []
            highway_cSet = np.array([[0]*200])
            for a in range(0, 100):
                contextSet= contextSet+data[a]['context']
                highway_cSet = np.concatenate((highway_cSet, data[a]['save_data']['highway_c'][0]))
            highway_cSet = np.delete(highway_cSet, 0, 0)
            model = TSNE(n_iter=2000)
            transformed = model.fit_transform(highway_cSet[:amount])
            xs = transformed[:,0]
            ys = transformed[:,1]
            return {'context': contextSet[:amount], "transform": transformed.tolist(), 'xs': (xs.tolist()), 'ys': (ys.tolist())}

        elif(getid == 'hq') :
            print("start tsnehq")
            data = np.load('./data/0_trainData.npy')
            questionSet = []
            highway_qSet = np.array([[0]*200])
            for a in range(0, 100):
                questionSet= questionSet+data[a]['question']
                highway_qSet = np.concatenate((highway_qSet, data[a]['save_data']['highway_q'][0]))
            highway_qSet = np.delete(highway_qSet, 0, 0)
            model = TSNE(n_iter=2000)
            transformed = model.fit_transform(highway_qSet[:amount])
            xs = transformed[:,0]
            ys = transformed[:,1]
            return {'question': questionSet[:amount], "transform": transformed.tolist(), 'xs': (xs.tolist()), 'ys': (ys.tolist())}

        elif(getid == 'q') :
            print("start tsneq")
            data = np.load('./data/0_trainData.npy')
            questionSet = []
            context_qSet = np.array([[0]*200])
            for a in range(0, 100):
                questionSet= questionSet+data[a]['question']
                context_qSet = np.concatenate((context_qSet, data[a]['save_data']['context_q'][0]))
            context_qSet = np.delete(context_qSet, 0, 0)
            model = TSNE(n_iter=2000)
            transformed = model.fit_transform(context_qSet[:amount])
            xs = transformed[:,0]
            ys = transformed[:,1]
            return {'question': questionSet[:amount], "transform": transformed.tolist(), 'xs': (xs.tolist()), 'ys': (ys.tolist())}

        elif(getid == 'wc') :
            print("start tsnewc")
            data = np.load('./data/0_trainData.npy')
            contextSet = []
            word_cSet = np.array([[0]*100])
            for a in range(0, 100):
                contextSet= contextSet+data[a]['context']
                word_cSet = np.concatenate((word_cSet, data[a]['save_data']['c_word'][0]))
            word_cSet = np.delete(word_cSet, 0, 0)
            model = TSNE(n_iter=2000)
            transformed = model.fit_transform(word_cSet[:amount])
            xs = transformed[:,0]
            ys = transformed[:,1]
            return {'context': contextSet[:amount], "transform": transformed.tolist(), 'xs': (xs.tolist()), 'ys': (ys.tolist())}

        elif(getid == 'wq') :
            print("start tsnewq")
            data = np.load('./data/0_trainData.npy')
            questionSet = []
            word_qSet = np.array([[0]*100])
            for a in range(0, 100):
                questionSet= questionSet+data[a]['question']
                word_qSet = np.concatenate((word_qSet, data[a]['save_data']['q_word'][0]))
            word_qSet = np.delete(word_qSet, 0, 0)
            model = TSNE(n_iter=2000)
            transformed = model.fit_transform(word_qSet[:amount])
            xs = transformed[:,0]
            ys = transformed[:,1]
            return {'question': questionSet[:amount], "transform": transformed.tolist(), 'xs': (xs.tolist()), 'ys': (ys.tolist())}

class data(Resource):
    def post(self):
        global temprate
        temprate = 0
        print(request.get_json())
        return {"success" : 1}

class datarate(Resource):
    def get(self):
        global temprate
        temprate += 10
        return temprate


api.add_resource(data, '/api/data')
api.add_resource(datarate, '/api/datarate')
api.add_resource(tsne, '/api/tsne/<string:id>')

@app.route('/api/bidaf')
def bidaf():
    if request.method == 'GET':
        result = execute_bidaf
        print(result)
        return result

@app.route('/api/graph/<string:page_name>')
def static_page(page_name):
    if request.method == 'GET':
        return render_template('%s.html' % page_name)

if __name__ == '__main__':
    app.run(debug=True)