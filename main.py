import os
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS
import werkzeug

import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import math

app = Flask(__name__)
cors = CORS(app)
api = Api(app)

amount = 5000

class tsne(Resource):
    def get(self, id):
        getid = id
        if(getid == 'c') :
            data = np.load('./0_trainData.npy')
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
            data = np.load('./0_trainData.npy')
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
            data = np.load('./0_trainData.npy')
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
            data = np.load('./0_trainData.npy')
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
            data = np.load('./0_trainData.npy')
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
            data = np.load('./0_trainData.npy')
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
            data = np.load('./0_trainData.npy')
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
            data = np.load('./0_trainData.npy')
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


api.add_resource(tsne, '/api/tsne/<string:id>')

if __name__ == '__main__':
    app.run(debug=True)