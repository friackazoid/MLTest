#!/usr/bin/env  python

from pprint import pprint
import numpy as np
import pandas as pd

from gensim.models import word2vec
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence

import logging
import os.path
import nltk
import itertools
from collections import namedtuple

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster.hierarchical import AgglomerativeClustering
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap

CATEGORIES=("comp.sys.mac.hardware", "soc.religion.christian", "rec.sport.hockey")
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3'}
MODEL_NAME='20news_300features_40minwords_10context'
DOC2VEC_MODEL_NAME='20news_300features_40minwords_10context_d2v'

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

MailDoc = namedtuple('MailDoc', 'words tags cluster')

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#=====================================================================
# Functions to show model and explain

def explain_word2vec_model (model):

    print "syn0 is a list of feature vectors for each word in the vocabulary"
#    print model.syn0.shape
    print model.wv.syn0norm.shape

    #it is numpy array
#    print type(model.syn0)
    print type(model.wv.syn0norm)

    print "Feature vector of word 'christ'"
    print model['christ']

    print "Feature vector of first word in matrix"
    print  model.syn0[0]

def explain_doc2vec_model (sentences, model):

    print "Labeled Sentence is:"
    pprint (sentences[:2])
    print

    print "Size of data:"
    pprint (len(sentences))
    print

    print "len model.docvecs"
    pprint (len(model.docvecs))

    for doc  in model.docvecs:
        print "Shape of document vector"
        print doc.shape
        doc_r = doc.reshape(1,-1)
        print "Reshape doc vector to fit TSNE alg"
        print doc_r.shape
        pprint (doc_r)
        np.nan_to_num(doc_r)
        print np.isnan(doc_r).all() #False
        print np.isnan(doc).all()
        print "==========\\nn"
        print np.isfinite(doc_r).all() #True
        print np.isinf(doc_r).all() #False
        print "===========\n"
        print np.asarray_chkfinite(doc_r).all()
        print np.asarray_chkfinite(doc).all()


    print "Vector of document ", sentences[3].tags[0] , "\n"
    pprint (model.docvecs[sentences[3].tags[0]])
    print

#--------------------------------------------------------------------
# Print content of dataset to show structure
def show_dataset (dataset):
    print "*** List ofcategories:"
    pprint (list(dataset.target_names))
    print

    print "*** The amount of records:"
    pprint (dataset.filenames.shape)
    pprint (dataset.target.shape)
    print

    print "Target is numerious label of categores:"
    pprint (dataset.target[:10])
    print

    print "Where the data is storied:"
    pprint (dataset.filenames[:2])
    print

    print "Content of data:"
    pprint (dataset.data[:2])

#---------------------------------------------------------------------
# Print content to learn model
def show_prepared (data):

    pprint(data[:2])

#=====================================================================

#=====================================================================
# Functions to prepare text

def sentance_to_wordlist( mail, remove_stopwords=False ):
    import string, re
    #table = string.maketrans("","")
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    #mail.translate(string.punctuation)
    mail  = regex.sub('', mail)
    words= mail.lower().split()
    return( words)

def mail_to_sentances(mail, tokenizer, remove_stopwords=False ):
    raw_sentances = tokenizer.tokenize(mail.strip())

    sentences = []
    for raw_sentance in raw_sentances:
        if len(raw_sentance) > 0:
            sentences.append(sentance_to_wordlist(raw_sentance))

    return sentences

#=====================================================================

#=====================================================================
# Function to build models

def build_doc2vec_model (dataset):

    sentences=[]
    for i in xrange(dataset.target.shape[0]):
        sentence = MailDoc(
                    words=sentance_to_wordlist(dataset.data[i]),
                    tags=[os.path.basename(dataset.filenames[i])],
                    cluster=[dataset.target[i]])
        sentences.append(sentence)

    if os.path.isfile(DOC2VEC_MODEL_NAME):
        print "Load model: ", DOC2VEC_MODEL_NAME
        model = Doc2Vec.load(DOC2VEC_MODEL_NAME)
    else:
        model = Doc2Vec(alpha=0.025, min_alpha=0.025, size=num_features, window=context,
                min_count=min_word_count, dm=1, workers=8, sample=downsampling)
        model.build_vocab(sentences)
        for epoch in range(10):
            model.train(sentences)
            model.alpha -= 0.002  # decrease the learning rate
            model.min_alpha = model.alpha  # fix the learning rate, no decay
        model.save(DOC2VEC_MODEL_NAME)
        
    #explain_doc2vec_model(sentences, model)

#    clusters = []
#    for md in sentences:
#        clusters.append(md.cluster[0])
#    plot_clusters (model.docvecs, clusters, 'real_cluster')
    return model

#====================================================================
def plot_clusters (X, clusters, title):

    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    xs, ys = X[:, 0], X[:, 1]

    plt.figure()
    ax = plt.subplot(111)
    ax.margins(0.05)

    for i, cluster in enumerate(clusters):
        ax.plot(xs[i], ys[i], marker='o',linestyle='',  ms=5,
                color=cluster_colors[cluster], mec='none')
        ax.set_aspect('auto')
        #ax.tick_params(\
        #        axis= 'x',          # changes apply to the x-axi
        #        which='both',      # both major and minor ticks are affected
        #        bottom='off',      # ticks along the bottom edge are off
        #        top='off',         # ticks along the top edge are off
        #        labelbottom='off')
        #ax.tick_params(\
        #        axis= 'y',         # changes apply to the y-axis
        #        which='both',      # both major and minor ticks are affected
        #        left='off',      # ticks along the bottom edge are off
        #        top='off',         # ticks along the top edge are off
        #        labelleft='off')

    #ax.legend(numpoints=1)  #show legend with only 1 point
    plt.title(title)
    plt.savefig(title + '.png')

#=====================================================================
def plot_clusters_old (doc_vecs, docs, title):


    # Build projection
    # We project our 300n vector to 2n vector to visualise it
    # Isomap alg shows best result
    n_neighbors = 30
    clf = Isomap(n_neighbors, n_components=2)
#    clf = MDS(n_components=2,  n_init=1, max_iter=100)
#    clf = TSNE(n_components=2, random_state=1)
    projected = clf.fit_transform(doc_vecs)
    x_min, x_max = np.min(projected, 0), np.max(projected, 0)
    projected = (projected - x_min) / (x_max - x_min)
    xs, ys = projected[:, 0], projected[:, 1]


    #create data frame that has the result of the MDS plus the cluster numbers and titles
#    df = pd.DataFrame(dict(x=xs, y=ys, doc=(docs)))

    #group by cluster
#    groups = df.groupby('label')
#    print df[:2]

#    doc_2d = TSNE(n_components=2, random_state=1).fit_transform(doc_vecs)
#    doc_2d = []
#    for doc in doc_vecs:
#        doc_2d.append(TSNE(n_components=2, random_state=1).fit_transform(doc.reshape(1,-1)))

#    pprint (doc_2d)

    plt.figure()
    ax = plt.subplot(111)
    ax.margins(0.05)

    i = 0
    for doc in docs:
#        pprint(doc_vecs[doc.tags[0]])
#        pos = TSNE(n_components=2, random_state=1).fit_transform(doc_vecs[doc.tags[0]].reshape(1,-1))
#        pos = clf.fit_transform(doc_vecs[doc.tags[0]])
#        pprint (pos)
#        pprint (xs[i])
#        pprint (ys[i])
        ax.plot(xs[i], ys[i], marker='o',linestyle='',  ms=5,
                label=CATEGORIES[doc.cluster[0]], color=cluster_colors[doc.cluster[0]],
                mec='none')
        i = i+ 1
        ax.set_aspect('auto')
        #ax.tick_params(\
        #        axis= 'x',          # changes apply to the x-axi
        #        which='both',      # both major and minor ticks are affected
        #        bottom='off',      # ticks along the bottom edge are off
        #        top='off',         # ticks along the top edge are off
        #        labelbottom='off')
        #ax.tick_params(\
        #        axis= 'y',         # changes apply to the y-axis
        #        which='both',      # both major and minor ticks are affected
        #        left='off',      # ticks along the bottom edge are off
        #        top='off',         # ticks along the top edge are off
        #        labelleft='off')

    #ax.legend(numpoints=1)  #show legend with only 1 point
    plt.title(title)
    plt.savefig(title + '.png')

#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_results(matrix, clusters, title):

    print("Computing MDS embedding")
    clf = MDS(n_components=2, n_init=1, max_iter=100)
    pos = clf.fit_transform(matrix)

    xs, ys = pos[:, 0], pos[:, 1]

    #create data frame that has the result of the MDS plus the cluster numbers and titles
    df = pd.DataFrame(dict(x=xs, y=ys, label=(clusters)))

    #group by cluster
    groups = df.groupby('label')

    plt.figure()
    ax = plt.subplot(111)
    ax.margins(0.05)

    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
                label=CATEGORIES[name], color=cluster_colors[name],
                mec='none')
        ax.set_aspect('auto')
        ax.tick_params(\
                axis= 'x',          # changes apply to the x-axi
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off')
        ax.tick_params(\
                axis= 'y',         # changes apply to the y-axis
                which='both',      # both major and minor ticks are affected
                left='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelleft='off')

    ax.legend(numpoints=1)  #show legend with only 1 point
    #add label in x,y position with the label as the film title
    for i in range(len(df)):
        ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['label'], size=8)

    plt.title(title)
    plt.savefig(title + '.png')


def agglomerative_clustering (matrix):
    print "====== Agglomerative Clustering ==============="

    model = AgglomerativeClustering(n_clusters=3, affinity='cosine', linkage='complete')
    preds = model.fit_predict(matrix)
    clusters =  model.labels_.tolist()

    return (preds, clusters)

def kmean_clustering (matrix):
    from sklearn.cluster import KMeans
    print "====== KMeans Clustering ===================="

    kmean_model = KMeans(n_clusters=3, random_state=42)
    pred_clusters = kmean_model.fit_predict(matrix)
    return pred_clusters

def em_clustering (matrix):
    from sklearn import mixture
    print "=========== EM Clustering =================="

    model =  mixture.GaussianMixture (n_components=3, covariance_type='full').fit(matrix)
    pred = model.predict(matrix)
    return (pred)


train_all = fetch_20newsgroups(subset='train')
simple_dataset = fetch_20newsgroups( subset='train', categories=CATEGORIES)

# Show structere of raw dataset
#show_dataset (simple_dataset)

sentences = []
for mail in simple_dataset.data:
    sentences += mail_to_sentances (mail, tokenizer) 

#show_prepared(sentences)
#raise SystemExit

num_features=300
min_word_count=40
context=10
downsampling=1e-3


#if os.path.isfile(MODEL_NAME):
#    print "Load model: ", MODEL_NAME
#    model = word2vec.Word2Vec.load(MODEL_NAME)
#else:
#    model = word2vec.Word2Vec(iter=1,size=num_features, min_count=min_word_count, \
#                window=context, sample=downsampling)
#    model.build_vocab(sentences)
#    model.train(sentences)
    #memory optimization
    #will not learn this model more
#    model.init_sims(replace=True)
#    model.save(MODEL_NAME)

#explain_word2vec_model(model)

model = build_doc2vec_model(simple_dataset)

# Build projection
# We project our 300n vector to 2n vector to visualise it
# Isomap alg shows best result
n_neighbors = 30
clf = Isomap(n_neighbors, n_components=2)
#    clf = MDS(n_components=2,  n_init=1, max_iter=100)
#    clf = TSNE(n_components=2, random_state=1)
X_projected = clf.fit_transform(model.docvecs)

plot_clusters(X_projected, simple_dataset.target, 'real_cluster')

#(pred, clusters) = agglomerative_clustering (model.syn0)
pred_clusters = kmean_clustering(model.docvecs)
print pred_clusters
print "================================"
print simple_dataset.target

#pred = em_clustering(model.syn0)
#print list(pred)
#print simple_dataset.target

plot_clusters (X_projected, pred_clusters, 'k_mean_cluster' )
