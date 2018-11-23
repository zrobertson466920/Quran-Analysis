import glob

import re, nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt

import networkx as nx


# Get name for path
def convert(path):
    return path[5:8]


# Convert list of n-grams strings to list of tuples
def ngram_convert(list):
    bi_list = []
    for item in list:
        bi_list.append(tuple(item.split()))
    return bi_list


if __name__ == '__main__':

    # Collect Paths to surahs in the Quran
    text_paths = []
    for filepath in glob.iglob('data/*.txt'):
        text_paths.append(filepath)
    text_paths.sort()

    # Create corpus dictionary sorted by surah
    corpus = {}
    for path in text_paths:
        label = convert(path)
        if label[0] not in ["0","1"]:
            continue
        fh = open(path, "r")
        try:
            corpus[label] += fh.readline().lower() + '\n'
        except:
            corpus[label] = []
            corpus[label] += fh.readline().lower() + '\n'
        fh.close()

    # Load information about location of surahs
    path = "data/location.txt"
    fh = open(path, "r")
    locations = fh.readlines()
    for n in range(0,len(locations)):
        locations[n] = re.sub('[^a-zA-Z]', '', locations[n])
    fh.close()

    # Load information about chronological order of surahs
    path = "data/chrono_order.txt"
    fh = open(path, "r")
    chrono = fh.readlines()
    for n in range(0,len(chrono)):
        chrono[n] = re.sub('[^0-9]', '', chrono[n])
    fh.close()

    # Create new corpus dictionary sorted by chronological order
    o_corpus = {}
    for index in range(0,len(chrono)):
        o_corpus[chrono[index].zfill(3)] = corpus[str(index+1).zfill(3)]

    # Create blob of All Surahs
    blob = ""
    count = []
    tokenizer = RegexpTokenizer(r'\w+')
    for surah in corpus:
        count.append(len(tokenizer.tokenize("".join(o_corpus[surah]))))
        blob += "".join(o_corpus[surah])

    vectorizer = CountVectorizer(stop_words = "english",ngram_range=(2,2))
    X = vectorizer.fit_transform([blob])
    y = np.array(vectorizer.get_feature_names())
    D = X.toarray()

    print("Median Length of Surahs")
    print(np.median(count))
    print()

    print("Phrases of the Quran")
    ngrams = ngram_convert(y[D[0].argsort()[-500:][::-1]])
    print(y[D[0].argsort()[-30:][::-1]])
    print()

    G = nx.Graph()
    edge_color = []
    for index in D[0].argsort()[-1000:][::-1]:
        item = tuple(y[index].split())
        for datum in item:
            G.add_node(datum)
        for j in range(0, len(item) - 1):
            G.add_edge(item[j], item[j + 1], weight=np.log(D[0][index]))

    cur_graph = G  # whatever graph you're working with

    if not nx.is_connected(cur_graph):
        # get a list of unconnected networks
        sub_graphs = list(nx.connected_component_subgraphs(cur_graph))

        main_graph = sub_graphs[0]

        # find the largest network in that list
        for sg in sub_graphs:
            if len(sg.nodes()) > len(main_graph.nodes()):
                main_graph = sg

    cur_graph.remove_nodes_from(["allah","lord","time"])
    remove = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] < np.log(6)]
    cur_graph.remove_edges_from(remove)
    remove = [node for node, degree in cur_graph.degree() if degree < 3]
    cur_graph.remove_nodes_from(remove)
    remove = [node for node, degree in cur_graph.degree() if degree < 1]
    cur_graph.remove_nodes_from(remove)

    G = cur_graph
    cmap = plt.cm.get_cmap("inferno")
    for (u, v, d) in G.edges(data=True):
        edge_color.append(d['weight'])

    print("Plotting Graph")
    plt.figure()
    nx.draw_spectral(G, node_color='#A0CBE2', with_labels=True, font_weight='bold', font_size=20, edge_cmap=cmap,
            edge_color=edge_color)
    plt.show()