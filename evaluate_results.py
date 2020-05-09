import networkx as nx
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sb
import os
from generate_train_test_datasets import load_pickle
from models import gcn
from sklearn.metrics import confusion_matrix
import pandas as pd
from tqdm import tqdm
import logging
from utils import *

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def evaluate_model_results(args=None):
    logger.info("Loading dataset and model for evaluation...")
    if args == None:
        args = load_pickle("gcc_args_02_description.pkl")

    ### Loads graph data
    G = load_pickle("gcc_text_graph_description.pkl")
    A = nx.to_numpy_matrix(G, weight="weight"); A = A + np.eye(G.number_of_nodes())
    degrees = []
    for d in G.degree(weight=None):
        if d == 0:
            degrees.append(0)
        else:
            degrees.append(d[1]**(-0.5))
    degrees = np.diag(degrees)
    X = np.eye(G.number_of_nodes()) # Features are just identity matrix
    A_hat = degrees@A@degrees
    f = X # (n X n) X (n X n) x (n X n) X (n X n) input of net
    f = torch.from_numpy(f).float()
    
    logger.info("Loading labels...")
    ### Loads labels
    test_idxs = load_pickle("gcc_test_idxs_02_description.pkl")
    selected = load_pickle("gcc_selected_02_description.pkl")
    labels_selected = load_pickle("gcc_labels_selected_02_description.pkl")
    labels_not_selected = load_pickle("gcc_labels_not_selected_02_description.pkl")
    
    ### predict the test
    weightfile = "./data/gcc_02_description.pkl"
    net = torch.load(weightfile)
    net.eval()
    pred_labels = net(f)
    _, predict = pred_labels[test_idxs].max(1)
    predict = predict.numpy()
    labels_not_selected = [x - 1 for x in labels_not_selected]
    calculate_f_score(labels_not_selected, predict, 5)
    

if __name__=="__main__":
    evaluate_model_results()
    