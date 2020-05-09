import os
import networkx as nx
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from generate_train_test_datasets import load_pickle, save_as_pickle, generate_text_graph
from models import gcn
from evaluate_results import evaluate_model_results
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def load_datasets(args):
    logger.info("Loading data...")
    df_data_path = "./data/gcc_df_data_description.pkl"
    graph_path = "./data/gcc_text_graph_description.pkl"
    if not os.path.isfile(df_data_path) or not os.path.isfile(graph_path):
        logger.info("Building datasets and graph from raw data... Note this will take quite a while...")
        generate_text_graph()
    df_data = load_pickle("gcc_df_data_description.pkl")
    G = load_pickle("gcc_text_graph_description.pkl")
    
    logger.info("Building adjacency and degree matrices...")
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
    
    logger.info("Splitting labels for training and inferring...")
    ### stratified test samples
    test_idxs = []
    for b_id in df_data["b"].unique():
        dum = df_data[df_data["b"] == b_id]
        if len(dum) >= 10:
            test_idxs.extend(list(np.random.choice(dum.index, size=round(args.test_ratio * len(dum)), replace=False)))
    save_as_pickle("gcc_test_idxs_02_description.pkl", test_idxs)
    # select only certain labelled nodes for semi-supervised GCN
    selected = []
    for i in range(len(df_data)):
        if i not in test_idxs:
            selected.append(i)
    save_as_pickle("gcc_selected_02_description.pkl", selected)
    f_selected = f[selected]; f_selected = torch.from_numpy(f_selected).float()
    labels_selected = [l for idx, l in enumerate(df_data["b"]) if idx in selected]
    f_not_selected = f[test_idxs]; f_not_selected = torch.from_numpy(f_not_selected).float()
    labels_not_selected = [l for idx, l in enumerate(df_data["b"]) if idx not in selected]
    f = torch.from_numpy(f).float()
    save_as_pickle("gcc_labels_selected_02_description.pkl", labels_selected)
    save_as_pickle("gcc_labels_not_selected_02_description.pkl", labels_not_selected)
    logger.info("Split into %d train and %d test lebels." % (len(labels_selected), len(labels_not_selected)))
    return f, X, A_hat, selected, labels_selected, labels_not_selected, test_idxs

def evaluate(output, labels_e):
    _, labels = output.max(1); labels = labels.numpy()
    return sum([(e-1) for e in labels_e] == labels) / len(labels)

def calculate_f_score(right, predict, class_num):
    _, predict = predict.max(1)
    predict = predict.numpy()
    # print(predict)
    right = [x - 1 for x in right]
    result = []
    for i in range(class_num):
        result.append([0, 0, 0])
    assert len(right) == len(predict)
    # print("precision/recall/F-measure")
    for i in range(len(right)):
        #print(right[i],predict[i])
        result[right[i]][1] += 1
        result[predict[i]][2] += 1
        if right[i] == predict[i]:
            result[predict[i]][0] += 1
    f_score = []
    average_f_score = 0
    for i in range(class_num):
        p = result[i][0] / (result[i][2] or 1)
        r = result[i][0] / (result[i][1] or 1)
        f_score.append((2*p*r) / ((p+r) or 1))
        average_f_score += (2*p*r) / ((p + r) or 1)
        # print(p, r, f_score[i])
    average_f_score = float(str(average_f_score / class_num))
    return average_f_score

if __name__ == "__main__":
    weightfile = "./data/gcc_02_description.pkl"
    parser = ArgumentParser()
    parser.add_argument("--hidden_size_1", type=int, default=200, help="Size of first GCN hidden weights")
    parser.add_argument("--hidden_size_2", type=int, default=100, help="Size of second GCN hidden weights")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of prediction classes")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Ratio of test to training nodes")
    parser.add_argument("--num_epochs", type=int, default=300, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.02, help="learning rate")
    parser.add_argument("--model_no", type=int, default=0, help="Model ID")
    args = parser.parse_args()
    save_as_pickle("gcc_args_02_description.pkl", args)
    
    f, X, A_hat, selected, labels_selected, labels_not_selected, test_idxs = load_datasets(args)
    net = gcn(X.shape[1], A_hat, args)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,200,2600,2700,2800,2900], gamma=0.77)
    logger.info("Starting training process...")
    net.train()
    for e in range(args.num_epochs):
        optimizer.zero_grad()
        output = net(f)
        loss = criterion(output[selected], torch.tensor(labels_selected).long() -1)
        loss.backward()
        optimizer.step()
        net.eval()
        pred_labels = net(f)
        untrained_accuracy = evaluate(pred_labels[test_idxs], labels_not_selected)
        trained_accuracy = evaluate(pred_labels[selected], labels_selected)
        ave_f_score = calculate_f_score(labels_not_selected, pred_labels[test_idxs], 5)
        print("epoch:", e, "loss:", loss.item(), "ave_f_score:", ave_f_score, trained_accuracy)
        torch.save(net, weightfile)
  
    logger.info("Finished training!")

    logger.info("Evaluate results...")

