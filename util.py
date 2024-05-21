# 作者: not4ya
# 时间: 2023/10/13 17:41

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix


def load_data():
    """加载数据集，获取同正样本相同数量的负样本"""
    herb_sim = pd.read_csv("data/sim/herb_herb_sim.csv", index_col=0, dtype=np.float32).to_numpy()
    target_sim = pd.read_csv("data/sim/target_target_sim.csv", index_col=0, dtype=np.float32).to_numpy()
    herb_target_ass = pd.read_csv("data/adj/adj.csv", index_col=0).to_numpy()
    herb_sim = herb_sim - np.diag(np.diag(herb_sim))
    target_sim = target_sim - np.diag(np.diag(target_sim))

    rng = np.random.default_rng(10086)
    pos_samples, edge_attr = dense2sparse(herb_target_ass)
    pos_samples_shuffled = rng.permutation(pos_samples, axis=1)

    neg_samples = np.where(herb_target_ass == 0)
    neg_samples_shuffled = rng.permutation(neg_samples, axis=1)[:, :pos_samples_shuffled.shape[1]]

    pos_samples = np.column_stack((pos_samples_shuffled.T, np.ones(pos_samples_shuffled.shape[1])))
    neg_samples = np.column_stack((neg_samples_shuffled.T, np.zeros(neg_samples_shuffled.shape[1])))
    samples = np.vstack((pos_samples, neg_samples))

    return herb_sim, target_sim, samples, pos_samples_shuffled, neg_samples_shuffled


def dense2sparse(matrix):
    """稀疏矩阵转化为索引"""
    mat_coo = coo_matrix(matrix)
    edge_idx = np.vstack((mat_coo.row, mat_coo.col))
    return edge_idx, mat_coo.data


def trans(sample):
    """样本转化为正负索引"""
    col_values, row_values, value_values = sample[:, 0], sample[:, 1], sample[:, 2]
    array_1 = np.column_stack((col_values[value_values == 1], row_values[value_values == 1])).T
    array_0 = np.column_stack((col_values[value_values == 0], row_values[value_values == 0])).T
    return array_1, array_0


def get_syn_sim(a, seq_sim, str_sim, mode):
    """拼接相似矩阵"""
    # GIP_c_sim = gip_kernel(a)
    # GIP_d_sim = gip_kernel(a.T)
    GIP_c_sim = gip_sparse(a)
    GIP_d_sim = gip_sparse(a.T)

    if mode == 0:
        return GIP_c_sim, GIP_d_sim

    syn_c = np.zeros((a.shape[0], a.shape[0]))
    syn_d = np.zeros((a.shape[1], a.shape[1]))

    for i in range(a.shape[0]):
        for j in range(a.shape[0]):
            if seq_sim[i, j] == 0:
                syn_c[i, j] = GIP_c_sim[i, j]
            else:
                syn_c[i, j] = (GIP_c_sim[i, j] + seq_sim[i, j]) / 2

    for i in range(a.shape[1]):
        for j in range(a.shape[1]):
            if str_sim[i, j] == 0:
                syn_d[i, j] = GIP_d_sim[i, j]
            else:
                syn_d[i, j] = (GIP_d_sim[i, j] + str_sim[i, j]) / 2
    return syn_c, syn_d


def gip_sparse(a):
    """GIP核计算(大型稀疏矩阵)"""
    a_sparse = csr_matrix(a)
    norms = np.sqrt(a_sparse.power(2).sum(axis=1).A1)
    sigma = np.mean(norms ** 2)
    diff_norms_squared = np.zeros((a.shape[0], a.shape[0]))
    for i in range(a.shape[0]):
        row_repeated = csr_matrix(np.repeat(a_sparse[i, :].toarray(), a.shape[0], axis=0))
        diff = a_sparse - row_repeated
        diff_norms_squared[i, :] = diff.power(2).sum(axis=1).A1
    matrix = np.exp(-diff_norms_squared / sigma)
    np.fill_diagonal(matrix, 1)
    return matrix


def gip_kernel(a):
    """GIP核计算"""
    nc = a.shape[0]
    matrix = np.zeros((nc, nc))
    r = row_norm(a)
    for i in range(nc):
        for j in range(nc):
            temp_up = np.square(np.linalg.norm(a[i, :] - a[j, :]))
            if r == 0:
                matrix[i][j] = 0
            elif i == j:
                matrix[i][j] = 1
            else:
                matrix[i][j] = np.e ** (-temp_up / r)
    return matrix


def row_norm(a):
    """计算行范数"""
    nc = a.shape[0]
    summ = 0
    for i in range(nc):
        x_norm = np.linalg.norm(a[i, :])
        x_norm = np.square(x_norm)
        summ = summ + x_norm
    r = summ / nc
    return r


def k_matrix(matrix, k=20):
    """k-近邻图：每个节点与其 k 个最相似（或最近的）节点相连"""
    num = matrix.shape[0]
    knn_graph = np.zeros_like(matrix)
    idx_sort = np.argsort(-(matrix - np.eye(num)), axis=1)
    for i in range(num):
        top_k_indices = idx_sort[i, :k]
        knn_graph[i, top_k_indices] = matrix[i, top_k_indices]
        knn_graph[top_k_indices, i] = matrix[top_k_indices, i]
    np.fill_diagonal(knn_graph, 1)
    return knn_graph


def construct_adj_mat(training_mask):
    """构建异构矩阵（不包含自相似矩阵）"""
    adj_tmp = training_mask.copy()
    herb_mat = np.zeros((training_mask.shape[0], training_mask.shape[0]))
    target_mat = np.zeros((training_mask.shape[1], training_mask.shape[1]))

    mat1 = np.hstack((herb_mat, adj_tmp))
    mat2 = np.hstack((adj_tmp.T, target_mat))
    ret = np.vstack((mat1, mat2))
    return ret


def construct_het_mat(training_mask, herb_mat, target_mat):
    """构建异构矩阵（包含自相似矩阵）"""
    mat1 = np.hstack((herb_mat, training_mask))
    mat2 = np.hstack((training_mask.T, target_mat))
    ret = np.vstack((mat1, mat2))
    return ret


def calculate_loss(predict, pos_edge_idx, neg_edge_idx):
    """Loss计算"""
    pos_predict = predict[pos_edge_idx[0], pos_edge_idx[1]]
    neg_predict = predict[neg_edge_idx[0], neg_edge_idx[1]]
    predict_scores = torch.hstack((pos_predict, neg_predict))
    true_labels = torch.hstack((torch.ones(pos_predict.shape[0]), torch.zeros(neg_predict.shape[0])))
    loss_fun = torch.nn.BCEWithLogitsLoss(reduction='mean')
    return loss_fun(predict_scores, true_labels)


def calculate_evaluation_metrics_group(predict, pos_edges, neg_edges, herb_h, herb_l, target_h, target_l):
    """结果评估"""
    pos_edges = pos_edges.astype(int)
    neg_edges = neg_edges.astype(int)

    results = {}
    groups = ["HH", "HL", "LH", "LL"]
    for group in groups:
        if group == "HH":
            pos_mask = np.isin(pos_edges[0], herb_h) & np.isin(pos_edges[1], target_h)
            neg_mask = np.isin(neg_edges[0], herb_h) & np.isin(neg_edges[1], target_h)
        elif group == "HL":
            pos_mask = np.isin(pos_edges[0], herb_h) & np.isin(pos_edges[1], target_l)
            neg_mask = np.isin(neg_edges[0], herb_h) & np.isin(neg_edges[1], target_l)
        elif group == "LH":
            pos_mask = np.isin(pos_edges[0], herb_l) & np.isin(pos_edges[1], target_h)
            neg_mask = np.isin(neg_edges[0], herb_l) & np.isin(neg_edges[1], target_h)
        else:  # "LL"
            pos_mask = np.isin(pos_edges[0], herb_l) & np.isin(pos_edges[1], target_l)
            neg_mask = np.isin(neg_edges[0], herb_l) & np.isin(neg_edges[1], target_l)
        pos_predict_group = predict[pos_edges[0][pos_mask], pos_edges[1][pos_mask]]
        neg_predict_group = predict[neg_edges[0][neg_mask], neg_edges[1][neg_mask]]
        predict_labels_group = np.hstack((pos_predict_group, neg_predict_group))
        true_labels_group = np.hstack((np.ones(pos_predict_group.shape[0]), np.zeros(neg_predict_group.shape[0])))
        results[group] = get_metrics(true_labels_group, predict_labels_group)

    return results


def calculate_evaluation_metrics(predict, pos_edges, neg_edges):
    """结果评估"""
    pos_edges = pos_edges.astype(int)
    neg_edges = neg_edges.astype(int)
    pos_predict = predict[pos_edges[0], pos_edges[1]]
    neg_predict = predict[neg_edges[0], neg_edges[1]]
    predict_labels = np.hstack((pos_predict, neg_predict))
    true_labels = np.hstack((np.ones(pos_predict.shape[0]), np.zeros(neg_predict.shape[0])))
    return get_metrics(true_labels, predict_labels)


def get_metrics(real_score, predict_score):
    """混淆矩阵"""
    real_score, predict_score = real_score.flatten(), predict_score.flatten()

    sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
    thresholds = sorted_predict_score[np.int32(len(sorted_predict_score) * np.arange(1, 1000) / 1000)]
    thresholds = np.mat(thresholds)

    predict_score_matrix = np.tile(predict_score, (thresholds.shape[1], 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1

    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN

    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]

    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T

    auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])

    recall_list = tpr
    precision_list = TP / (TP + FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]

    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)

    # plt.plot(x_ROC, y_ROC)
    # plt.plot(x_PR, y_PR)
    # plt.show()

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    return [auc[0, 0], aupr[0, 0], f1_score, accuracy, recall, specificity, precision], x_ROC, y_ROC, x_PR, y_PR


def compute_network_nsp(sim1, sim2, adj):
    """网络一致性投影"""
    temp_matrix1 = np.dot(sim1, adj)
    modulus1 = np.linalg.norm(adj, axis=0).reshape(1, -1)
    sim1_proj_result = temp_matrix1 / modulus1

    temp_matrix2 = np.dot(adj, sim2)
    modulus2 = np.linalg.norm(adj, axis=1).reshape(-1, 1)
    sim2_proj_result = temp_matrix2 / modulus2

    index_modulus = np.linalg.norm(sim1, axis=1).reshape(-1, 1)
    columns_modulus = np.linalg.norm(sim2, axis=0).reshape(1, -1)
    modulus_sum = index_modulus + columns_modulus

    result = np.nan_to_num((np.nan_to_num(sim2_proj_result) + np.nan_to_num(sim1_proj_result)) / modulus_sum)

    return result
