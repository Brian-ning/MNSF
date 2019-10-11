import numpy as np


def read_network(path):
    import pickle
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def link_prediction(edge_list_file, sg_output_path, pickle_file, ground_truth_file_path, layer_num, tmp_path,
                    ebd_output_path, node_num, K1, mu, K2, t, tsne=False):
    """

    :param edge_list_file: 边表文件
    :param sg_output_path: 单层网络边表输出路径
    :param pickle_file: pickle文件路径
    :param ground_truth_file_path: ground_truth路径
    :param layer_num: 网络层数
    :param tmp_path: 临时文件路径
    :param ebd_output_path: embedding输出路径
    :param node_num: 网络节点总数
    :param K1: 模型参数
    :param mu: 参数
    :param K2: 参数
    :param t: 参数
    :param tsne: 是否降维
    :return:
    """
    import ReadNormalData
    g_data = ReadNormalData.ReadNormalData(edge_list_file, sg_output_path, layer_num)
    output_path_list = g_data.get_output_path_list()
    from embedding import line
    ebd = line.line(output_path_list, tmp_path, node_num, ebd_output_path, representation_size=128)
    ebd_path_dic = ebd.run()
    ebd_path_list = [ebd_path_dic[ele] for ele in ebd_path_dic]
    from fusion import snf_wrapper as sw
    snf_wrapper = sw.snf_wrapper(ebd_path_list, ground_truth_file_path)
    network, ground_truth, best, second = snf_wrapper.fusion(K1, mu, K2, t, "")
    data = read_network(pickle_file)
    pos = data["test_edges"][:int(len(data["test_edges"]) / 2)]
    neg = data["test_edges"][int(len(data["test_edges"]) / 2):]
    train = pos[:int(len(pos) * 7 / 10)] + neg[:int(len(neg) * 7 / 10)]
    test = pos[int(len(pos) * 7 / 10):] + neg[int(len(neg) * 7 / 10):]
    label = data["test_labels"].tolist()
    pos_label = label[:int(len(label) / 2)]
    neg_label = label[int(len(label) / 2):]
    Y_train_ = pos_label[:int(len(pos_label) * 7 / 10)] + neg_label[:int(len(neg_label) * 7 / 10)]
    Y_test_ = pos_label[int(len(pos_label) * 7 / 10):] + neg_label[int(len(neg_label) * 7 / 10):]
    X_train_ = []
    X_test_ = []
    if tsne:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=128, random_state=4, method='exact')
        result = tsne.fit_transform(network)
    else:
        result = network
    for edge in train:
        u_ebd = result[int(edge[0]) - 1] * result[int(edge[1]) - 1]
        X_train_.append(u_ebd)
    print(len(X_train_))
    for edge in test:
        u_ebd = result[int(edge[0]) - 1] * result[int(edge[1]) - 1]
        X_test_.append(u_ebd)
    print(len(X_test_))
    from sklearn import metrics
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier()
    clf.fit(X_train_, Y_train_)
    y = clf.predict(X_test_)
    yp = clf.predict_proba(X_test_)
    yp_ = []
    for item in yp:
        prob = item[1]
        yp_.append(prob)
    fpr, tpr, thresholds = metrics.roc_curve(Y_test_, yp_, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print(auc)


# Example:
link_prediction("D:/graph/Dataset/CKM-Physicians-Innovation_multiplex.edges",
                "D:/graph/Dataset/layer/",
                "D:/graph/0_LinkPrediction_Graph/CKM_information.pickle",
                "D:/graph/Dataset/ground_truth.csv",
                3,
                "D:/graph/Dataset/tmp/",
                "D:/graph/Dataset/embedding/",
                246, 80, 0.38, 20, 20, tsne=True)
