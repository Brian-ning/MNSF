import numpy as np


def read_network(path):
    import pickle
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def genMat(edge_list, node_num):
    """
    输入为有向图
    :param edge_list:
    :param node_num:
    :return:
    """
    mat = np.full((node_num, node_num), 0.0001)
    for pair in edge_list:
        mat[pair[0] - 1][pair[1] - 1] = float(pair[2])
    return mat


def get_least_numbers_big_data_idx(alist, k):
    length = len(alist)
    if not alist or k <= 0 or k > length:
        return
    idx_list = []
    for i in range(k):
        min_num = 0.0
        idx = 0
        for j, item in enumerate(alist):
            if j not in idx_list:
                if item > min_num:
                    min_num = item
                    idx = j
        idx_list.append(idx)
    return idx_list


def gen_adjlist(edge_file_list, pickle_path, node_num, delete_edge=False):
    """
    通过边表建立邻接表
    :param edge_file_list: 每层网络的边表的list，例如：['D:/layer1.edgelist','D:/layer2.edgelist','D:/layer3.edgelist']
    :param pickle_path: 对应网络的pickle文件路径
    :param node_num: 对应网络的ground_truth文件路径
    :param delete_edge: 用于链路预测，如果为True，则删除测试边
    :return:
    """
    network_data = read_network(pickle_path)
    mat = []
    for edge_list in edge_file_list:
        edge = []
        with open(edge_list, "r") as f:
            for line in f:
                words = line.strip().split()
                if delete_edge:
                    if (int(words[0]), int(words[1])) not in network_data['test_edges']:
                        edge.append((int(words[0]), int(words[1]), int(1)))
                else:
                    edge.append((int(words[0]), int(words[1]), int(1)))
        ret = genMat(edge, node_num)
        mat.append(ret)
    return mat


def fusion_cluster(edge_list_file, sg_output_path, pickle_path, ground_truth_file_path, layer_num, node_num, K1, mu, K2,
                   t, tsne=False):
    """
    直接融合网络社团发现
    :param edge_file_list:每层网络的边表的list，例如：['D:/layer1.edgelist','D:/layer2.edgelist','D:/layer3.edgelist']
    :param pickle_path: 对应网络的pickle文件路径
    :param ground_truth_file_path: 对应网络的ground_truth文件路径
    :param node_num: 网络的节点数量
    :param K1: 融合参数
    :param mu: 融合参数
    :param K2: 融合参数
    :param t: 融合参数
    :param tsne: 为True则降至128维，为False则直接使用P进行社团发现
    :return:
    """
    import ReadNormalData
    data = ReadNormalData.ReadNormalData(edge_list_file, sg_output_path, layer_num)
    output_path_list = data.get_output_path_list()
    mat = gen_adjlist(output_path_list, pickle_path, node_num)
    # print(mat)
    from sklearn.utils import Bunch
    data = Bunch(data=mat, labels=np.loadtxt(ground_truth_file_path, dtype=int))
    from fusion import snf_wrapper as sw
    snf_wrapper = sw.snf_wrapper("", "")
    network, ground_truth, best, second = snf_wrapper.fusion(K1, mu, K2, t, data, load_data=False, draw_raw=False)
    if tsne:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=128, random_state=4, method='exact')
        result = tsne.fit_transform(network)
        from sklearn.cluster import SpectralClustering
        labels = SpectralClustering(n_clusters=best).fit_predict(result)
    else:
        import sklearn.cluster as sc
        labels = sc.spectral_clustering(network, n_clusters=best)
    from sklearn.metrics import adjusted_rand_score
    from sklearn.metrics import adjusted_mutual_info_score
    from sklearn.metrics import silhouette_score
    ari = adjusted_rand_score(ground_truth, labels)
    nmi = adjusted_mutual_info_score(ground_truth, labels)
    s = silhouette_score(network, labels)
    print("adjusted_rand_score: %.4f\nnmi: %.4f\nsilhouette_score: %.4f" % (ari, nmi, s))


def fusion_cluster_greedy(edge_list_file, pickle_path, sg_output_path, layer_num, node_num, ground_truth_file_path, k,
                          K1, mu, K2, t):
    """
    使用greedy_modularity_communities进行社团发现
    :param edge_file_list: 同上
    :param pickle_path: 同上
    :param node_num: 同上
    :param ground_truth_file_path: 同上
    :param k: 构造邻接矩阵时需要考虑的最相似的邻居个数
    :return:
    """
    import ReadNormalData
    data = ReadNormalData.ReadNormalData(edge_list_file, sg_output_path, layer_num)
    output_path_list = data.get_output_path_list()
    mat = gen_adjlist(output_path_list, pickle_path, node_num)
    from sklearn.utils import Bunch
    data = Bunch(data=mat, labels=np.loadtxt(ground_truth_file_path, dtype=int))
    from fusion import snf_wrapper as sw
    snf_wrapper = sw.snf_wrapper("", "")
    network, ground_truth, best, second = snf_wrapper.fusion(K1, mu, K2, t, data, load_data=False, draw_raw=False)
    import networkx as nx
    adj_list = np.full((node_num, node_num), int(0))
    for i, vec in enumerate(network):
        vec_tmp = vec.tolist()
        idx_list = get_least_numbers_big_data_idx(vec_tmp, k)
        for j, item in enumerate(idx_list):
            adj_list[i][item] = 1
    graph = nx.from_numpy_array(adj_list)
    ret = list(nx.algorithms.community.greedy_modularity_communities(graph))
    community = []
    for item in ret:
        community.append(list(item))
    label = []
    for i in range(node_num):
        for j in range(len(community)):
            if i in community[j]:
                label.append(j)
                break
    from sklearn.metrics import adjusted_rand_score
    from sklearn.metrics import adjusted_mutual_info_score
    ari = adjusted_rand_score(ground_truth, label)
    nmi = adjusted_mutual_info_score(ground_truth, label)
    print(ari, nmi)


def cluster(edge_list_file, sg_output_path, tmp_path, ebd_output_path, layer_num, ground_truth_file_path, node_num, K1,
            mu, K2, t, tsne=False):
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
    if tsne:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=128, random_state=0, method='exact')
        result = tsne.fit_transform(network)
        print(len(result))
        from sklearn.cluster import SpectralClustering
        labels = SpectralClustering(n_clusters=4).fit_predict(result)
    else:
        import sklearn.cluster as sc
        print(best, second)
        labels = sc.spectral_clustering(network, n_clusters=4)
    from sklearn.metrics import adjusted_rand_score
    from sklearn.metrics import adjusted_mutual_info_score
    from sklearn.metrics import silhouette_score
    ari = adjusted_rand_score(ground_truth, labels)
    nmi = adjusted_mutual_info_score(ground_truth, labels)
    s = silhouette_score(network, labels)
    print("adjusted_rand_score: %.4f\nnmi: %.4f\nsilhouette_score: %.4f" % (ari, nmi, s))


def cluster_greedy(edge_list_file, sg_output_path, tmp_path, ebd_output_path, layer_num, ground_truth_file_path,
                   node_num, k, K1, mu, K2, t):
    """
    使用greedy_modularity_communities进行社团发现
    :param edge_file_list: 同上
    :param pickle_path: 同上
    :param node_num: 同上
    :param ground_truth_file_path: 同上
    :param k: 构造邻接矩阵时需要考虑的最相似的邻居个数
    :return:
    """
    import ReadNormalData
    data = ReadNormalData.ReadNormalData(edge_list_file, sg_output_path, layer_num)
    output_path_list = data.get_output_path_list()
    # from embedding import line
    # ebd = line.line(output_path_list, tmp_path, node_num, ebd_output_path, representation_size=128)
    from embedding import deepwalk
    ebd = deepwalk.deepwalk(output_path_list, tmp_path, node_num, ebd_output_path, representation_size=128)
    ebd_path_dic = ebd.run()
    ebd_path_list = [ebd_path_dic[ele] for ele in ebd_path_dic]
    from fusion import snf_wrapper as sw
    snf_wrapper = sw.snf_wrapper(ebd_path_list, ground_truth_file_path)
    network, ground_truth, best, second = snf_wrapper.fusion(K1, mu, K2, t, "")
    import networkx as nx
    adj_list = np.full((node_num, node_num), int(0))
    for i, vec in enumerate(network):
        vec_tmp = vec.tolist()
        idx_list = get_least_numbers_big_data_idx(vec_tmp, k)
        for j, item in enumerate(idx_list):
            adj_list[i][item] = 1
    graph = nx.from_numpy_array(adj_list)
    ret = list(nx.algorithms.community.greedy_modularity_communities(graph))
    community = []
    for item in ret:
        community.append(list(item))
    label = []
    for i in range(node_num):
        for j in range(len(community)):
            if i in community[j]:
                label.append(j)
                break
    from sklearn.metrics import adjusted_rand_score
    from sklearn.metrics import adjusted_mutual_info_score
    ari = adjusted_rand_score(ground_truth, label)
    nmi = adjusted_mutual_info_score(ground_truth, label)
    print(ari, nmi)


# Example:

print("直接融合")
fusion_cluster("D:/graph/Dataset/CKM-Physicians-Innovation_multiplex.edges",
               "D:/graph/Dataset/layer/",
               "D:/graph/0_LinkPrediction_Graph/CKM_information.pickle",
               "D:/graph/Dataset/ground_truth.csv",
               3,246,80,0.38,20,20)

print("直接融合-降维")
fusion_cluster("D:/graph/Dataset/CKM-Physicians-Innovation_multiplex.edges",
               "D:/graph/Dataset/layer/",
               "D:/graph/0_LinkPrediction_Graph/CKM_information.pickle",
               "D:/graph/Dataset/ground_truth.csv",
               3,246,80,0.38,20,20,tsne=True)

print("直接融合-greedy_modularity_communities")
fusion_cluster_greedy("D:/graph/Dataset/CKM-Physicians-Innovation_multiplex.edges",
                      "D:/graph/0_LinkPrediction_Graph/CKM_information.pickle",
                      "D:/graph/Dataset/layer/",
                      3,246,
                      "D:/graph/Dataset/ground_truth.csv",
                      20,80,0.38,20,20)

print("表示学习")
cluster("D:/graph/Dataset/CKM-Physicians-Innovation_multiplex.edges",
        "D:/graph/Dataset/layer/",
        "D:/graph/Dataset/tmp/",
        "D:/graph/Dataset/embedding/",3,
        "D:/graph/Dataset/ground_truth.csv",246,80,0.38,20,20)

print("表示学习")
cluster("D:/graph/Dataset/CKM-Physicians-Innovation_multiplex.edges",
        "D:/graph/Dataset/layer/",
        "D:/graph/Dataset/tmp/",
        "D:/graph/Dataset/embedding/", 3,
        "D:/graph/Dataset/ground_truth.csv", 246, 80, 0.5, 20, 10)

print("表示学习-greedy_modularity_communities")
cluster_greedy("D:/graph/Dataset/CKM-Physicians-Innovation_multiplex.edges",
               "D:/graph/Dataset/layer/",
               "D:/graph/Dataset/tmp/",
               "D:/graph/Dataset/embedding/", 3,
               "D:/graph/Dataset/ground_truth.csv", 246, 30,80, 0.5, 20, 10)
