import gc
import ast
import copy
import pickle
import numpy as np
import networkx as nx
from Code.MNE import MNE
from Code.MNE import main
import openne.graph as opgraph
from Code.ohmnet import ohmnet
import Code.Node2vec as Node2vec
import openne.node2vec as node2vec
import Code.ReadMultiplexNetwork as RMN
from Code.MELL.MELL.MELL import MELL_model
import Code.Network_Fusion.Cluster as NFC
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils.validation import check_symmetric
from Code.Network_Fusion.fusion.snf.compute import _B0_normalized, _find_dominate_set

Parameter = {
    "p":2,
    "q":1,
    "num_walks":20,
    "walk_length":30,
    "dimensions":128,
}

def Main_function():

    # 数据加载阶段
    graphs_path = './Code/'
    graph_datasets = RMN.read_graph_pickle(graphs_path)

    # 表示学习参数设置阶段
    p = Parameter["p"]
    q = Parameter["q"]
    num_walks = Parameter["num_walks"]
    walk_length = Parameter["walk_length"]
    dimensions = Parameter["dimensions"]
    knei = [10, 15, 20, 25]
    mu = [0.4, 0.5, 0.6]
    for name, dets in graph_datasets.items():
        print("---------------%s---------------"%name)
        wvecs = []

        # 训练数据集的加载与测试
        nx_graph = dets['train_ng']
        merge_graph = dets['train_mg']

        # 测试验证集的加载与验证
        train_edges = []
        ground_truth = []
        test_edges = dets["test_edges"]
        test_labels = dets["test_labels"]

        # 对网络中的节点标签进行修改，需要进行排序
        nodes = sorted(list(merge_graph.nodes()))
        if nodes[0] > 0:
            train_edges.extend([[i, e[0] - 1, e[1] - 1, 1] for i in range(len(nx_graph)) for e in nx_graph[i].edges()])
            train_merge = nx.relabel_nodes(merge_graph, lambda x: int(x) - 1)
            train_nxgraph = [nx.relabel_nodes(g, lambda x: int(x) - 1) for g in nx_graph]
            test_edges = [[e[0]-1, e[1]-1] for i in test_edges for e in i]
            nodes = list(train_merge.nodes())
        else:
            train_edges.extend([[i, e[0], e[1], 1] for i in range(len(nx_graph)) for e in nx_graph[i].edges()])
            train_nxgraph = copy.deepcopy(nx_graph)
            train_merge = copy.deepcopy(merge_graph)

        # 有的节点编号并不是连续的，下面语句是为了使节点的编号连续
        restru_test_edges = []
        for i in test_edges:
            restru_test_edges.append([[nodes.index(e[0]), nodes.index(e[1])] for e in i])
        str_graph = nx.relabel_nodes(train_merge, lambda x: str(x))

        # 下面操作的是opennet定义的网络，为了使用现有的单层网络算法做对比
        G = opgraph.Graph()
        DG = str_graph.to_directed()
        G.read_g(DG)
        nx_para_graph = []
        for g in train_nxgraph:
            str_graph = nx.relabel_nodes(g, lambda x: str(x))
            G = opgraph.Graph()
            DG = str_graph.to_directed()
            G.read_g(DG)
            nx_para_graph.append(G)

        ################################对比实验部分###############################
        #1# merge_network
        auc = []
        for index, layer in enumerate(restru_test_edges):
            y_pred = []
            for e in layer:
                if e[0] in train_merge.nodes() and e[1] in train_merge.nodes():
                    y_pred.append(list(nx.adamic_adar_index(train_merge, [e]))[0][2])
                else:
                    y_pred.append(0) # 当不存在这个节点的时候，应该概率为0
            auc.append(roc_auc_score(test_labels[index], y_pred))
        print("merge-network:%f"%(sum(auc)/len(auc)))

        #2# Ohmnet 实现多层网络嵌入 Bioinformatics'2017
        ohmnet_walks = []
        orignal_walks = []
        LG = copy.deepcopy(train_nxgraph)
        on = ohmnet.OhmNet(LG, p=p, q=q, num_walks=num_walks,
            walk_length=walk_length, dimension=dimensions,
            window_size=10, n_workers=8, n_iter=5,out_dir= '.')
        for ns in on.embed_multilayer():
            orignal_walks.append(ns)
            on_walks = [n.split("_")[2] for n in ns]
            ohmnet_walks.append([str(step) for step in on_walks])
        Ohmnet_model = Node2vec.N2V.learn_embeddings(ohmnet_walks, dimensions, workers = 5, window_size=10, niter=5)
        Ohmnet_wvecs = np.array([Ohmnet_model.get_vector(str(i)) for i in nodes])
        y_pred = []
        auc = []
        for index, layer in enumerate(restru_test_edges):
            y_pred = []
            for e in layer:
                if str(e[0]) in Ohmnet_model.index2entity and str(e[1]) in Ohmnet_model.index2entity: # 如果关键字没有在字典Key中，则设置为0.5
                    y_pred.append(cosine_similarity([Ohmnet_model.get_vector(str(e[0])), Ohmnet_model.get_vector(str(e[1]))])[0][1])
                else:
                    y_pred.append(0)
            auc.append(roc_auc_score(test_labels[index], y_pred))
        print("ohmnet-network:%f"%(sum(auc)/len(auc)))
        #
        # #3# MNE 实现可扩展的Multiplex network的嵌入，IJCAI'2018
        edge_data_by_type = {}
        all_edges = list()
        all_nodes = list()
        for e in train_edges:
            if e[0] not in edge_data_by_type:
                edge_data_by_type[e[0]]=list()
            edge_data_by_type[e[0]].append((e[1],e[2]))
            all_edges.append((e[1], e[2]))
            all_nodes.append(e[1])
            all_nodes.append(e[2])
        all_nodes = list(set(all_nodes))
        all_edges = list(set(all_edges))
        edge_data_by_type['Base'] = all_edges
        MNE_model = MNE.train_model(edge_data_by_type)
        local_model = dict()

        auc = []
        for index, layer in enumerate(restru_test_edges):
            y_pred = []
            for pos in range(len(MNE_model['index2word'])):
                local_model[MNE_model['index2word'][pos]] = MNE_model['base'][pos] + 0.5 * np.dot(
                    MNE_model['addition'][index][pos], MNE_model['tran'][index])
            for e in layer:
                if str(e[0]) in MNE_model['index2word'] and str(e[1]) in MNE_model['index2word']: # 如果关键字没有在字典Key中，则设置为0.5
                    y_pred.append(cosine_similarity([local_model[str(e[0])], local_model[str(e[1])]])[0][1])
                else:
                    y_pred.append(0)
            auc.append(roc_auc_score(test_labels[index], y_pred))
        print("MNE:%f"%(sum(auc)/len(auc)))

        #4# PMNE的3种算法
        merged_networks = dict()
        merged_networks['training'] = dict()
        merged_networks['test_true'] = dict()
        merged_networks['test_false'] = dict()
        for index, g in enumerate(train_nxgraph):
            merged_networks['training'][index] = set(g.edges())
            merged_networks['test_true'][index] = restru_test_edges[index]
            merged_networks['test_false'][index] = test_edges[index][len(test_edges):]

        performance_1, performance_2, performance_3 = main.Evaluate_PMNE_methods(merged_networks)
        print("PMNE(n):%f" % (performance_1))
        print("PMNE(r):%f" % (performance_2))
        print("MNE(c):%f" % (performance_3))


        #5# MELL实现多层网络的节点表示学习，WWW’2018
        L = len(nx_graph)
        N = max([int(n) for n in train_merge.nodes()])+1
        N = max(N, train_merge.number_of_nodes()) # 为了构造邻接矩阵需要找到行的标准
        directed = True
        d = 128
        k = 3
        lamm = 10
        beta = 1
        gamma = 1
        MELL_wvecs = MELL_model(L, N, directed, train_edges, d, k, lamm, beta, gamma)
        MELL_wvecs.train(30) # 之前是500，但是有的数据集500会报错，因此设置为30
        auc = []
        for index, layer in enumerate(restru_test_edges):
            y_pred = []
            for e in layer:
                if e[0] in all_nodes and e[1] in all_nodes: # 如果关键字没有在字典Key中，则设置为0.5
                    y_pred.append(MELL_wvecs.predict((index, e[0], e[1])))
                else:
                    y_pred.append(0)
            auc.append(roc_auc_score(test_labels[index], y_pred))
        print("MELL:%f"%(sum(auc)/len(auc)))

        #6# 基本相似性度量方法：CN JC AA
        auc1 = []
        auc2 = []
        auc3 = []
        for index, layer in enumerate(restru_test_edges):
            y_pred_cn = []
            y_pred_jc = []
            y_pred_AA = []
            for e in layer:
                if e[0] in train_nxgraph[index].nodes() and e[1] in train_nxgraph[index].nodes():
                    y_pred_cn.append(len(list(nx.common_neighbors(train_nxgraph[index], e[0], e[1]))))
                    y_pred_jc.append(list(nx.jaccard_coefficient(train_nxgraph[index], [e]))[0][2])
                    # y_pred_AA.append(list(nx.adamic_adar_index(train_nxgraph[index], [e]))[0][2])
                else:
                    y_pred_cn.append(0) # 如果不存在这个节点，那么为共有邻居为0
                    y_pred_jc.append(0)
                    # y_pred_AA.append(0)

            auc1.append(roc_auc_score(test_labels[index], y_pred_cn))  # 计算AUC值
            auc2.append(roc_auc_score(test_labels[index], y_pred_jc))
            auc3.append(roc_auc_score(test_labels[index], y_pred_AA))
        print("CN-network:%f" %(sum(auc1) / len(auc1)))
        print("JC-network:%f" %(sum(auc2) / len(auc2)))
        print("AA-network:%f" %(sum(auc3) / len(auc3)))

        #7# Single-layer Node2vec
        auc = []
        for index, G in enumerate(nx_para_graph):
            model_nf = node2vec.Node2vec(G, walk_length, num_walks, dimensions, p=p, q=q, dw=True)
            index_num = sorted([int(i) for i in model_nf.vectors.keys()])
            g_embedding = [model_nf.vectors[str(i)] for i in index_num]
            y_pred = []
            for e in restru_test_edges[index]:
                if str(e[0]) in G.G.nodes() and str(e[1]) in G.G.nodes(): # 如果关键字没有在字典Key中，则设置为0.5
                    y_pred.append(cosine_similarity([model_nf.vectors[str(e[0])], model_nf.vectors[str(e[1])]])[0][1])
                else:
                    y_pred.append(0)
            auc.append(roc_auc_score(test_labels[index], y_pred))
        print("Node2vec: %f" % (sum(auc) / len(auc)))

        #7# Network + Embedding(N2V) + SNF4st 网络的表示学习
        for k in knei:
            for m in mu:
                auc_final = []
                for i in range(2, 10): # 为了求平均值
                    # 第一个参数是KNN的K值，第二个是mu值，第三个是其他过程使用的K值，最后一个参数使迭代次数，一般情况下20次就会达到收敛
                    network, groundtruth, best, second = NFC.cluster_E(nx_para_graph, ground_truth, Parameter, nodes, k, m, k, 30) # CKM\V(20, 0.5, 20, 20)
                    Network_Adj = _find_dominate_set(check_symmetric(network, raise_warning=False), K=k) # 从网络的相似性矩阵中构建邻接矩阵 CKM(20) Vickers(15)
                    g = nx.from_numpy_matrix(Network_Adj) # 基于邻接矩阵构建网络
                    auc = []
                    for index, layer in enumerate(restru_test_edges):
                        y_pred = []
                        for e in layer:
                            if e[0] in train_nxgraph[index].nodes() and e[1] in train_nxgraph[index].nodes():
                                y_pred.append(list(nx.adamic_adar_index(g, [(nodes.index(e[0]), nodes.index(e[1]))]))[0][2]) # 利用RA相似性计算测试集两点之间概率
                            else:
                                y_pred.append(0)
                        auc.append(roc_auc_score(test_labels[index], y_pred)) # 计算AUC值
                    auc_final.append(sum(auc) / len(auc))
                value = max(auc_final)
                average = sum(auc_final)/len(auc_final)
                print("K=%d Mu=%.2f Max:index({%d})->%f"%(k, m, auc_final.index(value), value))
                print("K=%d Mu=%f Ave:->%f" % (k, m, average))
