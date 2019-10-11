import networkx as nx
import numpy as np
import ReadMultiplexNetwork as RMN
import pickle
import copy
import random


def merge_g(nx_graphs):
    '''生成合并的图结构'''
    m_g = nx.Graph()
    for g in nx_graphs:
        m_g.add_nodes_from(g.nodes())
        m_g.add_edges_from(g.edges())
    return m_g

def get_selected_edges(pos_edge_list, neg_edge_list):
    '''边训练和测试的标签'''
    edges = pos_edge_list + neg_edge_list
    labels = np.zeros(len(edges))
    labels[:len(pos_edge_list)] = 1
    return edges, labels

def generate_pos_neg_links(nx_graph, merge_network, test_para):
    '''生成正负样例边'''
    Multi_Networks = copy.deepcopy(nx_graph)
    # train_g = copy.deepcopy(merge_network)
    selected_layer = random.randint(0, len(Multi_Networks))
    train_g = copy.deepcopy(Multi_Networks[selected_layer])
    train_ng = Multi_Networks.remove(train_g)
    # 获取网络中存在的边
    exit_edges = list(train_g.edges())
    num_exit = len(exit_edges)

    # 获取网络中不存在的边
    noexit_edges = list(nx.non_edges(train_g))
    num_noexit = len(noexit_edges)

    # 随机化列表的序列
    random.shuffle(exit_edges)
    random.shuffle(noexit_edges)

    # 正例边的采样
    pos_edge_list = []
    n_count = 0
    edges = exit_edges
    rnd = np.random.RandomState(seed=None)
    rnd_inx = rnd.permutation(edges)  # 基于随机种子产生下标
    for eii in rnd_inx:
        edge = eii
        # 删除该边
        data = train_g[edge[0]][edge[1]]
        train_g.remove_edge(*edge)

        # 测试存在的边在删除之后，整个网络能否联通
        if nx.is_connected(train_g):
            flag = True
            for g in Multi_Networks:
                if edge in g.edges():
                    gt = copy.deepcopy(g)
                    gt.remove_edge(*edge)
                    if nx.is_connected(gt) == False:
                        del gt
                        flag = False
                        break
            if flag:
                for g in Multi_Networks:
                    if edge in g.edges():
                        g.remove_edge(*edge)
                pos_edge_list.append(tuple(edge))
                n_count += 1
            else:
                train_g.add_edge(*edge, **data)
        else:
            train_g.add_edge(*edge, **data)

    # 正采样的边
    if not len(pos_edge_list): # 如果原始图都是空的，那么就没有意义，所以就随机选择一定数量的边
        pos_edge_list = exit_edges[:int(len(exit_edges)*test_para)]
        [g.remove_edge(*e) for g in Multi_Networks for e in pos_edge_list if e in g.edges()]
        [train_g.remove_edge(*e) for e in pos_edge_list]
        nneg = npos = len(pos_edge_list)
    else:
        # 确定测试边的个数
        if len(pos_edge_list) < num_noexit:
            npos = int(test_para * len(pos_edge_list))  # 正例的数量
        else:
            npos = int(test_para * num_noexit)
        nneg = npos  # 负例的数量
        pos_edge_list = pos_edge_list[:nneg]

    # 负采样的边
    neg_edge_list = noexit_edges[:nneg]
    # 测试边数据集和标签
    test_edges, labels = get_selected_edges(pos_edge_list, neg_edge_list)
    return Multi_Networks, train_g, pos_edge_list, neg_edge_list, test_edges, labels

def selected_pos_neg_links(nx_graph, test_para):
    '''
        nx_grahp:表示原始的网络
        test_para:表示测试的边的比例
    '''

    Multi_Networks = copy.deepcopy(nx_graph)
    # 选择要操作的网络
    selected_layer = random.randint(0, len(Multi_Networks)-1)
    train_g = copy.deepcopy(Multi_Networks[selected_layer])
    Multi_Networks.remove(Multi_Networks[selected_layer])

    # 重新得到剩余网络构建的图
    train_mg = merge_g(Multi_Networks)

    # 获取网络中存在的边
    exit_edges = list(train_g.edges())
    num_exit = len(exit_edges)

    # 获取网络中不存在的边
    noexit_edges = list(nx.non_edges(train_g))
    num_noexit = len(noexit_edges)

    # 随机化列表的序列
    random.shuffle(exit_edges)
    random.shuffle(noexit_edges)

    # 根据得到的正例的个数和负例的个数判断最终的测试边个数
    if num_noexit < num_exit:
        selected_edges_number = int(num_noexit * test_para)
    else:
        selected_edges_number = int(num_exit * test_para)

    pos_edge_list = exit_edges[:selected_edges_number]
    neg_edge_list = noexit_edges[:selected_edges_number]

    # 边测试集的标准化
    test_posedges_list = list(set(exit_edges) - set(pos_edge_list))
    test_negedges_list = noexit_edges[selected_edges_number:selected_edges_number+len(test_posedges_list)]
    test_edges, labels = get_selected_edges(test_posedges_list, test_negedges_list)
    return Multi_Networks, train_mg, pos_edge_list, neg_edge_list, test_edges, labels

def each_selected_pos_neg_links(nx_graph, test_para):
    '''
        nx_grahp:表示原始的网络
        test_para:表示测试的边的比例
    '''

    Multi_Networks = copy.deepcopy(nx_graph)
    test_edges_list = []
    test_edges_lable = []
    # 选择要操作的网络
    for train_g in Multi_Networks:
        # 获取网络中存在的边
        exit_edges = list(train_g.edges())
        num_exit = len(exit_edges)

        # 获取网络中不存在的边
        noexit_edges = list(nx.non_edges(train_g))
        num_noexit = len(noexit_edges)

        # 根据得到的正例的个数和负例的个数判断最终的测试边个数
        if num_noexit < num_exit:
            selected_edges_number = int(num_noexit * test_para)
        else:
            selected_edges_number = int(num_exit * test_para)

        # 随机选择测试边
        random.shuffle(exit_edges)
        random.shuffle(noexit_edges)

        # 正、负边的个数都相同
        pos_edge_list = exit_edges[:selected_edges_number]
        neg_edge_list = noexit_edges[:selected_edges_number]
        train_g.remove_edges_from(pos_edge_list)
        test_edges, labels = get_selected_edges(pos_edge_list, neg_edge_list)
        test_edges_list.append(test_edges)
        test_edges_lable.append(labels)

    # 重新得到剩余网络构建的图
    train_mg = merge_g(Multi_Networks)

    # 边测试集的标准化
    return Multi_Networks, train_mg, test_edges_list, test_edges_lable

if __name__ == '__main__':

    # 加载数据集,主要是将节点的初始编号从0开始,CKM数据集缺少154,165,195,201,203这五个节点，所以在构建网络的时候需要删除这些节点
    nx_graphs = RMN.read_f('../pierreauger_multiplex.edges')
    merge_graph = merge_g(nx_graphs)

    # 加载社区的标签为其他任务使用
    name = 'pierreauger_information'
    ground_truth_norm = []

    if sorted(list(merge_graph.nodes()))[0] > 0:
        merge_graph = nx.relabel_nodes(merge_graph, lambda x: x-1)
        nx_graphs = [nx.relabel_nodes(nx_graphs[i], lambda x: x-1) for i in range(len(nx_graphs))]

    # 将每一层网络中的节点补全，形成标准的Multiplex Network
    nx_graphs_Norm = []
    for g in nx_graphs:
        temp_g = nx.Graph()
        temp_g.add_nodes_from(merge_graph.nodes())
        temp_g.add_edges_from(g.edges())
        nx_graphs_Norm.append(temp_g)

    # 生成测试和训练时使用的正例、负例边
    train_ng, train_mg, test_edges, labels = each_selected_pos_neg_links(nx_graphs_Norm, 0.2)
    # 保存处理后的数据
    Dict_graph = {'merge_graph':merge_graph, 'nx_graph':nx_graphs_Norm, "comm_label":np.array(ground_truth_norm),'train_mg':train_mg, 'train_ng':train_ng, 'test_edges': test_edges, "test_labels": labels}
    pickle.dump(Dict_graph, open('./' + name+ '.pickle', '+wb'))


