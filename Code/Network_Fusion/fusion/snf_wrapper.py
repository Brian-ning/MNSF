from ..fusion import snf
from ..fusion.snf import datasets
import numpy as np

class snf_wrapper:
    def __init__(self, data, label):
        self.__data = data
        self.__label=label

    def fusion(self, K1, mu_, K2, t_, draw_raw = False):
        data = self.__data
        affinity_networks = []
        affinity_networks.append(snf.compute.make_affinity(data, K=K1, mu=mu_))

        if draw_raw==True:
            raw = (np.array(data[0]) + np.array(data[1]) + np.array(data[2]))/3
            fig = Visualization.plot_embedding(raw, self.__label, "(prediction)")
            fig.show()

        # fused_network = snf.snf(affinity_networks, K=K2, t=t_)
        fused_network = snf.compute.snf(affinity_networks, K=K2, t=t_)
        best, second = snf.get_n_clusters(np.nan_to_num(fused_network))
        return fused_network, self.__label, best, second

    def fusion_E(self, K1, mu_, K2, t_, draw_raw = False):
        data = self.__data
        affinity_networks = []
        affinity_networks.append(snf.compute.make_affinity(data, K=K1, mu=mu_))

        if draw_raw==True:
            raw = (np.array(data[0]) + np.array(data[1]) + np.array(data[2]))/3
            fig = Visualization.plot_embedding(raw, self.__label, "(prediction)")
            fig.show()

        # fused_network = snf.snf(affinity_networks, K=K2, t=t_)
        fused_network = snf.compute.snf4st(affinity_networks, K=K2, t=t_)
        best, second = snf.get_n_clusters(fused_network)
        return fused_network, self.__label, best, second

    def fusion_(self, K1, mu_, K2, t_, data_, load_data = True, draw_raw = False):
        if load_data:
            data = datasets.load_data(self.__network_file_list, self.__label)
        else:
            data = data_
        affinity_networks = snf.make_affinity(data.data, K=K1, mu=mu_)

        if draw_raw==True:
            raw = (np.array(data.data[0]) + np.array(data.data[1]) + np.array(data.data[2]))/3
            fig = Visualization.plot_embedding(raw, data.labels, "(prediction)")
            fig.show()

        fused_network_list = snf.snf_(affinity_networks, K=K2, t=t_)
        best, second = snf.get_n_clusters(fused_network_list[len(fused_network_list)-1])
        return fused_network_list, data.labels, best, second



