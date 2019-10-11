class ReadNormalData:
    def __init__(self, edge_list_path, output_path, layer_num):
        self.__edge_list_path = edge_list_path
        self.__output_path = output_path
        self.__layer_num = layer_num
        self.__output_path_list = self._process()

    def _process(self):
        ret = []
        tmp = []
        for i in range(self.__layer_num):
            tmp.append([])
        with open(self.__edge_list_path, "r") as f:
            for lines in f:
                words = lines.strip().split()
                layer_num = int(words[0]) - 1
                tmp[layer_num].append((int(words[1]),int(words[2])))
        layer_cnt = 1
        for layer in tmp:
            real_path = self.__output_path + "layer" + str(layer_cnt) + ".edges"
            ret.append(real_path)
            with open(real_path, "w") as f:
                for pair in layer:
                    f.writelines("%d %d\n"%(pair[0],pair[1]))
            layer_cnt = layer_cnt + 1
        return ret

    def get_output_path_list(self):
        return self.__output_path_list
