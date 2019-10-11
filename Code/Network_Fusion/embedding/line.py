import os
import random
import csv

class line:
    def __init__(self, edgelist, tmp_path, nodes_num, output_path, representation_size=128):
        self.__edge_list_path = edgelist
        self.__tmp_path = tmp_path
        self.__nodes_num = nodes_num
        self.__output_path = output_path
        self.__representation_size = representation_size
        self.__embedding = []

    def run(self):
        tmp_file_dict = dict()
        for path in self.__edge_list_path:
            file_name = (path.split("/")[-1]).split(".")[0]
            output_path = self.__tmp_path + file_name + ".tmpedges"
            command = "D:/line.exe -train \"%s\" -output \"%s\" -size %d -order 2 -samples 128 -threads 8" \
                      % (path, output_path, self.__representation_size)
            print(command)
            os.system(command)
            tmp_file_dict[file_name]=output_path
        embeddings = self._get_embedding(tmp_file_dict)
        embedding_path_dict = self._output_embedding_as_csv(embeddings)
        return embedding_path_dict

    def _get_embedding(self, tmp_file_dict):
        ret = dict()
        for file_name in tmp_file_dict:
            tmp = []
            indexList = []
            with open(tmp_file_dict[file_name], 'r') as f:
                for l in f:
                    words = l.strip().split()
                    words = [float(ele) for ele in words]
                    tmp.append(words)
                    if len(words) == 2:
                        indexList.append(0)
                        continue
                    indexList.append(int(words[0]))
            result = []
            for i in range(self.__nodes_num):
                if i+1 in indexList:
                    realIndex = indexList.index(i+1)
                    tmp[realIndex].pop(0)
                    result.append(tmp[realIndex])
                else:
                    result.append([random.uniform(-1.0, 1.0) for i in range(self.__representation_size)])
            ret[file_name] = result
        return ret

    def _output_embedding_as_csv(self, embeddings):
        ret = dict()
        for name in embeddings:
            file_name = self.__output_path + name + ".csv"
            with open(file_name, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for line in embeddings[name]:
                    writer.writerow(line)
            ret[name] = file_name
        return ret