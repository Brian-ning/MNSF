cluster_test.py
本算法的社团发现实验，包括：直接融合、直接融合-降维、直接融合-greedy、表示学习、表示学习-降维、表示学习-greedy 共六个实验。参数见代码注释及后面的例子。

link_test.py
链路预测实验，使用方法和参数含义见注释。

如需运行，要提前安装deepwalk==1.0.3
line 可执行文件在embedding中，如果需要在Linux下运行，请替换该文件（见embedding/line.py 19行）

表示学习算法deepwalk的效果比line好，可以试一试其他的表示学习算法。