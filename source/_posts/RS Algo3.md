RS Algo3



We consider matrix completion for recommender systems from the point of view of link prediction on graphs. Interaction data such as movie ratings can be represented by a bipartite user-item graph with labeled edges denoting observed ratings. Building on recent progress in deep learning on graph-structured data, we propose a graph auto-encoder framework based on differentiable message passing on the bipartite interaction graph. Our model shows competitive performance on standard collaborative filtering benchmarks. In settings where complimentary feature information or structured data such as a social network is available, our framework outperforms recent state-of-the-art methods.

我们从图上的链接预测的角度考虑推荐系统的矩阵完成。 互动数据（例如电影收视率）可以由两部分用户项目图表示，带有标记的边缘表示观察到的收视率。 基于图结构化数据深度学习的最新进展，我们提出了一种基于在双向交互图上传递的可区分消息的图自动编码器框架。 我们的模型显示出在标准协作过滤基准上的竞争表现。 在可以使用免费功能信息或结构化数据（例如社交网络）的环境中，我们的框架要优于最新的方法。

With the explosive growth of e-commerce and social media platforms, recommendation algorithms have become indispensable tools for many businesses. Two main branches of recommender algorithms are often distinguished: content-based recommender systems [24] and collaborative filtering models [9]. Content-based recommender systems use content information of users and items, such as their respective occupation and genre, to predict the next purchase of a user or rating of an item. Collaborative filtering models solve the matrix completion task by taking into account the collective interaction data to predict future ratings or purchases.

随着电子商务和社交媒体平台的爆炸性增长，推荐算法已成为许多企业必不可少的工具。 推荐程序算法的两个主要分支通常是有区别的：基于内容的推荐程序系统[24]和协作过滤模型[9]。 基于内容的推荐系统使用用户和项目的内容信息（例如其各自的职业和流派）来预测用户的下一次购买或项目的等级。 协作过滤模型通过考虑集体交互数据来预测未来的评级或购买，从而解决了矩阵完成任务。

In this work, we view matrix completion as a link prediction problem on graphs: the interaction data in collaborative filtering can be represented by a bipartite graph between user and item nodes, with observed ratings/purchases represented by links. Content information can naturally be included in this framework  in the form of node features. Predicting ratings then reduces to predicting labeled links in the bipartite user-item graph.

在这项工作中，我们将矩阵完成视为图上的链接预测问题：协作过滤中的交互数据可以由用户和项目节点之间的二部图表示，而观察到的评级/购买则由链接表示。 内容信息自然可以以节点特征的形式包含在此框架中。 预测收视率然后减少为预测二分用户项目图中的标记链接。

We propose graph convolutional matrix completion (GC-MC): a graph-based auto-encoder framework for matrix completion, which builds on recent progress in deep learning on graphs [2, 6, 19, 5, 15, 30, 14].
The auto-encoder produces latent features of user and item nodes through a form of message passing on the bipartite interaction graph. These latent user and item representations are used to reconstruct the rating links through a bilinear decoder.

我们提出图卷积矩阵完成（GC-MC）：一种基于图的矩阵完成自动编码器框架，该框架基于图上深度学习的最新进展[2、6、19、5、15、30、14]。
  自动编码器通过在双向交互图上传递的消息形式来生成用户和项目节点的潜在特征。 这些潜在的用户和项目表示用于通过双线性解码器重建评级链接。



The benefit of formulating matrix completion as a link prediction task on a bipartite graph becomes especially apparent when recommender graphs are accompanied with structured external information such as social networks. Combining such external information with interaction data can alleviate performance bottlenecks related to the cold start problem. We demonstrate that our graph auto-encoder model efficiently combines interaction data with side information, without resorting to recurrent frameworks as in [22].

当推荐图与结构化的外部信息（如社交网络）一起使用时，将矩阵完成公式化为二分图上的链接预测任务的好处变得尤为明显。 将此类外部信息与交互数据结合在一起可以缓解与冷启动问题相关的性能瓶颈。 我们证明了我们的图形自动编码器模型可以有效地将交互数据与边信息结合起来，而无需借助[22]中的递归框架。



Consider a rating matrix M of shape Nu × Nv, where Nu is the number of users and Nv is the number of items. Entries Mij in this matrix encode either an observed rating (user i rated item j) from a set of discrete possible rating values, or the fact that the rating is unobserved (encoded by the value 0). See Figure 1 for an illustration. The task of matrix completion or recommendation can be seen as predicting the value of unobserved entries in M.

考虑形状为Nu×Nv的评分矩阵M，其中Nu是用户数，Nv是项目数。 此矩阵中的条目Mij编码来自一组离散的可能等级值的观察等级（用户i等级j），或者未观察到等级（由值0编码）。 有关说明，请参见图1。 矩阵完成或推荐的任务可以看作是预测M中未观察到的条目的值。

![1585388184972](C:\Users\Vilic\AppData\Roaming\Typora\typora-user-images\1585388184972.png)

Figure 1: Left: Rating matrix M with entries that correspond to user-item interactions (ratings between 1-5) or missing observations (0). Right: User-item interaction graph with bipartite structure. Edges correspond to interaction events, numbers on edges denote the rating a user has given to a particular item. The matrix completion task (i.e. predictions for unobserved interactions) can be cast as a link prediction problem and modeled using an end-to-end trainable graph auto-encoder.

图1：左图：评分矩阵M，其条目对应于用户-项目交互（评分介于1-5之间）或缺少观察值（0）。 右图：具有二分结构的用户-项目交互图。 边缘对应于交互事件，边缘上的数字表示用户对特定项目的评价。 矩阵完成任务（即，对未观察到的交互的预测）可以转换为链接预测问题，并使用端到端可训练图自动编码器进行建模。

In this work, we have introduced graph convolutional matrix completion (GC-MC): a graph auto-encoder framework for the matrix completion task in recommender systems. The encoder contains a graph convolution layer that constructs user and item embeddings through message passing on the bipartite user-item interaction graph. Combined with a bilinear decoder, new ratings are predicted in the form of labeled edges.



The graph auto-encoder framework naturally generalizes to include side information for both users and items. In this setting, our proposed model outperforms recent related methods by a large margin, as demonstrated on a number of benchmark datasets with feature- and graph-based side information. We further show that our model can be trained on larger scale datasets through stochastic mini-batching. In this setting, our model achieves results that are competitive with recent state-of-the-art collaborative filtering.

在这项工作中，我们介绍了图卷积矩阵完成（GC-MC）：一种用于图集系统中矩阵完成任务的图自动编码器框架。 编码器包含一个图形卷积层，该图形卷积层通过在两方用户项交互图上传递的消息来构造用户和项的嵌入。 结合双线性解码器，以标记边缘的形式预测新的收视率。
   图形自动编码器框架自然可以概括为包括用户和项目的辅助信息。 在这种情况下，我们提出的模型在很大程度上优于最新的相关方法，正如许多具有特征和基于图形的边信息的基准数据集所证明的那样。 我们进一步表明，可以通过随机迷你批处理在较大规模的数据集上训练我们的模型。 在这种情况下，我们的模型所获得的结果可与最新的最新协作过滤相媲美。



