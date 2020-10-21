# RS Algo 4 MovieLens

## Part I

With the explosive growth of e-commerce and social media platforms, recommendation algorithms have become indispensable tools for many businesses. Two main branches of recommender algorithms are often distinguished: content-based recommender systems and collaborative filtering models. Content-based recommender systems use content information of users and items, such as their respective occupation and genre, to predict the next purchase of a user or rating of an item. Collaborative filtering models solve the matrix completion task by taking into account the collective interaction data to predict future ratings or purchases.

In this work, we view matrix completion as a link prediction problem on graphs: the interaction data in collaborative filtering can be represented by a bipartite graph between user and item nodes, with observed ratings/purchases represented by links. Content information can naturally be included in this framework  in the form of node features. Predicting ratings then reduces to predicting labeled links in the bipartite user-item graph.



## Part II

In this work a recommender system for a movie database using neo4j which will be able to: 

– suggest top N movies similar to a given movie title to users

– predict user votes for the movies they have not voted for.



## Memory-based collaborative filtering

Aside from the movie metadata we have another valuable source of information at our exposure: the user rating data. Our recommender system can recommend a movie that is similar to “Inception (2010)” on the basis of user ratings. In other words, what other movies have received similar ratings by other users? This would be an example of item-item collaborative filtering. You might have heard of it as “The users who liked this item also liked these other ones.” The data set of interest would be ratings.csv and we manipulate it to form items as vectors of input rates by the users. As there are many missing votes by users, we have imputed Nan(s) by 0 which would suffice for the purpose of our collaborative filtering. Here we have movies as vectors of length ~80000. Again as before we can apply a truncated SVD to this rating matrix and only keep the first 200 latent components which we will name the collab_latent matrix. The next step is to use a similarity measure and find the top N most similar movies to “Inception (2010)” on the basis of each of these filtering methods we introduced. Cosine similarity is one of the similarity measures we can use. To see a summary of other similarity criteria, read Ref [2]- page 93. In the following, you will see how the similarity of an input movie title can be calculated with both content and collaborative latent matrices. I have also added a hybrid filter which is an [average](https://blog.codecentric.de/en/2019/07/recommender-system-movie-lens-dataset/#) measure of similarity from both content and collaborative filtering standpoints. If I list the top 10 most similar movies to “Inception (2010)” on the basis of the hybrid measure, you will see the following list in the data frame. For me personally, the hybrid measure is predicting more reasonable titles than any of the other filters.

