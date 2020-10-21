# RS Algo

we can implement collaborative filtering using a library called fatasi developed by [Jeremy Howard](https://medium.com/u/34ab754f8c5e?source=post_page-----919da17ecefb----------------------) et al. This library is built on top of [**pytorch**](http://pytorch.org/) and is focused on easier implementation of **machine learning and deep learning models**.

Also, we’ll get to know how we can interpret and visualise embeddings using [**t-SNE**](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)**,** [**Plotly**](https://plot.ly/) **and** [**Bokeh**](https://bokeh.pydata.org/en/latest/) (*Python interactive visualisation library that targets modern web browsers for presentation*).

# Dataset

I have used [**Movielens**](https://grouplens.org/datasets/movielens/) **[1]** dataset ([ml-latest-small](http://files.grouplens.org/datasets/movielens/ml-latest-small.zip)). This dataset describes **5-star** rating and free-text tagging activity from [MovieLens](http://movielens.org/), a movie recommendation service. It contains **100004 ratings** and **1296 tag applications** across **9125 movies**. These data were created by **671 users** between January 09, 1995 and October 16, 2016.

We’ll be using 2 files: `ratings.csv` and `movies.csv`

# Collaborative filtering using fastai

Before we get started we need 2 things:

- A GPU enabled machine (local or AWS)
- Install fastai library on your machine: `pip install fastai`

**Note:** At the end of the post I have explained in detail as to how to setup your system for fastai

## Step 1: Data loading

<iframe src="https://towardsdatascience.com/media/654f2fd18439adc0d7f664dd9cd51e4d" allowfullscreen="" frameborder="0" height="342" width="680" title="loading data" class="s t u hf ai" scrolling="auto" style="box-sizing: inherit; position: absolute; top: 0px; left: 0px; width: 617px; height: 310.313px;"></iframe>

We are looking at 2 files: **ratings and movies

![1585387153484](C:\Users\Vilic\AppData\Roaming\Typora\typora-user-images\1585387153484.png)

Figure 1: Ratings

**Ratings** contain ratings by different users for different movies.

![1585387165881](C:\Users\Vilic\AppData\Roaming\Typora\typora-user-images\1585387165881.png)

Figure 2: Movies

**Movies** contains metadata about movies. `movieid` is the key to join the 2 datasets.

## Step 2: Model training

```
#fastai functionval_idxs = get_cv_idxs(len(ratings)) #get validation indices 
```

We’ll divide our data into train and validation set. Our validation is **20%** of our original dataset.

```
wd=2e-4 #weight decayn_factors = 50 #dimension of embedding vector
```

We’ll use **weight decay** to reduce **overfitting**. Also we have to define the dimension of our **embedding vector**.

```
#fastai functioncf = CollabFilterDataset.from_csv(path, 'ratings.csv', 'userId', 'movieId', 'rating') #creating a custom data loader
```

Now we have to create a data object for collaborative filtering. Think of it as something which will transform your raw data and prepare it in the form that is required by the model. `from_csv` implies that the input should be a **csv file**.

Parameters of the function:

- `path`: path to the location of the csv file
- `ratings.csv` : name of the csv file. It should be in the **long format** shown in figure 1
- `userID/movieID` : column names of the 2 entities
- `rating` : column name of the dependent variable that you want to predict

```
#create a learner (model) and specify the batch size and optimizer learn = cf.get_learner(n_factors, val_idxs, 64, opt_fn=optim.Adam) #fastai function
```

Next step is to create a model object which is a function of the data object that we have created. `learner` in fastai library is synonymous to model. This function takes the following parameters:

- `n_factors` : Dimension of the embedding vector (**50 in our case**)
- `val_idxs` : Row Indices from the ratings.csv file which have to be considered in validation
- `batch size` : Number of rows that will be passed to the optimiser for each step of gradient descent. In our case **64 rows** will be passed per iteration
- `opt_fn` : Optimiser that we want to use. In our case we are using [**Adam**](http://ruder.io/optimizing-gradient-descent/)**.** You have access to different optimisers in this library

![img](https://miro.medium.com/max/54/0*lgXXFb4_R0IVzmkU.png?q=20)

![img]()

Figure 3: Optimisers

```
#training with learning rate as 1e-2 learn.fit(1e-2, 2, wds=wd, cycle_len=1, cycle_mult=2, use_wd_sched=True) #fastai function
```

The final step of training is to actually train the model. Calling `fit` on the `learner` object trains the model and learn the **right values in the embedding and bias matrix**.

Parameters of the function:

- `learning rate` : 1e-2 is the learning rate that we use for optimisation
- `wd` : passing the weight decay
- `cycle_len/cycle_mult` : These are fastai goodies that incorporate the state of the art methods for **learning rate scheduling**. *The end of the post contains links to useful articles related to this.*
- `use_wd_sched` : Whether to use schedule for weight decay

When you run the above code, model will start training as shown below. You can observe **training (left) and validation (right) loss** after each epoch. Our loss function for optimisation is `MSE (Mean squared error)` .