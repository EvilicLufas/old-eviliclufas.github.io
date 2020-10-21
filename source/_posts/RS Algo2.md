# RS Algo

we can implement collaborative filtering using a library called fatasi . This library is built on top of [**pytorch**](http://pytorch.org/) and is focused on easier implementation of **machine learning and deep learning models**.

# Dataset

I have used [**Movielens**](https://grouplens.org/datasets/movielens/) **[1]** dataset ([ml-latest-small](http://files.grouplens.org/datasets/movielens/ml-latest-small.zip)). This dataset describes **5-star** rating and free-text tagging activity from [MovieLens](http://movielens.org/), a movie recommendation service. It contains **100004 ratings** and **1296 tag applications** across **9125 movies**. These data were created by **671 users** between January 09, 1995 and October 16, 2016.



1.1 description for each file 

 

| Filename    | ratings.csv                                                  |
| ----------- | ------------------------------------------------------------ |
| Description | User’s ratings on each movie                                 |
| Properties  |                                                              |
| userId      | The id of user                                               |
| movieId     | The id of movie                                              |
| Rating      | Ratings are made on a 5-star scale, with   half-star increments (0.5 stars - 5.0 stars). |
| timestamp   | Timestamps represent seconds since   midnight Coordinated Universal Time (UTC) of January 1, 1970. |

 

| Filename    | movies.csv                                                   |
| ----------- | ------------------------------------------------------------ |
| Description | offers the url for each movie from three   website sources   |
| Properties  |                                                              |
| movieId     | The id of movie                                              |
| title       | Movie titles are entered manually or   imported from <https://www.themoviedb.org/>, and include the year of   release in parentheses. Errors and inconsistencies may exist in these titles. |
| genres      | Genres are a pipe-separated list, and are   selected from the following:   * Action   * Adventure   * Animation   * Children's   * Comedy   * Crime   * Documentary   * Drama   * Fantasy   * Film-Noir   * Horror   * Musical   * Mystery   * Romance   * Sci-Fi   * Thriller   * War   * Western |

 

 

| Filename    | tags.csv                                                     |
| ----------- | ------------------------------------------------------------ |
| Description | All ratings are contained in the file   `ratings.csv`        |
| Properties  |                                                              |
| userId      | The id of user                                               |
| movieId     | The id of movie                                              |
| tag         | Tags are user-generated metadata about   movies. Each tag is typically a single word or short phrase. The meaning,   value, and purpose of a particular tag is determined by each user. |
| timestamp   | Timestamps represent seconds since   midnight Coordinated Universal Time (UTC) of January 1, 1970. |

 

| Filename    | links.csv                                                    |
| ----------- | ------------------------------------------------------------ |
| Description | offers the url for each movie from three   website sources   |
| Properties  |                                                              |
| movieId     | an identifier for movies used by   <https://movielens.org>.  |
| imdbId      | an identifier for movies used by   <http://www.imdb.com>.    |
| tmdbId      | an identifier for movies used by   <https://www.themoviedb.org>.. |

We’ll be using 2 files: `ratings.csv` and `movies.csv`

# Collaborative filtering using fastai

Before we get started we need 2 things:

- A GPU enabled machine (local or AWS)
- Install fastai library on your machine: `pip install fastai`

**Note:** At the end of the post I have explained in detail as to how to setup your system for fastai

## Step 1: Data loading

<iframe src="https://towardsdatascience.com/media/654f2fd18439adc0d7f664dd9cd51e4d" allowfullscreen="" frameborder="0" height="342" width="680" title="loading data" class="s t u hf ai" scrolling="auto" style="box-sizing: inherit; position: absolute; top: 0px; left: 0px; width: 617px; height: 310.313px;"></iframe>

We are looking at 2 files: **ratings and movies

<img src="C:\Users\Vilic\AppData\Roaming\Typora\typora-user-images\1585387153484.png" alt="1585387153484" style="zoom: 80%;" />

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

<img src="RS Algo2.assets/1585387344264.png" alt="1585387344264" style="zoom:67%;" />

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











```
Summary
=======

This dataset (ml-latest-small) describes 5-star rating and free-text tagging activity from [MovieLens](http://movielens.org), a movie recommendation service. It contains 100836 ratings and 3683 tag applications across 9742 movies. These data were created by 610 users between March 29, 1996 and September 24, 2018. This dataset was generated on September 26, 2018.

Users were selected at random for inclusion. All selected users had rated at least 20 movies. No demographic information is included. Each user is represented by an id, and no other information is provided.

The data are contained in the files `links.csv`, `movies.csv`, `ratings.csv` and `tags.csv`. More details about the contents and use of all these files follows.

This is a *development* dataset. As such, it may change over time and is not an appropriate dataset for shared research results. See available *benchmark* datasets if that is your intent.

This and other GroupLens data sets are publicly available for download at <http://grouplens.org/datasets/>.


Usage License
=============

Neither the University of Minnesota nor any of the researchers involved can guarantee the correctness of the data, its suitability for any particular purpose, or the validity of results based on the use of the data set. The data set may be used for any research purposes under the following conditions:

* The user may not state or imply any endorsement from the University of Minnesota or the GroupLens Research Group.
* The user must acknowledge the use of the data set in publications resulting from the use of the data set (see below for citation information).
* The user may redistribute the data set, including transformations, so long as it is distributed under these same license conditions.
* The user may not use this information for any commercial or revenue-bearing purposes without first obtaining permission from a faculty member of the GroupLens Research Project at the University of Minnesota.
* The executable software scripts are provided "as is" without warranty of any kind, either expressed or implied, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose. The entire risk as to the quality and performance of them is with you. Should the program prove defective, you assume the cost of all necessary servicing, repair or correction.

In no event shall the University of Minnesota, its affiliates or employees be liable to you for any damages arising out of the use or inability to use these programs (including but not limited to loss of data or data being rendered inaccurate).

If you have any further questions or comments, please email <grouplens-info@umn.edu>


Citation
========

To acknowledge use of the dataset in publications, please cite the following paper:

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. <https://doi.org/10.1145/2827872>


Further Information About GroupLens
===================================

GroupLens is a research group in the Department of Computer Science and Engineering at the University of Minnesota. Since its inception in 1992, GroupLens's research projects have explored a variety of fields including:

* recommender systems
* online communities
* mobile and ubiquitious technologies
* digital libraries
* local geographic information systems

GroupLens Research operates a movie recommender based on collaborative filtering, MovieLens, which is the source of these data. We encourage you to visit <http://movielens.org> to try it out! If you have exciting ideas for experimental work to conduct on MovieLens, send us an email at <grouplens-info@cs.umn.edu> - we are always interested in working with external collaborators.


Content and Use of Files
========================

Formatting and Encoding
-----------------------

The dataset files are written as [comma-separated values](http://en.wikipedia.org/wiki/Comma-separated_values) files with a single header row. Columns that contain commas (`,`) are escaped using double-quotes (`"`). These files are encoded as UTF-8. If accented characters in movie titles or tag values (e.g. Misérables, Les (1995)) display incorrectly, make sure that any program reading the data, such as a text editor, terminal, or script, is configured for UTF-8.


User Ids
--------

MovieLens users were selected at random for inclusion. Their ids have been anonymized. User ids are consistent between `ratings.csv` and `tags.csv` (i.e., the same id refers to the same user across the two files).


Movie Ids
---------

Only movies with at least one rating or tag are included in the dataset. These movie ids are consistent with those used on the MovieLens web site (e.g., id `1` corresponds to the URL <https://movielens.org/movies/1>). Movie ids are consistent between `ratings.csv`, `tags.csv`, `movies.csv`, and `links.csv` (i.e., the same id refers to the same movie across these four data files).


Ratings Data File Structure (ratings.csv)
-----------------------------------------

All ratings are contained in the file `ratings.csv`. Each line of this file after the header row represents one rating of one movie by one user, and has the following format:

    userId,movieId,rating,timestamp

The lines within this file are ordered first by userId, then, within user, by movieId.

Ratings are made on a 5-star scale, with half-star increments (0.5 stars - 5.0 stars).

Timestamps represent seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970.


Tags Data File Structure (tags.csv)
-----------------------------------

All tags are contained in the file `tags.csv`. Each line of this file after the header row represents one tag applied to one movie by one user, and has the following format:

    userId,movieId,tag,timestamp

The lines within this file are ordered first by userId, then, within user, by movieId.

Tags are user-generated metadata about movies. Each tag is typically a single word or short phrase. The meaning, value, and purpose of a particular tag is determined by each user.

Timestamps represent seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970.


Movies Data File Structure (movies.csv)
---------------------------------------

Movie information is contained in the file `movies.csv`. Each line of this file after the header row represents one movie, and has the following format:

    movieId,title,genres

Movie titles are entered manually or imported from <https://www.themoviedb.org/>, and include the year of release in parentheses. Errors and inconsistencies may exist in these titles.

Genres are a pipe-separated list, and are selected from the following:

* Action
* Adventure
* Animation
* Children's
* Comedy
* Crime
* Documentary
* Drama
* Fantasy
* Film-Noir
* Horror
* Musical
* Mystery
* Romance
* Sci-Fi
* Thriller
* War
* Western
* (no genres listed)


Links Data File Structure (links.csv)
---------------------------------------

Identifiers that can be used to link to other sources of movie data are contained in the file `links.csv`. Each line of this file after the header row represents one movie, and has the following format:

    movieId,imdbId,tmdbId

movieId is an identifier for movies used by <https://movielens.org>. E.g., the movie Toy Story has the link <https://movielens.org/movies/1>.

imdbId is an identifier for movies used by <http://www.imdb.com>. E.g., the movie Toy Story has the link <http://www.imdb.com/title/tt0114709/>.

tmdbId is an identifier for movies used by <https://www.themoviedb.org>. E.g., the movie Toy Story has the link <https://www.themoviedb.org/movie/862>.

Use of the resources listed above is subject to the terms of each provider.
```