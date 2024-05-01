import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)



#item-based collaborative filtering algorithm
#Item-based Collaborative Filtering focuses on finding similar movies instead of similar users to recommend to user ‘A’ based on their past preferences.
#It identifies pairs of movies rated/liked by the same users, measures their similarity across all users who rated both, and then suggests similar movies based on the similarity scores.

#datasets
movies = pd.read_csv("dataset/movies.csv")
ratings = pd.read_csv("dataset/ratings.csv")

#print(ratings.head())


#make a new dataframe where each column would represent each unique userId and each row represents each unique movieId.
final_dataset = ratings.pivot(index='movieId',columns='userId',values='rating')
#print(final_dataset.head())

#replace NaN fields with 0  
final_dataset.fillna(0,inplace=True)
#print(final_dataset.head())



#Removing Noise from the data

#get num of user votes on movies
no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
#get num of movies voted on by users
no_movies_voted = ratings.groupby('userId')['rating'].agg('count')

#select movies only if they have been rated a min 10 times by users
final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index,:]
#select users only if they have rated a min 50 movies
final_dataset=final_dataset.loc[:,no_movies_voted[no_movies_voted > 50].index]
#print(final_dataset)


#reduce the sparsity
#sprase = data that contains mostly zeroes
sample = np.array([[0,0,3,0,0],[4,0,0,0,2],[0,0,0,0,1]])
sparsity = 1.0 - ( np.count_nonzero(sample) / float(sample.size) )
#print(sparsity)


#To reduce the sparsity we use the csr_matrix function from the scipy library.
csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)
csr_sample = csr_matrix(sample)
#print(csr_sample)


#knn algorithm to compute similarity with cosine similarity
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)


def get_movie_recommendation(movie_name):
    # Number of movies to recommend
    n_movies_to_reccomend = 10
    
    # Find movies with titles containing the given movie name
    movie_list = movies[movies['title'].str.contains(movie_name)]  
    
    # If movies are found
    if len(movie_list):        
        # Get the index of the first movie in the list
        movie_idx= movie_list.iloc[0]['movieId']
        
        # Find the index of the movie in the final dataset
        movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
        
        # Compute the distances and indices of nearest neighbors
        distances , indices = knn.kneighbors(csr_data[movie_idx], n_neighbors=n_movies_to_reccomend+1)    
        
        # Extract recommended movie indices with their distances
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        
        # Create a list to store recommended movies and their distances
        recommend_frame = []
        recommend_frame2 = []
        # Iterate through recommended movie indices
        for val in rec_movie_indices:
            # Get the movie index
            movie_idx = final_dataset.iloc[val[0]]['movieId']
            # Find the index of the movie in the movies DataFrame
            idx = movies[movies['movieId'] == movie_idx].index
            # print movie title only
            #print(movies.iloc[idx]['title'].values[0])
            # Append recommended movie title and distance to the list
            recommend_frame.append({'Title':movies.iloc[idx]['title'].values[0],'Distance':val[1]})
            recommend_frame2.append(movies.iloc[idx]['title'].values[0])

        # Create a DataFrame from the recommendation list
        df = pd.DataFrame(recommend_frame,index=range(1,n_movies_to_reccomend+1))

        # returns ranking,movie title, and distance
       # return df
        return recommend_frame2
    else:
        # Return a message if no movies are found
        return "No movies found. Please check your input"



@app.route("/")

# default route when page is first opened this will run 
def index():
  # info = get_movie_recommendation('Star Wars')
  # info = "test"
   return render_template("index.html")

#when request is made on index this will handle
@app.route('/process_input', methods=['POST'])
def process_input():
    input_value = request.json.get('input')
    info = get_movie_recommendation(input_value)
    return jsonify({"message": info}), 200


app.run(host="0.0.0.0", port = 5500) 
