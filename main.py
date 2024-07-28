# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setting the style for seaborn plots
sns.set_style('white')


# Loading the user data
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=column_names)

# Checking for missing values in user data
print("Missing values in user data:", df.isnull().sum())

# Loading movie titles
movietitle = pd.read_csv('Movie_Id_Titles')

# Checking for missing values in movie titles
print("Missing values in movie titles:", movietitle.isnull().sum())

# Merging user data with movie titles
df = pd.merge(df, movietitle, on='item_id')

# Exploratory Data Analysis (EDA)
# Checking the first few rows of the merged data
print(df.head())

# Computing the average rating for each movie
average_ratings = df.groupby('title')['rating'].mean().sort_values(ascending=False)
print("Top 5 movies by average rating:\n", average_ratings.head())

# Computing the count of ratings for each movie
rating_counts = df.groupby('title')['rating'].count().sort_values(ascending=False)
print("Top 5 movies by rating count:\n", rating_counts.head())

# Creating a DataFrame for ratings
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())

# Visualization of the distribution of ratings
plt.figure(figsize=(10,4))
ratings['rating'].hist(bins=70)
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

# Creating a pivot table for user ratings
moviemat = df.pivot_table(index='user_id', columns='title', values='rating')
print(moviemat.head())

# Function to find similar movies
def find_similar_movies(movie_title, min_ratings=100):
    movie_user_ratings = moviemat[movie_title]
    similar_to_movie = moviemat.corrwith(movie_user_ratings)
    corr_movie = pd.DataFrame(similar_to_movie, columns=['Correlation'])
    corr_movie.dropna(inplace=True)
    corr_movie = corr_movie.join(ratings['num of ratings'])
    recommendations = corr_movie[corr_movie['num of ratings'] > min_ratings].sort_values('Correlation', ascending=False)
    return recommendations

# Finding similar movies to Star Wars (1977)
starwars_recommendations = find_similar_movies('Star Wars (1977)')
print("Recommendations for Star Wars (1977):\n", starwars_recommendations.head())

# Finding similar movies to Liar Liar (1997)
liarliar_recommendations = find_similar_movies('Liar Liar (1997)')
print("Recommendations for Liar Liar (1997):\n", liarliar_recommendations.head())

# Additional visualizations
plt.figure(figsize=(12,6))
sns.jointplot(x='rating', y='num of ratings', data=ratings, alpha=0.5)
plt.show()

print("Movie recommendation project completed successfully!")
