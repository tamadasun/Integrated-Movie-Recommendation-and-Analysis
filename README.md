# The Cinematic Nexus: Unveiling the Future of Movie Recommendations and Analysis


![Image Description](https://imgur.com/VMGcx6v.jpg)


---

## Overview

Welcome to the groundbreaking CineConnect project repository! The objective is nothing short of revolutionizing the movie streaming experience through the development of a dynamic recommendation and analysis platform. Unlike traditional models, CineConnect incorporates an advanced recommendation system that integrates collaborative, content-based, and genre cluster models. 

The heart of our innovation lies in providing a personalized and engaging movie-watching journey for users with a focus on collaborative, content-based, and genre cluster models that aims to offer a seamless and adaptive experience without compromising on personalization. Throughout this project, we tackle the challenges faced by existing recommendation systems head-on. From addressing data sparsity to overcoming the limitations of static models, we employ state-of-the-art techniques, including API interactions, extensive data wrangling, feature engineering, exploratory data analysis (EDA), and data cleaning.

This project aims to sets new standards for accuracy and diversity in movie recommendations and redefine the way users connect with the cinematic world. Join us on this transformative journey as we redefine the future of personalized entertainment. CineConnect is not just a platform; it's a cinematic revolution.


---

## Table of Content

1. [Notebooks](notebook/)
   - 1.1 [Data_Cleaning_and_EDA.ipynb](notebooks/01_Data_Cleaning_EDA.ipynb)
      - 1.1.1 Introduction
      - 1.1.2 Data Collection
      - 1.1.3 Data Cleaning and EDA
   - 1.2 [Data_Modeling.ipynb](notebooks/02_Data_Modeling.ipynb)
      - 1.2.1 Introduction
      - 1.2.2 Data Transformation/Engineering
      - 1.2.3 Data Modeling and Visualization
   - 1.3 [Evaluation_Summary.ipynb](notebooks/03_Evaluation_Summary.ipynb)
      - 1.3.1 Introduction
      - 1.3.2 Evaluation of Movie Recommendation System
      - 1.3.3 Summary of Findings
      - 1.3.4 Recommendations for Further Improvement

2. [Data](data/)
   - 2.1 Dataset Dictionary
   - 2.2 Data Files

3. [Images](images/)
   - 3.1 Visualizations
   - 3.2 Charts

4. [Presentation](presentation/)
   - 4.1 PowerPoint Presentation (PDF)

5. [Model](model/)
   - 5.1 movie_app.py
   - 5.2 movie_recommendation.py


--- 


## Data Description

<div align='center'>

| column         | type   | description                                          |
| -------------- | ------ | ---------------------------------------------------- |
| adult          | bool   | whether intended for adults                          |
| backdrop_path  | object | URL for a movie image backdrop                       |
| genre_ids      | object | genre ID for each movie                              |
| id             | int64  | unique identifier for each movie                     |
| original_lan   | object | original language in which the movie was produced    |
| original_title | object | original title of the movie in its original language |
| overview       | object | brief summary of movie plot                          |
| popularity     | float64| popularity of a movie based on mentions and views    |
| poster_path    | object | URL to the movie's poster image                      |
| release_date   | date   | release date of the movie                            |
| title          | object | title of the movie                                   |
| video          | bool   | whether the movie has a video                        |
| vote_average   | float64| average rating given to a movie                      |
| vote_count     | int64  | number of votes a movie has received                 |
    
 
</div>

---

## Problem Statement

I have embarked on an exciting journey with CineConnect, a cutting-edge startup poised to disrupt the movie streaming industry and rival giants like Netflix, Hulu, Apple TV+, and HBO max. My role as a consultant and data scientist involves spearheading the development of an innovative movie recommendation platform that goes beyond the ordinary, catering to the diverse needs of movie enthusiasts.

CineConnect, derived from "Cinematic Nexus," signifies the companies commitment to creating a cinematic hub that connects users with personalized and engaging movie experiences. With a focus on films ("Cine") and the power to connect users ("Connect"), the company's platform aims to redefine the way users discover and interact with movies.

The current landscape of recommendation systems faces challenges such as data sparsity and the static nature of existing models. CineConnect aspires to overcome these hurdles by introducing advanced collaborative, content-based, and genre cluster models, ensuring optimal accuracy and diversity in movie recommendations. The company's goal is not just to compete but to lead in offering users a dynamic and personalized connection to the cinematic world.

As we navigate the ever-evolving preferences of movie enthusiasts, the platform will set new standards for personalized content recommendation across various domains. The success of this project will be measured not only by the company's ability to compete with industry giants in the longrun but also by the significant improvements in recommendation accuracy and diversity for advanced user preferences.

CineConnect is not just a platform; it's a cinematic revolution, and I am thrilled to be at the forefront of this transformative journey.

---

## Hypothesis

**Null Hypothesis:** The accuracy of movie recommendations remain equivalent between collaborative filtering (user preference) and content-based filtering (advanced preference) models, suggesting no significant difference in the accuracy of recommendations.

**Alternative Hypothesis:** The accuracy of movie recommendations significantly differ between collaborative filtering (user preference) and content-based filtering (advanced preference) models, indicating a notable impact on the accuracy of suggestions based on the level of user input.

---

## Methods and Models

**Techniques**
- Collaborative Filtering:  make movie recommendations based on user preferences and similarities between users or items
- Content-Based Filtering: recommends items based on the features of the items and the user's preferences.

**Model Deployment**
- Use web frameworks like Streamlit to deploy the machine learning models into a web application. This allows users to interact with the recommender seamlessly. 

---
![Alt Text](https://asiainsurancepost.com/wp-content/uploads/2023/08/ai-4-980x654.webp)


## Part 1: Data Cleaning and Exploratory Data Analysis (EDA)

In this section, the project will detail the process of acquiring and preparing the data for our movie recommendation system. This includes data gathering, cleaning, and conducting exploratory data analysis.

- Data Collection
    - Retrieve movie data from TMDB API, ensuring comprehensive coverage of movie details.
    - Validate the integrity of the dataset to avoid missing or incomplete information.
    - Handle any API rate limitations, ensuring a smooth and ethical interaction with the TMDB API.
- Data Cleaning
    - Address missing or inconsistent data entries by applying appropriate imputation techniques.
    - Standardize and clean data formats, ensuring consistency across different data fields.
     - Handle outliers and anomalies that might impact the accuracy of recommendations.
    - Normalize numerical features for better model performance.
- Exploratory Data Analysis (EDA)
    - Conduct a thorough exploration of the dataset to gain insights into movie trends and characteristics.
    - Analyze distributions of key variables such as genre, release year, and user ratings.
    - Visualize the relationships between different features to identify potential patterns or correlations.
    - Extract meaningful statistics to inform the modeling process.

---

## Part 2: Data Modeling

In this section, the project will delve into the process of transforming and engineering the data for our movie recommendation system. Additionally, we will build and evaluate predictive models and employ data visualization techniques to gain insights into the performance and characteristics of the models.

- Data Transformation/Engineering
    - Feature Engineering: Create new features that might enhance the predictive power of the models, such as extracting information from movie titles, actors, or directors.
    - Handle Sparse Data: Address potential sparsity issues in user-item interaction matrices, as sparse data can impact collaborative filtering models.
    - Encoding: Encode categorical features, ensuring all data is in a format suitable for modeling.
- Data Modeling
    - Collaborative Filtering: Implement collaborative filtering techniques to make movie recommendations based on user preferences and similarities between users or items.
    - Content-Based Filtering: Apply content-based filtering approaches to recommend movies based on their features, such as overview, cast, or director and nlp technique.
    - K-Means model: Apply content-based genre clustering for improved recommendation accuracy.
- Data Visualization 
    - Model Evaluation: Visualize the performance of different recommendation models and showcase Silhouette and Inertia Scores.


---

## Part 3: Evalutaion Metric/Executive Summary/Recommendations

In this section, the project will evaluate the performance of our movie recommendation system, provide an executive summary of our findings, offer recommendations for further improvement, and conclude our project.

- Evaluation of Movie Recommendation System
    - Analyze the performance of collaborative filtering and content-based filtering models.
    - Hypothesis testing for advance user preference and the result of the hypothesis test   
- Summary of Findings
    - Summarize key findings from the evaluation, highlighting successful aspects and areas for improvement.
    - Provide insights into user preferences, popular genres, and the effectiveness of the recommendation system.
- Recommendations for Further Improvement
    - Integration of Chatbot that utilize user feedback and preferences as valuable inputs for future system enhancements.
    - Hybrid Model Integration
    - Fine-Tuning of K-Means Genre Cluster Approach



---

## Conclusion

Throughout this arduous process, the development and evaluation of the movie recommendation system has uncovered valuable insights into the effectiveness of different approaches. The hypothesis testing revealed a differences in the accuracy of movie recommendations between the collaborative (user preference) and content-based (advanced preference) filtering models.

- Collaborative Filtering: Proved powerful in capturing user preferences, offering personalized recommendations based on historical interactions.
- Content-Based Filtering: Excelled in recommending movies based on inherent characteristics, providing diversity beyond user history.

The conclusion of this project marks a milestone in the development of a movie recommendation system, and has the outlined a path for an even more sophisticated and user-centric movie recommendation system. The incorporation of a chatbot, refinement of existing models, and exploration of hybrid approaches promise an exciting path forward in delivering unparalleled cinematic recommendations and truly creating a cinematic nexus.


## Technology Requirements

[List the technology stack and tools required for the project]

- Python 3.x
- Jupyter Notebooks
- Natural Language Processing Libraries (e.g., NLTK)
- Machine Learning Frameworks (e.g., Scikit-learn)
- Data Visualization Tools (e.g., Matplotlib, Seaborn)
- Streamlit

<img src="https://t3.ftcdn.net/jpg/05/74/18/34/240_F_574183420_pc0caByueQA0QjQMsJr0lY5txOaQoBmo.jpg" alt="Alt Text" width="900">

