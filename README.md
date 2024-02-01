# The Cinematic Nexus: Unveiling the Future of Movie Recommendations and Analysis

![Alt Text](https://asiainsurancepost.com/wp-content/uploads/2023/08/ai-4-980x654.webp)


# REMEMBER TO CHANGE TO PAST TENSE WHEN TASK COMPLETED

---

## Overview

Welcome to the my project repository! The goal of this project is to develop a dynamic movie recommendation and analysis platform that leverages an interactive chatbot. The chatbot, powered by automation tools, will engage users in a conversational manner to gather preferences, provide real-time recommendations, and enhance the overall user experience. By incorporating automation in the form of a chatbot, the platform aims to create a more personalized and adaptive movie-watching journey for each user.

---

## Table of Content

1. [Data_Cleaning and EDA](01_Data_Cleaning_EDA.ipynb)
   - 1.1 Introduction
   - 1.2 Data Collection
   - 1.3 Data Cleaning
   - 1.4 EDA

2. [Data Modeling](02_Data_Modeling.ipynb)
   - 2.1 Introduction
   - 2.2 Data Tranformation/Engineering
   - 2.3 Data Modeling
   - 2.4 Data Visualization
  
3. [Chatbot Implementation](03_Chatbot.ipynb)
   - 3.1 Introduction
   - 3.2 Chatbot Development
   - 3.3 Integration with Movie Recommendation System
 
4. [Chatbot Implementation](03_Chatbot.ipynb)
   - 3.1 Introduction
   - 3.2 Chatbot Development
   - 3.3 Integration with Movie Recommendation System

3. [Data](data/)
   - 3.1 Dataset Dictionary
   - 3.2 Data Files

4. [Images](images/)
   - 4.1 Visualizations
   - 4.2 Charts

5. [Presentation](presentation/)
   - 5.1 PowerPoint Presentation (PDF)

6. [Model](model/)
   - 6.1 Trained Model (.pkl)
   

---

## Data Description

<div align='center'>

| column         | type   | description                       |
| -------------- | ------ | --------------------------------- |
| airline        | object | airline company                   |
| flight         | object | flight number                     |
| origin         | object | departure city                    |
| departure_time | object | departure time of day             |
| stops          | int    | number of stops                   |
| arrival_time   | object | arrival time of day               |
| destination    | object | arrival city                      |
| class          | int    | economy (0) or business (1) class |
| duration       | float  | flight duration in minutes        |
| price          | int    | ticket price in USD               |

</div>

---

## Problem Statement

Movie enthusiasts face challenges in discovering personalized content tailored to their individual preferences. Traditional recommendation systems often fall short in capturing the nuanced tastes of users and suffers from data sparsity(the problem of having insufficient or missing ratings or interactions between users and items), leading to a less-than-optimal user experience. Additionally, the static nature of these systems may not effectively adapt to evolving user preferences over time.

To address these issues, this project propose the development of an innovative movie recommendation and analysis platform that integrates an interactive chatbot for user interaction. The success of this platform will be measured by the improvement in the accuracy of movie recommendations provided by the chatbot, assessed through precision and recall metrics.

---

## Success Metric

The success metric will be the improvement in the accuracy of movie recommendations provided by the chatbot. This will be measured using the precision and recall metrics, with a target precision of 0.75 and recall of 0.80 within the first few months of implementing the chatbot. Aiming for this balance ensures that the chatbot not only recommends movies the user is likely to enjoy (high precision) but also doesn't miss out on many movies the user would like (high recall).

--- 

## Target Metric

The target metric will focus on reducing the false-positive rate in movie recommendations generated by the chatbot. The goal is to achieve a 15% decrease in the false-positive rate (out of out of every 10 movie suggestions, approximately 1.55 suggestions are not likely to match the user's preferences if baseline fp rate is 30%), indicating that users receive more accurate and relevant movie suggestions.

---

## Hypothesis

**Null Hypothesis:** The mean precision and recall of movie recommendations remain unchanged before and after the implementation of the interactive chatbot, suggesting that the chatbot does not significantly impact the accuracy of recommendations.

**Alternative Hypothesis:** The mean precision and recall of movie recommendations significantly improve after the implementation of the interactive chatbot, indicating a positive impact on the accuracy of suggestions.

---

## Methods and Models

**Techniques**
- Collaborative Filtering:  make movie recommendations based on user preferences and similarities between users or items
- Content-Based Filtering: recommends items based on the features of the items and the user's preferences. I
- Utilize automation tools and natural language processing (NLP) techniques to implement an interactive chatbot capable of understanding and responding to user queries.
    - Use NLTK for intent recognition (preprocess user input, tokenize it, and identify the intent based on keywords, patterns, or machine learning models)
    - Use NLTK for response generation (response generation mechanism based on the recognized intent that could involve rule-based responses)

**Model Deployment**
- Use web frameworks like Streamlit to deploy the machine learning models and chatbot into a web application. This allows users to interact with the chatbot seamlessly. 

---

## Part 1: Data Cleaning and Exploratory Data Analysis (EDA)

In this section, the project will detail the process of acquiring and preparing the data for our movie recommendation system. This includes data gathering, cleaning, and any necessary transformations.

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
    - Content-Based Filtering: Apply content-based filtering approaches to recommend movies based on their features, such as genre, cast, or director.
    - Hybrid Models: Explore the development of hybrid models that combine collaborative and content-based filtering for improved recommendation accuracy.
- Data Visualization 
    - Model Evaluation: Visualize the performance of different recommendation models using metrics such as precision, recall, and accuracy.
    - Feature Importance: Gain insights into the importance of different features in the models through visualizations, aiding in model interpretation.
    - User-Item Interaction: Visualize patterns in user-item interaction matrices to understand user preferences and item popularity.

---

## Part 3: Chatbot Implementation

In this section, the project will focus on developing and integrating an interactive chatbot to enhance user interaction with our movie recommendation system.

- Chatbot Development
    - Utilize NLP library such as NLTK for intent recognition and response generation.
    - Implement dialog management to maintain context and create a smooth conversational experience.
    - Fine-tune the chatbot's responses to provide relevant and user-friendly recommendations.
- Integration with Movie Recommendation System
    - Connect the chatbot with the underlying movie recommendation models to fetch real-time personalized movie suggestions.
    - Implement a user interface within the chatbot for users to interact with and receive movie recommendations seamlessly.
    - Ensure that the chatbot integrates with the recommendation system's data processing components for dynamic and adaptive responses.

---

## Part 4: Streamlit Application

In this section, the project will focus on developing a Streamlit application that serves as the user interface for our movie recommendation and analysis platform.

- Streamlit Development
    - Design an intuitive layout that incorporates chatbot interaction, movie suggestions, and additional features for an engaging user experience.
    - Implement responsive components within the Streamlit app to showcase real-time movie recommendations based on user preferences.
- Integration with Movie Recommendation System
    - Integrate the Streamlit application with the underlying movie recommendation and chatbot systems.
    - Ensure seamless communication between the Streamlit app and Jupyter Notebook
    - Implement interactive elements within the app for users to provide feedback and refine preferences

--- 

## Part 5: Evalutaion Metric/Executive Summary/Recommendations

In this section, the project will evaluate the performance of our movie recommendation system, provide an executive summary of our findings, offer recommendations for further improvement, and conclude our project.

- Evaluation of Movie Recommendation System
    - Assess the precision, recall, and other relevant metrics to measure the effectiveness of our recommendation models.
    - Analyze the performance of collaborative filtering and content-based filtering models.
    - Hypothesis testing for chatbot impact and the result of the hypothesis test   
- Summary of Findings
    - Summarize key findings from the evaluation, highlighting successful aspects and areas for improvement.
    - Provide insights into user preferences, popular genres, and the effectiveness of the interactive chatbot.
- Recommendations for Further Improvement
    - Utilize user feedback and preferences as valuable inputs for future system enhancements.
    - Integrate machine learning models to enable the chatbot to adapt to changing user preferences over time, enhancing the accuracy of movie recommendations (Reinforcement Learning and Real Time Learning).

---

## Conclusion



## Technology Requirements

[List the technology stack and tools required for the project]

- Python 3.x
- Jupyter Notebooks
- Natural Language Processing Libraries (e.g., NLTK)
- Machine Learning Frameworks (e.g., Scikit-learn)
- Data Visualization Tools (e.g., Matplotlib, Seaborn)
- Streamlit

<img src="https://t3.ftcdn.net/jpg/05/74/18/34/240_F_574183420_pc0caByueQA0QjQMsJr0lY5txOaQoBmo.jpg" alt="Alt Text" width="900" height="900">



