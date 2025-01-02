import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

data = pd.read_csv('Coursera.csv')
df = pd.DataFrame(data)

df['Course Rating'] = pd.to_numeric(df['Course Rating'], errors='coerce')

df['Course Rating'].fillna(df['Course Rating'].mean(), inplace=True)

df['Normalized Rating'] = df['Course Rating'] / 5.0

df['Skills'] = df['Skills'].apply(lambda x: x.lower())  

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Skills'])

def compute_cosine_similarity(user_skills, course_skills_vector):
    user_skills_vector = vectorizer.transform([user_skills.lower()])
    return cosine_similarity(user_skills_vector, course_skills_vector)[0][0]

st.title("Course Recommendation System")

preferred_difficulty = st.selectbox("Select Preferred Difficulty Level", ["Beginner", "Intermediate", "Advanced", "Conversant"])

user_skills = st.text_input("Enter your skills (space-separated)")

def check_skills_match(user_skills, df):
    user_skills_set = set(user_skills.lower().split())
    course_skills_set = set(' '.join(df['Skills']).lower().split())
    return user_skills_set.intersection(course_skills_set)

def get_suggested_skills(df, num_suggestions=10):
    all_skills = ' '.join(df['Skills']).lower().split()
    unique_skills = set(all_skills)
    
    return list(unique_skills)[:num_suggestions]

if user_skills:
    matching_skills = check_skills_match(user_skills, df)
    
    if not matching_skills:
        st.error("No matching skills found. Please check and correct your skills.")
        
        suggested_skills = get_suggested_skills(df)
        st.write(f"Suggested skills to help you get better recommendations: {', '.join(suggested_skills)}")
        
    else:
        df['Cosine Similarity'] = df['Skills'].apply(lambda x: compute_cosine_similarity(user_skills, vectorizer.transform([x])))

        cosine_weight = 0.5
        rating_weight = 0.3
        difficulty_weight = 0.2

        def compute_final_score(row, preferred_difficulty):
            cosine_similarity = row['Cosine Similarity']
            normalized_rating = row['Normalized Rating']
            difficulty_score = 1 if row['Difficulty Level'] == preferred_difficulty else 0  
            final_score = (cosine_similarity * cosine_weight) + (normalized_rating * rating_weight) + (difficulty_score * difficulty_weight)
            return final_score

        df['Final Score'] = df.apply(compute_final_score, axis=1, preferred_difficulty=preferred_difficulty)
        
        recommended_courses = df[['Course Name', 'Final Score', 'Course Rating', 'Normalized Rating']].sort_values(by='Final Score', ascending=False)
        
        recommended_courses_unique = recommended_courses.drop_duplicates(subset=['Course Name'])
        
        top_recommended_courses = recommended_courses_unique.head(5)
        
        st.subheader("Top Recommended Courses")
        st.table(top_recommended_courses[['Course Name', 'Final Score', 'Course Rating']])  
        
        predicted_ratings = recommended_courses_unique['Final Score']
        actual_ratings = recommended_courses_unique['Course Rating']

        rmse = sqrt(mean_squared_error(actual_ratings, predicted_ratings))
        mae = mean_absolute_error(actual_ratings, predicted_ratings)

        st.subheader("Recommendation System Evaluation")
        st.write(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
        st.write(f"MAE (Mean Absolute Error): {mae:.4f}")


else:
    st.write("Enter your skills to get course recommendations.")
