# movie-recommendation-apriori
# 🎬 Movie Recommendation System using Apriori Algorithm

## 📌 Project Overview  
This project implements a movie recommendation system using Association Rule Mining. It analyzes user rating patterns from the MovieLens 100k dataset to discover relationships between movies that are frequently liked together. The system generates rules such as: “If a user likes Movie A, they are likely to like Movie B.”

## 🎯 Problem Statement  
With the increasing number of movies on online platforms, users often find it difficult to choose relevant content. Traditional recommendation systems fail to capture hidden relationships between movies. This project solves this problem by applying the Apriori Algorithm to identify frequent co-rating patterns and generate meaningful recommendations.

## 🎯 Objectives  
- Analyze user rating data  
- Identify frequently co-rated movies  
- Implement Apriori algorithm  
- Generate association rules  
- Evaluate rules using support, confidence, and lift  
- Provide insights for recommendation systems  

## 📊 Dataset  
MovieLens 100k Dataset  
- 100,000 ratings  
- 943 users  
- 1,682 movies  

Dataset Link: https://grouplens.org/datasets/movielens/100k/  

Files Used:  
- u.data → user ratings  
- u.item → movie details  

## ⚙️ Technologies Used  
- Python  
- Pandas  
- NumPy  
- Mlxtend  

## 🧠 Algorithm Used  
Apriori Algorithm (Association Rule Mining)  
- Finds frequent itemsets  
- Generates association rules  
- Uses metrics: Support, Confidence, Lift  

## 🔄 Project Workflow  
1. Load dataset  
2. Preprocess data (convert ratings to liked/not liked)  
3. Create user-movie matrix  
4. Filter popular movies  
5. Apply Apriori algorithm  
6. Generate association rules  
7. Display and save results  

## 📈 Sample Output  
Example Rule:  
If a user likes “Aladdin (1992)” → they are likely to like “The Lion King (1994)”  
Support: 0.086  
Confidence: 0.562  
Lift: 3.954  

## ▶️ How to Run the Project  
Step 1: Clone the repository  
git clone https://github.com/your-username/movie-recommendation-apriori.git  
cd movie-recommendation-apriori  

Step 2: Install dependencies  
pip install -r requirements.txt  

Step 3: Download dataset  
Download from: https://grouplens.org/datasets/movielens/100k/  
Extract the folder and place it in the project directory  

Step 4: Run the code  
python AI.py  

## 📁 Project Structure  
project/  
AI.py  
association_rules.csv  
requirements.txt  
README.md  

## 📊 Results  
- Generated 2576 frequent itemsets  
- Identified strong movie relationships  
- Achieved high confidence and lift values  
- Discovered meaningful user behavior patterns  

## 💡 Key Insights  
- Users who like certain movies tend to like related movies  
- Strong associations exist between popular films  
- Association Rule Mining is effective for recommendation systems  

## 🧩 Future Scope  
- Integrate with real-time recommendation systems  
- Use larger datasets for better accuracy  
- Combine with collaborative filtering  
- Build a web-based interface  

## 👥 Team Members  
- Ammar Kazi  
- Soham Patil  
- Affan Ansari  

## 📚 References  
MovieLens Dataset: https://grouplens.org/datasets/movielens/100k/  
Mlxtend Documentation: https://rasbt.github.io/mlxtend/  
Apriori Algorithm Paper: Agrawal & Srikant (1994)  

## ⭐ Acknowledgment  
We thank the GroupLens Research team for providing the dataset used in this project.  

## 📌 License  
This project is for academic purposes only.  
