#Imports
import joblib
import warnings

warnings.filterwarnings("ignore")
#Load the logistic model
joblib_logistic_regression_model = "joblib_logistic_regression_model.pkl"
lr_model = joblib.load(joblib_logistic_regression_model)

#Load TF-IDF vectorizer
joblib_tfidf_vector = "tfidf_vector.pkl"
tfidf_vectorizer = joblib.load(joblib_tfidf_vector)

print("Type q to exit.")
while True:
    user_input = input("Please enter your issue: ")
    if user_input == "q":
        break
    vectorized_issue = tfidf_vectorizer.transform([user_input])
    print(lr_model.predict(vectorized_issue)[0])
    print()