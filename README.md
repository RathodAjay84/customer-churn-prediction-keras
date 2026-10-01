📌 Project Overview
Customer churn is one of the most critical challenges in the telecom industry. It refers to customers discontinuing services and switching to competitors. Accurately predicting churn enables companies to take proactive retention measures, reducing revenue loss.
This project builds a robust churn prediction system using Artificial Neural Networks (ANN) with Keras. It demonstrates a complete end-to-end machine learning pipeline — from data preprocessing and visualization to model building, evaluation, and performance improvement.

🎯 Objectives

Analyze customer data to identify churn patterns
Build predictive models to classify churn vs non-churn customers
Handle real-world challenges like class imbalance
Evaluate and improve model performance using deep learning techniques


📊 Dataset Description
Dataset: Telco Customer Churn Dataset
The dataset contains customer-level information including:

Demographics: Gender, Senior Citizen
Account Information: Tenure, Monthly Charges, Total Charges
Services: Internet Service, Payment Method
Target Variable:

Churn → Whether the customer left the service (1) or not (0)




🔍 Exploratory Data Analysis (EDA)
The following insights were explored:

Distribution of churn vs non-churn customers using pie chart
Distribution of Internet Service types using bar plot
Identification of high-risk customer segments using conditional filtering


🧹 Data Preprocessing
Key preprocessing steps include:

Removed irrelevant column (customerID)
Converted TotalCharges to numeric format
Handled missing values using median imputation
Encoded categorical variables using Label Encoding
Feature scaling applied (StandardScaler) for neural network optimization


🤖 Model Development
Three different Artificial Neural Network (ANN) models were built:

🔹 Model 1: Basic ANN

Feature: Tenure
Architecture:

Input Layer: 12 neurons (ReLU)
Hidden Layer: 8 neurons (ReLU)
Output Layer: 1 neuron (Sigmoid)


Purpose: Baseline model


🔹 Model 2: ANN with Dropout

Same architecture as Model 1
Dropout layers added:

0.3 after input layer
0.2 after hidden layer


Purpose: Reduce overfitting


🔹 Model 3: Multi-Feature ANN

Features used:

Tenure
Monthly Charges
Total Charges


Applied feature scaling
Purpose: Improve prediction using more informative inputs


⚙️ Model Optimization
To address real-world challenges, the following improvements were applied:

Class imbalance handled using class weights
Prediction threshold adjusted from 0.5 to 0.3
Feature scaling applied for better convergence

These enhancements significantly improved the model’s ability to detect churn customers instead of predicting only the majority class.

📈 Model Evaluation
Models were evaluated using:

Confusion Matrix → to analyze classification performance
Accuracy Score → overall prediction correctness
Accuracy vs Epochs Graph → training performance over time


📊 Key Results

Initial models showed bias toward non-churn class due to data imbalance
After applying class balancing and scaling:

Models improved in identifying churn customers
Performance became more realistic and usable




⚠️ Important Insight
Accuracy alone is not a reliable metric for imbalanced datasets.
The confusion matrix revealed that initial models predicted only the majority class. After optimization, the model was able to correctly classify churn cases, making it more effective for real-world usage.

📂 Project Structure
Customers Churn Project/
├── Data/
│   └── Telco-Customer-Churn.csv
├── outputs/
│   ├── Accuracy_vs_Epochs.png
│   ├── Model2_Accuracy_vs_Epochs.png
│   ├── Model3_Accuracy_vs_Epochs.png
│   ├── Churn_Distribution.png
│   ├── Internet_Service_Distribution.png
├── src/
│   └── churn_model.py
├── README.md
├── requirements.txt

🚀 Technologies Used

Python
Pandas, NumPy
Matplotlib, Seaborn
Scikit-learn
TensorFlow / Keras


💡 Conclusion
This project demonstrates how deep learning can be applied to solve a real-world business problem like customer churn prediction. It highlights the importance of proper data preprocessing, handling class imbalance, and evaluating models beyond accuracy.

🔗 Future Improvements

Use One-Hot Encoding instead of Label Encoding
Apply advanced models (XGBoost, Random Forest)
Hyperparameter tuning for better performance
Deploy model using Streamlit or Flask


👤 Author
Ajay Rathod
B.Tech CSE (AI & ML)

⭐ If you found this project useful
Give it a star on GitHub and connect on LinkedIn 🚀

