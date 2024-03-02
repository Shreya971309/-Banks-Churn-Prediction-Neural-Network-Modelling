
# Churn Prediction in Banking Using Neural Networks


## Overview
This project aims to predict whether bank customers will leave the service provider within the next six months. Churn prediction is crucial for businesses like banks to understand customer behavior and improve service quality to retain customers. The project utilizes neural network-based classifiers to achieve this objective.
## Problem Statement
Businesses like banks face the challenge of customer churn, where customers leave for alternative service providers. Understanding the factors influencing customer churn is essential for targeted improvement efforts. The objective is to build a neural network-based classifier that predicts whether a bank customer will leave within the next six months.
## Data Description

The dataset, taken from Kaggle, contains 10,000 sample points with 14 distinct features. These are:
| Number | Attribute         | Description                                                                                       |
|--------|-------------------|---------------------------------------------------------------------------------------------------|
| 1.     | **CustomerId**    | Unique ID assigned to each customer.                                                             |
| 2.     | **Surname**       | Last name of the customer.                                                                       |
| 3.     | **CreditScore**   | Defines the credit history of the customer.                                                       |
| 4.     | **Geography**     | Customer's location.                                                                              |
| 5.     | **Gender**        | Gender of the customer.                                                                           |
| 6.     | **Age**           | Age of the customer.                                                                              |
| 7.     | **Tenure**        | Number of years for which the customer has been with the bank.                                     |
| 8.     | **NumOfProducts** | Number of products that a customer has purchased through the bank.                                 |
| 9.     | **Balance**       | Account balance.                                                                                  |
| 10.    | **HasCrCard**     | Categorical variable indicating whether the customer has a credit card (1 for yes, 0 for no).     |
| 11.    | **EstimatedSalary** | Estimated salary of the customer.                                                              |
| 12.    | **IsActiveMember** | Categorical variable indicating whether the customer is an active member of the bank (1 for yes, 0 for no). |
| 13.    | **Exited**        | Categorical variable indicating whether the customer left the bank within six months (1 for yes, 0 for no). |

## Project structure
<br> (click to expand)

<details>
<summary>Exploratory Data Analysis</summary>

<h2>Exploratory Data Analysis (EDA) Insights</h2>


#### Overall Churn Rate
- **Bank Churn Rate:** 20.4% of customers have exited the bank, while 79.6% remain.
 <center>

![exited_not](https://github.com/Shreya971309/-Banks-Churn-Prediction-Neural-Network-Modelling/assets/156785157/ecac545c-42b5-470f-b59f-1d5eb14954ca)
</center>

#### Gender Disparity
- **Gender Churn Analysis:** Women exhibit a higher churn rate (25%) compared to men (16.4%).
<center>
  
  ![gender](https://github.com/Shreya971309/-Banks-Churn-Prediction-Neural-Network-Modelling/assets/156785157/040ec1db-ee94-4a19-933d-b9b0f3b6af57)

</center>

#### Geographic Influence
- **Regional Churn Patterns:** Germany experiences the highest churn rate (32%) among all countries, followed by France (16.1%) and Spain (16.6%).
<center>
  
  ![geography](https://github.com/Shreya971309/-Banks-Churn-Prediction-Neural-Network-Modelling/assets/156785157/22cd4de6-e40b-4f6e-8d83-2d17e099f313)

  
</center>

#### Tenure Impact
- **Tenure and Churn Relationship:** Customers with shorter tenure, particularly 0 and 1 year, show elevated churn rates.
  <center>
![tenure](https://github.com/Shreya971309/-Banks-Churn-Prediction-Neural-Network-Modelling/assets/156785157/4fb17255-b9c1-4276-bbac-7f5f20d3da08)

  </center>


#### Product Purchases
- **Product Impact on Churn:** All customers with 4 product purchases have churned, whereas those with 2 products exhibit the lowest churn rate.
<center>
  
  ![no_of_prods](https://github.com/Shreya971309/-Banks-Churn-Prediction-Neural-Network-Modelling/assets/156785157/93fd5c04-5767-463a-a6ef-51d3bd64c5c9)

</center>

#### Membership Activity
- **Active Membership vs. Churn:** Non-active members demonstrate a higher propensity to churn.
<center>
  
  ![active_non_active](https://github.com/Shreya971309/-Banks-Churn-Prediction-Neural-Network-Modelling/assets/156785157/528e42f5-dcf9-4a06-8e52-5ea03897afb3)

</center>

#### Credit Card Ownership
- **Credit Card Influence on Churn:** Churn rates are comparable between customers with and without credit cards.
<center>
  
  ![credit_card](https://github.com/Shreya971309/-Banks-Churn-Prediction-Neural-Network-Modelling/assets/156785157/39b00797-7af8-4b06-98fd-6dd8663e8574)

</center>


</details>

<details>
<summary>Data Pre-processing</summary>

<h2>Data Preprocessing</h2>

### Missing Value and Duplicate Data Treatment

No missing or duplicated values were found in the dataset, so no treatment was necessary.

### Outlier Removal

Although outliers were detected in the "Age" and "CreditScore" features, they were retained. These outliers may offer valuable insights into the diversity of the dataset.

### Dummy Creation for Geography and Gender

Categorical variables such as "Geography" and "Gender" were converted into dummy variables.

### Data Split

The dataset was split into training, validation, and test sets:

- **Training Set (X_train, y_train):** 64% of the data
- **Validation Set (X_val, y_val):** 16% of the data
- **Test Set (X_test, y_test):** 20% of the data

### Scaling

The numerical columns were scaled using the standard scaler to ensure consistency in feature magnitudes.


</details>

<details>
<summary>Model Building</summary>

## Model Building

### Model Evaluation Criterion

Recall is prioritized, especially when the cost of false negatives is high. In customer churn prediction, reducing false negatives is crucial as it minimizes missed opportunities to retain customers by providing incentives to prevent them from leaving.

### Models

- **SGD Optimizer**: Uses Stochastic Gradient Descent (SGD) to update model weights in small batches, aiming for gradual loss function minimization.

- **Adam Optimizer**: Utilizes Adam optimizer, adapting learning rates for each parameter, leading to faster convergence and improved performance, especially on large datasets.

- **Adam Optimizer with Dropout (Dropout rate: 0.2)**: Incorporates dropout regularization, randomly removing 20% of neurons during training to prevent overfitting by reducing dependency on specific neurons.

- **Adam Optimizer with Hyperparameter Tuning**: Employs grid search to find the best hyperparameter combination (batch size and learning rate) for optimizing model performance.

- **Balanced Data with SMOTE and Adam Optimizer (Oversampling)**: Addresses class imbalance by oversampling the minority class using SMOTE in combination with Adam optimizer during training, enhancing the model's ability to learn from minority class instances.




</details>



<details>
<summary>Final Model Selection</summary>
  
<h2>Final Model Selection</h2> 

### Evaluation Metrics

- **ROC-AUC**: Assesses the model's ability to rank customers based on their likelihood of churning, providing insights into how well the model discriminates between churners and non-churners.

- **Accuracy**: Reflects the proportion of correct predictions made by the model, offering a general measure of its overall correctness.

- **Precision**: Indicates the model's precision in identifying customers who actually churn, capturing the proportion of true churners among all predicted churners.

- **Recall**: Measures the model's ability to capture all customers who are likely to churn, revealing its capability to identify all positive instances correctly.

- **F1-score**: Combines precision and recall into a single metric, offering a balanced assessment of the model's performance, especially in scenarios where there is an imbalance between churned and non-churned customers.
### Model-1: SGD Optimizer
| Metric   | Accuracy | Precision | Recall   | F1 Score | AUC Score |
|----------|----------|-----------|----------|----------|-----------|
| Score    | 0.718125 | 0.392796  | 0.702454 | 0.50385  | 0.80      |

### Model-2: Adam Optimizer
| Metric   | Accuracy | Precision | Recall   | F1 Score | AUC Score |
|----------|----------|-----------|----------|----------|-----------|
| Score    | 0.780625 | 0.474227  | 0.705521 | 0.567201 | 0.87      |

### Model-3: Adam Optimizer with Dropout (Dropout rate: 0.2)
| Metric   | Accuracy | Precision | Recall   | F1 Score | AUC Score |
|----------|----------|-----------|----------|----------|-----------|
| Score    | 0.7775   | 0.47012   | 0.723926 | 0.570048 | 0.86      |

### Model-4: Adam Optimizer with Hyperparameter Tuning
| Metric   | Accuracy | Precision | Recall   | F1 Score | AUC Score |
|----------|----------|-----------|----------|----------|-----------|
| Score    | 0.765    | 0.451172  | 0.708589 | 0.551313 | 0.86      |

### Model-5: Balanced Data with SMOTE and Adam Optimizer (Oversampling)
| Metric   | Accuracy | Precision | Recall   | F1 Score | AUC Score |
|----------|----------|-----------|----------|----------|-----------|
| Score    | 0.73625  | 0.413978  | 0.708589 | 0.522624 | 0.87      |

### Final Model
After evaluating all considered models, we determined that the model incorporating the Adam optimizer and dropout rate of 0.2 (Model 3) exhibited the highest recall value. In the context of our specific problem statement, where correctly identifying customers likely to leave is crucial, recall serves as an important metric. Therefore, we select Model 3 as our final model for predicting values on the test set.

| Metric    | Score    |
|-----------|----------|
| Accuracy  | 0.73625  |
| Precision | 0.413978 |
| Recall    | 0.708589 |
| F1 Score  | 0.522624 |
| AUC       | 0.83     |

- **Accuracy (0.73625)**: The model correctly predicted the customer churn for about 73.625% of the customers in the test set.
- **Precision (0.413978)**: When the model predicts a customer will churn, it is correct about 41.3978% of the time. The model may be overestimating customer churn, leading to unnecessary retention efforts.
- **Recall (0.708589)**: The model correctly identified 70.8589% of the customers who actually churned. This is crucial to identify as many churning customers as possible.
- **F1 Score (0.522624)**: An F1 score of 52.2624% indicates that there is room for improvement in achieving a better balance.
- **AUC (0.83)**: An AUC score of 0.83 on the test set indicates relatively good performance of the model in distinguishing between positive and negative instances. Therefore, the model has a high probability of ranking a randomly chosen positive instance higher than a randomly chosen negative instance.
<center>
  
![AUC_ROC](https://github.com/Shreya971309/-Banks-Churn-Prediction-Neural-Network-Modelling/assets/156785157/ce78e0bf-b3dd-4f3d-8d56-4648de72a3a5)

  
</center>

<center>
  
  ![confusion_matrix](https://github.com/Shreya971309/-Banks-Churn-Prediction-Neural-Network-Modelling/assets/156785157/96c7c569-7838-4857-a73e-7fb90693e91b)

</center>


</details>

<details>
<summary>Insights and Recommendations</summary>
<h2> Insights and Recommendations</h2>

#### Engagement of Dormant Members
The bank may consider launching a campaign to re-engage dormant members and convert them into active clients. This could involve reaching out to them with exclusive deals, incentives, or personalized financial guidance to help them make the most out of their accounts.

#### Product Retention and Diversification
Encouraging customers to diversify their product holdings could be beneficial, especially considering that a significant proportion (51%) of customers only own one product. Implementing retention strategies to retain clients with multiple products, such as offering incentives or bundled services, could be effective.

#### Services Tailored to Age
Given the positive correlation between leaving a bank and age, the bank should consider offering age-specific services or incentives to retain customers across different age groups. Customizing services to cater to various life stages could enhance client retention.

#### Retention Strategies Based on Tenure
Customers with shorter tenures, specifically one year and zero years, exhibit higher rates of churn. Implementing promotions, personalized services, or onboarding programs targeted at acquiring and retaining customers during the early years of their banking relationship could mitigate churn.

</details>

<details>
<summary>TL;DR: Churn Prediction Project Overview</summary>
  
### TL;DR: Churn Prediction Project Overview

- **Objective**: Develop neural network models to predict bank customer churn within the next six months.

- **Key Steps**:
  - Explored dataset, handled preprocessing tasks like missing values and encoding.
  - Built neural network models using different optimizers like SGD and Adam.
  - Enhanced model performance through techniques like dropout and hyperparameter tuning.
  - Selected the model with the highest recall (Adam optimizer with dropout rate of 0.2) for predicting customer churn.

- **Actual Insights & Recommendations**:
  - **Engagement of Dormant Members**: Re-engage dormant members with exclusive deals or personalized financial guidance.
  - **Product Retention and Diversification**: Encourage customers to diversify their product holdings and implement retention techniques.
  - **Services Tailored to Age**: Offer age-specific services or incentives to retain customers across different age groups.
  - **Retention Strategies Based on Tenure**: Implement promotions, personalized services, or onboarding programs for new customers.

</details>
