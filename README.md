# Hotel Revenue Optimisation

Analyzing over 110,000 hotel bookings to predict cancellations, segment customers, and optimize pricing. Developed in R using advanced machine learning models including Random Forest, XGBoost, and K-Means clustering.

## Key Features

* Customer segmentation into 4 distinct behavioral groups  
* High-accuracy cancellation prediction with 96% AUC  
* Average Daily Rate (ADR) pricing models using Linear Regression and CART  
* Loyalty prediction for repeat guests  

**Models Used:** Logistic Regression, Random Forest, XGBoost, K-Means  
**Dataset:** Kaggle Hotel Booking Demand (87,000+ records after cleaning) ([Kaggle link](https://www.kaggle.com/datasets/mojtaba142/hotel-booking))  

Includes a full data cleaning pipeline, exploratory data analysis (EDA) visualizations, and comprehensive feature engineering.

## Key Insights

### Cancellation Patterns (Quantified)

* Lead time risk rises from 21% for short lead times to over 50% for bookings made 180+ days in advance for city hotels  
* Online Travel Agency (OTA) bookings are 3.3 times more likely to cancel compared to direct bookings  
* Group size risk peaks at 45% cancellation rate for 4-person bookings, dropping significantly beyond 5 persons  
* Non-refundable deposits show 95% cancellation, likely reflecting a data quality issue with no-shows  

### Model Performance Comparison

#### Cancellation Prediction

| Model         | Accuracy | AUC   | Recall | False Negatives |
| ------------- | -------- | ----- | ------ | --------------- |
| Logistic      | 82.8%    | 0.901 | 74.8%  | -               |
| Random Forest | 89.6%    | 0.960 | 84.9%  | 1,385           |
| **XGBoost**   | 88.5%    | 0.960 | 88.8%  | 1,030           |

XGBoost identifies 355 more risky bookings than Random Forest, crucial for revenue optimization.

#### Loyalty Prediction

| Model         | AUC   | Precision | Recall | PR-AUC |
| ------------- | ----- | --------- | ------ | ------ |
| Random Forest | 0.988 | 82.2%     | 89.8%  | 0.897  |
| **XGBoost**   | 0.993 | 96.2%     | 94.0%  | 0.986  |

XGBoost reduces false positives by 74% per true positive, enhancing targeting accuracy.

### Feature Importance (XGBoost Gain)

**Cancellation Drivers:**  
* Online Travel Agent (OTA): 15.3%  
* Lead time  
* ADR  
* Market segment  

**Loyalty Drivers:**  
* Previous bookings: 64.3% (dominant)  
* Lead time: 11.4%  
* ADR: 5.1%  
* Prior cancellations: 3.9%  

### Segmentation Stats

* Number of clusters: 4 (determined by elbow method)  
* Cluster 3 (OTA): Highest ADR and highest cancellation rate  
* Cluster 1 (Corporate): Lowest ADR and lowest cancellation rate  
* Repeat guests have a 29.8% room change rate versus 12.2% for new guests (2.4x difference)  

### Pricing Model

* Linear regression \(R^2\) of 0.64 with RMSE of 31.44  
* CART’s top split based on arrival month highlights strong seasonal effects  
* Repeat guests receive a negative coefficient, confirming loyalty discounts  
