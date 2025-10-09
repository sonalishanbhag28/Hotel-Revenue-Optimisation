# Hotel Booking Analytics
Analyzing 110k+ hotel bookings to predict cancellations, segment customers, and optimize pricing. Built with R using Random Forest, XGBoost, and K-Means clustering.

### Key Features:
Customer segmentation into 4 behavioral groups
Cancellation prediction (96% AUC)
ADR pricing models (Linear + CART)
Loyalty prediction for repeat guests

**Models**: Logistic Regression, Random Forest, XGBoost, K-Means

**Dataset**: Kaggle Hotel Booking Demand (87k records after cleaning) (*https://www.kaggle.com/datasets/mojtaba142/hotel-booking*)

Includes full data cleaning pipeline, EDA visualizations, and feature engineering. 

## Key Insights:

### Cancellation Patterns (Quantified)
- **Lead time risk**: 21% (short) → 50%+ (180+ days) for city hotels
- **Channel risk ratio**: OTA is **3.3x more likely** to cancel vs direct bookings
- **Group size sweet spot**: 4-person bookings = 45% cancel rate (highest), 5+ drops significantly
- **Deposit impact**: Non-refundable shows 95% cancellation (data quality issue - likely no-shows)

### Model Performance Comparison

**Cancellation Prediction**
| Model | Accuracy | AUC | Recall | False Negatives |
|-------|----------|-----|--------|-----------------|
| Logistic | 82.8% | 0.901 | 74.8% | - |
| Random Forest | 89.6% | 0.960 | 84.9% | 1,385 |
| **XGBoost** | **88.5%** | **0.960** | **88.8%** | **1,030** |

**XGBoost catches 355 more risky bookings** (critical for revenue)

**Loyalty Prediction**
| Model | AUC | Precision | Recall | PR-AUC |
|-------|-----|-----------|--------|--------|
| Random Forest | 0.988 | 82.2% | 89.8% | 0.897 |
| **XGBoost** | **0.993** | **96.2%** | **94.0%** | **0.986** |

**74% fewer false positives** per true positive (better targeting)

### Feature Importance (XGBoost Gain)

**Cancellation drivers:**
1. Online TA: 15.3%
2. Lead time
3. ADR
4. Market segment

**Loyalty drivers:**
1. Previous bookings: **64.3%** (dominates)
2. Lead time: 11.4%
3. ADR: 5.1%
4. Prior cancellations: 3.9%

### Segmentation Stats
- **K=4 clusters** (elbow method)
- Cluster 3 (OTA): Highest ADR + highest cancel rate
- Cluster 1 (Corporate): Lowest ADR + lowest cancel rate
- **Repeat guests**: 29.8% room change rate vs 12.2% for new (2.4x difference)

### Pricing Model
- **Linear R²**: 0.64, RMSE: 31.44
- **CART top split**: Arrival month (seasonal effect strongest)
- Repeat guests get **negative coefficient** (loyalty discounts confirmed)
