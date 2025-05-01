# Credit Card Default Model 

## Instructions:
To clone this repo, run 
```
git clone https://github.com/Jakej6262/Credit_Card_Default_Model
```

Install Dependencies 
```
pip install -r requirements.txt
```

## Results and Context
- The initial model performed well on the validation set with an **initial accuracy** of **84.4%** 
- However, there was only **82% recall on positive instances** (2868/3491 defaults captured). Capturing defaults should be prioritized and the threshold should be adjusted accordingly.
- After adjusting the threshold to prioritize recall, the model captures 3078/3491 **(88%)** of defaults while maintaining a **83%** accuracy

## About:
Random Forest model aiming to predict default status given the following variables over a 6 month span.
- Age
- Gender
- Marital Status 
- Education 
- Credit Limit
- Monthy bill amount 
- Monthly payment amount
- Payment status for each month 
    - (-1 = paid on time, 0= no payment, 1+= 1 or more months behind)

## Methodology:
- Chose random forest as it deals well with tightly correlated variables and is resistant to overfitting
- Implemented SMOTE to deal with feature imbalance (far less istances of default than non-default)
- Determine optimal prediction threshold to allign model with business needs 


