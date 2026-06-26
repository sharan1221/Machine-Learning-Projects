# Customer Segmentation — K-Means Clustering

Built a K-Means Clustering model to segment mall customers 
into 5 distinct groups based on income and spending behaviour.

## Results
- Optimal Clusters : 5 (found using Elbow Method)
- Dataset         : 200 mall customers
- Segments found  : High spender, Low spender, Average, Impulsive, Saver

## What I did
- Loaded and explored customer data
- Selected Annual Income and Spending Score as features
- Scaled features using StandardScaler
- Used Elbow Method to find optimal K = 5
- Trained K-Means model and assigned cluster labels
- Profiled each cluster to understand customer personality
- Built interactive segment predictor for new customers

## Tech Stack
Python · scikit-learn · pandas · NumPy · matplotlib

## How to Run
```bash
pip install pandas numpy scikit-learn matplotlib
python customer_segmentation.py
```
Download dataset from:
https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
