
import pandas as pd              
import numpy as np                
from sklearn.cluster import KMeans          
from sklearn.preprocessing import StandardScaler  
import matplotlib                           
matplotlib.use('Agg')                       
import matplotlib.pyplot as plt            
import os                                 

print("=" * 55)
print("   CUSTOMER SEGMENTATION - K-Means Clustering")
print("=" * 55)

if not os.path.exists('Mall_Customers.csv'):
    print("ERROR: Mall_Customers.csv not found in this folder.")
    exit()

df = pd.read_csv('Mall_Customers.csv')
print(f"\nDataset loaded: {df.shape[0]} customers, {df.shape[1]} columns")
print(f"\n   Columns in dataset:")
for col in df.columns:
    print(f"   - {col}")

print(f"\n{'='*55}")
print("   DATASET OVERVIEW")
print(f"{'='*55}")
print(df.describe().round(2))

print(f"\n   Gender breakdown:")
print(df['Gender'].value_counts().to_string())

print(f"\n   Missing values: {df.isnull().sum().sum()} (none - clean dataset!)")


X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nFeatures selected and scaled:")
print(f"   Annual Income (k$)     : min={X[:,0].min()}, max={X[:,0].max()}")
print(f"   Spending Score (1-100) : min={X[:,1].min()}, max={X[:,1].max()}")

print(f"\n{'='*55}")
print("   ELBOW METHOD (Finding best K)")
print(f"{'='*55}")

inertias = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    print(f"   K={k:2d}  |  Inertia: {km.inertia_:,.2f}")

print(f"\n   Best K = 5 (clear elbow point in the curve)")
optimal_k = 5

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans.fit(X_scaled)

df['Cluster'] = kmeans.labels_

print(f"\nK-Means model trained with K={optimal_k}")
print(f"\n   Customers per cluster:")
for i in range(optimal_k):
    count = (df['Cluster'] == i).sum()
    print(f"   Cluster {i}: {count} customers")

cluster_labels = {
    0: "Low Income,  Low Spender    (Careful)",
    1: "High Income, Low Spender    (Saver)",
    2: "Medium Income, Medium Spender (Average)",
    3: "Low Income,  High Spender   (Impulsive)",
    4: "High Income, High Spender   (Target Customer)"
}

print(f"\n{'='*55}")
print("   CLUSTER PROFILES")
print(f"{'='*55}")

cluster_means = df.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)', 'Age']].mean().round(1)
for i in range(optimal_k):
    row = cluster_means.loc[i]
    count = (df['Cluster'] == i).sum()
    print(f"\n   Cluster {i} ({count} customers)")
    print(f"   Avg Income : ${row['Annual Income (k$)']:.0f}k")
    print(f"   Avg Spend  : {row['Spending Score (1-100)']:.0f}/100")
    print(f"   Avg Age    : {row['Age']:.0f} years")

colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
cluster_names = ['Low Income\nLow Spender', 'High Income\nLow Spender',
                 'Medium Income\nMedium Spender', 'Low Income\nHigh Spender',
                 'High Income\nHigh Spender']

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle('Customer Segmentation - K-Means Clustering', fontsize=15, fontweight='bold', y=0.98)

for i in range(optimal_k):
    mask = df['Cluster'] == i
    axes[0,0].scatter(
        df[mask]['Annual Income (k$)'],
        df[mask]['Spending Score (1-100)'],
        c=colors[i], label=cluster_names[i], s=60, alpha=0.8, edgecolors='white', lw=0.5
    )

centres_original = scaler.inverse_transform(kmeans.cluster_centers_)
axes[0,0].scatter(
    centres_original[:,0], centres_original[:,1],
    c='black', marker='X', s=200, zorder=5, label='Cluster Centres'
)
axes[0,0].set_xlabel("Annual Income (k$)", fontsize=11)
axes[0,0].set_ylabel("Spending Score (1-100)", fontsize=11)
axes[0,0].set_title("Customer Segments by Income vs Spending", fontsize=12, fontweight='bold')
axes[0,0].legend(fontsize=8, loc='upper left')
axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(list(K_range), inertias, 'bo-', markersize=7, linewidth=2)
axes[0,1].axvline(x=5, color='red', linestyle='--', lw=2, label='Optimal K=5')
axes[0,1].scatter([5], [inertias[4]], color='red', s=150, zorder=5)
axes[0,1].set_xlabel("Number of Clusters (K)", fontsize=11)
axes[0,1].set_ylabel("Inertia (Total distance to centre)", fontsize=11)
axes[0,1].set_title("Elbow Method — Finding Best K", fontsize=12, fontweight='bold')
axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)
axes[0,1].set_xticks(list(K_range))

cluster_counts = df['Cluster'].value_counts().sort_index()
bars = axes[1,0].bar(range(optimal_k), cluster_counts.values,
                      color=colors, edgecolor='black', alpha=0.85)
axes[1,0].set_xlabel("Cluster", fontsize=11)
axes[1,0].set_ylabel("Number of Customers", fontsize=11)
axes[1,0].set_title("Number of Customers per Cluster", fontsize=12, fontweight='bold')
axes[1,0].set_xticks(range(optimal_k))
axes[1,0].set_xticklabels([f'Cluster {i}' for i in range(optimal_k)], fontsize=9)
for bar, val in zip(bars, cluster_counts.values):
    axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(val), ha='center', fontweight='bold', fontsize=11)
axes[1,0].grid(True, alpha=0.3, axis='y')

x_pos = np.arange(optimal_k)
width = 0.35
bars1 = axes[1,1].bar(x_pos - width/2, cluster_means['Annual Income (k$)'],
                       width, label='Avg Income (k$)', color='steelblue',
                       edgecolor='black', alpha=0.85)
bars2 = axes[1,1].bar(x_pos + width/2, cluster_means['Spending Score (1-100)'],
                       width, label='Avg Spending Score', color='coral',
                       edgecolor='black', alpha=0.85)
axes[1,1].set_xlabel("Cluster", fontsize=11)
axes[1,1].set_ylabel("Value", fontsize=11)
axes[1,1].set_title("Avg Income vs Spending Score per Cluster", fontsize=12, fontweight='bold')
axes[1,1].set_xticks(x_pos)
axes[1,1].set_xticklabels([f'Cluster {i}' for i in range(optimal_k)], fontsize=9)
axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('customer_segmentation_results.png', dpi=150, bbox_inches='tight')
print(f"\nCharts saved as 'customer_segmentation_results.png'")

print(f"\n{'='*55}")
print("   PREDICT A NEW CUSTOMER'S SEGMENT")
print(f"{'='*55}")

def get_int(prompt, min_val, max_val):
    while True:
        try:
            val = int(input(prompt))
            if min_val <= val <= max_val:
                return val
            print(f"  ⚠  Enter a value between {min_val} and {max_val}")
        except ValueError:
            print("  ⚠  Please enter a whole number")

print("\nEnter details of a new customer:\n")
income  = get_int("  Annual Income in k$ (e.g. 60 means $60,000): ", 1, 200)
spending = get_int("  Spending Score (1 to 100): ", 1, 100)

new_customer = scaler.transform([[income, spending]])

predicted_cluster = kmeans.predict(new_customer)[0]

descriptions = [
    "Low Income + Low Spender — Careful, budget-conscious customer",
    "High Income + Low Spender — Wealthy but conservative spender",
    "Medium Income + Average Spender — Typical average customer",
    "Low Income + High Spender — Impulsive, spends beyond means",
    "High Income + High Spender — Premium target customer"
]

print(f"\n{'='*55}")
print("   NEW CUSTOMER RESULT")
print(f"{'='*55}")
print(f"  Annual Income  : ${income}k")
print(f"  Spending Score : {spending}/100")
print(f"\n  Assigned to : Cluster {predicted_cluster}")
print(f"  👤 Profile     : {descriptions[predicted_cluster]}")
profile = cluster_means.loc[predicted_cluster]
print(f"\n  This cluster's averages:")
print(f"  - Avg Income  : ${profile['Annual Income (k$)']:.0f}k")
print(f"  - Avg Spending: {profile['Spending Score (1-100)']:.0f}/100")
print(f"  - Avg Age     : {profile['Age']:.0f} years")
print(f"\n  Total customers in this segment: {(df['Cluster']==predicted_cluster).sum()}")
print("=" * 55)
print("\nDone! Check 'customer_segmentation_results.png' for charts.")
