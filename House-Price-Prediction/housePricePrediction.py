import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

if not os.path.exists('train.csv'):
    print("Error: train.csv not found")
    exit()
df = pd.read_csv('train.csv')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

features=["GrLivArea","OverallQual","OverallCond","YearBuilt","YearRemodAdd","TotalBsmtSF","GarageCars","GarageArea","Fireplaces","LotArea","TotRmsAbbvGrd","BedroomAbvGr","FullBath","HalfBath","CentralAir","Neighbourhood"]
target="SalePrice"

df_model = df[features + [target]].copy()
 
df_model['CentralAir'] = df_model['CentralAir'].map({'Y': 1, 'N': 0})

le_neighborhood = LabelEncoder()
df_model['Neighborhood'] = le_neighborhood.fit_transform(
    df_model['Neighborhood'].astype(str)
)
neighborhood_list = list(le_neighborhood.classes_)  
 
df_model = df_model.dropna()
 
X = df_model[features]
y = df_model[target]
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training set: {len(X_train)} houses")
print(f"Testing set : {len(X_test)} houses")

model = LinearRegression()
model.fit(X_train, y_train)
print(f"\nModel trained successfully!")

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
 
print("\n" + "=" * 55)
print("   MODEL PERFORMANCE")
print("=" * 55)
print(f"  R² Score : {r2:.4f}  (basic model was 0.6335)")
print(f"  RMSE     : ${rmse:,.2f}  (basic model was $53,018)")
print(f"  Improvement: R² went up by {(r2-0.6335)*100:.1f}%!")

print("\n" + "=" * 55)
print("   WHAT THE MODEL LEARNED")
print("=" * 55)
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
}).sort_values('Coefficient', ascending=False)
 
for _, row in coef_df.iterrows():
    direction = "▲" if row['Coefficient'] > 0 else "▼"
    print(f"  {direction} {row['Feature']:20s}: ${row['Coefficient']:>12,.2f}")
print(f"\n  Base price (intercept): ${model.intercept_:,.2f}")

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle('House Price Prediction - Enhanced Linear Regression', 
             fontsize=15, fontweight='bold', y=0.98)

axes[0, 0].scatter(y_test, y_pred, alpha=0.4, color='steelblue', s=20)
axes[0, 0].plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect fit')
axes[0, 0].set_xlabel("Actual Price ($)")
axes[0, 0].set_ylabel("Predicted Price ($)")
axes[0, 0].set_title("Actual vs Predicted Prices")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

residuals = y_test.values - y_pred
axes[0, 1].scatter(y_pred, residuals, alpha=0.4, color='coral', s=20)
axes[0, 1].axhline(0, color='black', lw=1.5, linestyle='--')
axes[0, 1].set_xlabel("Predicted Price ($)")
axes[0, 1].set_ylabel("Error (Actual - Predicted)")
axes[0, 1].set_title("Residual Plot (Errors)")
axes[0, 1].grid(True, alpha=0.3)
 
colors = ['green' if c > 0 else 'red' for c in coef_df['Coefficient']]
axes[1, 0].barh(coef_df['Feature'], coef_df['Coefficient'],
                color=colors, alpha=0.7, edgecolor='black')
axes[1, 0].axvline(0, color='black', lw=1)
axes[1, 0].set_xlabel("Coefficient Value ($)")
axes[1, 0].set_title("Feature Impact on Price")
axes[1, 0].grid(True, alpha=0.3, axis='x')

model_names  = ['Basic Model\n(4 features)', 'Enhanced Model\n(16 features)']
r2_scores    = [0.6335, r2]
bar_colors   = ['#ff7f7f', '#7fbfff']
bars = axes[1, 1].bar(model_names, r2_scores, color=bar_colors,
                       edgecolor='black', width=0.4)
axes[1, 1].set_ylim(0, 1)
axes[1, 1].set_ylabel("R² Score")
axes[1, 1].set_title("Basic vs Enhanced Model")
axes[1, 1].grid(True, alpha=0.3, axis='y')
for bar, score in zip(bars, r2_scores):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.02,
                    f'{score:.4f}', ha='center', fontweight='bold')
 
plt.tight_layout()
plt.savefig('house_price_results.png', dpi=150, bbox_inches='tight')
print("\nCharts saved as 'house_price_results.png'")
 
print("\n" + "=" * 55)
print("   PREDICT YOUR OWN HOUSE PRICE")
print("=" * 55)
 
def get_int(prompt, min_val, max_val):
    while True:
        try:
            val = int(input(prompt))
            if min_val <= val <= max_val:
                return val
            print(f"   Please enter a value between {min_val} and {max_val}")
        except ValueError:
            print("    Please enter a whole number")
 
def get_yes_no(prompt):
    while True:
        val = input(prompt).strip().upper()
        if val in ['Y', 'N']:
            return 1 if val == 'Y' else 0
        print("   Please enter Y or N")
 
def get_neighborhood():
    print("\n  Available Neighborhoods:")
    for i, n in enumerate(neighborhood_list):
        print(f"    {i:2d}. {n}")
    while True:
        try:
            idx = int(input(f"\n  Enter number (0-{len(neighborhood_list)-1}): "))
            if 0 <= idx < len(neighborhood_list):
                return idx, neighborhood_list[idx]
            print(f"   Enter a number between 0 and {len(neighborhood_list)-1}")
        except ValueError:
            print("   Please enter a number")
 
print("\nEnter the details of the house you want to price:\n")
 
grlivarea    = get_int("  Above-ground living area (sqft, e.g. 1500): ", 300, 6000)
overallqual  = get_int("  Overall quality (1=Poor, 5=Average, 10=Excellent): ", 1, 10)
overallcond  = get_int("  Overall condition (1=Poor, 5=Average, 10=Excellent): ", 1, 10)
yearbuilt    = get_int("  Year built (e.g. 1990): ", 1872, 2010)
yearremod    = get_int("  Year of last remodel (same as YearBuilt if never remodeled): ", 1950, 2010)
totalbsmtsf  = get_int("  Total basement area in sqft (0 if no basement): ", 0, 3000)
garagecars   = get_int("  Garage capacity (0=No garage, 1, 2, or 3 cars): ", 0, 3)
garagearea   = get_int("  Garage area in sqft (0 if no garage): ", 0, 1500)
fireplaces   = get_int("  Number of fireplaces (0, 1, 2, or 3): ", 0, 3)
lotarea      = get_int("  Lot area in sqft (e.g. 8000): ", 1000, 50000)
totrms       = get_int("  Total rooms above ground (not counting bathrooms): ", 2, 15)
bedrooms     = get_int("  Bedrooms above ground: ", 0, 8)
fullbath     = get_int("  Full bathrooms: ", 0, 3)
halfbath     = get_int("  Half bathrooms: ", 0, 2)
centralair   = get_yes_no("  Central air conditioning? (Y/N): ")
neigh_idx, neigh_name = get_neighborhood()
 
user_input = pd.DataFrame([{
    'GrLivArea'   : grlivarea,
    'OverallQual' : overallqual,
    'OverallCond' : overallcond,
    'YearBuilt'   : yearbuilt,
    'YearRemodAdd': yearremod,
    'TotalBsmtSF' : totalbsmtsf,
    'GarageCars'  : garagecars,
    'GarageArea'  : garagearea,
    'Fireplaces'  : fireplaces,
    'LotArea'     : lotarea,
    'TotRmsAbvGrd': totrms,
    'BedroomAbvGr': bedrooms,
    'FullBath'    : fullbath,
    'HalfBath'    : halfbath,
    'CentralAir'  : centralair,
    'Neighborhood': neigh_idx,
}])
 
predicted_price = model.predict(user_input)[0]
predicted_price = max(0, predicted_price)  
 
print("\n" + "=" * 55)
print("   YOUR HOUSE SUMMARY")
print("=" * 55)
print(f"  Living Area   : {grlivarea:,} sqft")
print(f"  Quality       : {overallqual}/10  |  Condition: {overallcond}/10")
print(f"  Year Built    : {yearbuilt}  |  Remodeled: {yearremod}")
print(f"  Basement      : {totalbsmtsf:,} sqft")
print(f"  Garage        : {garagecars} car(s), {garagearea} sqft")
print(f"  Fireplaces    : {fireplaces}")
print(f"  Lot Area      : {lotarea:,} sqft")
print(f"  Rooms         : {totrms} total | {bedrooms} bed | {fullbath} full bath | {halfbath} half bath")
print(f"  Central Air   : {'Yes' if centralair else 'No'}")
print(f"  Neighborhood  : {neigh_name}")
print()
print(f"  PREDICTED PRICE: ${predicted_price:,.2f}")
print(f"   Model accuracy : R² = {r2:.4f} (predictions typically")
print(f"                      within ±${rmse:,.0f})")
print("=" * 55)
print("\nDone! Check 'house_price_enhanced_results.png' for charts.")