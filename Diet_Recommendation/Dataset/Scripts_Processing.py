import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("Food_and_Nutrition.csv")

# Trim spaces from column names
df.columns = df.columns.str.strip()

# Print columns to debug
print("Columns in dataset:", df.columns.tolist())

# Define numeric columns and check for existence
numeric_cols = ['Height', 'Weight', 'Calories', 'Carbohydrate', 'Protein', 'Sugar', 'Sodium', 'Fiber', 'Fat']
existing_numeric_cols = [col for col in numeric_cols if col in df.columns]
print("Using numeric columns:", existing_numeric_cols)  # Debugging line

# Fill missing values in numeric columns
df[existing_numeric_cols] = df[existing_numeric_cols].fillna(df[existing_numeric_cols].mean())
print(df[existing_numeric_cols])

# Handle missing values in categorical columns properly
df['Activity Level'] = df.get('Activity Level', pd.Series()).fillna("Unknown")
df['Dietary Preference'] = df.get('Dietary Preference', pd.Series()).fillna("Unknown")
df['Disease'] = df.get('Disease', pd.Series()).fillna("Unknown")

# Add BMI column safely
if 'Height' in df.columns and 'Weight' in df.columns:
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    print(df['BMI'])
    df.to_csv("Food_and_Nutrition_with_BMI.csv", index=False)


# Encode categorical data
for col in ['Activity Level', 'Dietary Preference', 'Disease']:
    if col in df.columns:
        df[col] = df[col].astype("category").cat.codes

# Normalize numeric columns (including BMI)
scaler = MinMaxScaler()
numeric_columns = ['Calories', 'Carbohydrate', 'Protein', 'Sugar', 'Sodium', 'Fiber', 'Fat', 'BMI']
existing_numeric_columns = [col for col in numeric_columns if col in df.columns]
df[existing_numeric_columns] = scaler.fit_transform(df[existing_numeric_columns])

# Save preprocessed dataset
df.to_csv("Food_and_Nutrition_Processed.csv", index=False)

print("Data preprocessing complete.")
