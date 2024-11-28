import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys

# Set up argument parsing
parser = argparse.ArgumentParser(description='Handle missing data in a CSV file.')
parser.add_argument('--file', type=str, required=True, help='Path to the CSV file.')
parser.add_argument('--method', type=str, choices=['drop', 'median', 'mean'], required=True, help='Method to handle missing data: drop, median, or mean.')

args = parser.parse_args()

df = pd.read_csv("2023-7-16-9126FCE9-F954-49EE-BF79-56B319560E0C.csv") #dataframe

print(df.columns)

# Handle missing data
if args.method == 'drop':
    df = df.dropna()
elif args.method == 'median':
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
elif args.method == 'mean':
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())


df['Sex'] = df['Sex'].map({'male':  0, 'female':  1}) #Map 'Sex' to numerical values
categorical_cols = df.select_dtypes(include=['object']).columns #Identify categorical columns

# Print the shape of the DataFrame to see the number of rows and columns before encoding
print("Shape of the DataFrame before one-hot encoding:", df.shape)

# Perform one-hot encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


# Print the shape of the DataFrame to see the number of rows and columns
print("Shape of the DataFrame after one-hot encoding:", df_encoded.shape)

# Save the result
output_file = "processed_" + args.file
df_encoded.to_csv(output_file, index=False)
print(f"Processed data saved to {output_file}")

numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns #Select only the numeric columns for PCA

# Standardize the numeric columns
scaler = StandardScaler()
df_encoded_scaled = scaler.fit_transform(df_encoded[numeric_cols])

# Apply PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(df_encoded_scaled)

# Create a DataFrame with the principal components
df_pca = pd.DataFrame(data = principalComponents, columns = ['principal component  1', 'principal component  2'])

df_pca['Survived'] = df_encoded['Survived'] #Add the Survived column to the DataFrame for coloring

# Plotting
plt.figure(figsize=(10,  8))
plt.scatter(df_pca['principal component  1'], df_pca['principal component  2'], c=df_pca['Survived'], cmap='viridis')
plt.xlabel('Principal Component  1')
plt.ylabel('Principal Component  2')
plt.title('PCA of Titanic Dataset')
plt.show()

# Plotting histograms for numeric features
for col in numeric_cols:
    plt.figure(figsize=(10, 6))
    plt.hist(df_encoded[col], bins=30, color='skyblue', edgecolor='black')

    # Checks if it's "sex" column, it will customize the label for male and female
    if col == 'Sex':
        plt.title(f'Histogram of {col}')
        plt.xlabel('Sex')
        plt.xticks([0, 1], ['Male', 'Female'])
    else:
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)

    plt.ylabel('Frequency')
    plt.show()