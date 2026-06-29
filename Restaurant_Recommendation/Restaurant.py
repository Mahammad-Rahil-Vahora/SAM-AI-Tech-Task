import pandas as pd

df = pd.read_csv("restaurants.csv")

location = input("Enter Location: ")
cuisine = input("Enter Cuisine: ")
rating = float(input("Enter Minimum Rating: "))

recommendation = df[
    (df["Location"].str.lower() == location.lower()) &
    (df["Cuisine"].str.lower() == cuisine.lower()) &
    (df["Rating"] >= rating)
]

# Display result
if recommendation.empty:
    print("\nNo restaurants found.")
else:
    print("\nRecommended Restaurants:\n")
    print(recommendation.to_string(index=False))