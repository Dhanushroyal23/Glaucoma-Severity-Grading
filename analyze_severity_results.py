import pandas as pd
import matplotlib.pyplot as plt

# === Load the results ===
csv_path = r"C:\Users\deswa\OneDrive\Desktop\capstone\glaucoma_severity_results.csv"
df = pd.read_csv(csv_path)

print("📊 Summary of Glaucoma Severity Results\n")
print(df.head())

# === Count how many in each severity category ===
severity_counts = df['Severity'].value_counts()
total_images = len(df)

print("\n🩺 Severity Distribution:\n", severity_counts)
print(f"\nTotal images analyzed: {total_images}")

# === CDR Statistics ===
print("\n📈 Cup-to-Disc Ratio Stats:")
print(df['CDR'].describe())  # <- Updated column name

# === Bar Chart: Severity Distribution with Percentages ===
plt.figure(figsize=(6,4))
bars = plt.bar(severity_counts.index, severity_counts.values,
                color=['green', 'orange', 'red'], edgecolor='black')

plt.title("Glaucoma Severity Distribution")
plt.xlabel("Severity Level")
plt.ylabel("Number of Images")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add percentage labels on bars
for bar in bars:
    height = bar.get_height()
    percentage = (height / total_images) * 100
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.5,
             f"{percentage:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# === Histogram of CDR values ===
plt.figure(figsize=(6,4))
plt.hist(df['CDR'], bins=20, edgecolor='black', color='skyblue')
plt.title("Distribution of Cup-to-Disc Ratios (CDR)")
plt.xlabel("CDR Value")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
