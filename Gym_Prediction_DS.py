import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("images", exist_ok=True)



df = pd.read_csv("gym_user_dropout_dataset.csv")

df["reason_for_joining"] = None
df["reason_for_quitting"] = None

# Function for joining reason
def assign_join_reason(age):
    if age < 25:
        return random.choice(["Muscle gain", "Sports training", "Weight loss"])
    elif age <= 40:
        return random.choice(["General fitness", "Stress relief", "Weight loss"])
    else:
        return random.choice(["Medical advice", "General fitness", "Stress relief"])

# Function for quitting reason
def assign_quit_reason(row):
    if row["dropout"] == 0:
        return None
    elif row["sessions_per_week"] <= 1:
        return random.choice(["No time", "Lack of motivation"])
    elif row["sessions_per_week"] >= 4:
        return random.choice(["Achieved goal", "Relocated"])
    else:
        return random.choice(["High cost", "Unsatisfied with trainers", "Health issues"])

# Apply logic
df["reason_for_joining"] = df["age"].apply(assign_join_reason)
df["reason_for_quitting"] = df.apply(assign_quit_reason, axis=1)


df.head()
df.info()

df.isnull().sum()
df.describe()
df['dropout'].value_counts(normalize=True)
bins = [18, 25, 35, 45, 60]
labels = ['18-25','26-35','36-45','46-60']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)

#“Middle-aged members showed higher dropout, possibly due to time and injury.”
sns.barplot(x='age_group', y='dropout', data=df)
plt.title("Dropout Rate by Age Group")
plt.show()

#“Dropout users clearly have lower workout consistency.”
sns.boxplot(x='dropout', y='sessions_per_week', data=df)
plt.title("Workout Frequency vs Dropout")
plt.show()

#“Members who don’t see visible progress are more likely to quit.”
sns.histplot(data=df, x='progress_score', hue='dropout', bins=20, kde=True)
plt.title("Progress Score Distribution by Dropout")
plt.show()

#“Negative post-workout mood is a strong churn indicator.”
sns.countplot(x='mood_after', hue='dropout', data=df)
plt.title("Mood After Workout vs Dropout")
plt.show()

#“Injuries significantly increase dropout probability — safety programs matter.”
sns.barplot(x='injury', y='dropout', data=df)
plt.title("Injury Impact on Dropout")
plt.show()

#“Fitness goals are diverse, so personalized plans are critical.”
sns.countplot(y='reason_for_joining', data=df,
              order=df['reason_for_joining'].value_counts().index)
plt.title("Top Reasons for Joining Gym")
plt.show()

#“Time constraints and lack of motivation are top churn drivers.”
quit_df = df[df['dropout'] == 1]

sns.countplot(y='reason_for_quitting', data=quit_df,
              order=quit_df['reason_for_quitting'].value_counts().index)
plt.title("Top Reasons for Quitting")
plt.show()

#“Progress score and sessions per week show strong negative correlation with dropout.”
plt.figure(figsize=(8,6))
sns.heatmap(df[['age','sessions_per_week','avg_session_duration',
                'progress_score','dropout']].corr(),
            annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

df['relocated_flag'] = (df['reason_for_quitting'] == 'Relocated').astype(int)


corr_features = df[
    ['age',
     'sessions_per_week',
     'avg_session_duration',
     'progress_score',
     'dropout',
     'relocated_flag']
]
plt.figure(figsize=(9,6))

sns.heatmap(
    corr_features.corr(),
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5
)

plt.title("Feature Correlation Heatmap: Relocation-Driven Gym Dropout")
plt.show()

#for GitHub

plt.savefig("images/quitting_reasons.png", bbox_inches="tight")
plt.show()

#Final Conclustion “The dominant reason for quitting is relocation, which indicates that a large portion of gym members are geographically mobile, especially younger working professionals and students.

#This suggests that the gym’s current location strategy may not be aligned with its target audience’s lifestyle. Since younger users tend to relocate frequently for jobs or education, gyms should be strategically located near IT hubs, colleges, residential rental zones, and metro-connected areas.

#Instead of treating relocation as an unavoidable churn factor, gyms can reduce location-driven dropout by opening multiple nearby branches, offering location transfer memberships, or partnering with gyms in other areas.

#Therefore, the data highlights that location planning and accessibility play a critical role in member retention, especially for a younger demographic.”


