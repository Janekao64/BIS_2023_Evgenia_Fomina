#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as nd
import pm4py
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import Counter
import os
import seaborn as sns


# In[2]:


#UPLOAD DATA FROM FILE

complaint_log_raw = pd.read_csv("C:\\Users\90545\Documents\Business information systems\Dataset\dataset_2\BPI2016_Complaints.csv", sep=';', encoding='latin1')

complaint_log = complaint_log_raw [["CustomerID", "AgeCategory", "Gender", "ContactDate", "ComplaintID", "ComplaintThemeID", "ComplaintSubthemeID", "ComplaintTopicID", "ComplaintTheme_EN", "ComplaintSubtheme_EN", "ComplaintTopic_EN"]]

selected_columns = ['ComplaintThemeID', 'ComplaintSubthemeID', 'ComplaintTopicID']
complaint_log['Activity'] = complaint_log[selected_columns].apply(tuple, axis=1).copy()

complaint_log


# In[3]:


# CHECK THAT THE COLUMNS CustomerID, AgeCategory, Gender, ContactDate, ComplaintTheme_EN, ComplaintSubtheme_EN, ComplaintTopic_EN DO NOT HAVE MISSING VALUES

columns_to_check = ["CustomerID", "AgeCategory", "Gender", "ContactDate", "ComplaintTheme_EN", "ComplaintSubtheme_EN", "ComplaintTopic_EN"]

null_values = complaint_log[columns_to_check].isnull().sum()

for column, count in null_values.items():
    print(f"Column '{column}' has {count} missing values.")


# In[4]:


# THE COMPLAINTS DISTRIBUTION OVER THE TIME

complaint_log['ContactDate'] = pd.to_datetime(complaint_log['ContactDate'])

complaints_by_month = complaint_log.groupby(complaint_log['ContactDate'].dt.to_period("M"))['ComplaintID'].count()

plt.figure(figsize=(12, 6))
complaints_by_month.plot(kind='line', marker='o', color='blue')
plt.xlabel('Month')
plt.ylabel('Number of Complaints')
plt.title('Distribution of Complaints Over Time (Monthly)')
plt.grid(True)  
plt.tight_layout()

plt.show()


# In[5]:


# THE COMPLAINTS DISTRIBUTION FOR THE CUSTOMERS OF THE AGES 18-29 OVER THE TIME

complaint_log_18_29 = complaint_log[complaint_log["AgeCategory"] == "18-29"]

complaint_log_18_29['ContactDate'] = pd.to_datetime(complaint_log_18_29['ContactDate'])

complaints_by_month_18_29 = complaint_log_18_29.groupby(complaint_log_18_29['ContactDate'].dt.to_period("M"))['ComplaintID'].count()

plt.figure(figsize=(12, 6))
complaints_by_month_18_29.plot(kind='line', marker='o', color='blue')
plt.xlabel('Month')
plt.ylabel('Number of Complaints for the Customers of the ages 18-29')
plt.title('Distribution of Complaints for the Customers of the ages 18-29 Over Time (Monthly)')
plt.grid(True)  
plt.tight_layout()

plt.show()


# In[6]:


# THE COMPLAINTS DISTRIBUTION FOR THE CUSTOMERS OF THE AGES 30-39 OVER THE TIME

complaint_log_30_39 = complaint_log[complaint_log["AgeCategory"] == "30-39"]

complaint_log_30_39['ContactDate'] = pd.to_datetime(complaint_log_30_39['ContactDate'])

complaints_by_month_30_39 = complaint_log_30_39.groupby(complaint_log_30_39['ContactDate'].dt.to_period("M"))['ComplaintID'].count()

plt.figure(figsize=(12, 6))
complaints_by_month_30_39.plot(kind='line', marker='o', color='blue')
plt.xlabel('Month')
plt.ylabel('Number of Complaints for the Customers of the ages 30-39')
plt.title('Distribution of Complaints for the Customers of the ages 30-39 Over Time (Monthly)')
plt.grid(True)  
plt.tight_layout()

# Show the plot
plt.show()


# In[7]:


# THE COMPLAINTS DISTRIBUTION FOR THE CUSTOMERS OF THE AGES 40-49 OVER THE TIME

complaint_log_40_49 = complaint_log[complaint_log["AgeCategory"] == "40-49"]

complaint_log_40_49['ContactDate'] = pd.to_datetime(complaint_log_40_49['ContactDate'])

complaints_by_month_40_49 = complaint_log_40_49.groupby(complaint_log_40_49['ContactDate'].dt.to_period("M"))['ComplaintID'].count()

plt.figure(figsize=(12, 6))
complaints_by_month_40_49.plot(kind='line', marker='o', color='blue')
plt.xlabel('Month')
plt.ylabel('Number of Complaints for the Customers of the ages 40-49')
plt.title('Distribution of Complaints for the Customers of the ages 40-49 Over Time (Monthly)')
plt.grid(True)  
plt.tight_layout()

plt.show()


# In[8]:


# THE COMPLAINTS DISTRIBUTION FOR THE CUSTOMERS OF THE AGES 50-65 OVER THE TIME

complaint_log_50_65 = complaint_log[complaint_log["AgeCategory"] == "50-65"]

complaint_log_50_65['ContactDate'] = pd.to_datetime(complaint_log_50_65['ContactDate'])

complaints_by_month_50_65 = complaint_log_50_65.groupby(complaint_log_50_65['ContactDate'].dt.to_period("M"))['ComplaintID'].count()

plt.figure(figsize=(12, 6))
complaints_by_month_50_65.plot(kind='line', marker='o', color='blue')
plt.xlabel('Month')
plt.ylabel('Number of Complaints for the Customers of the ages 50-65')
plt.title('Distribution of Complaints for the Customers of the ages 50-65 Over Time (Monthly)')
plt.grid(True)  
plt.tight_layout()

plt.show()


# In[9]:


# STATISTICS OF THE COLUMN AgeCategory

grouped_AgeCategory = complaint_log.value_counts("AgeCategory")


# In[10]:


# COMPLAINT DISTRIBUTION BY AgeCategory

plt.pie(grouped_AgeCategory, labels=grouped_AgeCategory.index, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  
plt.title('COMPLAINT DISTRIBUTION BY AGE CATEGORY')
plt.show()


# In[11]:


# STATISTICS OF THE COLUMN Gender

grouped_Gender = complaint_log.value_counts("Gender")


# In[12]:


# COMPLAINT DISTRIBUTION BY GENDER

# Create a pie plot   
plt.pie(grouped_Gender, labels=grouped_Gender.index, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  
plt.title('COMPLAINT DISTRIBUTION BY GENDER')
plt.show()


# In[13]:


# STATISTICS OF THE COLUMN ComplaintTheme_EN

grouped_Complaint_Theme = complaint_log.value_counts("ComplaintTheme_EN").reset_index(name="Count")
grouped_Complaint_Theme


# In[14]:


# COMPLAINT DISTRIBUTION BY COMPLAINT THEME (PLOT PIE)

counts = grouped_Complaint_Theme['Count']
themes = grouped_Complaint_Theme['ComplaintTheme_EN']

plt.figure(figsize=(8, 8))
plt.pie(counts, labels=themes, autopct='%1.1f%%', startangle=140)

plt.axis('equal')

plt.title('Complaint Distribution by Theme')

plt.show()


# In[15]:


# COMPLAINT DISTRIBUTION BY COMPLAINT THEME (PLOT BAR)

grouped_Complaint_Theme_sorted = grouped_Complaint_Theme.sort_values(by='Count', ascending=False)

activity_names_Complaint_Theme = grouped_Complaint_Theme_sorted['ComplaintTheme_EN']
activity_counts_Complaint_Theme = grouped_Complaint_Theme_sorted['Count']

plt.figure(figsize=(12, 6))
plt.barh(activity_names_Complaint_Theme, activity_counts_Complaint_Theme, color='orange')
plt.xlabel('Count')
plt.ylabel('Complaint Theme')
plt.title('Complaint Distribution By Complaint Theme')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# In[16]:


# COMPLAINT DISTRIBUTION BY COMPLAINT THEME AND AGE CATEGORY

grouped_complaint_theme_age = complaint_log.groupby(["ComplaintTheme_EN", "AgeCategory"])["AgeCategory"].count()

grouped_complaint_theme_age = grouped_complaint_theme_age.rename("Count")

grouped_complaint_theme_age = grouped_complaint_theme_age.reset_index()

grouped_complaint_theme_age


# In[17]:


# COMPLAINT DISTRIBUTION BY COMPLAINT THEME AND AGE CATEGORY (PLOT_DETAILED)

sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))
sns.barplot(
    data=grouped_complaint_theme_age,  
    x="ComplaintTheme_EN",
    y="Count",  
    hue="AgeCategory",         
    palette="viridis",        
)

plt.title("Complaint Distribution by Complaint Theme and Age Category")
plt.xlabel("Complaint Theme")
plt.ylabel("Number of Complaints")
plt.xticks(rotation=90) 

plt.legend(title="Age Category")
plt.tight_layout()
plt.show()


# In[18]:


# COMPLAINT DISTRIBUTION BY COMPLAINT THEME AND GENDER

grouped_complaint_theme_gender = complaint_log.groupby(["ComplaintTheme_EN", "Gender"])["Gender"].count()

grouped_complaint_theme_gender = grouped_complaint_theme_gender.rename("Count")

grouped_complaint_theme_gender = grouped_complaint_theme_gender.reset_index()

grouped_complaint_theme_gender


# In[19]:


# COMPLAINT DISTRIBUTION BY COMPLAINT THEME AND GENDER (PLOT_DETAILED)

sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))
sns.barplot(
    data=grouped_complaint_theme_gender,  
    x="ComplaintTheme_EN",
    y="Count",  
    hue="Gender",        
    palette="viridis",        
)

plt.title("Complaint Distribution by Complaint Theme and Gender")
plt.xlabel("Complaint Theme")
plt.ylabel("Number of Complaints")
plt.xticks(rotation=90)  

plt.legend(title="Gender")
plt.tight_layout()
plt.show()


# In[20]:


# STATISTICS OF THE COLUMN ComplaintSubtheme_EN

grouped_Complaint_SubTheme = complaint_log.value_counts("ComplaintSubtheme_EN").reset_index(name="Count")
grouped_Complaint_SubTheme


# In[21]:


# COMPLAINT DISTRIBUTION BY COMPLAINT SUBTHEME (PLOT PIE)

grouped_Complaint_SubTheme = complaint_log['ComplaintSubtheme_EN'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(grouped_Complaint_SubTheme, labels=grouped_Complaint_SubTheme.index, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title('COMPLAINT SUBTHEME DISTRIBUTION', x=0, y=1)
plt.show()


# In[22]:


# COMPLAINT DISTRIBUTION BY COMPLAINT SUBTHEME (PLOT BAR)

grouped_Complaint_SubTheme_dict = dict(sorted(grouped_Complaint_SubTheme.items(), key=lambda x: x[1], reverse=True))

activity_names_Complaint_SubTheme = list(grouped_Complaint_SubTheme_dict.keys())
activity_counts_Complaint_SubTheme = list(grouped_Complaint_SubTheme_dict.values())

plt.figure(figsize=(12, 6))
plt.barh(activity_names_Complaint_SubTheme, activity_counts_Complaint_SubTheme, color='orange')
plt.xlabel('Count')
plt.ylabel('Complaint SubTheme')
plt.title('Complaint Distribution By Complaint SubTheme')
plt.gca().invert_yaxis()  
plt.tight_layout()

plt.show()


# In[23]:


# COMPLAINT DISTRIBUTION BY COMPLAINT SUBTHEME AND AGE CATEGORY

grouped_complaint_subtheme_age = complaint_log.groupby(["ComplaintSubtheme_EN", "AgeCategory"])["AgeCategory"].count()

grouped_complaint_subtheme_age = grouped_complaint_subtheme_age.rename("Count")

grouped_complaint_subtheme_age = grouped_complaint_subtheme_age.reset_index()

grouped_complaint_subtheme_age


# In[24]:


# COMPLAINT DISTRIBUTION BY COMPLAINT SUBTHEME AND AGE CATEGORY (PLOT_DETAILED)

sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))
sns.barplot(
    data=grouped_complaint_subtheme_age,  
    x="ComplaintSubtheme_EN",
    y="Count",  
    hue="AgeCategory",         
    palette="viridis",       
)

plt.title("Complaint Distribution by Complaint SubTheme and Age Category")
plt.xlabel("Complaint SubTheme")
plt.ylabel("Number of Complaints")
plt.xticks(rotation=90)  

plt.legend(title="Age Category")
plt.tight_layout()
plt.show()


# In[25]:


# COMPLAINT DISTRIBUTION BY COMPLAINT SUBTHEME AND GENDER

grouped_complaint_subtheme_gender = complaint_log.groupby(["ComplaintSubtheme_EN", "Gender"])["Gender"].count()

grouped_complaint_subtheme_gender = grouped_complaint_subtheme_gender.rename("Count")

grouped_complaint_subtheme_gender = grouped_complaint_subtheme_gender.reset_index()

grouped_complaint_subtheme_gender


# In[26]:


# COMPLAINT DISTRIBUTION BY COMPLAINT SUBTHEME AND GENDER (PLOT_DETAILED)

sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))
sns.barplot(
    data=grouped_complaint_subtheme_gender,  
    x="ComplaintSubtheme_EN",
    y="Count",  
    hue="Gender",         
    palette="viridis",        
)

plt.title("Complaint Distribution by Complaint SubTheme and Gender")
plt.xlabel("Complaint SubTheme")
plt.ylabel("Number of Complaints")
plt.xticks(rotation=90)  

plt.legend(title="Gender")
plt.tight_layout()
plt.show()


# In[27]:


# STATISTICS OF THE COLUMN ComplaintTopic_EN

grouped_Complaint_Topic = complaint_log.value_counts("ComplaintTopic_EN").reset_index(name="Count")
grouped_Complaint_Topic


# In[28]:


# TOP 10 COMPLAINT DISTRIBUTION BY COMPLAINT SUBTHEME (PLOT BAR)

grouped_Complaint_Topic_df = grouped_Complaint_Topic.reset_index()  
sorted_data = grouped_Complaint_Topic_df.sort_values(by='Count', ascending=False).head(10) 

plt.figure(figsize=(12, 6))
plt.barh(sorted_data['ComplaintTopic_EN'], sorted_data['Count'], color='orange')
plt.xlabel('Count')
plt.ylabel('Complaint Topic')
plt.title('Top 10 Complaint Distribution By Complaint Topic')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# In[29]:


# COMPLAINT DISTRIBUTION BY COMPLAINT TOPIC AND AGE CATEGORY

grouped_complaint_topic_age = complaint_log.groupby(["ComplaintTopic_EN", "AgeCategory"])["AgeCategory"].count()

grouped_complaint_topic_age = grouped_complaint_topic_age.rename("Count")

grouped_complaint_topic_age = grouped_complaint_topic_age.reset_index()

grouped_complaint_topic_age


# In[30]:


# COMPLAINT DISTRIBUTION BY COMPLAINT TOPIC AND AGE CATEGORY (PLOT_DETAILED)

sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))
sns.barplot(
    data=grouped_complaint_topic_age,  
    x="ComplaintTopic_EN",
    y="Count",  
    hue="AgeCategory",         
    palette="viridis",       
)

plt.title("Complaint Distribution by Complaint Topic and Age Category")
plt.xlabel("Complaint SubTheme")
plt.ylabel("Number of Complaints")
plt.xticks(rotation=90)  

plt.legend(title="Age Category")
plt.tight_layout()
plt.show()


# In[31]:


# COMPLAINT DISTRIBUTION BY COMPLAINT TOPIC AND GENDER

grouped_complaint_topic_gender = complaint_log.groupby(["ComplaintTopic_EN", "Gender"])["Gender"].count()

grouped_complaint_topic_gender = grouped_complaint_topic_gender.rename("Count")

grouped_complaint_topic_gender = grouped_complaint_topic_gender.reset_index()

grouped_complaint_topic_gender


# In[32]:


# COMPLAINT DISTRIBUTION BY COMPLAINT TOPIC AND GENDER (PLOT_DETAILED)

sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))
sns.barplot(
    data=grouped_complaint_topic_gender,  
    x="ComplaintTopic_EN",
    y="Count",  
    hue="Gender",         
    palette="viridis",       
)

plt.title("Complaint Distribution by Complaint Topic and Gender")
plt.xlabel("Complaint Topic")
plt.ylabel("Number of Complaints")
plt.xticks(rotation=90)  

plt.legend(title="Gender")
plt.tight_layout()
plt.show()


# In[33]:


# STATICTICS ABOUT ACTIVITY

activity_counts = complaint_log["Activity"].value_counts()

sorted_activity_counts = activity_counts.sort_values(ascending=True)

count_of_counts = sorted_activity_counts.value_counts()

count_of_counts_df = count_of_counts.reset_index()

count_of_counts_df.columns = ['Activity Count', 'Frequency']

print(count_of_counts_df)


# In[34]:


# STATISTICS OF THE COLUMN ComplaintTopic_EN

grouped_activity = complaint_log.value_counts("Activity").reset_index(name="Count")

grouped_activity['Percentage'] = ((grouped_activity['Count'] / grouped_activity['Count'].sum()) * 100).round(2)

grouped_activity = grouped_activity.rename(columns={"Action": "Action", "Percentage": "Percentage of Occurrence"})
grouped_activity.sort_values("Count", ascending = False)
grouped_activity.head(10)


# In[35]:


# TOP 5 USED COMPLAINTS THEME + SUBTHEME + TOPIC

grouped_activity_percentage = grouped_activity[grouped_activity['Percentage of Occurrence'] > 5]
grouped_activity_percentage = grouped_activity_percentage.sort_values(by ='Percentage of Occurrence', ascending = False)
grouped_activity_percentage = grouped_activity_percentage.reset_index()

grouped_activity_percentage


# In[36]:


# MAKE A LINE PLOT OF THE FREQUENCIES OF THE ACTIVITIES

grouped_activity_plot = complaint_log.value_counts("Activity").reset_index(name="Count")

plt.figure(figsize=(16, 16))
act_plot = grouped_activity_plot.plot(kind='line', marker='o', color='blue')
plt.xlabel('Activity')
plt.ylabel('Count')
plt.title('Complaint Distribution By Complaint path')

x_labels = grouped_activity_plot.index  
plt.xticks(range(len(x_labels)), x_labels, rotation=90)  

plt.show()


# In[37]:


# COMPLAINT DISTRIBUTION BY ACTIVITY AND AGE CATEGORY

grouped_activity_filtered = grouped_activity[grouped_activity["Count"] > 5]

grouped_complaint_activity_age = complaint_log.groupby(["Activity", "AgeCategory"])["AgeCategory"].count()

grouped_complaint_activity_age = grouped_complaint_activity_age.rename("Count")

grouped_complaint_activity_age = grouped_complaint_activity_age.reset_index()

filtered_activities = grouped_activity_filtered["Activity"].tolist()
grouped_complaint_activity_age = grouped_complaint_activity_age[grouped_complaint_activity_age["Activity"].isin(filtered_activities)]

grouped_complaint_activity_age


# In[38]:


# COMPLAINT DISTRIBUTION BY ACTIVITY AND AGE CATEGORY

grouped_activity_filtered = grouped_activity[grouped_activity["Percentage of Occurrence"] > 5]

grouped_complaint_activity_age = complaint_log.groupby(["Activity", "AgeCategory"])["AgeCategory"].count()

grouped_complaint_activity_age = grouped_complaint_activity_age.rename("Count")

grouped_complaint_activity_age = grouped_complaint_activity_age.reset_index()

filtered_activities = grouped_activity_filtered["Activity"].tolist()
grouped_complaint_activity_age = grouped_complaint_activity_age[grouped_complaint_activity_age["Activity"].isin(filtered_activities)]

grouped_complaint_activity_age


# In[39]:


# COMPLAINT DISTRIBUTION BY ACTIVITY AND AGE CATEGORY (PLOT_DETAILED)

def flatten_activity(activity):
    return ", ".join(map(str, activity))

grouped_complaint_activity_age['Activity'] = grouped_complaint_activity_age['Activity'].apply(flatten_activity)

sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
sns.barplot(
    data=grouped_complaint_activity_age,
    x="Activity",
    y="Count",
    hue="AgeCategory",
    palette="viridis"
)
plt.title("Complaint Distribution by Activity and Age Category")
plt.xlabel("Activity")
plt.xticks(rotation=90)
plt.show()


# In[40]:


# COMPLAINT DISTRIBUTION BY ACTIVITY AND GENDER

grouped_activity_filtered = grouped_activity[grouped_activity["Percentage of Occurrence"] > 5]

grouped_complaint_activity_gender = complaint_log.groupby(["Activity", "Gender"])["Gender"].count()

grouped_complaint_activity_gender = grouped_complaint_activity_gender.rename("Count")

grouped_complaint_activity_gender = grouped_complaint_activity_gender.reset_index()

filtered_activities = grouped_activity_filtered["Activity"].tolist()
grouped_complaint_activity_gender = grouped_complaint_activity_gender[grouped_complaint_activity_gender["Activity"].isin(filtered_activities)]

grouped_complaint_activity_gender


# In[41]:


# COMPLAINT DISTRIBUTION BY ACTIVITY AND GENDER (PLOT_DETAILED)

grouped_complaint_activity_gender['Activity'] = grouped_complaint_activity_gender['Activity'].apply(flatten_activity)

sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))
sns.barplot(
    data=grouped_complaint_activity_gender,  
    x="Activity",
    y="Count",  
    hue="Gender",         
    palette="viridis",        
)

plt.title("Complaint Distribution by Gender")
plt.xlabel("Complaint")
plt.ylabel("Number of Complaints")
plt.xticks(rotation=90)  

plt.legend(title="Gender")
plt.tight_layout()
plt.show()


# In[42]:


# TOP 5 USED COMPLAINTS THEME + SUBTHEME + TOPIC

def map_activity(activity):
    matching_row = complaint_log[
        (complaint_log['ComplaintThemeID'] == activity[0]) &
        (complaint_log['ComplaintSubthemeID'] == activity[1]) &
        (complaint_log['ComplaintTopicID'] == activity[2])
    ]
    
    if not matching_row.empty:
        return (
            matching_row['ComplaintTheme_EN'].values[0],
            matching_row['ComplaintSubtheme_EN'].values[0],
            matching_row['ComplaintTopic_EN'].values[0]
        )
    else:
        return None

grouped_activity_percentage['Detailed Activity'] = grouped_activity_percentage['Activity'].apply(map_activity)
pd.set_option('display.max_colwidth', None)

grouped_activity_percentage


# In[43]:


for index, row in grouped_activity_percentage.iterrows():
    activity_value = row['Activity']
    detailed_activity_value = row['Detailed Activity']
    
    print(f"{activity_value} -> {detailed_activity_value}")


# In[44]:


# TO GET THE LIST OF CUSTOMERS WITH THE ACTIVITY = (3, 4, 32)

desired_activity = (3, 4, 32)
filtered_customers_3_4_32 = complaint_log[complaint_log['Activity'] == desired_activity]['CustomerID'].tolist()

print(filtered_customers_3_4_32)

