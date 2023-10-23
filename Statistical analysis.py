#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pm4py
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import Counter
import os
import seaborn as sns
import matplotlib.dates as mdates
from collections import defaultdict
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.visualization.petri_net import visualizer as pn_visualizer


# In[2]:


#UPLOAD DATA FROM FILE

mixed_type_columns = [11, 13, 14, 15, 16, 17, 18, 19]

dtype_mapping = {col_idx: 'object' for col_idx in mixed_type_columns}

click_log_raw = pd.read_csv("C:\\Users\90545\Documents\Business information systems\Dataset\dataset_1\BPI2016_Clicks_Logged_In.csv~\BPI2016_Clicks_Logged_In.csv", sep=';', encoding='latin1', dtype=dtype_mapping)

click_log = click_log_raw [["CustomerID", "AgeCategory", "Gender", "Office_U", "Office_W", "SessionID", "IPID", "TIMESTAMP", "VHOST", "URL_FILE", "PAGE_NAME", "REF_URL_category", "page_load_error", "page_action_detail_EN", "service_detail_EN", "tip_EN", "xps_info"]]
click_log.head(10)


# In[3]:


# CHECK THAT THE COLUMNS CustomerID, AgeCategory, Gender, TIMESTAMP, VHOST, URL_FILE DO NOT HAVE MISSING VALUES

columns_to_check = ["CustomerID", "AgeCategory", "Gender", "TIMESTAMP", "VHOST", "URL_FILE"]

null_values = click_log[columns_to_check].isnull().sum()

for column, count in null_values.items():
    print(f"Column '{column}' has {count} missing values.")


# In[4]:


# CONVERT THE TIMESTAMP AND SORT IT

click_log['TIMESTAMP'] = pd.to_datetime(click_log['TIMESTAMP'])
click_log.sort_values(by=['SessionID', 'TIMESTAMP'], ascending=[True, True], inplace=True)


# In[5]:


# CREATE ACTIVITY COLUMN

selected_columns = ['VHOST', 'PAGE_NAME', 'REF_URL_category', 'page_action_detail_EN', 'service_detail_EN']
click_log['Activity'] = click_log[selected_columns].apply(tuple, axis=1).copy()
click_log


# In[6]:


# FILTER PAGE LOAD ERRORS WITH > 80 % OF OCCURRENCE

error_click_log = click_log[(click_log['page_load_error'] == 1)]

error_stat = error_click_log['Activity'].value_counts().reset_index()
error_stat.columns = ['Activity', 'Count_of_errors']

pages_stat = click_log['Activity'].value_counts().reset_index()
pages_stat.columns = ['Activity', 'Count']
page_error_stat = pd.merge(error_stat, pages_stat, how="left", on="Activity")

page_error_stat['Percentage'] = ((page_error_stat['Count_of_errors'] / page_error_stat['Count']) * 100).round(2)

page_error_percentage = page_error_stat[page_error_stat['Percentage'] > 80]

page_error_percentage = page_error_percentage.rename(columns={"Activity": "Activity", "Percentage": "Error Percentage"})

error_percentage = page_error_percentage[["Activity", "Error Percentage"]]
error_percentage = error_percentage.sort_values(by ='Error Percentage', ascending = False)

error_percentage.head(10)


# In[7]:


# CHANNEL DISTRIBUTION OVER 8 MONTHS (BY CLICKS)

digid_werk_nl_click = click_log[click_log["VHOST"] == "digid.werk.nl"][["VHOST", "TIMESTAMP"]]

#digid_werk_nl_click['TIMESTAMP'] = pd.to_datetime(digid_werk_nl_click['TIMESTAMP'])

digid_werk_nl_click['YearMonth'] = digid_werk_nl_click['TIMESTAMP'].dt.strftime('%Y-%m')

digid_werk_nl_monthly_counts_click = digid_werk_nl_click.groupby('YearMonth').size().reset_index(name='Count')

www_werk_nl_click = click_log[click_log["VHOST"] == "www.werk.nl"][["VHOST", "TIMESTAMP"]]

#www_werk_nl_click["TIMESTAMP"] = pd.to_datetime(www_werk_nl_click["TIMESTAMP"])

www_werk_nl_click['YearMonth'] = www_werk_nl_click["TIMESTAMP"].dt.strftime('%Y-%m')

www_werk_nl_monthly_counts_click = www_werk_nl_click.groupby('YearMonth').size().reset_index(name='Count')

digid_werk_nl_monthly_counts_click = digid_werk_nl_monthly_counts_click.sort_values('YearMonth')
www_werk_nl_monthly_counts_click = www_werk_nl_monthly_counts_click.sort_values('YearMonth')

plt.figure(figsize=(15, 6))

plt.plot(digid_werk_nl_monthly_counts_click['YearMonth'], digid_werk_nl_monthly_counts_click['Count'], linestyle='-', color='b', label='digid.werk.nl')

plt.plot(www_werk_nl_monthly_counts_click['YearMonth'], www_werk_nl_monthly_counts_click['Count'], linestyle='-', color='r', label='www.werk.nl')

plt.legend()

plt.yticks(digid_werk_nl_monthly_counts_click['Count'], ha='right')

plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(base=100000))

plt.ticklabel_format(axis='y', style='plain', useOffset=False)

plt.title('CHANNEL DISTRIBUTION OVER 8 MONTHS (BY CLICKS)')

plt


# In[8]:


# CHANNEL DISTRIBUTION BY CLICKS

value_counts_clicks = click_log["VHOST"].value_counts()

mycolors = ["#32CD32", "#FFD700"]
plt.pie(value_counts_clicks, labels=value_counts_clicks.index, autopct='%1.1f%%', startangle=140, colors = mycolors)
plt.axis('equal')  
plt.title('CHANNEL DISTRIBUTION BY CLICKS')
plt.show()


# In[9]:


# CHANNEL DISTRIBUTION OF CLICKS BY AGE CATEGORY

grouped_AgeCategory_click = click_log.groupby(["VHOST", "AgeCategory"])["AgeCategory"].count()

reshaped_AgeCategory_click = grouped_AgeCategory_click.unstack(level=1)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].pie(reshaped_AgeCategory_click.loc["digid.werk.nl"], labels=reshaped_AgeCategory_click.columns, autopct='%1.1f%%', startangle=140)
axs[0].set_title("digid.werk.nl")

axs[1].pie(reshaped_AgeCategory_click.loc["www.werk.nl"], labels=reshaped_AgeCategory_click.columns, autopct='%1.1f%%', startangle=140)
axs[1].set_title("www.werk.nl")

axs[0].axis('equal')
axs[1].axis('equal')

fig.suptitle('CHANNEL DISTRIBUTION OF CLICKS BY AGE CATEGORY')

plt.show()


# In[10]:


# CHANNEL DISTRIBUTION OF CLICKS BY GENDER

grouped_Gender_click = click_log.groupby(["VHOST", "Gender"])["Gender"].count()

reshaped_Gender_click = grouped_Gender_click.unstack(level=1)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

mycolors_gender = ["#8FBC8F", "#FFA500"]

axs[0].pie(reshaped_Gender_click.loc["digid.werk.nl"], labels=reshaped_Gender_click.columns, autopct='%1.1f%%', startangle=140, colors = mycolors_gender)
axs[0].set_title("digid.werk.nl")

axs[1].pie(reshaped_Gender_click.loc["www.werk.nl"], labels=reshaped_Gender_click.columns, autopct='%1.1f%%', startangle=140, colors = mycolors_gender)
axs[1].set_title("www.werk.nl")

axs[0].axis('equal')
axs[1].axis('equal')

fig.suptitle('CHANNEL DISTRIBUTION OF CLICKS BY GENDER')

plt.show()


# In[11]:


#STATISTICS ABOUT FREQUENCY OF ACTIVITIES ON THE PAGES (WITH CLICKS)

activity_counts = click_log['Activity'].value_counts().reset_index()
activity_counts.columns = ['Activity', 'Count']
activity_counts


# In[12]:


#STATISTICS ABOUT FREQUENCY PERCENTAGE OF ACTIVITIES (WITH CLICKS)

activity_counts['Percentage'] = ((activity_counts['Count'] / activity_counts['Count'].sum()) * 100).round(2)
activity_counts.head(10)


# In[13]:


# GET THE STATISTICS

clicks_stat = click_log.groupby('CustomerID').agg({
    'SessionID': 'nunique',        # Number of sessions
    'VHOST': 'nunique',           # Number of unique VHOST
    'PAGE_NAME': 'nunique',     # Number of unique pages
    #'page_action_detail_EN': 'nunique',      # Number of unique actions
    #'service_detail_EN': 'nunique'        # Number of unique services
}).reset_index()

clicks_stat.columns = ['CustomerID', 'Number of sessions', 'Number of unique VHOST', 'Number of unique pages']

clicks_stat['Number of clicks'] = click_log.groupby('CustomerID').size().values

clicks_stat


# In[14]:


# ADD THE COLUMN INDICATING THE CASE DURATION DAYS

customer_activity_range = click_log.groupby('CustomerID')['TIMESTAMP'].agg(['min', 'max']).reset_index()

clicks_stat['case_duration_days'] = ((customer_activity_range['max'] - customer_activity_range['min']).dt.days + 1)

clicks_stat


# In[15]:


# DELETE THE EXTREME OUTLIERS

def find_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (column < lower_bound) | (column > upper_bound)

numeric_columns = ['Number of sessions', 'Number of unique pages', 'Number of clicks', 'case_duration_days']
outliers = clicks_stat[numeric_columns].apply(find_outliers)

clicks_stat_filtered = clicks_stat[~outliers.any(axis=1)]

clicks_stat_filtered


# In[16]:


# FILTER CLICK_LOG BY THE CUSTOMER WHICH IS PRESENT IN THE CLICKS_STAT_FILTERED

merged_data = click_log.merge(clicks_stat_filtered[['CustomerID']], on='CustomerID')

merged_data


# In[17]:


# FILTER PAGES WITH LOAD ERROR

no_error_click_log = merged_data[merged_data["page_load_error"] == 0]
no_error_click_log


# In[18]:


# FIND THE SESSIONS WITH NULL DURATION

session_durations = no_error_click_log.groupby('SessionID')['TIMESTAMP'].apply(lambda x: x.max() - x.min())

session_durations = session_durations.reset_index()

session_durations.columns = ['SessionID', 'session_duration']

session_durations


# In[19]:


# FILTER SESSIONS WITH NULL DURATION FROM DATASET

zero_durations = session_durations[session_durations['session_duration'] == pd.Timedelta(days=0)]

session_ids_to_delete = zero_durations['SessionID'].tolist()

null_duration_filtered = no_error_click_log[~no_error_click_log['SessionID'].isin(session_ids_to_delete)]

null_duration_filtered


# In[20]:


# FILTER CONCURRENT REPEATING CLICKS IN THE SAME SESSION

click_log_filtered_mask = null_duration_filtered.copy()

mask_top1 = click_log_filtered_mask['Activity'].eq(click_log_filtered_mask['Activity'].shift()) & (click_log_filtered_mask['SessionID'] == click_log_filtered_mask['SessionID'].shift(1)) & (click_log_filtered_mask['Activity'] == click_log_filtered_mask['Activity'].shift(1))
mask_top1.iloc[0] = False

click_log_filtered = click_log_filtered_mask[~mask_top1]

click_log_filtered.reset_index(drop=True, inplace=True)

click_log_filtered


# In[21]:


# THE RESULT OF THE FILTERING APPLICATION

percentage = 100 - (click_log_filtered.shape[0] / click_log.shape[0]) * 100

rounded_percentage = math.ceil(percentage)

print(f"Filtering steps has significantly reduced the dataset size to approximately {rounded_percentage}% of its original volume.")


# In[80]:


#STATISTICS ABOUT FREQUENCY OF ACTIVITIES ON THE PAGES (WITH VISITS)

activity_counts_visits = click_log_filtered['Activity'].value_counts().reset_index()
activity_counts_visits.columns = ['Activity', 'Count']
activity_counts_visits


# In[82]:


#STATISTICS ABOUT FREQUENCY PERCENTAGE OF ACTIVITIES (WITH CLICKS)

activity_counts_visits['Percentage'] = ((activity_counts_visits['Count'] / activity_counts_visits['Count'].sum()) * 100).round(2)
activity_counts_visits.head(10)


# In[81]:


# GET THE MEAN COUNT OF SESSIONS FOR EACH AGE CATEGORY AND GENDER

session_counts = click_log_filtered.groupby(['CustomerID', 'AgeCategory', 'Gender'])['SessionID'].nunique().reset_index()

session_counts.rename(columns={'SessionID': 'SessionCount'}, inplace=True)

session_counts.sort_values("SessionCount", ascending = False)

mean_session_count_by_age_gender = session_counts.groupby(['AgeCategory', 'Gender'])['SessionCount'].mean().reset_index()

mean_session_count_by_age_gender


# In[23]:


# THE MEAN COUNT OF SESSIONS FOR EACH AGE CATEGORY AND GENDER (PLOT)

groups_age_gender = mean_session_count_by_age_gender.groupby(['AgeCategory', 'Gender'])

plt.figure(figsize=(12, 6))
x = np.arange(len(groups_age_gender))
width = 0.35

plt.bar(x, groups_age_gender['SessionCount'].mean(), width, label='Mean Session Count', color='skyblue')

xtick_labels = [f'{age}-{gender}' for (age, gender) in groups_age_gender.groups.keys()]
plt.xticks(x, xtick_labels, rotation=45)

plt.xlabel('Age Category and Gender')
plt.ylabel('Mean Session Count')
plt.title('Mean Session Count by Age Category and Gender')

plt.legend()

plt.tight_layout()
plt.show()


# In[24]:


# GET THE MEAN COUNT OF SESSION FOR EACH AGE CATEGORY

mean_session_count_by_gender = session_counts.groupby('Gender')['SessionCount'].mean().reset_index()
mean_session_count_by_gender


# In[25]:


# THE MEAN COUNT OF SESSIONS FOR EACH GENDER(PLOT)

gender_categories_plt = mean_session_count_by_gender['Gender']
session_counts_gender_plt = mean_session_count_by_gender['SessionCount']

plt.figure(figsize=(10, 6))
plt.bar(gender_categories_plt, session_counts_gender_plt, color='skyblue')
plt.xlabel('Gender')
plt.ylabel('Mean Session Count')
plt.title('Mean Session Count by Gender')

plt.xticks(rotation=0)

plt.show()


# In[26]:


# GET THE MEAN TIME OF SESSION FOR EACH CUSTOMER

click_log_filtered['TIMESTAMP'] = pd.to_datetime(click_log_filtered['TIMESTAMP'])

click_log_filtered_sort = click_log_filtered.sort_values(by=['CustomerID', 'TIMESTAMP'])

click_log_filtered_sort.reset_index(drop=True, inplace=True)

click_log_filtered_sort['SessionDuration'] = click_log_filtered_sort.groupby('CustomerID')['TIMESTAMP'].diff().fillna(pd.Timedelta(seconds=0))

mean_session_duration = click_log_filtered_sort.groupby(['CustomerID', 'AgeCategory', 'Gender'])['SessionDuration'].mean().reset_index()
mean_session_duration


# In[27]:


# GET THE MEAN TIME OF SESSIONS FOR EACH AGE CATEGORY AND GENDER

mean_session_time_by_age_gender = mean_session_duration.groupby(['AgeCategory', 'Gender'])['SessionDuration'].mean().reset_index()

mean_session_time_by_age_gender.sort_values('SessionDuration')
mean_session_time_by_age_gender


# In[28]:


# THE MEAN DURATION OF SESSIONS FOR EACH AGE CATEGORY AND GENDER (PLOT)

mean_session_time_by_age_gender['SessionDuration'] = pd.to_timedelta(mean_session_time_by_age_gender['SessionDuration'])

mean_duration_by_group = mean_session_time_by_age_gender.groupby(['AgeCategory', 'Gender'])['SessionDuration'].mean().reset_index()

mean_duration_by_group['SessionDuration'] = mean_duration_by_group['SessionDuration'].dt.total_seconds()

mean_duration_by_group.sort_values(by='SessionDuration', inplace=True)

plt.figure(figsize=(12, 6))

x = np.arange(len(mean_duration_by_group))
mean_durations = mean_duration_by_group['SessionDuration']

plt.bar(x, mean_durations, width=0.35, label='Mean Session Duration', color='skyblue')

xtick_labels = [f'{age}-{gender}' for age, gender in zip(mean_duration_by_group['AgeCategory'], mean_duration_by_group['Gender'])]
plt.xticks(x, xtick_labels, rotation=45)

plt.xlabel('Age Category and Gender')
plt.ylabel('Mean Session Duration (seconds)')
plt.title('Mean Session Duration by Age Category and Gender')

plt.legend()

plt.tight_layout()
plt.show()


# In[29]:


# CHANNEL DISTRIBUTION OVER 8 MONTHS (BY VISITS)

digid_werk_nl_visits = click_log_filtered[click_log_filtered["VHOST"] == "digid.werk.nl"][["VHOST", "TIMESTAMP"]]

digid_werk_nl_visits['YearMonth'] = digid_werk_nl_visits['TIMESTAMP'].dt.strftime('%Y-%m')

digid_werk_nl_monthly_counts_visits = digid_werk_nl_visits.groupby('YearMonth').size().reset_index(name='Count')

www_werk_nl_visits = click_log_filtered[click_log_filtered["VHOST"] == "www.werk.nl"][["VHOST", "TIMESTAMP"]]

www_werk_nl_visits['YearMonth'] = www_werk_nl_visits["TIMESTAMP"].dt.strftime('%Y-%m')

www_werk_nl_monthly_counts_visits = www_werk_nl_visits.groupby('YearMonth').size().reset_index(name='Count')

digid_werk_nl_monthly_counts_visits = digid_werk_nl_monthly_counts_visits.sort_values('YearMonth')
www_werk_nl_monthly_counts_visits = www_werk_nl_monthly_counts_visits.sort_values('YearMonth')

plt.figure(figsize=(15, 6))

plt.plot(digid_werk_nl_monthly_counts_visits['YearMonth'], digid_werk_nl_monthly_counts_visits['Count'], linestyle='-', color='b', label='digid.werk.nl')

plt.plot(www_werk_nl_monthly_counts_visits['YearMonth'], www_werk_nl_monthly_counts_visits['Count'], linestyle='-', color='r', label='www.werk.nl')

plt.legend()

plt.yticks(digid_werk_nl_monthly_counts_visits['Count'], ha='right')

plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(base=10000))

plt.ticklabel_format(axis='y', style='plain', useOffset=False)

plt.title('CHANNEL DISTRIBUTION OVER 8 MONTHS (BY VISITS)')

plt


# In[30]:


# CHANNEL DISTRIBUTION BY VISITS

value_counts_visits = click_log_filtered["VHOST"].value_counts()

mycolors = ["#32CD32", "#FFD700"]
plt.pie(value_counts_visits, labels=value_counts_visits.index, autopct='%1.1f%%', startangle=140, colors = mycolors)
plt.axis('equal')  
plt.title('CHANNEL DISTRIBUTION BY VISITS')
plt.show()


# In[31]:


# CHANNEL DISTRIBUTION OF VISITS BY AGE CATEGORY

grouped_AgeCategory_visit = click_log_filtered.groupby(["VHOST", "AgeCategory"])["AgeCategory"].count()

reshaped_AgeCategory_visit = grouped_AgeCategory_visit.unstack(level=1)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].pie(reshaped_AgeCategory_visit.loc["digid.werk.nl"], labels=reshaped_AgeCategory_visit.columns, autopct='%1.1f%%', startangle=140)
axs[0].set_title("digid.werk.nl")

axs[1].pie(reshaped_AgeCategory_visit.loc["www.werk.nl"], labels=reshaped_AgeCategory_visit.columns, autopct='%1.1f%%', startangle=140)
axs[1].set_title("www.werk.nl")

axs[0].axis('equal')
axs[1].axis('equal')

fig.suptitle('CHANNEL DISTRIBUTION OF VISITS BY AGE CATEGORY')

plt.show()


# In[32]:


# CHANNEL DISTRIBUTION OF VISITS BY GENDER

grouped_Gender_visits = click_log_filtered.groupby(["VHOST", "Gender"])["Gender"].count()

reshaped_Gender_visits = grouped_Gender_visits.unstack(level=1)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

mycolors_gender = ["#8FBC8F", "#FFA500"]

axs[0].pie(reshaped_Gender_visits.loc["digid.werk.nl"], labels=reshaped_Gender_visits.columns, autopct='%1.1f%%', startangle=140, colors = mycolors_gender)
axs[0].set_title("digid.werk.nl")

axs[1].pie(reshaped_Gender_visits.loc["www.werk.nl"], labels=reshaped_Gender_visits.columns, autopct='%1.1f%%', startangle=140, colors = mycolors_gender)
axs[1].set_title("www.werk.nl")

axs[0].axis('equal')
axs[1].axis('equal')

fig.suptitle('CHANNEL DISTRIBUTION OF VISITS BY GENDER')

plt.show()


# In[33]:


# THE LIST OF ALL VALUES OF THE COLUMN page_action_detail_EN

page_action_detail_stat = click_log_filtered["page_action_detail_EN"].value_counts()
page_action_detail_stat


# In[34]:


# STATISTICS OF THE PERCENTAGE OF ACTION OCCURRENCE < 1%

page_action_detail_stat = page_action_detail_stat.reset_index()
page_action_detail_stat.columns = ['Action', 'Count']

page_action_detail_stat['Percentage'] = ((page_action_detail_stat['Count'] / page_action_detail_stat['Count'].sum()) * 100).round(2)

page_action_detail_stat = page_action_detail_stat.rename(columns={"Action": "Action", "Percentage": "Percentage of Occurrence"})

page_action_detail_stat_percentage = page_action_detail_stat[["Action", "Percentage of Occurrence"]]

page_action_detail_stat_percentage = page_action_detail_stat_percentage[page_action_detail_stat_percentage['Percentage of Occurrence'] < 1]
page_action_detail_stat_percentage = page_action_detail_stat_percentage.sort_values(by ='Percentage of Occurrence', ascending = False)
page_action_detail_stat_percentage = page_action_detail_stat_percentage.reset_index()

page_action_detail_stat_percentage


# In[35]:


# THE PERCENTAGE OF LEAST-USED ACTIONS

percentage_action_least_use = 100 - (page_action_detail_stat_percentage.shape[0] / page_action_detail_stat.shape[0]) * 100

rounded_percentage_action_least_use = math.ceil(percentage_action_least_use)

print(f"The percentage of non-used actions is {rounded_percentage_action_least_use}%.")


# In[36]:


# THE LIST OF ALL VALUES OF THE COLUMN page_action_detail_EN AND VHOST

action_vhost_stat = click_log_filtered.groupby(["VHOST", "page_action_detail_EN"])["CustomerID"].count()
action_vhost_stat


# In[37]:


# ACTION USAGE DISTRIBUTION BY AGE CATEGORY

non_empty_rows_actions = click_log_filtered[click_log_filtered["page_action_detail_EN"].notna()]
action_age_full = non_empty_rows_actions.groupby(["AgeCategory"])["CustomerID"].nunique()
action_age_full


# In[38]:


# ACTION USAGE DISTRIBUTION BY AGE CATEGORY (PLOT)

plt.pie(action_age_full, labels=action_age_full.index, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  
plt.title('ACTION USAGE DISTRIBUTION BY AGE CATEGORY')
plt.show()


# In[39]:


# ACTION USAGE DISTRIBUTION BY ACTION AND AGE CATEGORY

action_age_detailed = non_empty_rows_actions.groupby(["page_action_detail_EN", "AgeCategory"])["CustomerID"].nunique()
action_age_detailed


# In[40]:


# ACTION USAGE DISTRIBUTION BY ACTION AND AGE CATEGORY (PLOT_DETAILED)

sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))
sns.barplot(
    data=action_age_detailed.reset_index(), 
    x="page_action_detail_EN",
    y="CustomerID",
    hue="AgeCategory",
    palette="viridis",  
)

plt.title("Action Usage Distribution by Action and Age Category")
plt.xlabel("Action")
plt.ylabel("Number of Unique Sessions")
plt.xticks(rotation=90)  

plt.legend(title="Age Category")
plt.tight_layout()
plt.show()


# In[41]:


# ACTION USAGE DISTRIBUTION BY GENDER

action_gender_full = non_empty_rows_actions.groupby(["Gender"])["CustomerID"].nunique()
action_gender_full


# In[42]:


# ACTION USAGE DISTRIBUTION BY GENDER (PLOT)

plt.pie(action_gender_full, labels=action_gender_full.index, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  
plt.title('ACTION USAGE DISTRIBUTION BY GENDER')
plt.show()


# In[43]:


# ACTION USAGE DISTRIBUTION BY ACTION AND GENDER

actions_gender_detailed = non_empty_rows_actions.groupby(["page_action_detail_EN", "Gender"])["CustomerID"].nunique()
actions_gender_detailed


# In[44]:


# ACTION USAGE DISTRIBUTION BY ACTION AND GENDER (PLOT_DETAILED)

sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))
sns.barplot(
    data=actions_gender_detailed.reset_index(), 
    x="page_action_detail_EN",
    y="CustomerID",
    hue="Gender",
    palette="viridis",  
)

plt.title("Action Usage Distribution by Action and Gender")
plt.xlabel("Action")
plt.ylabel("Number of Unique Sessions")
plt.xticks(rotation=90)  

plt.legend(title="Gender")
plt.tight_layout()
plt.show()


# In[71]:


# THE LIST OF ALL VALUES OF THE COLUMN service_detail_EN

page_service_detail_stat = click_log_filtered["service_detail_EN"].value_counts()
page_service_detail_stat


# In[48]:


# THE LIST OF ALL VALUES OF THE COLUMN service_detail_EN AND VHOST

service_vhost_stat = click_log_filtered.groupby(["VHOST", "service_detail_EN"])["CustomerID"].count()
service_vhost_stat


# In[72]:


# STATISTICS OF THE PERCENTAGE OF SERVICES OCCURRENCE < 2%

page_service_detail_stat = page_service_detail_stat.reset_index()
page_service_detail_stat.columns = ['Service', 'Count']

page_service_detail_stat['Percentage'] = ((page_service_detail_stat['Count'] / page_service_detail_stat['Count'].sum()) * 100).round(2)

#page_service_detail_stat = page_service_detail_stat.rename(columns={"Service": "Service", "Percentage": "Percentage of Occurrence"})

percentage_service_least_use = page_service_detail_stat[page_service_detail_stat['Percentage'] < 2]

percentage_service_least_use


# In[67]:


# THE PERCENTAGE OF LEAST-USED SERVICES


percentage_service_least_use_stat = 100 - (percentage_service_least_use.shape[0] / page_service_detail_stat.shape[0]) * 100

rounded_percentage_service_least_use = math.ceil(percentage_service_least_use_stat)

print(f"The percentage of non-used services is {rounded_percentage_service_least_use}%.")


# In[46]:


# SERVICE USAGE DISTRIBUTION BY AGE CATEGORY

non_empty_rows_service = click_log_filtered[click_log_filtered["service_detail_EN"].notna()]

service_age_full = non_empty_rows_service.groupby(["AgeCategory"])["CustomerID"].nunique()
service_age_full


# In[47]:


# SERVICE USAGE DISTRIBUTION BY AGE CATEGORY (PLOT)

plt.pie(service_age_full, labels=service_age_full.index, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  
plt.title('SERVICE USAGE DISTRIBUTION BY AGE CATEGORY')
plt.show()


# In[49]:


# SERVICE USAGE DISTRIBUTION BY GENDER

service_gender_full = non_empty_rows_service.groupby(["Gender"])["CustomerID"].nunique()
service_gender_full


# In[50]:


# SERVICE USAGE DISTRIBUTION BY GENDER (PLOT)

plt.pie(service_gender_full, labels=service_gender_full.index, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  
plt.title('SERVICE USAGE DISTRIBUTION BY GENDER')
plt.show()


# In[51]:


# SERVICE USAGE DISTRIBUTION BY AGE CATEGORY

age_service_detailed = non_empty_rows_service.groupby(["service_detail_EN", "AgeCategory"])["CustomerID"].nunique()
age_service_detailed.sort_values()
age_service_detailed


# In[53]:


# SERVICE USAGE DISTRIBUTION BY AND AGE CATEGORIES (PLOT_DETAILED)

sns.set(style="whitegrid")

# Create a bar plot
plt.figure(figsize=(12, 6))
sns.barplot(
    data=age_service_detailed.reset_index(),  
    x="service_detail_EN",
    y="CustomerID",
    hue="AgeCategory",
    palette="viridis",  
)

plt.title("Service Usage Distribution by Age Categories")
plt.xlabel("Service")
plt.ylabel("Number of Unique Session")
plt.xticks(rotation=90)  

# Show the plot
plt.legend(title="Age Category")
plt.tight_layout()
plt.show()


# In[54]:


# SERVICE USAGE DISTRIBUTION BY GENDER

service_gender_detailed = non_empty_rows_service.groupby(["service_detail_EN", "Gender"])["CustomerID"].nunique()
service_gender_detailed


# In[55]:


# SERVICE USAGE DISTRIBUTION BY GENDER (PLOT_DETAILED)

sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))
sns.barplot(
    data=service_gender_detailed.reset_index(),  
    x="service_detail_EN",
    y="CustomerID",
    hue="Gender",
    palette="viridis",  
)

plt.title("Service Usage Distribution by Gender")
plt.xlabel("Service")
plt.ylabel("Number of Unique Sessions")
plt.xticks(rotation=90) 

plt.legend(title="Gender")
plt.tight_layout()
plt.show()


# In[73]:


# CONVERT TO THE EVENT LOG WITH SESSION CASES

event_log_SessionID = click_log_filtered.copy()

event_log_SessionID.rename(columns={'TIMESTAMP': 'time:timestamp', 'SessionID': 'case:concept:name', 
                          'Activity': 'concept:name'}, inplace=True)

event_log_SessionID = event_log_SessionID.sort_values(by=['case:concept:name', 'time:timestamp'])

event_log_SessionID = pm4py.format_dataframe(event_log_SessionID, case_id='case:concept:name', activity_key='concept:name', timestamp_key='time:timestamp')

event_log_SessionID


# In[74]:


# THE DISTRIBUTION OF 20-TOP START ACTIVITIES OF THE EVENT LOG WITH SESSIONID CASES

event_log_start_activities_SessionID = pm4py.stats.get_start_activities(event_log_SessionID)

top_20_start_activities_SessionID = dict(sorted(event_log_start_activities_SessionID.items(), key=lambda x: x[1], reverse=True)[:20])

activity_names = list(top_20_start_activities_SessionID.keys())
activity_counts = list(top_20_start_activities_SessionID.values())

plt.figure(figsize=(12, 6))
plt.barh(activity_names, activity_counts, color='orange')
plt.xlabel('Count')
plt.ylabel('Start Activities')
plt.title('Top 20 Start Activities for the SessionID cases')
plt.gca().invert_yaxis() 
plt.tight_layout()

plt.show()


# In[75]:


# THE DISTRIBUTION OF TOP-10 START ACTIVITIES OF THE EVENT LOG WITH SESSIONID CASES (PERCENTAGE)

top_10_start_activities_SessionID = dict(sorted(event_log_start_activities_SessionID.items(), key=lambda x: x[1], reverse=True)[:10])

activity_counts_10 = list(top_10_start_activities_SessionID.values())

plt.figure(figsize=(8, 8))
plt.pie(activity_counts_10, autopct='%1.1f%%', startangle=140)
plt.axis('equal') 

plt.title('Distribution of Top 10 Start Activities for the SessionID cases')

plt.show()


# In[76]:


# THE DISTRIBUTION OF 20-TOP END ACTIVITIES OF THE EVENT LOG WITH SESSIONID CASES

event_log_end_activities_SessionID = pm4py.stats.get_end_activities(event_log_SessionID)

top_20_end_activities_SessionID = dict(sorted(event_log_end_activities_SessionID.items(), key=lambda x: x[1], reverse=True)[:20])

activity_names_end_20 = list(top_20_end_activities_SessionID.keys())
activity_counts_end_20 = list(top_20_end_activities_SessionID.values())

plt.figure(figsize=(12, 6))
plt.barh(activity_names_end_20, activity_counts_end_20, color='orange')
plt.xlabel('Count')
plt.ylabel('End Activities')
plt.title('Top 20 End Activities for the SessionID cases')
plt.gca().invert_yaxis()  
plt.tight_layout()

plt.show()


# In[77]:


# THE DISTRIBUTION OF TOP-10 END ACTIVITIES OF THE EVENT LOG WITH SESSIONID CASES (PERCENTAGE)

top_10_end_activities_SessionID = dict(sorted(event_log_end_activities_SessionID.items(), key=lambda x: x[1], reverse=True)[:10])

activity_counts_end_10 = list(top_10_end_activities_SessionID.values())

plt.figure(figsize=(8, 8))
plt.pie(activity_counts_end_10, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  

plt.title('Distribution of Top 10 End Activities for the SessionID cases')

plt.show()


# In[78]:


# TOP 10 VARIANT DISTRIBUTION (PLOT) FOR EVENT LOG WITH SESSIONID CASES

event_log_variants_SessionID = pm4py.stats.get_variants(event_log_SessionID)

variant_counts_SessionID = Counter(event_log_variants_SessionID)

sorted_variants_SessionID = sorted(variant_counts_SessionID.items(), key=lambda x: x[1], reverse=True)

top_variants_SessionID = sorted_variants_SessionID[:10]

variants_SessionID = [', '.join(variant[0]) for variant in top_variants_SessionID]
counts_SessionID = [variant[1] for variant in top_variants_SessionID]

plt.figure(figsize=(18, 6))
plt.barh(variants_SessionID, counts_SessionID, color='orange')
plt.xlabel('Count')
plt.ylabel('Variants')
plt.title('Top 10 Variants in the Event Log (Session-Based Cases)')
plt.gca().invert_yaxis()  
plt.tight_layout()

plt.show()


# In[79]:


# DISTRIBUTION OF THE VARIANTS OF THE EVENT LOG WITH SESSIONID CASES

variants_df_SessionID = pd.DataFrame.from_records(sorted_variants_SessionID).rename(columns={0: 'Variants', 1: 'Count'})
variants_df_SessionID

variant_plot_adj_SessionID = variants_df_SessionID[0:100].index
frequency_plot_adj_SessionID = variants_df_SessionID[0:100]['Count']

frequency_log_SessionID = [math.log(i, 2) for i in frequency_plot_adj_SessionID] 

fig = plt.figure(figsize = (15, 5))
 
plt.bar(variant_plot_adj_SessionID, frequency_plot_adj_SessionID, color ='orange',
        width = 0.4)
 
plt.xlabel("Variants sorted by frequency")
plt.ylabel("Frequency")
plt.title("Bar chart of variants frequency for session-based cases")
plt.show()


# In[ ]:




