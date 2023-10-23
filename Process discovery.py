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

# Define the column indices with mixed types
mixed_type_columns = [11, 13, 14, 15, 16, 17, 18, 19]

# Create a  with the column indices and their intended data type
dtype_mapping = {col_idx: 'object' for col_idx in mixed_type_columns}

# Read the CSV file with the specified data types
click_log_raw = pd.read_csv("C:\\Users\90545\Documents\Business information systems\Dataset\dataset_1\BPI2016_Clicks_Logged_In.csv~\BPI2016_Clicks_Logged_In.csv", sep=';', encoding='latin1', dtype=dtype_mapping)

click_log = click_log_raw [["CustomerID", "AgeCategory", "Gender", "Office_U", "Office_W", "SessionID", "IPID", "TIMESTAMP", "VHOST", "URL_FILE", "PAGE_NAME", "REF_URL_category", "page_load_error", "page_action_detail_EN", "service_detail_EN", "tip_EN", "xps_info"]]
click_log.head(10)


# In[3]:


# CONVERT THE TIMESTAMP AND SORT IT

click_log['TIMESTAMP'] = pd.to_datetime(click_log['TIMESTAMP'])
click_log.sort_values(by=['SessionID', 'TIMESTAMP'], ascending=[True, True], inplace=True)


# In[4]:


# CREATE ACTIVITY COLUMN

selected_columns = ['VHOST', 'PAGE_NAME', 'REF_URL_category', 'page_action_detail_EN', 'service_detail_EN']
click_log['Activity'] = click_log[selected_columns].apply(tuple, axis=1).copy()
click_log


# In[5]:


# MAKE DICTIONARY FOR ACTIVITY

activity_array = click_log['Activity'].unique()

activity_array_df = pd.DataFrame(activity_array, columns = ['Activity'])

unique_numbers = {element: index for index, element in enumerate(activity_array_df.apply(tuple, axis=1).unique())}
activity_array_df['UniqueNumber'] = activity_array_df.apply(tuple, axis=1).map(unique_numbers)

activity_array_df


# In[6]:


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


# In[7]:


# ADD THE COLUMN INDICATING THE CASE DURATION DAYS

customer_activity_range = click_log.groupby('CustomerID')['TIMESTAMP'].agg(['min', 'max']).reset_index()

clicks_stat['case_duration_days'] = ((customer_activity_range['max'] - customer_activity_range['min']).dt.days + 1)

clicks_stat


# In[8]:


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


# In[9]:


# FILTER CLICK_LOG BY THE CUSTOMER WHICH IS PRESENT IN THE CLICKS_STAT_FILTERED

merged_data = click_log.merge(clicks_stat_filtered[['CustomerID']], on='CustomerID')

merged_data


# In[10]:


# FILTER PAGES WITH LOAD ERROR

no_error_click_log = merged_data[merged_data["page_load_error"] == 0]
no_error_click_log


# In[11]:


# FIND THE SESSIONS WITH NULL DURATION

session_durations = no_error_click_log.groupby('SessionID')['TIMESTAMP'].apply(lambda x: x.max() - x.min())

session_durations = session_durations.reset_index()

session_durations.columns = ['SessionID', 'session_duration']

session_durations


# In[12]:


# FILTER SESSIONS WITH NULL DURATION FROM DATASET

zero_durations = session_durations[session_durations['session_duration'] == pd.Timedelta(days=0)]

session_ids_to_delete = zero_durations['SessionID'].tolist()

null_duration_filtered = no_error_click_log[~no_error_click_log['SessionID'].isin(session_ids_to_delete)]

null_duration_filtered


# In[13]:


# FILTER CONCURRENT REPEATING CLICKS IN THE SAME SESSION

click_log_filtered_mask = null_duration_filtered.copy()

mask_top1 = click_log_filtered_mask['Activity'].eq(click_log_filtered_mask['Activity'].shift()) & (click_log_filtered_mask['SessionID'] == click_log_filtered_mask['SessionID'].shift(1)) & (click_log_filtered_mask['Activity'] == click_log_filtered_mask['Activity'].shift(1))
mask_top1.iloc[0] = False

click_log_filtered = click_log_filtered_mask[~mask_top1]

click_log_filtered.reset_index(drop=True, inplace=True)

click_log_filtered


# In[14]:


# ADD COLUMN OF CYPHED ACTIVITY

click_log_filtered['activity_cypher'] = click_log_filtered['Activity'].map(activity_array_df.set_index('Activity')['UniqueNumber'])

click_log_filtered


# In[15]:


# CONVERT TO THE EVENT LOG WITH SESSION CASES

event_log_SessionID = click_log_filtered.copy()

event_log_SessionID.rename(columns={'TIMESTAMP': 'time:timestamp', 'SessionID': 'case:concept:name', 
                          'activity_cypher': 'concept:name'}, inplace=True)

event_log_SessionID = event_log_SessionID.sort_values(by=['case:concept:name', 'time:timestamp'])

event_log_SessionID = pm4py.format_dataframe(event_log_SessionID, case_id='case:concept:name', activity_key='concept:name', timestamp_key='time:timestamp')

event_log_SessionID


# In[16]:


# NUMBER OF EVENTS AND CASES

num_events_SessionID = len(event_log_SessionID)
num_cases_SessionID = len(event_log_SessionID['case:concept:name'].unique())
print("Number of events: {}\nNumber of cases: {}".format(num_events_SessionID, num_cases_SessionID))


# In[17]:


# GET THE START ACTIVITIES OF THE EVENT LOG WITH SESSION CASES

event_log_start_activities_SessionID = pm4py.stats.get_start_activities(event_log_SessionID)
event_log_start_activities_SessionID


# In[18]:


# GET THE END ACTIVITIES OF THE EVENT LOG WITH SESSION CASES

event_log_end_activities_SessionID = pm4py.stats.get_end_activities(event_log_SessionID)
event_log_end_activities_SessionID


# # FILTERING
# # START ACTIVITY (www.werk.nl, home, nan, nan, nan) AND END ACTIVITY 'www.werk.nl', 'home', 'Logged Out', nan, nan 

# In[19]:


# FILTER THE CASES WITH START ACTIVITY (www.werk.nl, home, nan, nan, nan) FOR THE EVENT LOG WITH SESSION CASES

event_log_start_activity_top1_SessionID = pm4py.filter_start_activities(event_log_SessionID, {'9'})
event_log_start_activity_top1_SessionID


# In[20]:


# FILTER THE CASES WITH START ACTIVITY 'digid.werk.nl', 'taken', nan, nan, nan AND END ACTIVITY 'www.werk.nl', 'home', 'Logged Out', nan, nan 

event_log_activity_top1_SessionID = pm4py.filter_end_activities(event_log_start_activity_top1_SessionID, {'7'})
event_log_activity_top1_SessionID


# In[21]:


# SORT ACTIVITIES IN THE ASCENDING WAY

sorted_event_log_SessionID_top1 = event_log_activity_top1_SessionID.groupby('case:concept:name').apply(lambda x: x.sort_values(by='time:timestamp', ascending=True))

sorted_event_log_SessionID_top1.reset_index(drop=True, inplace=True)

sorted_event_log_SessionID_top1


# In[22]:


# GET THE VARIANTS OF THE CASES WITH START ACTIVITY 'digid.werk.nl', 'taken', nan, nan, nan AND END ACTIVITY 'www.werk.nl', 'home', 'Logged Out', nan, nan WITH THE ACTIVITIES IN THE page_action_detail_EN AND service_detail_EN 

sorted_event_log_SessionID_filtered_variants_top1 = pm4py.get_variants(sorted_event_log_SessionID_top1)

variant_counts_top1 = Counter(sorted_event_log_SessionID_filtered_variants_top1)

variant_count_data_top1 = [{'Variant': variant, 'Count': count} for variant, count in variant_counts_top1.items()]

variant_count_df_top1 = pd.DataFrame(variant_count_data_top1)

variant_count_df_top1 = variant_count_df_top1.sort_values(by='Count', ascending=False)

variant_count_df_top1


# In[24]:


# MAKE THE DATASET OF SESSION AND ACTIVITIES SEQUENCES 

sorted_event_log_SessionID_top1 = sorted_event_log_SessionID_top1.sort_values(by=['case:concept:name', 'time:timestamp'])

session_activity_data_top1 = []

current_session_top1 = None
session_activities_top1 = []

current_customer_id_top1 = None

for index, row in sorted_event_log_SessionID_top1.iterrows():
    session_id_top1 = row['case:concept:name']
    concept_name_top1 = row['concept:name']
    customer_id_top1 = row['CustomerID']  
    
    if session_id_top1 != current_session_top1:
        if current_session_top1 is not None:
            session_activity_data_top1.append([current_session_top1, current_customer_id_top1, session_activities_top1])
        current_session_top1 = session_id_top1
        session_activities_top1 = [concept_name_top1]
        current_customer_id_top1 = customer_id_top1  
    else:
        session_activities_top1.append(concept_name_top1)

if current_session_top1 is not None:
    session_activity_data_top1.append([current_session_top1, current_customer_id_top1, session_activities_top1])

session_activities_table_top1 = pd.DataFrame(session_activity_data_top1, columns=['SessionID', 'CustomerID', 'ActivitySequence'])

session_activities_table_top1


# In[25]:


# MAKE COMMON STATISTICS ABOUT ACTIVITY SEQUENCE

session_activities_table_top1['ActivitySequenceTuple'] = session_activities_table_top1['ActivitySequence'].apply(tuple)

sequence_customer_count_top1 = session_activities_table_top1.groupby('ActivitySequenceTuple')['CustomerID'].nunique().reset_index()

sequence_customer_count_top1 = sequence_customer_count_top1.rename(columns={'CustomerID': 'CustomerCount'})

sequence_customer_count_sorted_top1 = sequence_customer_count_top1.sort_values('CustomerCount', ascending = False)

sequence_customer_count_sorted_top1.columns = ['ActivitySequence', 'CustomerCount']

variant_count_df_top1['ActivitySequence'] = variant_count_df_top1['Variant'].apply(tuple)

merged_data_top1 = sequence_customer_count_sorted_top1.merge(variant_count_df_top1, on='ActivitySequence', how='left')

merged_data_top1.rename(columns={'Count': 'VariantCount'}, inplace=True)

merged_data_top1 = merged_data_top1.drop(['Variant'], axis=1)
merged_data_top1


# In[26]:


# DOTTED PLOT OF ACTIVITY SEQUENCE VS CUSTOMER COUNT

merged_data_plot = merged_data_top1.copy()

merged_data_plot['ActivitySequence'] = merged_data_plot['ActivitySequence'].apply(lambda x: ' -> '.join(map(str, x)))

activity_sequence_top1 = merged_data_plot['ActivitySequence']
customer_count_top1 = merged_data_top1['CustomerCount']

plt.figure(figsize=(10, 6))
plt.plot(activity_sequence_top1, customer_count_top1, marker='o', linestyle='', markersize=5)

plt.xlabel('Activity Sequence')
plt.ylabel('Customer Count')

plt.title('Dotted Plot of Activity Sequence vs Customer Count')

plt.xticks([])

plt.tight_layout()
plt.show()


# In[27]:


# THE PERCENTAGE OF CUSTOMERS WHICH USED THE "POPULAR" ACTIVITY (>50 CUSTOMERS)

merged_data_top1_50 = merged_data_top1[merged_data_top1['CustomerCount'] >= 50]

res_perc = (merged_data_top1_50['CustomerCount'].sum() / merged_data_top1['CustomerCount'].sum()) * 100

res_perc


# In[29]:


# THE LIST OF SESSIONS WITH ACTIVITY SEQUENCE USED MORE THAN 50 CUSTOMERS

session_activities_table_copy_top1 = session_activities_table_top1.copy()
merged_data_copy_top1 = merged_data_top1.copy()

session_activities_table_copy_top1['ActivitySequence'] = session_activities_table_copy_top1['ActivitySequence'].apply(tuple)

merged_dataset_top1 = session_activities_table_copy_top1.merge(merged_data_copy_top1[['ActivitySequence', 'CustomerCount']], on='ActivitySequence', how='left')

merged_dataset_top1['ActivitySequence'] = merged_dataset_top1['ActivitySequence'].apply(list)

filtered_sessions_top1 = merged_dataset_top1[merged_dataset_top1['CustomerCount'] >= 50]

session_id_list_top1 = filtered_sessions_top1['SessionID'].tolist()

filtered_event_log_50_top1 = sorted_event_log_SessionID_top1[sorted_event_log_SessionID_top1['case:concept:name'].isin(session_id_list_top1)]
filtered_event_log_50_top1


# In[30]:


# CONVERT TO THE EVENT LOG WITH SESSION CASES FOR VISUALIZATION

filtered_event_log_50_copy_top1 = filtered_event_log_50_top1.copy()

filtered_event_log_50_copy_top1.rename(columns={'concept:name': 'activity_cypher'}, inplace=True)

filtered_event_log_50_copy_top1.rename(columns={'time:timestamp': 'time:timestamp', 'case:concept:name': 'case:concept:name', 
                          'Activity': 'concept:name'}, inplace=True)

filtered_event_log_50_copy_top1 = filtered_event_log_50_copy_top1.sort_values(by=['case:concept:name', 'time:timestamp'])

filtered_event_log_50_copy_top1 = pm4py.format_dataframe(filtered_event_log_50_copy_top1, case_id='case:concept:name', activity_key='concept:name', timestamp_key='time:timestamp')

filtered_event_log_50_copy_top1


# # PROCESS DISCOVERY
# # START ACTIVITY ('digid.werk.nl', 'taken', nan, nan, nan) AND END ACTIVITY 'www.werk.nl', 'home', 'Logged Out', nan, nan 

# In[31]:


# ALPHA MINER

net_alpha_top1, im_alpha_top1, fm_alpha_top1 = pm4py.discovery.discover_petri_net_alpha(filtered_event_log_50_copy_top1)
pm4py.view_petri_net(net_alpha_top1, im_alpha_top1, fm_alpha_top1, format = 'png')


# In[32]:


# CONFORMANCE ANALYSIS (FITNESS FOR ALPHA MINER)

fitness_alpha_top1 = pm4py.fitness_token_based_replay(filtered_event_log_50_copy_top1, net_alpha_top1, im_alpha_top1, fm_alpha_top1)
fitness_alpha_top1


# In[33]:


# CONFORMANCE ANALYSIS (PRECISION FOR ALPHA MINER)

prec_alpha_top1 = pm4py.precision_token_based_replay(filtered_event_log_50_copy_top1, net_alpha_top1, im_alpha_top1, fm_alpha_top1)
prec_alpha_top1


# In[34]:


# CONFORMANCE ANALYSIS (GENERALIZATION FOR ALPHA MINER)

gen_alpha_top1 = generalization_evaluator.apply(filtered_event_log_50_copy_top1, net_alpha_top1, im_alpha_top1, fm_alpha_top1)
gen_alpha_top1


# In[35]:


# CONFORMANCE ANALYSIS (SIMPLICITY FOR ALPHA MINER)

simp_alpha_top1 = simplicity_evaluator.apply(net_alpha_top1)
simp_alpha_top1


# In[36]:


# INDUCTIVE MINER

net_induc_top1, im_induc_top1, fm_induc_top1 = pm4py.discover_petri_net_inductive(filtered_event_log_50_copy_top1)
pm4py.view_petri_net(net_induc_top1, im_induc_top1, fm_induc_top1, format = 'png')


# In[43]:


# INDUCTIVE MINER (PROCESS TREE)

tree_inductive = pm4py.discover_process_tree_inductive(filtered_event_log_50_copy_top1)
pm4py.view_process_tree(tree_inductive)


# In[38]:


# CONFORMANCE ANALYSIS (FITNESS FOR INDUCTIVE MINER)

fitness_induc_top1 = pm4py.fitness_token_based_replay(filtered_event_log_50_copy_top1, net_induc_top1, im_induc_top1, fm_induc_top1)
fitness_induc_top1


# In[39]:


# CONFORMANCE ANALYSIS (PRECISION FOR INDUCTIVE MINER)

prec_induc_top1 = pm4py.precision_token_based_replay(filtered_event_log_50_copy_top1, net_induc_top1, im_induc_top1, fm_induc_top1)
prec_induc_top1


# In[40]:


# CONFORMANCE ANALYSIS (GENERALIZATION FOR INDUCTIVE MINER)

gen_induc_top1 = generalization_evaluator.apply(filtered_event_log_50_copy_top1, net_induc_top1, im_induc_top1, fm_induc_top1)
gen_induc_top1


# In[41]:


# CONFORMANCE ANALYSIS (SIMPLICITY FOR INDUCTIVE MINER)

simp_induc_top1 = simplicity_evaluator.apply(net_induc_top1)
simp_induc_top1


# In[63]:


# HEURISTIC MINER (PETRI NET)

net_heu_top1, im_heu_top1, fm_heu_top1 = pm4py.discover_petri_net_heuristics(filtered_event_log_50_copy_top1, dependency_threshold=0.9)
pm4py.view_petri_net(net_heu_top1, im_heu_top1, fm_heu_top1)


# In[64]:


# CONFORMANCE ANALYSIS (FITNESS FOR HEURISTIC MINER)

fitness_heu_top1 = pm4py.fitness_token_based_replay(filtered_event_log_50_copy_top1, net_heu_top1, im_heu_top1, fm_heu_top1)
fitness_heu_top1


# In[65]:


# CONFORMANCE ANALYSIS (PRECISION FOR HEURISTIC MINER)

prec_heu_top1 = pm4py.precision_token_based_replay(filtered_event_log_50_copy_top1, net_heu_top1, im_heu_top1, fm_heu_top1)
prec_heu_top1


# In[66]:


# CONFORMANCE ANALYSIS (GENERALIZATION FOR HEURISTIC MINER)

gen_heu_top1 = generalization_evaluator.apply(filtered_event_log_50_copy_top1, net_heu_top1, im_heu_top1, fm_heu_top1)
gen_heu_top1


# In[67]:


# CONFORMANCE ANALYSIS (SIMPLICITY FOR HEURISTIC MINER)

simp_heu_top1 = simplicity_evaluator.apply(net_heu_top1)
simp_heu_top1


# In[48]:


# DFG GRAPH

dfg_top1, start_dfg_top1, end_dfg_top1 = pm4py.discover_dfg(filtered_event_log_50_copy_top1)
pm4py.view_dfg(dfg_top1, start_dfg_top1, end_dfg_top1, format='png')

