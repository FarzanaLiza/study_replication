#!/usr/bin/env python
# coding: utf-8

# ### Read data: Healthcare providers' performance from Lu et al. (2020)
# https://www.frontiersin.org/articles/10.3389/fped.2020.00544/full

# In[1]:


import pandas as pd


filename = "/Users/.../Data_HC_paper.csv"

#Read csv file
pdf = pd.read_csv(filename)
print ("Shape of dataset: ", pdf.shape)


# ### Data cleaning

# - Removing participant_id column from the datafile

# In[2]:


df = pd.DataFrame(pdf)


# In[4]:


# Selecting all the features except the participant_id

df_new = df[df.columns[1:]]


# ### Handling missing values
# - **For continuous features: replacing missing values with mean values**
# - **For categorical features: replacing missing values with mode values** 
# 

# In[5]:


df_new = df_new.fillna( {
    'Months_LastNRPcourse':df_new[['Months_LastNRPcourse']].mean().iloc[0],
    'Years_NeonatalCare':df_new[['Years_NeonatalCare']].mean().iloc[0],
    'HoursVideoMo':df_new[['HoursVideoMo']].mean().iloc[0],
    'YearsVideoExp':df_new[['YearsVideoExp']].mean().iloc[0],
                       
    'Mo2Test0F1P':df_new['Mo2Test0F1P'].mode().iloc[0],
    'Mo5Test0F1P':df_new['Mo5Test0F1P'].mode().iloc[0],
    'StressfulScen1_5':df_new['StressfulScen1_5'].mode().iloc[0]
} )

df_new.dtypes


# In order to use the Gower package, we need to create the categorical features as string features
# - First, separate continuous features from the file
# - Second, convert the rest of the features as strings or objects
# - third, add back the features that were removed in the first step 

# In[6]:


df_str = df_new[df_new.columns.difference(['Months_LastNRPcourse', 'Years_NeonatalCare', 'HoursVideoMo', 'YearsVideoExp'])]


# In[7]:


df_str2= df_str.astype(str)


# In[8]:


# Combine back with the four features
#df_new.insert(0, 'Participant_ID', df['Participant_ID'])
df_str2.insert(0, 'Months_LastNRPcourse', df_new['Months_LastNRPcourse'])
df_str2.insert(1, 'Years_NeonatalCare', df_new['Years_NeonatalCare'])
df_str2.insert(2, 'HoursVideoMo', df_new['HoursVideoMo'])
df_str2.insert(3, 'YearsVideoExp', df_new['YearsVideoExp'])
#df_str2


# In[9]:


# Making sure these features are saved as string
df_str2.dtypes
df_str2['DecisionsQuickly1Y2N'][0]


# In[10]:


# Adding labels to the scaled data

scaled_df = pd.DataFrame(df_str2, columns= df_new.columns)
#scaled_df


# ### Gower package for catagorical features

# We can see that the variables are a mix of categorical and continuous values. We will use the Gower package to calculate the distance matrix for our feature sets.<br>
# (link: https://towardsdatascience.com/clustering-on-numerical-and-categorical-features-6e0ebcf1cbad)

# In[330]:


pip install gower


# In[11]:


import gower

dist_matrix = gower.gower_matrix(scaled_df)
dist_matrix


# ### Clustering using scikit-learn 

# In[13]:


import numpy as np 
from scipy import ndimage 
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 
from matplotlib import pyplot as plt 
from sklearn import manifold, datasets 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.datasets import make_blobs 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


Z_using_dist_matrix = hierarchy.linkage(dist_matrix, 'ward')


# In[15]:


import pylab
fig = pylab.figure(figsize=(18,10))
dendro = hierarchy.dendrogram(Z_using_dist_matrix, leaf_rotation=90, leaf_font_size =8, orientation = 'top')


# In[137]:


#Z_using_dist_matrix
#help(hierarchy.linkage)


# In[16]:


agglom = AgglomerativeClustering(n_clusters = 3, linkage = 'ward')
agglom.fit(dist_matrix)

agglom.labels_


# In[17]:


df_str2['cluster_'] = agglom.labels_
#df_str2


# In[18]:


# Add back the Paricipant_id column to the datafile to compare the cluster labels

df_str2.insert(0, 'Participant_ID', df['Participant_ID'])


# In[625]:


df_str2.to_csv("/Users/.../Data_HC_Ward.csv")


# In[19]:


linkage_matrix = hierarchy.linkage(agglom.children_)


# - **Use 'Truncate_mode' options to choose how many clusters we can choose.**

# In[27]:


plt.figure(figsize=(15,6))
dendro= hierarchy.dendrogram(linkage_matrix, truncate_mode = 'lastp', p=10)


# In[ ]:




