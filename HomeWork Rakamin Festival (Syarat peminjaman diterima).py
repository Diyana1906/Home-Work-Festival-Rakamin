#!/usr/bin/env python
# coding: utf-8

# # Import Library

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# # Load DataSet¶

# In[2]:


df = pd.read_csv('loan_customer.csv')
df.head()


# In[3]:


df.info()


# # Data Cleaning

# ## Missing Value

# In[4]:


df.isna().sum()


# In[5]:


#mengubah tipe kolom birth_date menjadi datetime (integral)
df['birth_date']=pd.to_datetime(df['birth_date'])


# In[6]:


df.info()


# ### Jika ada kolom dengan data kosong yang sangat banyak, maka kolom tersebut bisa dihapus aja

# In[7]:


df.drop('has_credit_card', axis=1, inplace=True)


# In[8]:


df


# In[9]:


# drop baris dengan missing values
df = df.dropna()


# In[10]:


df.shape


# In[11]:


df.isna().sum()


# ## Handling Duplicated Data

# In[12]:


# cek jumlah duplicated rows 
# dari semua kolom 
df.duplicated().sum()


# In[13]:


print ('jumlah data duplikat')
print (df.duplicated().sum())
print ('Jumlah data setelah duplikat di hapus')
df.drop_duplicates(inplace=True)
print (df.duplicated().sum())


# In[14]:


df.info()


# ### applicant_income

# In[15]:


sns.boxplot(x='applicant_income', data=df)


# In[16]:


print (f'Jumlah baris sebelum memfilter outlier:', len(df))

Q1 = df['applicant_income'].quantile(0.25)
Q3 = df['applicant_income'].quantile(0.75)

IQR = Q3-Q1

low_limit = Q1 - 1.5*IQR
high_limit = Q3 + 1.5*IQR

df = df[(df["applicant_income"] >= low_limit) &
        (df["applicant_income"] <= high_limit)
        ]

print(f'Jumlah baris setelah memfilter outlier: {len(df)}')


# In[17]:


sns.boxplot(x='applicant_income', data=df)


# ### coapplicant_income

# In[18]:


sns.boxplot(x='coapplicant_income', data=df)


# In[19]:


print (f'Jumlah baris sebelum memfilter outlier:', len(df))

Q1 = df['coapplicant_income'].quantile(0.25)
Q3 = df['coapplicant_income'].quantile(0.75)

IQR = Q3-Q1

low_limit = Q1 - 1.5*IQR
high_limit = Q3 + 1.5*IQR

df = df[(df["coapplicant_income"] >= low_limit) &
        (df["coapplicant_income"] <= high_limit)
        ]

print(f'Jumlah baris setelah memfilter outlier: {len(df)}')


# In[20]:


sns.boxplot(x='coapplicant_income', data = df)


# ### loan_amount

# In[21]:


sns.boxplot(x='loan_amount', data=df)


# In[22]:


print ('Jumlah baris sebelum memfilter outlier:', len(df))

Q1 = df['loan_amount'].quantile(0.25)
Q3 = df['loan_amount'].quantile(0.75)

IQR = Q3-Q1

low_limit = Q1 - 1.5*IQR
high_limit = Q3 + 1.5*IQR

df = df[(df["loan_amount"] >= low_limit) &
        (df["loan_amount"] <= high_limit)]

print(f'Jumlah baris setelah memfilter outlier: {len(df)}')


# In[23]:


sns.boxplot(x='loan_amount', data=df)


# ### loan_term_month

# In[24]:


sns.boxplot(x='loan_term_month', data=df)


# ## Feature Encoding

# In[25]:


df1 = df.copy()

# membuat daftar nama kolom yang bertipe kategorikal 
cats = ['gender','married','dependents','education','self_employed','property_type','loan_status']

#melihat kategori dari setiap kolom
for i in cats:
    print('Kolom', i, df1[i].unique())


# In[26]:


# pengecekan nilai/entri dari kolom-kolom kategorikal

for col in cats:
    print(f'value counts of column {col}')
    print(df[col].value_counts())
    print('---'*10,'\n')


# #### Membuat label encoding untuk setiap kategori

# In[27]:


df1


# In[28]:


mapping_gender = {
    'Male' :0,
    'Female' :1
}

mapping_married = {
    'No' : 0,
    'Yes' : 1
}

mapping_dependents = {
    '0' : 0,
    '1' : 1,
    '2' : 2,
    '3+' : 3
}

mapping_education = {
    'Not Graduate' : 0,
    'Graduate' :1
}

mapping_self_employed = {
    'No' : 0,
    'Yes' : 1,
}

mapping_property_type = {
    'house' :0,
    'studio' :1,
    'apartment' :2
}

df1['gender'] = df1['gender'].map(mapping_gender)
df1['married'] = df1['married'].map(mapping_married)
df1['dependents'] = df1['dependents'].map(mapping_dependents)
df1['education'] = df1['education'].map(mapping_education)
df1['self_employed'] = df1['self_employed'].map(mapping_self_employed)
df1['property_type'] = df1['property_type'].map(mapping_property_type)


# In[29]:


df1


# # Visualization

# ## Business Insight

# ### Analisis syarat pelanggan yang sesuai berdasarkan pada tipe properti

# In[30]:


df_gr1 = df.groupby(['property_type','loan_status'])['loan_id'].count().reset_index()
df_gr1.columns = ['gender','property_type','count']
df_gr1


# In[31]:


# Visualisasi data diatas menggunakan bar chart (perbandingan)

sns.set(style="darkgrid")
f, ax = plt.subplots(1,1,figsize=(12,8))
sns.countplot(x = 'property_type', data=df, hue='loan_status', palette="Set1")


# ### Analisis pengaruh status perkawinan dan  jangka waktu peminjaman terhadap peminjaman di kabulkan 

# In[32]:


df_gr2 = df.groupby(['loan_term_year','loan_status'])['loan_id'].count().reset_index()
df_gr2.columns = [ 'loan_term_year','Loan_status', 'number_of_applicant']
df_gr2


# In[33]:


sns.set(style="darkgrid")
f, ax = plt.subplots(1,1,figsize=(12,8))

sns.histplot(data=df, x="loan_term_year", bins=10, color="pink")


# In[34]:


# membuat kategori untuk loan_term_year
loan_term_year_group = {}
for i in range(1,41) :
    if i<5:
        loan_term_year_group[i]='<15'
    elif i>=15 and i<=25:
        loan_term_year_group[i]= '15-25'
    elif i==30:
        loan_term_year_group[i]= '30'
    elif i>30:
        loan_term_year_group[i]= '>30'
    else:
        pass
    
# membuat kolom baru berdasarkan pada grouping sebelumnya
df_gr2['loan_term_year_group'] = df_gr2['loan_term_year'].replace(loan_term_year_group)


# In[35]:


plt.figure(figsize=(15,10))
sns.barplot(x='loan_term_year_group', y='number_of_applicant', hue='Loan_status', data=df_gr2)

plt.tick_params(axis='both', which='major', labelsize=12)
plt.grid()
plt.legend(title='Loan Status', title_fontsize=14, prop={'size':12}, loc=9)

plt.xlabel('Loan Term Group (in Year)', fontsize=14)
plt.ylabel('Number of Applicant', fontsize=14)
plt.text(x=-0.5, y=250, s="Most applicants who are married or not have chosen a loan term of 30 years", 
         fontsize=20, fontweight='bold') 


# ### Analisis pengaruh pendapatan pemohon terhadap penerimaan permohonan

# In[36]:


df_gr3 = df.groupby(['applicant_income','loan_status'])['loan_id'].count().reset_index()
df_gr3.columns = ['applicant_income', 'loan_status', 'number_of_applicant']
df_gr3


# In[37]:


sns.histplot(data=df, x="applicant_income", bins=10, color="pink")


# Berdasarkan grafik di atas, maka kita dapat mengelompokkan data menjadi beberapa kategori yaitu:
# 
# - <2000
# - 2000-4000
# - 4000-6000
# - 6000-8000
# - (>8000)

# In[38]:


# membuat kategori untuk applicant_income 
applicant_income_group = {}
for i in range(0,100001):
    if i<2000:
        applicant_income_group[i]='<2000'
    elif i>=2000 and i<=4000:
        applicant_income_group[i]='2000-4000'
    elif i>=4000 and i<=6000:
        applicant_income_group[i]='4000-6000'
    elif i>=6000 and i<=8000:
        applicant_income_group[i]='6000-8000'
    elif i>8000:
        applicant_income_group[i]='>8000'
    else:
        pass
# membuat kolom baru berdasarkan pada grouping sebelumnya 
df_gr3['applicant_income_group'] = df_gr3['applicant_income'].replace(applicant_income_group)


# In[39]:


df_gr3


# In[40]:


df_gr4 = df_gr3.groupby([ 'applicant_income_group','loan_status'])['applicant_income'].count().reset_index()
df_gr4.columns = ['applicant_income_group','loan_status', 'number_of_applicant']
df_gr4


# In[41]:


plt.figure(figsize=(15,10))
sns.barplot(x='applicant_income_group', y='number_of_applicant', hue='loan_status', data=df_gr4)

plt.tick_params(axis='both', which='major', labelsize=12)
plt.grid()
plt.legend(title='Loan Status', title_fontsize=14, prop={'size':12}, loc=9)

plt.xlabel('Applicant Income Group(in US Dollar)', fontsize=14)
plt.ylabel('Number of Applicant', fontsize=14)
plt.text(x=-0.5, y=150, s="Most Applicants who are approved or not have applicant income 2000-4000",
        fontsize=20, fontweight='bold')


# ### Analisis pengaruh pendapatan pemohon bersama (coapplicant_income) terhadap penerimaan permohonan

# In[42]:


df_gr5=df.groupby(['coapplicant_income','loan_status'])['loan_id'].count().reset_index()
df_gr5.columns = ['coapplicant_income', 'loan_status', 'number_of_applicant']
df_gr5


# In[43]:


sns.histplot(data=df, x="coapplicant_income", bins=10, color="pink")


# Berdasarkan grafik di atas, maka kita dapat mengelompokkan data menjadi beberapa kategori yaitu:
# 
# - <1000
# - 1000-2000
# - 2000-3000
# - 3000-4000
# - 4000-5000
# - (>5000)

# In[44]:


# membuat kategori untuk applicant_income 
coapplicant_income_group = {}
for i in range(0,100001):
    if i<1000:
        coapplicant_income_group[i]='<1000'
    elif i>=1000 and i<=2000:
        coapplicant_income_group[i]='1000-2000'
    elif i>=2000 and i<=3000:
        coapplicant_income_group[i]='2000-3000'
    elif i>=3000 and i<=4000:
        coapplicant_income_group[i]='3000-4000'
    elif i>=4000 and i<=5000:
        coapplicant_income_group[i]='4000-5000'
    elif i>5000:
        coapplicant_income_group[i]='>5000'
    else:
        pass
# membuat kolom baru berdasarkan pada grouping sebelumnya 
df_gr5['coapplicant_income_group'] = df_gr5['coapplicant_income'].replace(applicant_income_group)


# In[45]:


df_gr5


# In[46]:


df_gr6 = df_gr5.groupby(['coapplicant_income_group','loan_status'])['coapplicant_income'].count().reset_index()
df_gr6.columns = ['coapplicant_income_group','loan_status', 'number_of_applicant']
df_gr6


# In[47]:


plt.figure(figsize=(15,10))
sns.barplot(x='coapplicant_income_group', y='number_of_applicant', hue='loan_status', data=df_gr6)

plt.tick_params(axis='both', which='major', labelsize=12)
plt.grid()
plt.legend(title='Loan Status', title_fontsize=14, prop={'size':12}, loc=9)

plt.xlabel('Applicant Income Group(in US Dollar)', fontsize=14)
plt.ylabel('Number of Applicant', fontsize=14)
plt.text(x=-0.5, y=60, s="Most coapplicants who are approved or not have applicant income 2000-4000",
        fontsize=20, fontweight='bold')


# ### Analisis pengaruh jumlah pinjaman terhadap penerimaan permohonan

# In[48]:


df_gr7=df.groupby(['loan_amount','loan_status'])['loan_id'].count().reset_index()
df_gr7.columns = ['loan_amount', 'loan_status', 'number_of_applicant']
df_gr7


# In[49]:


sns.set(style="darkgrid")
f, ax = plt.subplots(1,1,figsize=(12,8))

sns.histplot(data=df, x="loan_amount", bins=10, color="pink")


# Berdasarkan grafik di atas, maka kita dapat mengelompokkan data menjadi beberapa kategori yaitu:
# 
# - <50000
# - 50000-100000
# - 100000-150000
# - 150000-200000
# - 200000-250000

# In[50]:


# membuat kategori untuk loan_term_year
loan_amount_group = {}
for i in range(1,1000001) :
    if i<50000:
        loan_amount_group[i]='<50000'
    elif i>=50000 and i<=100000:
        loan_amount_group[i]= '50000-100000'
    elif i>=100000 and i<=150000:
        loan_amount_group[i]= '100000-150000'
    elif i>=150000 and i<=200000:
        loan_amount_group[i]= '150000-200000'
    elif i>=200000 and i<=250000:
        loan_amount_group[i]= '200000-250000'
    elif i>250000:
        loan_amount_group[i]= '>250000'
    else:
        pass
    
# membuat kolom baru berdasarkan pada grouping sebelumnya
df_gr7['loan_amount_group'] = df_gr7['loan_amount'].replace(loan_amount_group)


# In[51]:


df_gr7


# In[53]:


df_gr8 = df_gr7.groupby(['loan_amount_group','loan_status'])['loan_amount'].count().reset_index()
df_gr8.columns = ['loan_amount_group','loan_status','number_of_applicant']
df_gr8


# In[57]:


plt.figure(figsize=(15,10))
sns.barplot(x='loan_amount_group', y='number_of_applicant', hue='loan_status', data=df_gr8)

plt.tick_params(axis='both', which='major', labelsize=12)
plt.grid()
plt.legend(title='Loan Status', title_fontsize=14, prop={'size':12}, loc=9)

plt.xlabel('Loan Amount Group (in US Dollar)', fontsize=14)
plt.ylabel('Number of Applicant', fontsize=14)
plt.text(x=-0.5, y=50, s="Most applicants who are approved or not have a loan amount of 100000-150000", 
         fontsize=20, fontweight='bold') 


# Kesimpulan :
#     Persyaratan permohonan peminjam pelanggan di setujui diantaranya adalah :
# - 
