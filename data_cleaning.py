# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 11:26:43 2021

@author: kapil
"""

import pandas as pd

df = pd.read_csv("glassdoor_jobs.csv")

#Salary parsing

df['Hourly'] = df['Salary Estimate'].apply(lambda x: 1 if "per hour" in x.lower() else 0)
df['Employer Provided'] = df['Salary Estimate'].apply(lambda x:1 if 'employer provided' in x.lower() else 0)

df = df[df["Salary Estimate"]!= "-1"]

salary = df["Salary Estimate"].apply(lambda x: x.split("(")[0])
minus_kd = salary.apply(lambda x: x.replace('K','').replace('$',''))

min_hr = minus_kd.apply(lambda x: x.lower().replace('per hour','').replace('employer provided salary:',''))

df['Min_Salary'] = min_hr.apply(lambda x: int(x.split('-')[0]))
df['Max_Salary'] = min_hr.apply(lambda x: int(x.split('-')[1]))

df['Min_Salary'].dtype

df['Avg_Salary'] = (df.Min_Salary + df.Max_Salary)/2

#Company name text only
df['Company_Txt'] = df.apply(lambda x: x['Company Name'] if x['Rating']<0 else x['Company Name'][:-3], axis = 1 )


#State field
df['Job_State'] = df['Location'].apply(lambda x: x.split(',')[1])
df['Same_State'] = df.apply(lambda x: 1 if x['Location'] == x['Headquarters'] else 0, axis =1)

#Age of company
df['Age'] = df['Founded'].apply(lambda x: x if x<0 else 2020-x)

#Parsing of job description (python, etc)

#python
df['Python_yn'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)

#r studio
df['Rstudio_yn'] = df['Job Description'].apply(lambda x: 1 if 'r studio' in x.lower() else 0)

#spark
df['Spark_yn'] = df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)

#aws
df['Aws_yn'] = df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)

#excel
df['Excel_yn'] = df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)

df_out = df.drop(['Unnamed: 0'], axis = 1)

#Clean data to csv
df_out.to_csv("salary_data_cleaned.csv", index = False)



