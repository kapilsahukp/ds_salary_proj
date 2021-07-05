# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 18:21:30 2021

@author: kapil
"""

import glassdoor_scraper as gs
import pandas as pd

path = "E:/DS_Projects/ds_salary_proj/chromedriver"

df = gs.get_jobs("data scientist", 15, False, path, 15 )

df.to_csv('glassdoor_jobs.csv', index = False)