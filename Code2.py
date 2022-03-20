import numpy as np
import pandas as pd  
import math as ma
from scipy import stats

data = pd.read_csv('chennaiMobility.csv', parse_dates=['date'], index_col=['date'])

def Shapiro_Normality_test(array):
    
    shapiro_test = stats.shapiro(array)
    p_val = shapiro_test.pvalue
    return p_val

def cliffsDelta(lst1, lst2, **dull):

    """Returns delta and true if there are more than 'dull' differences"""
    if not dull:
        dull = {'small': 0.147, 'medium': 0.33, 'large': 0.474} # effect sizes from (Hess and Kromrey, 2004)
    m, n = len(lst1), len(lst2)
    lst2 = sorted(lst2)
    j = more = less = 0
    for repeats, x in runs(sorted(lst1)):
        while j <= (n - 1) and lst2[j] < x:
            j += 1
        more += j*repeats
        while j <= (n - 1) and lst2[j] == x:
            j += 1
        less += (n - j)*repeats
    d = (more - less) / (m*n)
    size = lookup_size(d, dull)
    return d, size


def lookup_size(delta: float, dull: dict) -> str:
    """
    :type delta: float
    :type dull: dict, a dictionary of small, medium, large thresholds.
    """
    delta = abs(delta)
    if delta < dull['small']:
        return 'negligible'
    if dull['small'] <= delta < dull['medium']:
        return 'small'
    if dull['medium'] <= delta < dull['large']:
        return 'medium'
    if delta >= dull['large']:
        return 'large'


def runs(lst):
    """Iterator, chunks repeated values"""
    for j, two in enumerate(lst):
        if j == 0:
            one, i = two, 0
        if one != two:
            yield j - i, one
            i = j
        one = two
    yield j - i + 1, two
    

def cohens_D_effect_size(arr1, arr2):
        
    diff = abs(arr1.mean() - arr2.mean())
    stddev = ma.sqrt(((len(arr1)-1)*arr1.std()**2 + (len(arr2)-1)*arr2.std()**2)/(len(arr1)+len(arr2)-2))
    effect = diff/stddev
    return effect
dull = {'small': 0.147, 'medium': 0.33, 'large': 0.474}
x = data.loc['2020-03-23':'2020-04-30']
y = data.loc['2020-05-01':'2020-06-08']
z = data.loc['2021-04-10':'2021-05-05']
z1 = data.loc['2021-05-06':'2021-06-07']
z2 = data.loc['2021-06-08':'2021-07-06']

for category in data.columns:         #Looping through each category
    
    #print(i, "\n")
    c1 = np.array(x[category].tolist())         #Complete Lockdown 1
    p1 = np.array(y[category].tolist())         #Partial Lockdown 1
    c = z[category].tolist()          
    c2 = np.array(z1[category].tolist())        #Complete lockdown 2
    e = z2[category].tolist() 
    p2 = np.array(c+e)                          #Partial Lockdown 2 (PL-I + PL - II)
    #print(p1[:4], c1[:4], p2[:4], c2[:4])

    print("Shapiro Test")
    print(Shapiro_Normality_test(p1), Shapiro_Normality_test(p2), Shapiro_Normality_test(c1), Shapiro_Normality_test(c1))
    print("\n")
    print("CohensD effect size")
    print(cohens_D_effect_size(p1, c1), cohens_D_effect_size(p2, c2), cohens_D_effect_size(c1, c2), cohens_D_effect_size(p1, p2))
    print("\n")
    print("Cliff's Delta")
    print(cliffsDelta(p1, c1, **dull), cliffsDelta(p2, c2, **dull), cliffsDelta(c1, c2, **dull), cliffsDelta(p1, p2, **dull))
    print("\n")
    print("\n")
