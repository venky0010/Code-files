# Importing packages

import os
import json
import sklearn
import time
import mxnet as mx
import pandas as pd
import matplotlib
from itertools import combinations
import random
import matplotlib.pyplot as plt

from numpy import mean
from pathlib import Path
from datetime import datetime
from gluonts.mx.trainer import Trainer
from gluonts.evaluation import Evaluator
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator

from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.mx.distribution.neg_binomial import NegativeBinomialOutput
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error

# Checking GPU

os.environ["MXNET_CUDNN_LIB_CHECKING"]="0"
print("\nWe have {} GPUs here!\n".format(mx.context.num_gpus()))

# Importing data

print(time.time())
start = time.perf_counter()
os.chdir('XXX')

def parser(s):
  return datetime.strptime(s,'%d-%m-%Y')

data = pd.read_csv(r'XXX.csv',parse_dates=[0],date_parser=parser,index_col=0)

# Lockdown dates

pre_lockdown_20 = data[pd.to_datetime('2020-01-01'):pd.to_datetime('2020-03-22')]
total_lockdown_20 = data[pd.to_datetime('2020-03-23'):pd.to_datetime('2020-04-30')]
partial_lockdown_20 = data[pd.to_datetime('2020-05-01'):pd.to_datetime('2020-06-08')]
post_lockdown_20 = data[pd.to_datetime('2020-09-01'):pd.to_datetime('2020-10-08')]

pre_lockdown_21 = data[pd.to_datetime('2021-03-20'):pd.to_datetime('2021-04-09')]
partial_lockdown_21_1 = data[pd.to_datetime('2021-04-10'):pd.to_datetime('2021-05-05')]
complete_lockdown_21 = data[pd.to_datetime('2021-05-06'):pd.to_datetime('2021-06-07')]
partial_lockdown_21_2 = data[pd.to_datetime('2021-06-08'):pd.to_datetime('2021-07-06')]
post_lockdown_21 = data[pd.to_datetime('2021-09-01'):pd.to_datetime('2020-09-30')]

os.chdir('XXX')

# Renaming columns

data.columns = ['Road traffic death', 'Road traffic injury (Grievous)','Road traffic injury (Minor)']

# Defining train and test set

test_stat = {}
train, test, agg_metrics, item_metrics, prediction = {},{},{},{},{}
train_time = pd.to_datetime('2019-12-31')
end_time = pd.to_datetime('2020-12-31')
atm = pd.to_datetime('2019-06-01')

for crime in data.columns:
    train[f'{crime}'] = ListDataset([{"start":data[f'{crime}'].index[0], "target": data[f'{crime}'][:train_time]}], freq = '1D')
    test[f'{crime}'] = ListDataset([{'start':data[f'{crime}'].index[0], "target": data[f'{crime}']}], freq = '1D')

# Defining WMAPE

def wmape(actual, forecast):
    # we take two series and calculate an output a wmape from it

    # make a series called mape
    se_mape = abs(actual-forecast)/actual

    # get a float of the sum of the actual
    ft_actual_sum = actual.sum()

    # get a series of the multiple of the actual & the mape
    se_actual_prod_mape = actual * se_mape

    # summate the prod of the actual and the mape
    ft_actual_prod_mape_sum = se_actual_prod_mape.sum()

    # float: wmape of forecast
    ft_wmape_forecast = ft_actual_prod_mape_sum / ft_actual_sum

    # return a float
    return ft_wmape_forecast

# Defining Cliff's D

def cliffsDelta(lst1, lst2, **dull):

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
    
    for j, two in enumerate(lst):
        if j == 0:
            one, i = two, 0
        if one != two:
            yield j - i, one
            i = j
        one = two
    yield j - i + 1, two

# Evaluator and estimator

estimator = {}

epochs = range(1,300)
learning_rate = range(1e-3, 1e-20)
patience = range(10,40)
decay_factor = range(0.2,0.9)

for nn in range(0,100):

    E = epochs[random.randint(1,300)]
    L = learning_rate[random.random(1e-20, 1e-3)]
    P = patience[random.randint(10,40)]
    D = decay_factor[round(random.random(0.2,0.9),1)]

for crime in data.columns:

    estimator[f'{crime}'] =  DeepAREstimator(freq = '1D',
                                               prediction_length = 639,
                                               context_length = 7,
                                               batch_size = 16,
                                               trainer = Trainer(epochs = E, ctx = mx.context.gpu(), minimum_learning_rate = L, learning_rate_decay_factor = D, patience = P),
                                               num_layers = 3,
                                               num_cells = 80)


    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9],
                          custom_eval_fn = {'WMAPE':[wmape, 'mean', 'mean']})

    # Train and forecast

    forecast_b, ts_b = {},{}
    predictor, forecast, ts = {},{},{}

    PREL_pred_2020,CL_pred_2020,PL_pred_2020,POSTL_pred_2020 = {},{},{},{}
    PREL_test_2020,CL_test_2020,PL_test_2020,POSTL_test_2020 = {},{},{},{}
    PREL_pred_2021,PL_pred_2021_1,CL_pred_2021,PL_pred_2021_2 = {},{},{},{}
    PREL_test_2021,PL_test_2021_1,CL_test_2021,PL_test_2021_2 = {},{},{},{}

    POSTL_pred_2021, POSTL_test_2021 = {},{}
    
    # Training the model
    print(f'{crime}')
    print(time.time())
    predictor[f'{crime}'] = estimator[f'{crime}'].train(training_data=train[f'{crime}'])
    predictor[f'{crime}'].serialize(Path(f'XXX'))

    # Forecasting
    forecast_b[f'{crime}'], ts_b[f'{crime}'] = make_evaluation_predictions(dataset = test[f'{crime}'], predictor = predictor[f'{crime}'], num_samples = 100)
    forecast[f'{crime}'], ts[f'{crime}'] = list(forecast_b[f'{crime}']), list(ts_b[f'{crime}'])

    # Splitting predictions into periods
    prediction[f'{crime}'] = pd.DataFrame({'prediction':forecast[f'{crime}'][0].mean}, index = pd.date_range('2020-01-01', periods=639, freq='D'))
    test_stat[f'{crime}'] = ts[f'{crime}'][0][-639:]

    # 2020 lockdown

    PREL_pred_2020[f'{crime}'] = prediction[f'{crime}'].loc[pd.date_range('2020-01-01', periods=82, freq='D')]
    CL_pred_2020[f'{crime}'] = prediction[f'{crime}'].loc[pd.date_range('2020-03-23', periods=38, freq='D')]
    PL_pred_2020[f'{crime}'] = prediction[f'{crime}'].loc[pd.date_range('2020-05-01', periods=38, freq='D')]
    POSTL_pred_2020[f'{crime}'] = prediction[f'{crime}'].loc[pd.date_range('2020-08-28', periods=38, freq='D')]

    PREL_test_2020[f'{crime}'] = test_stat[f'{crime}'].loc[pd.date_range('2020-01-01', periods=82, freq='D')]
    CL_test_2020[f'{crime}'] = test_stat[f'{crime}'].loc[pd.date_range('2020-03-23', periods=38, freq='D')]
    PL_test_2020[f'{crime}'] = test_stat[f'{crime}'].loc[pd.date_range('2020-05-01', periods=38, freq='D')]
    POSTL_test_2020[f'{crime}'] = test_stat[f'{crime}'].loc[pd.date_range('2020-09-01', periods=38, freq='D')]
    print('\n')

    # 2021 lockdown

    PREL_pred_2021[f'{crime}'] = prediction[f'{crime}'].loc[pd.date_range('2021-03-20', '2021-04-09', freq='D')]
    PL_pred_2021_1[f'{crime}'] = prediction[f'{crime}'].loc[pd.date_range('2021-04-10', '2021-05-05', freq='D')]
    CL_pred_2021[f'{crime}'] = prediction[f'{crime}'].loc[pd.date_range('2021-05-06', '2021-06-07', freq='D')]
    PL_pred_2021_2[f'{crime}'] = prediction[f'{crime}'].loc[pd.date_range('2021-06-08', '2021-07-06', freq='D')]
    POSTL_pred_2021[f'{crime}'] = prediction[f'{crime}'].loc[pd.date_range('2021-09-01', '2021-09-30', freq='D')]

    PREL_test_2021[f'{crime}'] = test_stat[f'{crime}'].loc[pd.date_range('2021-03-20', '2021-04-09', freq='D')]
    PL_test_2021_1[f'{crime}'] = test_stat[f'{crime}'].loc[pd.date_range('2021-04-10', '2021-05-05', freq='D')]
    CL_test_2021[f'{crime}'] = test_stat[f'{crime}'].loc[pd.date_range('2021-05-06', '2021-06-07', freq='D')]
    PL_test_2021_2[f'{crime}'] = test_stat[f'{crime}'].loc[pd.date_range('2021-06-08', '2021-07-06', freq='D')]
    POSTL_test_2021[f'{crime}'] = test_stat[f'{crime}'].loc[pd.date_range('2021-09-01', '2021-09-30', freq='D')]
    print('\n')

    end = time.perf_counter()
    print(end-start)

# Statistical tests

from scipy.stats import shapiro, wilcoxon

Periods_pred = {'PREL_pred_2020':PREL_pred_2020, 'CL_pred_2020':CL_pred_2020, 'PL_pred_2020':PL_pred_2020, 'POSTL_pred_2020':POSTL_pred_2020,
                'PREL_pred_2021':PREL_pred_2021, 'PL_pred_2021_1':PL_pred_2021_1, 'CL_pred_2021':CL_pred_2021, 'PL_pred_2021_2':PL_pred_2021_2,
                'POSTL_pred_2021':POSTL_pred_2021}  

Periods_test = {'PREL_test_2020':PREL_test_2020, 'CL_test_2020':CL_test_2020, 'PL_test_2020':PL_test_2020, 'POSTL_test_2020':POSTL_test_2020,
                'PREL_test_2021':PREL_test_2021, 'PL_test_2021_1':PL_test_2021_1, 'CL_pred_2021':CL_pred_2021, 'PL_test_2021_2':PL_test_2021_2,
                'POSTL_test_2021':POSTL_test_2021}  

stat_tests_1 = {'Shapiro': shapiro}
stat_tests_3 = {'Mean':mean}
stat_tests_2 = {'Cliff':cliffsDelta, 'Wilcoxon':wilcoxon}
predicted_table = pd.DataFrame()
actual_table = pd.DataFrame()

# Mean and shapiro

b,c = pd.DataFrame(),pd.DataFrame()

for fun_key, fun in stat_tests_1.items():
  for crime in data.columns:
      for period_1,value_1 in Periods_pred.items():
          stat,p = stat_tests_1[fun_key](Periods_pred[period_1][f'{crime}'].iloc[:,0])
          a = {'stat': stat, 'p': p}
          print(f'{crime} ,{period_1} ,{fun_key} Predicted Statistics=%.3f, p=%.3f' % (stat, p))
          b = pd.concat([b, pd.DataFrame([a])],axis = 1)
      for period_2,value_2 in Periods_test.items():
          stat,p = stat_tests_1[fun_key](Periods_test[period_2][f'{crime}'].iloc[:,0])
          d = {'stat': stat, 'p': p}
          print(f'{crime} ,{period_2} ,{fun_key} Actual Statistics=%.3f, p=%.3f' % (stat, p))
          c = pd.concat([c,pd.DataFrame([d])],axis = 1)

      predicted_table = predicted_table.append(b, ignore_index = True)
      actual_table = actual_table.append(c, ignore_index = True)
      b,c = pd.DataFrame(),pd.DataFrame()

predicted_table.to_csv('predicted stats shapiro.csv')
actual_table.to_csv('actual stats shapiro.csv')

predicted_table, actual_table = pd.DataFrame(), pd.DataFrame()
b,c = pd.DataFrame(),pd.DataFrame()

for fun_key, fun in stat_tests_3.items():
  for crime in data.columns:
      for period_3,value_3 in Periods_test.items():
          meanz = stat_tests_3[fun_key](Periods_test[period_3][f'{crime}'].iloc[:,0])
          a = {'mean':meanz}
          print(f'{crime} ,{period_3} ,{fun_key} = %.3f' % (a['mean']))
          c = pd.concat([c,pd.DataFrame([a])],axis = 1)

      for period_3,value_3 in Periods_pred.items():
          meanz = stat_tests_3[fun_key](Periods_pred[period_3][f'{crime}'].iloc[:,0])
          d = {'mean':meanz}
          print(f'{crime} ,{period_3} ,{fun_key} = %.3f' % (a['mean']))
          b = pd.concat([b,pd.DataFrame([d])], axis = 1)

      predicted_table = predicted_table.append(b, ignore_index = True)
      actual_table = actual_table.append(c, ignore_index = True)
      b,c =pd.DataFrame(),pd.DataFrame()

predicted_table.to_csv('predicted stats mean.csv')
actual_table.to_csv('actual stats mean.csv')

# Cliff's Delta and wilcoxon

cliff_table = pd.DataFrame()
cliff_pairs = {'PREL_pred_2020':PREL_test_2020, 'CL_pred_2020':CL_test_2020, 'PL_pred_2020':PL_test_2020, 'POSTL_pred_2020':POSTL_test_2020,
               'PREL_pred_2021':PREL_test_2021, 'PL_pred_2021_1':PL_test_2021_1,'CL_pred_2021':CL_test_2021 ,'PL_pred_2021_2':PL_test_2021_2,
               'POSTL_pred_2021':POSTL_test_2021}

b = pd.DataFrame()

for fun_key, fun in stat_tests_2.items():
  for crime in data.columns:
    for period_1,value_1 in Periods_pred.items():
      p,q = stat_tests_2[fun_key](cliff_pairs[period_1][f'{crime}'].iloc[:,0], Periods_pred[period_1][f'{crime}'].iloc[:,0])
      a = {'statistic': p ,'p':q}
      print(f'{crime} ,{period_1} ,{fun_key} = {p}')
      b = pd.concat([b,pd.DataFrame([a])], axis = 1)

    cliff_table = cliff_table.append(b, ignore_index = True)
    b = pd.DataFrame()

cliff_table.to_csv('cliff and wilcoxon stats.csv')

# Plotting the results

plt.rcParams.update({'font.size': 16})
plt.rcParams["axes.labelweight"] = "bold"
matplotlib.font_manager.FontEntry(weight='bold')

dates = ['2020-01-01','2020-03-22','2020-04-30','2020-06-08','2020-09-01','2020-10-08','2021-03-20','2021-04-10','2021-05-05',
         '2021-06-07','2021-07-06','2021-09-01','2021-09-30']

# Plotting both the years together

for crime in data.columns:

    fig, ax = plt.subplots(1, 1, figsize=(18, 6))
    ts[f'{crime}'][0][-639:].plot(ax=ax, color = 'b') 
    forecast[f'{crime}'][0].plot(prediction_intervals=[95], color='r')

    for i in dates:
        plt.axvline(x = pd.to_datetime(f'{i}'), ls='--', color='k', linewidth=0.7)

    plt.axhline(y = mean(Periods_pred['PREL_pred'][f'{crime}']).values[0], ls='-.', color='k', linewidth=3, xmin=0.002, xmax=0.123)
    plt.axhline(y = mean(Periods_pred['CL_pred'][f'{crime}']).values[0], ls='-.', color='k', linewidth=3, xmin=0.123,xmax=0.221)
    plt.axhline(y = mean(Periods_pred['PL_pred'][f'{crime}']).values[0], ls='-.', color='k', linewidth=3, xmin=0.221,xmax=0.331)
    plt.axhline(y = mean(Periods_pred['POSTL_pred'][f'{crime}']).values[0], ls='-.', color='k', linewidth=3, xmin=0.665,xmax=0.762)
    plt.axhline(y = mean(Periods_test['PREL_test'][f'{crime}']).values[0], ls='-.', color='k', linewidth=3, xmin=0.002, xmax=0.123)
    plt.axhline(y = mean(Periods_test['CL_test'][f'{crime}']).values[0], ls='-.', color='k', linewidth=3, xmin=0.123,xmax=0.221)
    plt.axhline(y = mean(Periods_test['PL_test'][f'{crime}']).values[0], ls='-.', color='k', linewidth=3, xmin=0.221,xmax=0.331)
    plt.axhline(y = mean(Periods_test['POSTL_test'][f'{crime}']).values[0], ls='-.', color='k', linewidth=3, xmin=0.665,xmax=0.762)

    plt.title(f'Plot of Actual and predicted cases of {crime} in the four periods')
    legend_properties = {'weight':'bold'}
    plt.legend(('Actual', 'Predicted'), fontsize=14, loc=1, prop=legend_properties)
    plt.savefig(f'{crime}.png')


