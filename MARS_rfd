import numpy as np
from pyearth import Earth
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

df = pd.read_csv('raw_geodata.csv', sep = ',', header = 0, index_col = 0)
categoricals = ['mezolayers', 'quartlayers']
df = df.drop(categoricals, axis = 1)

#rfd = kriging radon flux with both convective and diffusive data, 
#rfdk = kriging diffusive radon. rfdd - real radon data (707 points)
df = df.drop(['rfd', 'rfdd'], axis = 1)
df2 = df.sample(frac = 0.25)
df_trn = df2.sample(frac=0.8)
df_tst = df2.drop(df_trn.index)
trn_f = df_trn.copy()
tst_f = df_tst.copy()
trn_l = trn_f.pop('rfdk')
tst_l = tst_f.pop('rfdk')
x = df_trn.pop('X')
y = df_trn.pop('Y')

model = Earth(max_terms = 250, max_degree = 2)
MARS = model.fit(trn_f, trn_l)
print(model.trace())
print(MARS.summary())
tst_f = df_tst.copy()
tst_f = tst_f.drop(['rfdk'], axis = 1)

pred_pre = model.predict(tst_f)
a = plt.axes(aspect = 'equal')
plt.scatter(tst_l, pred_pre)
plt.xlabel('True')
plt.ylabel('Predictions')
plt.show()
print(r2_score(tst_l, pred_pre))

df_pred = pd.read_csv('raw_geodata.csv', sep = ',', header = 0, index_col = 0)
df_pred = df_pred[~df_pred.index.isin(df2.index)]
df_pred = df_pred.drop(['rfdd', 'rfdk', 'rfd', mezolayers, quartlayers], axis = 1)
predix = model.predict(df_pred)
df_pred['rfdpred'] = predix
df_pred.to_csv('MARS250.csv')
