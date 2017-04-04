import sys
sys.path.append("../lib/")

import oocHotEncoder as he
import sparseMatrix as sm
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import  linear_model
from keras.models import Sequential  
from keras.layers.core import Dense, Activation


c=['3SsnPorch', 'MasVnrType', 'LotConfig', 'BedroomAbvGr', 'GarageQual',
 'Exterior1st', 'GarageFinish', 'GarageYrBlt', 'HalfBath', 'OverallQual',
  'OverallCond', 'LotArea', 'Electrical', 'BsmtQual', 'HouseStyle', 
  'YearRemodAdd', 'GarageCond', 'Foundation', 'BsmtFullBath', 'BsmtCond',
  'HeatingQC', 'LotFrontage', 'MiscVal', 'CentralAir', 'BsmtExposure',
  'Utilities', 'KitchenAbvGr', 'Fireplaces', 'LotShape', 'BsmtHalfBath',
  'TotRmsAbvGrd', 'MiscFeature', 'YrSold', 'PavedDrive', 'LandSlope',
  'BldgType', 'FullBath', 'SaleType', 'MSZoning', 'Exterior2nd', 'Heating',
  'MSSubClass', 'GarageCars', 'YearBuilt', 'KitchenQual', 'BsmtFinType2',
  'BsmtFinType1', 'Condition2', 'Condition1', 'GarageType', 'LandContour',
  'PoolArea', 'Neighborhood', 'ScreenPorch', 'Fence', 'MoSold', 'SaleCondition',
  'ExterQual', 'Functional', 'Alley', 'RoofStyle', 'Street', 'ExterCond', 'PoolQC', 'FireplaceQu', 'RoofMatl']


data_dir="/home/ahmedyassinekhaili/kaggle/housePrising/input/"
g=he.ooche(columns=c,target="SalePrice",Id="Id")
_=g.hotEncode(data_dir+"train.csv",data_dir+"train",sep=",")
test_id=g.hotEncode(data_dir+"test.csv",data_dir+"test",sep=",")

print g.oldToNew.keys()

train_fl=sm.sparseMatrix(file=data_dir+"train",feature_number=g.nextInd)
test_fl=sm.sparseMatrix(file=data_dir+"test",feature_number=g.nextInd)

train,target=train_fl.nextChunk(chunk_size=25000)
test,_=test_fl.nextChunk(chunk_size=25000)


########################### NN
fcl=[train.shape[1],128,256,64]
model = Sequential()
for i in range(len(fcl)-1):
    model.add(Dense(fcl[i+1], input_dim=fcl[i], init='normal', activation='relu'))
model.add(Dense(1, init='normal'))
# Compile model
model.compile(loss='mean_squared_error', optimizer='Adagrad')

model.fit(train, target,epochs=1000)
train_pred_nn=model.predict(train)
test_pred_nn=model.predict(test)


########################### xgb

print "training XGB"
regr = xgb.XGBRegressor(
                 colsample_bytree=0.2,
                 gamma=0.0,
                 learning_rate=0.05,
                 max_depth=6,
                 min_child_weight=1.5,
                 n_estimators=7200,                                                                  
                 reg_alpha=0.9,
                 reg_lambda=0.6,
                 subsample=0.2,
                 seed=42)


regr.fit(train, target)
train_pred_xgb = regr.predict(train)
test_pred_xgb = regr.predict(test)


################################FS

print "Features selection"
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(train, target)
importances = forest.feature_importances_

threshold=0.005

restricted=[]
for i in range(len(importances)):
	if importances[i]>threshold:
		restricted.append(i)


x=np.concatenate((train[:,restricted], train_pred_xgb.reshape((train.shape[0],1))), axis=1)
t=np.concatenate((test[:,restricted], test_pred_xgb.reshape((test.shape[0],1))), axis=1)
x=np.concatenate((x, np.array(train_pred_nn).reshape((x.shape[0],1))), axis=1)
t=np.concatenate((t, np.array(test_pred_nn).reshape((t.shape[0],1))), axis=1)





############################## Restricted xgb

print "training XGB"
regr = xgb.XGBRegressor(
                 colsample_bytree=0.2,
                 gamma=0.0,
                 learning_rate=0.05,
                 max_depth=6,
                 min_child_weight=1.5,
                 n_estimators=1200,                                                                  
                 reg_alpha=0.9,
                 reg_lambda=0.6,
                 subsample=0.2,
                 seed=42)


regr.fit(x, target)
test_pred_xgb = regr.predict(t)








############Create final prediction
pred_df = pd.DataFrame(np.array(test_pred_xgb), index=test_id, columns=["SalePrice"])
pred_df.to_csv('my_new_output.csv', header=True, index_label='Id')
