import pandas as pd
csv_file_path = 'C:/Users/karni/Downloads/insurance.csv'
data = pd.read_csv(csv_file_path)
#print(data.head(1))
#println(data.tail(1)) #To check the number of rows present No=1337
print(f'Row-{data.shape[0]} Col-{data.shape[1]}')
data.info()
print(data.isnull().sum())
print(data.describe(include='all'))
#Making the string into int form
print(data['sex'].unique())
data['sex']=data['sex'].map({'female':0,'male':1})
print(data['smoker'].unique())
data['smoker']=data['smoker'].map({'no':0,'yes':1})
print(data['region'].unique())
data['region']=data['region'].map({'southwest':0,'southeast':1,'northwest':2,'northeast':3})
#Table has been created
x=data.drop(['charges'],axis=1)#dependent variable
y=data['charges']#independent variable
print(y)
#Splitting of training dataset and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
#scaling to improve
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Feature scaling for x (independent variables)
scaler_x = StandardScaler()
x_train_scaled = scaler_x.fit_transform(x_train)
x_test_scaled = scaler_x.transform(x_test)
# Target scaling for y (dependent variable)
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
#linear regression model
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train_scaled,y_train_scaled)
y_pre_lr=lr.predict(x_test_scaled)
from sklearn.svm import SVR
svm=SVR()
svm.fit(x_train_scaled,y_train_scaled)
y_pre_svm=svm.predict(x_test_scaled)
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()
rf.fit(x_train_scaled,y_train_scaled)
y_pre_rf=rf.predict(x_test_scaled)
from sklearn.ensemble import GradientBoostingRegressor
gr=GradientBoostingRegressor()
paragrid={'n_estimators':[50,100,200],
        'learning_rate':[0.001,0.01,0.1],
          'max_depth':[3,4,5]
}
from sklearn.model_selection import GridSearchCV
grid_search=GridSearchCV(gr,paragrid,cv=5)
grid_search.fit(x_train_scaled,y_train_scaled)
improve_gr=grid_search.best_estimator_
y_pre_gr=improve_gr.predict(x_test_scaled)

y_pre_lr_original_scale = scaler_y.inverse_transform(y_pre_lr.reshape(-1, 1)).flatten()
y_pre_svm_original_scale = scaler_y.inverse_transform(y_pre_svm.reshape(-1, 1)).flatten()
y_pre_rf_original_scale = scaler_y.inverse_transform(y_pre_rf.reshape(-1, 1)).flatten()
y_pre_gr_original_scale = scaler_y.inverse_transform(y_pre_gr.reshape(-1, 1)).flatten()

# Create DataFrame for visualization
df1 = pd.DataFrame({'Actual': y_test,
                    'Predict_lr': y_pre_lr_original_scale,
                    'Predict_svm': y_pre_svm_original_scale,
                    'Predict_rf': y_pre_rf_original_scale,
                    'Predict_gr': y_pre_gr_original_scale})

#Visulaization
import matplotlib.pyplot as plt
plt.subplot(421)
plt.plot(df1['Actual'],label='actual')
plt.plot(df1['Predict_lr'],label='lr')
plt.legend()
plt.subplot(422)
plt.plot(df1['Actual'],label='actual')
plt.plot(df1['Predict_svm'],label='svm')
plt.legend()
plt.subplot(423)
plt.plot(df1['Actual'],label='actual')
plt.plot(df1['Predict_rf'],label='rf')
plt.legend()
plt.subplot(424)
plt.plot(df1['Actual'],label='actual')
plt.plot(df1['Predict_gr'],label='gr')
plt.legend()
plt.subplot(425)
plt.plot(df1['Actual'].iloc[0:11],label='actual')
plt.plot(df1['Predict_lr'].iloc[0:11],label='lr')
plt.legend()
plt.subplot(426)
plt.plot(df1['Actual'].iloc[0:11],label='actual')
plt.plot(df1['Predict_svm'].iloc[0:11],label='svm')
plt.legend()
plt.subplot(427)
plt.plot(df1['Actual'].iloc[0:11],label='actual')
plt.plot(df1['Predict_rf'].iloc[0:11],label='rf')
plt.legend()
plt.subplot(428)
plt.plot(df1['Actual'].iloc[0:11],label='actual')
plt.plot(df1['Predict_gr'].iloc[0:11],label='gr')
plt.legend()

plt.tight_layout()
plt.show()

from sklearn import metrics
#high r2 value is good
s1=metrics.r2_score(y_test_scaled,y_pre_gr)
s2=metrics.r2_score(y_test_scaled,y_pre_svm)
s3=metrics.r2_score(y_test_scaled,y_pre_lr)
s4=metrics.r2_score(y_test_scaled,y_pre_rf)
print(f'R2 SCORE \n GradientBoostingRegressor:{s1}\n SVM:{s2}\n LinearRegression:{s3}\n RandomForestRegressor:{s4}')
#less mean is good
r1=metrics.mean_absolute_error(y_test_scaled,y_pre_gr)
r2=metrics.mean_absolute_error(y_test_scaled,y_pre_svm)
r3=metrics.mean_absolute_error(y_test_scaled,y_pre_lr)
r4=metrics.mean_absolute_error(y_test_scaled,y_pre_rf)
print(f'MEAN ABSOLUTE ERROR \n GradientBoostingRegressor:{r1}\n SVM:{r2}\n LinearRegression:{r3}\n RandomForestRegressor:{r4}')

# Add a column for percentage accuracy
LR = (1 - metrics.mean_absolute_error(y_test_scaled, y_pre_lr)) * 100
print(LR)
SVM = (1 - metrics.mean_absolute_error(y_test_scaled, y_pre_svm)) * 100
print(SVM)
RF= (1 - metrics.mean_absolute_error(y_test_scaled, y_pre_rf)) * 100
print(RF)
GR= (1 - metrics.mean_absolute_error(y_test_scaled, y_pre_gr)) * 100
print(GR)

## From analysing with different models with got that RANDOMFOREST REGRESSION gives good accuracy % and mertices for regression also says the same
#Saving our model
gr=GradientBoostingRegressor()
paragrid={'n_estimators':[50,100,200],
        'learning_rate':[0.001,0.01,0.1],
          'max_depth':[3,4,5]
}
from sklearn.model_selection import GridSearchCV
grid_search=GridSearchCV(gr,paragrid,cv=5)
grid_search.fit(x,y)
improve_gr=grid_search.best_estimator_
import joblib
joblib.dump(improve_gr,'model1')
model=joblib.load('model1')
input={'age':20,
       'sex':1,
       'bmi':40.30,
       'children':1,
       'smoker':1,
       'region':1,
}
dataframe=pd.DataFrame([input])
t=model.predict(dataframe)
print(f'The value predicted for the given input:{t}')
