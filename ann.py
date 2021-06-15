############################### problem 1 ################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation,Layer,Lambda

startup = pd.read_csv("C:/Users/usach/Desktop/ANN/50_Startups (2).csv")

##Creating dummy variables for the state column

startup_dummy = pd.get_dummies(startup["State"])

Startup = pd.concat([startup,startup_dummy],axis=1)

Startup.drop(["State"],axis=1, inplace =True)

Data = Startup.describe()
Data
##The scales of the data are differnt so we normalise
def norm_func(i):
    x = (i - i.min())/(i.max()-i.min())
    return (x)

Start_up = norm_func(Startup)

##Using this Data set and we build the model. 
def prep_model(hidden_dim):
    model = Sequential()
    for i in range(1,len(hidden_dim)-1):
        if (i==1):
            model.add(Dense(hidden_dim[i],input_dim=hidden_dim[0],kernel_initializer="normal",activation="relu"))
        else:
            model.add(Dense(hidden_dim[i],activation="relu"))
    model.add(Dense(hidden_dim[-1]))
    model.compile(loss="mean_squared_error",optimizer="adam",metrics = ["accuracy"])
    return (model)

predictors = Start_up.iloc[:,[0,1,2,4,5,6]]
target = Start_up.iloc[:,3]

##Partitioning the data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(predictors,target,test_size = 0.25)

first_model = prep_model([6,50,1])
first_model.fit(np.array(x_train),np.array(y_train),epochs=900)
#predicting on train data
pred_train = first_model.predict(np.array(x_train))
pred_train = pd.Series([i[0] for i in pred_train])
rmse_value = np.sqrt(np.mean((pred_train-y_train)**2))#0.3032473097680326
np.corrcoef(pred_train,y_train) ## 0.98698937

#Visualising 
plt.plot(pred_train,y_train,"bo")

##Predicting on test data
pred_test = first_model.predict(np.array(x_test))
pred_test = pd.Series([i[0] for i in pred_test])
rmse_test = np.sqrt(np.mean((pred_test-y_test)**2))#0.21990964982399788
np.corrcoef(pred_test,y_test)#0.97927285

##Visualizing
plt.plot(pred_test,y_test,"bo")
############################### problem 2 ################################
import pandas as pd
import numpy as np
#loading the dataset
forest = pd.read_csv("C:/Users/usach/Desktop/ANN/fireforests.csv")

#2.	Work on each feature of the dataset to create a data dictionary as displayed in the below image
data_details =pd.DataFrame({"column name":forest.columns,
                            "data type(in Python)": forest.dtypes})

#3.	Data Pre-forestcessing
#3.1 Data Cleaning, Feature Engineering, etc
#details of forest 
forest.info()
forest.describe()          
forest.nunique()

#data types        
forest.dtypes
#checking for na value
forest.isna().sum()
forest.isnull().sum()
#checking unique value for each columns
forest.nunique()
"""	Exploratory Data Analysis (EDA):
	Summary
	Univariate analysis
	Bivariate analysis """
EDA ={"column ": forest.columns,
      "mean": forest.mean(),
      "median":forest.median(),
      "mode":forest.mode(),
      "standard deviation": forest.std(),
      "variance":forest.var(),
      "skewness":forest.skew(),
      "kurtosis":forest.kurt()}

EDA
# covariance for data set 
covariance = forest.cov()
covariance
# Correlation matrix 
Correlation = forest.corr()
Correlation
# according to correlation coefficient no correlation of  Administration & State with model_dffit
#According scatter plot strong correlation between model_dffit and rd_spend and 
#also some relation between model_dffit and m_spend.

#variance for each column
forest.var()                   #rain column has low variance 
#droping rain colunm and month and day columns  due to thoes are already present in dummy
forest.drop(["month","day"], axis = 1, inplace = True)
####### graphidf repersentation 

##historgam and scatter plot
import seaborn as sns
sns.pairplot(forest.iloc[:,0:9])


#boxplot for every columns
forest.columns
forest.nunique()

#boxplot for every column
# Boxplot of independent variable distribution for each category of size_category
forest.boxplot(column=['FFMC', 'DMC', 'DC', 'ISI','temp', 'RH', 'wind', 'area','rain'])  
#normal
# Normalization function using z std. all are continuous data.
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)
# Normalized data frame (considering the numerical part of data)
df = norm_func(forest.iloc[:,0:8])
df.describe()
#final dataframe
model_df = pd.concat([forest.iloc[:,[8]],df,forest.iloc[:,9:28] ], axis =1)
##################################
"""5.	Model Building:
5.1	Perform Artificial Neural Network on the given datasets.
5.2	Use TensorFlow keras to build your model in Python and use Neural net package in R
5.3	Briefly explain the output in the documentation for each step in your own words.
5.4	Use different activation functions to get the best model.
"""
from tensorflow.keras import Sequential
from tensorflow.keras.layers import  Dense

np.random.seed(10)

X= model_df.iloc[:,1:]
Y= model_df.iloc[:,0]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2,random_state = 457) # 20% test data
 
import sklearn.metrics as skl_mtc
from tensorflow import keras 
import matplotlib.pyplot as plt

model = keras.models.Sequential()
model.add(keras.layers.Dense(5000, activation='relu', input_dim=27))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(500, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(50, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(1, kernel_initializer='uniform'))
model.compile(loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Nadam(
        learning_rate=0.0005,
        beta_1=0.8,
        beta_2=0.999),metrics=["mse"])

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    verbose=1,
    patience=20,
    mode='auto',
    restore_best_weights=True)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=10,
    verbose=1,
    mode='auto',
    min_delta=0.0005,
    cooldown=0,
    min_lr=1e-6)
# fitting model on train data
model.fit(x=x_train,y=y_train,batch_size=15,epochs=100)
# Evaluating the model on test data  
eval_score_test = model.evaluate(x_test,y_test,verbose = 1)

# accuracy on test data set

# accuracy score on train data 
eval_score_train = model.evaluate(x_train,y_train,verbose=1)
predict_y = model.predict(x_test)

#R2-score
result = skl_mtc.r2_score(y_test, predict_y)
print(f'R2-score in test set: {np.round(result, 4)}')

# test residual values 

# accuracy on train data set 

pred_df = pd.DataFrame(predict_y, columns =['predict_y'])
pred_y= pred_df.iloc[:,0]

test_resid = pred_y - y_test
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse

#graph for eppchs
history = model.fit(x_train, y_train, epochs=10, batch_size=50,  verbose=1, validation_split=0.2)

print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
############################### problem 3 ################################
import pandas as pd
import numpy as np

#loading the dataset
concrete = pd.read_csv("C:/Users/usach/Desktop/ANN/concrete.csv")

# according to correlation coefficient no correlation of  Administration & State with model_dffit
#According scatter plot strong correlation between model_dffit and rd_spend and 
#also some relation between model_dffit and m_spend.

#variance for each column
concrete.var()                   #rain column has low variance 

#normal
# Normalization function using z std. all are continuous data.
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Normalized data frame (considering the numerical part of data)
df = norm_func(concrete.iloc[:,0:8])
df.describe()
#final dataframe
model_df = pd.concat([concrete.iloc[:,[8]],df ], axis =1)
np.random.seed(10)

X= model_df.iloc[:,1:]
Y= model_df.iloc[:,0]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2,random_state = 457) # 20% test data
 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import  Dense
import sklearn.metrics as skl_mtc
from tensorflow import keras 
import matplotlib.pyplot as plt

model = keras.models.Sequential()
model.add(keras.layers.Dense(5000, activation='relu', input_dim=8))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(500, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(50, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(1, kernel_initializer='uniform'))
model.compile(loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Nadam(
        learning_rate=0.0005,
        beta_1=0.8,
        beta_2=0.999),metrics=["mse"])

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    verbose=1,
    patience=20,
    mode='auto',
    restore_best_weights=True)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=10,
    verbose=1,
    mode='auto',
    min_delta=0.0005,
    cooldown=0,
    min_lr=1e-6)

# fitting model on train data
model.fit(x=x_train,y=y_train,batch_size=50,epochs=100)
# Evaluating the model on test data  
eval_score_test = model.evaluate(x_test,y_test,verbose = 1)
print ("Accuracy: %.3f%%" %(eval_score_test[1]*100)) 
# accuracy on test data set

# accuracy score on train data 
eval_score_train = model.evaluate(x_train,y_train,verbose=1)
print ("Accuracy: %.3f%%" %(eval_score_train[1]*100)) 
predict_y = model.predict(x_test)
#R2-score
result = skl_mtc.r2_score(y_test, predict_y)
print(f'R2-score in test set: {np.round(result, 4)}')

# test residual values 
# accuracy on train data set 

pred_df = pd.DataFrame(predict_y, columns =['predict_y'])
pred_y= pred_df.iloc[:,0]

test_resid = pred_y - y_test
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse
#graph for eppchs
history = model.fit(x_train, y_train, epochs=10, batch_size=50,  verbose=1, validation_split=0.2)

print(history.history.keys())

############################### problem 4 ################################
import pandas as pd
import numpy as np


#loading the dataset
rpl = pd.read_csv("C:/Users/usach/Desktop/ANN/RPL.csv")

# covariance for data set 
covariance = rpl.cov()
covariance

# Correlation matrix 
Correlation = rpl.corr()
Correlation

# according to correlation coefficient no correlation of  Administration & State with model_dffit
#According scatter plot strong correlation between model_dffit and rd_spend and 
#also some relation between model_dffit and m_spend.

#variance for each column
rpl.var()                   #rain column has low variance 
rpl.drop(["RowNumber","CustomerId","Surname"], axis = 1, inplace = True)

#normal

# Normalization function using z std. all are continuous data.
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Normalized data frame (considering the numerical part of data)
df = norm_func(rpl.iloc[:,[0,3,4,5,9]])
df.describe()
#categorical
enc_df = pd.get_dummies(rpl.iloc[:,[1,2,6,7,8]])
enc_df.columns

#final dataframe
model_df = pd.concat([rpl.iloc[:,[10]],df,enc_df], axis =1)
# from keras.datasets import mnist

from tensorflow.keras import Sequential
from tensorflow.keras.layers import  Dense
from keras.utils import np_utils
from keras.layers import Dropout,Flatten

np.random.seed(10)

from sklearn.model_selection import train_test_split

model_df_train, model_df_test = train_test_split(model_df, test_size = 0.2,random_state = 457) # 20% test data
 
x_train = model_df_train.iloc[:,1:].values.astype("float32")
y_train = model_df_train.iloc[:,0].values.astype("float32")
x_test = model_df_test.iloc[:,1:].values.astype("float32")
y_test = model_df_test.iloc[:,0].values.astype("float32")

# one hot encoding outputs for both train and test data sets 
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


# Storing the number of classes into the variable num_of_classes 
num_of_classes = y_test.shape[1]


# Creating a user defined function to return the model for which we are
# giving the input to train the ANN mode
def design_mlp():
    # Initializing the model 
    model = Sequential()
    model.add(Dense(150,input_dim =13,activation="relu"))
    model.add(Dense(200,activation="tanh"))
    model.add(Dense(120,activation="tanh"))
    model.add(Dense(200,activation="tanh"))
    model.add(Dense(num_of_classes,activation="softmax"))
    model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
    return model

# building a cnn model using train data set and validating on test data set
model = design_mlp()

# fitting model on train data
model.fit(x=x_train,y=y_train,batch_size=500,epochs=5)


# Evaluating the model on test data  
eval_score_test = model.evaluate(x_test,y_test,verbose = 1)
print ("Accuracy: %.3f%%" %(eval_score_test[1]*100)) 
# accuracy on test data set

# accuracy score on train data 
eval_score_train = model.evaluate(x_train,y_train,verbose=0)
print ("Accuracy: %.3f%%" %(eval_score_train[1]*100)) 
# accuracy on train data set 
