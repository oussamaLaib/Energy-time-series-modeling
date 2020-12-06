import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn import cluster, datasets
from pandas import DataFrame as df
from IPython.display import clear_output

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.core import Activation, Dropout  
from keras.optimizers import RMSprop
from keras.layers import Embedding, RepeatVector
from keras.layers import LSTM
from keras.layers.recurrent import GRU
from keras.optimizers import SGD



def import_data():
    dataset = pd.read_csv('data/DP.csv',usecols=[27],engine='python',skipfooter=None)
    return dataset
def mean_absolute_percentage_error(y_true, y_pred): 

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true): 
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def getData_Clustered():
    data = import_data()
    data= np.array(data)
    data= data.reshape(-1,24)

    daydata= df(data)
    daydata.index= pd.date_range('2014-1-1', periods=365, freq='D')


    clusters = cluster.KMeans(n_clusters=3).fit_predict(daydata)
    pca= PCA(n_components=3)
    pca.fit(daydata)
    data_pca =  pca.transform(daydata)

    clusters[[45,46,47]]= clusters[0]
    clusters[[84,85,86]]= clusters[87]
    # pdClusters= pd.DataFrame(clusters,index=daydata.index)
    print(clusters)
    fig = plt.figure(1, figsize=(12, 9))

    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=45, azim=135)

    ax.scatter(data_pca[:,0],data_pca[:,1],data_pca[:,2],cmap=plt.cm.spectral,c=clusters)

    plt.show()

    daydata['cluster']=clusters
    return daydata

def prepareInputs(daydata, season, UsedInputs):
    nbrInputs= 0
    
    previousHours = UsedInputs[0]
    previousDay = UsedInputs[1]
    previousWeek = UsedInputs[2]
    temp = UsedInputs[3]
    tempMax = UsedInputs[4]
    tempMin = UsedInputs[5]
    dayIndicator = UsedInputs[6]
    
    if previousHours == True: nbrInputs= nbrInputs+ 1
    if previousDay == True: nbrInputs= nbrInputs+1
    if previousWeek == True : nbrInputs= nbrInputs+ 1
    if temp == True: nbrInputs= nbrInputs+1
    if tempMax == True: nbrInputs= nbrInputs+1
    if tempMin == True: nbrInputs= nbrInputs+1
    if dayIndicator== True: nbrInputs= nbrInputs+7

    hourclusters= np.empty([(daydata.index.size*24),1])

    hourdataindex= pd.DataFrame(index=pd.date_range('2014-1-8 00:00:00', periods=(365)*24, freq='H'))

    for x in range(0,daydata.index.size):
        for y in range(0,24):
            hourclusters[(x * 24) + y,0] = daydata.iloc[x,24]
    hourclusters.size

    tempAlgiers=  pd.read_csv('data/tempAlgiers.csv')
    tempA= tempAlgiers.loc[:,'Hour_1':'Hour_24']
    tempnp= np.array(tempA)
    tempnp= tempnp.reshape(-1,1)
    tempdata= pd.DataFrame(tempnp)

    tempmax= tempAlgiers.loc[:,'Tmax']
    tempmin= tempAlgiers.loc[:,'Tmin']




    tempmx= np.random.random([tempmax.size*24,1])
    tempmn= np.random.random([tempmin.size*24,1])



    for x in range(0,tempmax.size):
        for y in range(0,24):
            tempmx[(x * 24) + y,0] = tempmax.iloc[x]

    for x in range(0,tempmin.size):
        for y in range(0,24):
            tempmn[(x * 24) + y,0] = tempmin.iloc[x]
        

    samples = daydata.index.size*24
    daydata2= daydata.copy()
    del(daydata2['cluster'])

    data= pd.DataFrame(np.array(daydata2).reshape(-1,1))

    maxcons= data.values.max()
    mincons= data.values.min()

    maxtemp= np.max(tempdata.values)
    mintemp= tempdata.values.min()

    maxtempmax= np.max(tempmx)
    mintempmax= np.min(tempmx)

    maxtempmin= np.max(tempmn)
    mintempmin= np.min(tempmn)

    sigxx= np.empty((samples - 168 , nbrInputs))
    sigyy= np.empty((samples - 168 , 1))

    i= 0
    for x in list(range(168,samples)):
        i=0
        if previousHours == True: 
            sigxx[x - 168 , i] = (data.iloc[x - 1 , 0])/(2*maxcons)
            i= i+ 1
        if previousDay == True: 
            sigxx[x - 168 , i] = (data.iloc[x - 24 , 0])/(2*maxcons)
            i= i+1
        if previousWeek == True : 
            sigxx[x - 168 , i] = (data.iloc[x - 168 , 0])/(2*maxcons)
            i= i+ 1
        if temp == True: 
            sigxx[x - 168 , i] = (tempdata.iloc[x])/(2*maxtemp)
            i= i+1
        if tempMax == True: 
            sigxx[x - 168 , i] = (tempmx[x])/(2*maxtempmax)
            i= i+1
        if tempMin == True: 
            sigxx[x - 168 , i] = (tempmn[x])/(2*maxtempmin)
            i= i+1
        if dayIndicator == True:
            ind=0
            for y in range(0,7):
                sigxx[x - 168 , i+ind]= 0
                ind = ind + 1
            sigxx[x - 168 , i+pd.datetime.weekday(hourdataindex.index[x])]=1

    
    for x in list(range(168,samples)):
        sigyy[x - 168 , 0]= (data.iloc[x , 0])/(2*maxcons)

    sigmoidxx= df(sigxx.copy())
    sigmoidyy= df(sigyy.copy())

    sigmoidxx.index= pd.date_range('2014-1-8 00:00:00', periods=(365-7)*24, freq='H')
    sigmoidyy.index= pd.date_range('2014-1-8 00:00:00', periods=(365-7)*24, freq='H')

    sigmoidxx['cluster'] = hourclusters[168:]
    sigmoidyy['cluster'] = hourclusters[168:]
    dfhourclusters = df(hourclusters)
    
    temp1= sigmoidyy[sigmoidyy.cluster==0]
    temp2= sigmoidyy[sigmoidyy.cluster==1]
    temp3= sigmoidyy[sigmoidyy.cluster==2]

    if season == 'summer':
        if temp1.index[0] == pd.datetime(2014,4,9,0,0,0): SigmoidInputs =sigmoidxx[sigmoidxx.cluster==0].copy()
        elif temp2.index[0] == pd.datetime(2014,4,9,0,0,0): SigmoidInputs =sigmoidxx[sigmoidxx.cluster==1].copy()
        elif temp3.index[0] == pd.datetime(2014,4,9,0,0,0): SigmoidInputs =sigmoidxx[sigmoidxx.cluster==2].copy()
    elif season == 'winter':
        if temp1.index[0] == pd.datetime(2014,1,8,0,0,0): SigmoidInputs =sigmoidxx[sigmoidxx.cluster==0].copy()
        elif temp2.index[0] == pd.datetime(2014,1,8,0,0,0): SigmoidInputs =sigmoidxx[sigmoidxx.cluster==1].copy()
        elif temp3.index[0] == pd.datetime(2014,1,8,0,0,0): SigmoidInputs =sigmoidxx[sigmoidxx.cluster==2].copy()
    elif season == 'spring and autumn':
        if temp1.index[0] == pd.datetime(2014,3,18,0,0,0): SigmoidInputs =sigmoidxx[sigmoidxx.cluster==0].copy()
        elif temp2.index[0] == pd.datetime(2014,3,18,0,0,0): SigmoidInputs =sigmoidxx[sigmoidxx.cluster==1].copy()
        elif temp3.index[0] == pd.datetime(2014,3,18,0,0,0): SigmoidInputs =sigmoidxx[sigmoidxx.cluster==2].copy()
    
    SigmoidOutputs= sigmoidyy[sigmoidyy.cluster==SigmoidInputs.loc[SigmoidInputs.index[0],'cluster']]
    del(SigmoidInputs['cluster'],SigmoidOutputs['cluster'])
    
    learningoutputs = pd.DataFrame(SigmoidOutputs.iloc[:int(SigmoidOutputs.size-168)].values.copy(),
                            index=SigmoidOutputs.iloc[:int(SigmoidOutputs.size-168)].index)
    testoutputs = pd.DataFrame(SigmoidOutputs.iloc[int(SigmoidOutputs.size-168):].values.copy(),
                            index=SigmoidOutputs.iloc[int(SigmoidOutputs.size-168):].index)

    learninginputs = pd.DataFrame(SigmoidInputs.iloc[:int(SigmoidOutputs.size-168)].values.copy(),
                            index=SigmoidOutputs.iloc[:int(SigmoidOutputs.size-168)].index)
    testinputs = pd.DataFrame(SigmoidInputs.iloc[int(SigmoidOutputs.size-168):].values.copy(),
                            index=SigmoidOutputs.iloc[int(SigmoidOutputs.size-168):].index)

    print('-------Input preparation process complet-------')
    return learninginputs, learningoutputs, testinputs, testoutputs, nbrInputs
    
def create_TrainModel(inputx1,inputx2,outputx1,outputx2,nbrInputs, epoch):
    mlpmodel = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    mlpmodel.add(Dense(13, input_dim=nbrInputs,activation='sigmoid'))
    mlpmodel.add(Dense(10, init='uniform'))
    mlpmodel.add(Dense(1, init='uniform'))
    mlpmodel.add(Activation('sigmoid'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    mlpmodel.compile(loss='mse', optimizer='Adam')

    tempMLPmodels= np.empty((epoch),dtype=object)
    tempErrorHistory= np.empty((epoch),dtype=object)

    counter=0
    for i in range(0,epoch):
        print('epoch:   %d .' %counter)
        ErrorHistory= mlpmodel.fit(X=inputx1, y=outputx1, nb_epoch=1,batch_size=12
                                   ,validation_data=(inputx2,outputx2),verbose= 2)
        tempMLPmodels[i]= mlpmodel
        tempErrorHistory[i]=ErrorHistory.history['val_loss']
        counter = counter +1
    return tempMLPmodels, tempErrorHistory

def create_TrainLSTM_Model(inputx1,inputx2,outputx1,outputx2,nbrInputs, epoch):
    
    model = Sequential()
    model.add(LSTM(15, return_sequences=True, input_shape=(1, nbrInputs)))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(10, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(1)) # output shape: (nb_samples, 10))
    optimizer = RMSprop(lr=0.01,clipnorm=10) 
    model.compile(optimizer='Adam', loss='mse')

    epoch= 80

    tempRNmodels= np.empty((epoch),dtype=object)
    tempErrorHistory= np.empty((epoch),dtype=object)

    counter=0
    for i in range(0,epoch):
        print('epoch:   %d .' %counter)
        ErrorHistory= model.fit(X=inputx1, y=outputx1, nb_epoch=1,batch_size=12,validation_data=(inputx2,outputx2))
        tempRNmodels[i]= model
        tempErrorHistory[i]=ErrorHistory.history['val_loss']
        counter = counter +1

    return tempMLPmodels, tempErrorHistory

def getWinterForecastingResults(learninginputs,learningoutputs,testinputs,testoutputs,tempMLPmodels, tempErrorHistory, nbrInputs, IsShift,previoushour,season):
    
    arg= tempErrorHistory.argmin()
    print(arg)
    
    predictionPeriod = learninginputs.index.size-168
    temp=learninginputs.iloc[:predictionPeriod].resample('D').sum()
    temp= temp.dropna()

    LearningForecastingError=pd.DataFrame(np.random.randn((predictionPeriod)/24, 1),index= temp.index)

    learningInputs= learninginputs.iloc[:learningoutputs.index.size-168].copy()
    learningOutputs= learningoutputs.iloc[:learningoutputs.index.size-168].copy()
    LearningInputs_24= learninginputs.iloc[:learninginputs.index.size-168].copy()


    print('===========================================================================')
    print('\t \t learning subset results')
    print('===========================================================================')

    for y in range(0,predictionPeriod/24-1):

        dailyError = 0

        for x in range(0,24):

            if IsShift == True:
                Testforecast=tempMLPmodels[arg].predict(LearningInputs_24.iloc[x + (y * 24) + 1].reshape(-1,nbrInputs))
                forecast=tempMLPmodels[arg].predict(learningInputs.iloc[x + (y * 24) + 1].reshape(-1,nbrInputs))
            else :
                Testforecast=tempMLPmodels[arg].predict(LearningInputs_24.iloc[x + (y * 24)].reshape(-1,nbrInputs))
                forecast=tempMLPmodels[arg].predict(learningInputs.iloc[x + (y * 24)].reshape(-1,nbrInputs))
                
            dailyError = dailyError + np.abs(((learningOutputs.iloc[x + (y * 24),0] - Testforecast[0]) / 
                                              learningOutputs.iloc[x + (y * 24),0]))
            if previoushour == True: 
                if IsShift == True: 
                    if x + 2 < 24: LearningInputs_24.iloc[(x + (y * 24)) + 2,0] = forecast[0]
                else :
                    if x + 1 < 24: LearningInputs_24.iloc[(x + (y * 24)) + 1,0] = forecast[0]


        dailyError = (dailyError / 24) * 100

        LearningForecastingError.iloc[y,0] = dailyError

    
    LearningMAPE = LearningForecastingError.mean()
    print('Daily Learning mean absolute percentage error: %f ' %LearningMAPE[0])

    fig = plt.figure(figsize=(12,3))
    plt.plot(LearningForecastingError,label='Error')

    if season == 'winter':
        dstart='2014-12-25 00:00:00'
        dend = '2014-12-30 23:00:00'
        dend2= '2014-12-31 00:00:00'
    elif season == 'spring and autumn':
        dstart='2014-11-28 00:00:00'
        dend = '2014-12-03 23:00:00'
        dend2 = '2014-12-04 00:00:00'
    elif season == 'summer':
        dstart='2014-10-29 00:00:00'
        dend = '2014-11-02 23:00:00'
        dend2 = '2014-11-03 00:00:00'
        
    testperiod=len(pd.date_range(dstart,dend,freq='H'))


    forecasting_history=pd.DataFrame(np.random.randn(testperiod, 2),columns=['real', 'forecast']
                                     ,index= pd.date_range(dstart,dend,freq='H'))

    validationInputs= testinputs.loc[dstart:dend2,:].copy()
    validationOutputs= testoutputs.loc[dstart:dend,:].copy()

    for x in pd.date_range(dstart,dend,freq='H'):
        
        if IsShift == True:
            forecast=tempMLPmodels[arg].predict(validationInputs.loc[x+ pd.DateOffset(hours=1)].reshape(-1,nbrInputs))
        else :
            forecast=tempMLPmodels[arg].predict(validationInputs.loc[x+ pd.DateOffset(hours=0)].reshape(-1,nbrInputs))
        
        forecasting_history.loc[x,'forecast']= forecast[0][0]
        forecasting_history.loc[x,'real']= testoutputs.loc[x,0]
        
        if previoushour == True: 
            if IsShift == True:
                if (pd.Timedelta(x-(x + pd.DateOffset(hours=2))).seconds/3600) + 1 < 168 - 1: 
                    validationInputs.loc[x + pd.DateOffset(hours=2),0] = forecast[0]
            else :
                if (pd.Timedelta(x-(x + pd.DateOffset(hours=1))).seconds/3600) + 1 < 168 - 1: 
                    validationInputs.loc[x + pd.DateOffset(hours=1),0] = forecast[0]



    testRMSS = rmse(forecasting_history.iloc[:,0], forecasting_history.iloc[:,1])
    testMAPE= mean_absolute_percentage_error(forecasting_history.iloc[:,0], forecasting_history.iloc[:,1])
    
    print('===========================================================================')
    print('\t \t Test subset results')
    print('===========================================================================')


    print('Test mean squared error: %f' %testRMSS)
    print('Test mean absolute percentage error: %f ' %testMAPE)



    fig = plt.figure(figsize=(12,4))
    plt.plot(forecasting_history[:testperiod], label=['real','forecast'])
    plt.legend(labels=['real','forecast'],loc= 'best')
    plt.show()
    
    return LearningMAPE, testMAPE






