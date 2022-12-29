# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 19:36:19 2018

@author: Kenny
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import copy

def groupDf(dataFrame):
    '''Group dataframe to separate web, hotel, and gds data'''
    
    groupFrame = dataFrame.groupby(by='Distribution_Channel')
    dfWeb = groupFrame.get_group('WEB')
    dfHotel = groupFrame.get_group('CRO/Hotel')
    dfGDS = groupFrame.get_group('GDS')
    
    return dfWeb,dfHotel,dfGDS

def summaryStats(name,web,hotel,gds):
    '''Prints the mean advance purchase values'''
    
    print(name,':\t','Mean:\t\t\t','Count:')
    print('Web:\t',web['Advance_Purchase'].mean(),'\t',web['Advance_Purchase'].count())
    print('Hotel:\t',hotel['Advance_Purchase'].mean(),'\t',hotel['Advance_Purchase'].count())
    print('GDS:\t',gds['Advance_Purchase'].mean(),'\t',gds['Advance_Purchase'].count(),'\n')
    
def myOneHotMultiEncoder(mydf, colNameStr):
    '''Turns a categorical value with only two options into a binary column'''
    
    count=0
    dfNew = pd.get_dummies(mydf[colNameStr],drop_first=False)
    for colname in dfNew.columns.values:
        mydf[colNameStr + '_' + str(colname)]=dfNew.iloc[:,count]
        count+=1        
    mydf.drop(colNameStr,axis=1, inplace=True)
    
def myOneHotSingleEncoder(mydf, colNameStr):
    '''For categorical data, creates a column for each category option'''
    
    count=0
    dfNew = pd.get_dummies(mydf[colNameStr],drop_first=False)
    for colname in dfNew.columns.values:
        mydf[colNameStr + '_' + str(colname)]=dfNew.iloc[:,count]
        count+=1        
    mydf.drop([colNameStr,colNameStr + '_' + str(colname)],axis=1, inplace=True)

def replaceNan(mydf):
    '''Replaces Nan values with mean values and value counts'''
    
    for colName in list(mydf):
        if type(mydf[colName][1]) is int or float:
            mydf[colName].fillna(round(mydf[colName].mean(),0),inplace=True)
        elif type(mydf[colName][1]) is str:
            mydf[colName].fillna('Unknown',inplace=True)
        else:
            mydf[colName].fillna(mydf[colName].value_counts().idxmax(),inplace=True)


def cleanMyHotel(myDf):
    '''Drops columns without any predictive value and one hot encodes
    columns with categorical values with predictive value. Output is a 
    clean dataframe with X attributes and two Y target dataframes in 
    two different formats'''
    
    myDf.drop(['Room_Type','Hotel_ID','Booking_ID','Product_ID','Purchased_Product','Booking_Date','Check_In_Date','Check_Out_Date','Purchased_Rate_Code','Rate_Description','Enrollment_Date','VIP_Enrollment_Date'],axis=1,inplace=True)
    myOneHotMultiEncoder(myDf,'Distribution_Channel')
    myOneHotMultiEncoder(myDf,'Purchased_Room_Type')
    myOneHotMultiEncoder(myDf,'Membership_Status')
    myOneHotMultiEncoder(myDf,'VIP_Membership_Status')
    myDf.dropna(inplace=True)
    dfRateCode = pd.DataFrame(myDf['Rate_Code'])
    dfRateCodeMulti = pd.DataFrame(myDf['Rate_Code'])
    myOneHotMultiEncoder(dfRateCodeMulti,'Rate_Code')
    return myDf,dfRateCode, dfRateCodeMulti

def rateGrouping(hotel):
    '''Groups the data for each hotel by their rate codes'''
    
    dfGrouped = hotel.groupby('Rate_Code')
    rateList = []
    try:
        rate1 = dfGrouped.get_group('Rate 1')
        rateList += [rate1]
        rate2 = dfGrouped.get_group('Rate 2')
        rateList += [rate2]
        rate3 = dfGrouped.get_group('Rate 3')
        rateList += [rate3]
        rate4 = dfGrouped.get_group('Rate 4')
        rateList += [rate4]
        rate5 = dfGrouped.get_group('Rate 5')
        rateList += [rate5]
        rate6 = dfGrouped.get_group('Rate 6')
        rateList += [rate6]
        rate7 = dfGrouped.get_group('Rate 7')
        rateList += [rate7]
        rate8 = dfGrouped.get_group('Rate 8')
        rateList += [rate8]
    except:
        pass
    dfHotel = []
    
    for i in rateList:
        dfHotel += [i['Nightly_Rate'].mean()]
        
    dfHotel = pd.DataFrame(dfHotel)
    dfHotel = np.array(dfHotel)
    return dfHotel

def randForestResults(randForestBinResult):
    '''Puts the random forest results into a format that can be used
    to represent the probability a customer selects a certain rate code'''
    
    colNames = []
    dfRandForest = randForestBinResult[0][:,1]
    dfRandForest = pd.DataFrame(dfRandForest)
    for i in range(1,len(randForestBinResult)):
        dfRandForest[i] = randForestBinResult[i][:,1]
    for j in range(1,len(randForestBinResult)+1):
        colNames += ['Rate_Code'+str(j)]
    dfRandForest2 = pd.DataFrame(dfRandForest.values,columns=colNames)
    return dfRandForest2

def profitDif(dfExpectedVal):
    '''Finds the difference in expected revenue between the maximum expected 
    revenue of all possible rate codes and the expected revenue from
    the purchased rate code'''
    
    profitDif = 0
    name = ''
    for idx, row in dfExpectedVal.iterrows():
        for colName in dfExpectedVal.columns.values:
            if max(row[:-1]) == row[colName]:
                name = row[-1]
        name = name[:4] + '_Code' + name[5:]
        profitDif += (max(row[:-1] - row[name]))
    return profitDif
    
def buckets(dfExpectedVal):
    '''Bins each rate code based on purchased rate code and the customer's
    highest expected value rate code'''
    
    actual1, actual2, actual3, actual4, actual5, actual6, actual7, actual8 = 0,0,0,0,0,0,0,0
    from1to1,from1to2,from1to3,from1to4,from1to5,from1to6,from1to7,from1to8 = 0,0,0,0,0,0,0,0
    from2to1,from2to2,from2to3,from2to4,from2to5,from2to6,from2to7,from2to8 = 0,0,0,0,0,0,0,0
    from3to1,from3to2,from3to3,from3to4,from3to5,from3to6,from3to7,from3to8 = 0,0,0,0,0,0,0,0
    from4to1,from4to2,from4to3,from4to4,from4to5,from4to6,from4to7,from4to8 = 0,0,0,0,0,0,0,0
    from5to1,from5to2,from5to3,from5to4,from5to5,from5to6,from5to7,from5to8 = 0,0,0,0,0,0,0,0
    from6to1,from6to2,from6to3,from6to4,from6to5,from6to6,from6to7,from6to8 = 0,0,0,0,0,0,0,0
    from7to1,from7to2,from7to3,from7to4,from7to5,from7to6,from7to7,from7to8 = 0,0,0,0,0,0,0,0
    from8to1,from8to2,from8to3,from8to4,from8to5,from8to6,from8to7,from8to8 = 0,0,0,0,0,0,0,0
    for idx,row in dfExpectedVal.iterrows():
        if row[-1] == 'Rate 1':
            actual1 += 1
            if row[0] == max(row[:-1]):
                from1to1 += 1
            elif row[1] == max(row[:-1]):
                from1to2 += 1
            elif row[2] == max(row[:-1]):
                from1to3 += 1
            elif row[3] == max(row[:-1]):
                from1to4 += 1
            elif row[4] == max(row[:-1]):
                from1to5 += 1
            elif row[5] == max(row[:-1]):
                from1to6 += 1
            elif row[6] == max(row[:-1]):
                from1to7 += 1
            elif row[7] == max(row[:-1]):
                from1to8 += 1
        elif row[-1] == 'Rate 2':
            actual2 += 1
            if row[0] == max(row[:-1]):
                from2to1 += 1
            elif row[1] == max(row[:-1]):
                from2to2 += 1
            elif row[2] == max(row[:-1]):
                from2to3 += 1
            elif row[3] == max(row[:-1]):
                from2to4 += 1
            elif row[4] == max(row[:-1]):
                from2to5 += 1
            elif row[5] == max(row[:-1]):
                from2to6 += 1
            elif row[6] == max(row[:-1]):
                from2to7 += 1
            elif row[7] == max(row[:-1]):
                from2to8 += 1
        elif row[-1] == 'Rate 3':
            actual3 += 1   
            if row[0] == max(row[:-1]):
                from3to1 += 1
            elif row[1] == max(row[:-1]):
                from3to2 += 1
            elif row[2] == max(row[:-1]):
                from3to3 += 1
            elif row[3] == max(row[:-1]):
                from3to4 += 1
            elif row[4] == max(row[:-1]):
                from3to5 += 1
            elif row[5] == max(row[:-1]):
                from3to6 += 1
            elif row[6] == max(row[:-1]):
                from3to7 += 1
            elif row[7] == max(row[:-1]):
                from3to8 += 1
        elif row[-1] == 'Rate 4':
            actual4 += 1
            if row[0] == max(row[:-1]):
                from4to1 += 1
            elif row[1] == max(row[:-1]):
                from4to2 += 1
            elif row[2] == max(row[:-1]):
                from4to3 += 1
            elif row[3] == max(row[:-1]):
                from4to4 += 1
            elif row[4] == max(row[:-1]):
                from4to5 += 1
            elif row[5] == max(row[:-1]):
                from4to6 += 1
            elif row[6] == max(row[:-1]):
                from4to7 += 1
            elif row[7] == max(row[:-1]):
                from4to8 += 1
        elif row[-1] == 'Rate 5':
            actual5 += 1
            if row[0] == max(row[:-1]):
                from5to1 += 1
            elif row[1] == max(row[:-1]):
                from5to2 += 1
            elif row[2] == max(row[:-1]):
                from5to3 += 1
            elif row[3] == max(row[:-1]):
                from5to4 += 1
            elif row[4] == max(row[:-1]):
                from5to5 += 1
            elif row[5] == max(row[:-1]):
                from5to6 += 1
            elif row[6] == max(row[:-1]):
                from5to7 += 1
            elif row[7] == max(row[:-1]):
                from5to8 += 1
        elif row[-1] == 'Rate 6':
            actual6 += 1
            if row[0] == max(row[:-1]):
                from6to1 += 1
            elif row[1] == max(row[:-1]):
                from6to2 += 1
            elif row[2] == max(row[:-1]):
                from6to3 += 1
            elif row[3] == max(row[:-1]):
                from6to4 += 1
            elif row[4] == max(row[:-1]):
                from6to5 += 1
            elif row[5] == max(row[:-1]):
                from6to6 += 1
            elif row[6] == max(row[:-1]):
                from6to7 += 1
            elif row[7] == max(row[:-1]):
                from6to8 += 1
        elif row[-1] == 'Rate 7':
            actual7 += 1
            if row[0] == max(row[:-1]):
                from7to1 += 1
            elif row[1] == max(row[:-1]):
                from7to2 += 1
            elif row[2] == max(row[:-1]):
                from7to3 += 1
            elif row[3] == max(row[:-1]):
                from7to4 += 1
            elif row[4] == max(row[:-1]):
                from7to5 += 1
            elif row[5] == max(row[:-1]):
                from7to6 += 1
            elif row[6] == max(row[:-1]):
                from7to7 += 1
            elif row[7] == max(row[:-1]):
                from7to8 += 1
        elif row[-1] == 'Rate 8':
            actual8 += 1
            if row[0] == max(row[:-1]):
                from8to1 += 1
            elif row[1] == max(row[:-1]):
                from8to2 += 1
            elif row[2] == max(row[:-1]):
                from8to3 += 1
            elif row[3] == max(row[:-1]):
                from8to4 += 1
            elif row[4] == max(row[:-1]):
                from8to5 += 1
            elif row[5] == max(row[:-1]):
                from8to6 += 1
            elif row[6] == max(row[:-1]):
                from8to7 += 1
            elif row[7] == max(row[:-1]):
                from8to8 += 1
                
    
    actualList = [actual1, actual2, actual3, actual4, actual5, actual6, actual7, actual8]
    oneList = [from1to1,from1to2,from1to3,from1to4,from1to5,from1to6,from1to7,from1to8]
    twoList = [from2to1,from2to2,from2to3,from2to4,from2to5,from2to6,from2to7,from2to8]
    threeList = [from3to1,from3to2,from3to3,from3to4,from3to5,from3to6,from3to7,from3to8]
    fourList = [from4to1,from4to2,from4to3,from4to4,from4to5,from4to6,from4to7,from4to8]
    fiveList = [from5to1,from5to2,from5to3,from5to4,from5to5,from5to6,from5to7,from5to8]
    sixList = [from6to1,from6to2,from6to3,from6to4,from6to5,from6to6,from6to7,from6to8]
    sevenList = [from7to1,from7to2,from7to3,from7to4,from7to5,from7to6,from7to7,from7to8]
    eightList = [from8to1,from8to2,from8to3,from8to4,from8to5,from8to6,from8to7,from8to8]
    
    hotelList = [actualList,oneList,twoList,threeList,fourList,fiveList,sixList,sevenList,eightList]
    hotelList = copy.deepcopy(hotelList)
    return hotelList



#Phase One



#Import the hotel data
h1 = pd.read_csv('Bodea - Choice based Revenue Management - Data Set - Hotel 1.csv')
h2 = pd.read_csv('Bodea - Choice based Revenue Management - Data Set - Hotel 2.csv')
h3 = pd.read_csv('Bodea - Choice based Revenue Management - Data Set - Hotel 3.csv')
h4 = pd.read_csv('Bodea - Choice based Revenue Management - Data Set - Hotel 4.csv')
h5 = pd.read_csv('Bodea - Choice based Revenue Management - Data Set - Hotel 5.csv')

#We only want data for the rooms that were actually booked
h1 = h1[h1.Purchased_Product != 0]
h2 = h2[h2.Purchased_Product != 0]
h3 = h3[h3.Purchased_Product != 0]
h4 = h4[h4.Purchased_Product != 0]
h5 = h5[h5.Purchased_Product != 0]

#Concatenating all frames into one dataframe
frames = [h1,h2,h3,h4,h5]
dfMerged = pd.concat(frames)

#heatmap
import seaborn as sns 
sns.set(style='white')
sns.set(style='whitegrid',color_codes=True)
ax = sns.heatmap(dfMerged.corr()) 

#summary statistics for hotel data
dfWeb,dfHotel,dfGDS = groupDf(dfMerged)
summaryStats('dfMerged',dfWeb,dfHotel,dfGDS)

h1Web,h1Hotel,h1GDS = groupDf(h1)
summaryStats('h1',h1Web,h1Hotel,h1GDS)

h2Web,h2Hotel,h2GDS = groupDf(h2)
summaryStats('h2',h2Web,h2Hotel,h2GDS)

h3Web,h3Hotel,h3GDS = groupDf(h3)
summaryStats('h3',h3Web,h3Hotel,h3GDS)

h4Web,h4Hotel,h4GDS = groupDf(h4)
summaryStats('h4',h4Web,h4Hotel,h4GDS)

h5Web,h5Hotel,h5GDS = groupDf(h5)
summaryStats('h5',h5Web,h5Hotel,h5GDS)

#drop columns without prediction value
dfMerged.drop(['Room_Type','Hotel_ID','Booking_ID','Product_ID','Purchased_Product','Booking_Date','Check_In_Date','Check_Out_Date','Purchased_Rate_Code','Rate_Description','Enrollment_Date','VIP_Enrollment_Date'],axis=1,inplace=True)

#onehotencode columns with categorical data
myOneHotMultiEncoder(dfMerged,'Distribution_Channel')
myOneHotMultiEncoder(dfMerged,'Purchased_Room_Type')
myOneHotMultiEncoder(dfMerged,'Membership_Status')
myOneHotMultiEncoder(dfMerged,'VIP_Membership_Status')

#drop data that is missing values
dfMerged.dropna(inplace=True)

#grouping the data by their rate codes
dfGrouped = dfMerged.groupby('Rate_Code')
rate1 = dfGrouped.get_group('Rate 1')
rate2 = dfGrouped.get_group('Rate 2')
rate3 = dfGrouped.get_group('Rate 3')
rate4 = dfGrouped.get_group('Rate 4')
rate5 = dfGrouped.get_group('Rate 5')
rate6 = dfGrouped.get_group('Rate 6')
rate7 = dfGrouped.get_group('Rate 7')
rate8 = dfGrouped.get_group('Rate 8')

#average total revenue for each rate code
print('Rate 1', rate1['Total_Revenue'].mean())
print('Rate 2', rate2['Total_Revenue'].mean())
print('Rate 3', rate3['Total_Revenue'].mean())
print('Rate 4', rate4['Total_Revenue'].mean())
print('Rate 5', rate5['Total_Revenue'].mean())
print('Rate 6', rate6['Total_Revenue'].mean())
print('Rate 7', rate7['Total_Revenue'].mean())
print('Rate 8', rate8['Total_Revenue'].mean())

#average nightly rate for each rate code
print('Rate 1', rate1['Nightly_Rate'].mean())
print('Rate 2', rate2['Nightly_Rate'].mean())
print('Rate 3', rate3['Nightly_Rate'].mean())
print('Rate 4', rate4['Nightly_Rate'].mean())
print('Rate 5', rate5['Nightly_Rate'].mean())
print('Rate 6', rate6['Nightly_Rate'].mean())
print('Rate 7', rate7['Nightly_Rate'].mean())
print('Rate 8', rate8['Nightly_Rate'].mean())

#count of the number of bookings in each rate code
print('Rate 1', rate1['Merge_Indicator'].count())
print('Rate 2', rate2['Merge_Indicator'].count())
print('Rate 3', rate3['Merge_Indicator'].count())
print('Rate 4', rate4['Merge_Indicator'].count())
print('Rate 5', rate5['Merge_Indicator'].count())
print('Rate 6', rate6['Merge_Indicator'].count())
print('Rate 7', rate7['Merge_Indicator'].count())
print('Rate 8', rate8['Merge_Indicator'].count())


#creating a df for the rate codes in one column
dfRateCode = pd.DataFrame(dfMerged['Rate_Code'])

#onehotencoding the rate codes
myOneHotMultiEncoder(dfMerged,'Rate_Code')

#creating a df for the rate codes as binary categorical values
dfMergedY = dfMerged.iloc[:,-8:].copy()

#remove the rate code categories from the main df
dfMerged.drop(list(dfMergedY),axis=1,inplace=True)  


#Running algorithms to find the probabilities that a customer will accept each rate code                                                                                                #if room_type == purchased_room_type


#train/test split
xTrain,xTest = train_test_split(dfMerged, test_size=0.3, random_state=56)
yTrain,yTest = train_test_split(dfMergedY, test_size=0.3, random_state=56)

#Random Forest
randForestClf = RandomForestClassifier(n_estimators=1000, criterion='entropy',max_depth=20,random_state=56) 
randForestClf.fit(xTrain,yTrain)
randForestBinResult = randForestClf.predict_proba(xTest)

#getting the random forest results
dfRandForest = randForestBinResult[0][:,1]
dfRandForest = pd.DataFrame(dfRandForest)
for i in range(1,8):
    dfRandForest[i] = randForestBinResult[i][:,1]
dfRandForest.sum(axis=1)

randForestClf.score(xTest,yTest)

#Uncomment the following code to view the optimization process
#It is commented out because it takes a while to run
#for depth in np.arange(1,51):
#    randForestClf = RandomForestClassifier(n_estimators=1000, criterion='entropy',max_depth=depth,random_state=56) 
#    randForestClf.fit(xTrain,yTrain)
#    randForestBinResult = randForestClf.predict_proba(xTest)
#    dfRandForest = randForestBinResult[0][:,1]
#    dfRandForest = pd.DataFrame(dfRandForest)
#    for i in range(1,8):
#        dfRandForest[i] = randForestBinResult[i][:,1]
#    dfRandForest.sum(axis=1)
#    
#    print([depth,randForestClf.score(xTest,yTest)])

#Neural Network
MLPclf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(20,8), random_state=56)
MLPclf.fit(xTrain, yTrain) 
mlpBinResult = MLPclf.predict_proba(xTest)

MLPclf.score(xTest,yTest) #I don't like our results

#KNN
knnModel = KNeighborsClassifier(n_neighbors=13)
knnModel.fit(xTrain,yTrain)
knnBinResult = knnModel.predict_proba(xTest)

knnModel.score(xTest,yTest)



#Phase Two



#Import the hotel data
h1 = pd.read_csv('Bodea - Choice based Revenue Management - Data Set - Hotel 1.csv')
h2 = pd.read_csv('Bodea - Choice based Revenue Management - Data Set - Hotel 2.csv')
h3 = pd.read_csv('Bodea - Choice based Revenue Management - Data Set - Hotel 3.csv')
h4 = pd.read_csv('Bodea - Choice based Revenue Management - Data Set - Hotel 4.csv')
h5 = pd.read_csv('Bodea - Choice based Revenue Management - Data Set - Hotel 5.csv')

#We only want data for the rooms that were actually booked
h1 = h1[h1.Purchased_Product != 0]
h2 = h2[h2.Purchased_Product != 0]
h3 = h3[h3.Purchased_Product != 0]
h4 = h4[h4.Purchased_Product != 0]
h5 = h5[h5.Purchased_Product != 0]

#mean nightly rate for each hotel rate code
h1Mean = rateGrouping(h1)
h2Mean = rateGrouping(h2)
h3Mean = rateGrouping(h3)
h4Mean = rateGrouping(h4)
h5Mean = rateGrouping(h5)

#cleaning the hotel data
dfH1,dfRateCodeH1,dfRateCodeH1Multi = cleanMyHotel(h1)
dfH2,dfRateCodeH2,dfRateCodeH2Multi = cleanMyHotel(h2)
dfH3,dfRateCodeH3,dfRateCodeH3Multi = cleanMyHotel(h3)
dfH4,dfRateCodeH4,dfRateCodeH4Multi = cleanMyHotel(h4)
dfH5,dfRateCodeH5,dfRateCodeH5Multi = cleanMyHotel(h5)


#train/test split


#Hotel 1
xH1Train,xH1Test = train_test_split(dfH1, test_size=0.3, random_state=56)
yH1Train,yH1Test = train_test_split(dfRateCodeH1Multi, test_size=0.3, random_state=56)

#Hotel 2
xH2Train,xH2Test = train_test_split(dfH2, test_size=0.3, random_state=56)
yH2Train,yH2Test = train_test_split(dfRateCodeH2Multi, test_size=0.3, random_state=56)

#Hotel 3
xH3Train,xH3Test = train_test_split(dfH3, test_size=0.3, random_state=56)
yH3Train,yH3Test = train_test_split(dfRateCodeH3Multi, test_size=0.3, random_state=56)

#Hotel 4
xH4Train,xH4Test = train_test_split(dfH4, test_size=0.3, random_state=56)
yH4Train,yH4Test = train_test_split(dfRateCodeH4Multi, test_size=0.3, random_state=56)

#Hotel 5
xH5Train,xH5Test = train_test_split(dfH5, test_size=0.3, random_state=56)
yH5Train,yH5Test = train_test_split(dfRateCodeH5Multi, test_size=0.3, random_state=56)


#Keep rate codes in index order


#Hotel 1
h1SplitCodes = xH1Test['Rate_Code']
xH1Test.drop(['Rate_Code'],axis=1,inplace=True)
xH1Train.drop(['Rate_Code'],axis=1,inplace=True)

#Hotel 2
h2SplitCodes = xH2Test['Rate_Code']
xH2Test.drop(['Rate_Code'],axis=1,inplace=True)
xH2Train.drop(['Rate_Code'],axis=1,inplace=True)

#Hotel 3
h3SplitCodes = xH3Test['Rate_Code']
xH3Test.drop(['Rate_Code'],axis=1,inplace=True)
xH3Train.drop(['Rate_Code'],axis=1,inplace=True)

#Hotel 4
h4SplitCodes = xH4Test['Rate_Code']
xH4Test.drop(['Rate_Code'],axis=1,inplace=True)
xH4Train.drop(['Rate_Code'],axis=1,inplace=True)

#Hotel 5
h5SplitCodes = xH5Test['Rate_Code']
xH5Test.drop(['Rate_Code'],axis=1,inplace=True)
xH5Train.drop(['Rate_Code'],axis=1,inplace=True)


#Random Forest predictions


#Hotel 1
randForestClf = RandomForestClassifier(n_estimators=1000, criterion='entropy',max_depth=20,random_state=56) 
randForestClf.fit(xH1Train,yH1Train)
randForestResultH1 = randForestClf.predict_proba(xH1Test)

randForestClf.score(xH1Test,yH1Test)
#.717 accuracy

#Hotel 2
randForestClf = RandomForestClassifier(n_estimators=1000, criterion='entropy',max_depth=20,random_state=56) 
randForestClf.fit(xH2Train,yH2Train)
randForestResultH2 = randForestClf.predict_proba(xH2Test)

randForestClf.score(xH2Test,yH2Test)
#.766 accuracy

#Hotel 3
randForestClf = RandomForestClassifier(n_estimators=1000, criterion='entropy',max_depth=20,random_state=56) 
randForestClf.fit(xH3Train,yH3Train)
randForestResultH3 = randForestClf.predict_proba(xH3Test)

randForestClf.score(xH3Test,yH3Test)
#.664 accuracy

#Hotel 4
randForestClf = RandomForestClassifier(n_estimators=1000, criterion='entropy',max_depth=20,random_state=56) 
randForestClf.fit(xH4Train,yH4Train)
randForestResultH4 = randForestClf.predict_proba(xH4Test)

randForestClf.score(xH4Test,yH4Test)
#.897 accuracy

#Hotel 5
randForestClf = RandomForestClassifier(n_estimators=1000, criterion='entropy',max_depth=20,random_state=56) 
randForestClf.fit(xH5Train,yH5Train)
randForestResultH5 = randForestClf.predict_proba(xH5Test)

randForestClf.score(xH5Test,yH5Test)
#.878 accuracy


#Random Forest Results
randForestResultH1 = randForestResults(randForestResultH1)
randForestResultH2 = randForestResults(randForestResultH2)
randForestResultH3 = randForestResults(randForestResultH3)
randForestResultH4 = randForestResults(randForestResultH4)
randForestResultH5 = randForestResults(randForestResultH5)


#Expected Values
dfExpectedValuesH1 = h1Mean.T*randForestResultH1
dfExpectedValuesH2 = h2Mean.T*randForestResultH2
dfExpectedValuesH3 = h3Mean.T*randForestResultH3
dfExpectedValuesH4 = h4Mean.T*randForestResultH4
dfExpectedValuesH5 = h5Mean.T*randForestResultH5


#Insert Rate Codes back into data


#Hotel 1
h1SplitCodes = pd.DataFrame(h1SplitCodes)
dfExpectedValuesH1['Rate_Code'] = h1SplitCodes.values

#Hotel 2
h2SplitCodes = pd.DataFrame(h2SplitCodes)
dfExpectedValuesH2['Rate_Code'] = h2SplitCodes.values

#Hotel 3
h3SplitCodes = pd.DataFrame(h3SplitCodes)
dfExpectedValuesH3['Rate_Code'] = h3SplitCodes.values

#Hotel 4
h4SplitCodes = pd.DataFrame(h4SplitCodes)
dfExpectedValuesH4['Rate_Code'] = h4SplitCodes.values

#Hotel 5
h5SplitCodes = pd.DataFrame(h5SplitCodes)
dfExpectedValuesH5['Rate_Code'] = h5SplitCodes.values

#Mean Expected Value for each rate code
dfExpectedValuesH1.mean()
dfExpectedValuesH2.mean()
dfExpectedValuesH3.mean()
dfExpectedValuesH4.mean()
dfExpectedValuesH5.mean()

#Expected value differences for each hotel    
print('Expected Value Differences for Each Hotel:')
print(profitDif(dfExpectedValuesH1))
print(profitDif(dfExpectedValuesH2))
print(profitDif(dfExpectedValuesH3))
print(profitDif(dfExpectedValuesH4))
print(profitDif(dfExpectedValuesH5))

#Average expected value increase per customer
print('Average Expected Value Increase per Customer:')
print(profitDif(dfExpectedValuesH1) / len(dfExpectedValuesH1))
print(profitDif(dfExpectedValuesH2) / len(dfExpectedValuesH2))
print(profitDif(dfExpectedValuesH3) / len(dfExpectedValuesH3))
print(profitDif(dfExpectedValuesH4) / len(dfExpectedValuesH4))
print(profitDif(dfExpectedValuesH5) / len(dfExpectedValuesH5))

#Sorting non-Rate 2 customers into buckets
h1BucketList = buckets(dfExpectedValuesH1)
h2BucketList = buckets(dfExpectedValuesH2)
h3BucketList = buckets(dfExpectedValuesH3)
h4BucketList = buckets(dfExpectedValuesH4)
h5BucketList = buckets(dfExpectedValuesH5)
