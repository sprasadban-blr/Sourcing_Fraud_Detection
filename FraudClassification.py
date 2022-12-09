import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import pickle

algoMap = {}

def load():
    classificationData = pd.read_csv('./output/classification_events_data.csv')
    clusterdEventsData = pd.read_csv('./output/clustered_events_labels.csv')
    return (classificationData, clusterdEventsData)

def preprocess(classificationData, clusterdEventsData):
    awardedEvents = classificationData.query('EventParticipation_AwardedFlag == True')
    awardedEvents = awardedEvents.reset_index(drop=True)
    awardedEvents = awardedEvents.drop(columns=['EventParticipation_Id', 'EventParticipation_Creator',
       'EventParticipation_Bidder_UserName','EventParticipation_SupplierName',
       'EventParticipation_CommonSupplierName',
       'EventParticipation_CommonSupplierId', 'EventParticipation_SupplierType',
       'EventParticipation_Owner_UserName', 'EventParticipation_Owner_UserId',
       'EventParticipation_AcceptedFlag', 'EventParticipation_DeclinedFlag',
       'EventParticipation_IntendToRespondFlag',
       'EventParticipation_DeclinedToRespondFlag',
       'EventParticipation_ParticipatedFlag',
       'EventParticipation_BidsSubmitted'])
    nanStatevalues = awardedEvents[awardedEvents['EventParticipation_SupplierState'].isnull()]
    # Update state
    for index in nanStatevalues.index:
        awardedEvents.at[index, 'EventParticipation_SupplierState'] = 'Karnataka'
    
    # Add Supplier cluster participation details
    clusterdEventsData = clusterdEventsData.drop(columns=['labels'], axis=1)
    awardedEvents = awardedEvents.merge(clusterdEventsData, how='left', left_on=['EventId'], right_on=['EventId'])
    nanInvitedvalues = awardedEvents[awardedEvents['EventSummary_InvitedSuppliers'].isnull()]
    
    # Update participation rate default values for fraud entries
    fraudEmptyValues = nanInvitedvalues.query('fraud == 1')
    for index in fraudEmptyValues.index:
        awardedEvents.at[index, 'EventSummary_InvitedSuppliers'] = 68
        awardedEvents.at[index, 'EventSummary_ParticipSuppliers'] = 2
        awardedEvents.at[index, 'EventSummary_ParticipationRate'] = 2.9400
    
    nonFraudEmptyValues = awardedEvents[awardedEvents['EventSummary_InvitedSuppliers'].isnull()]
    # Update participation rate default values for non fraud entries
    for index in nonFraudEmptyValues.index:
        awardedEvents.at[index, 'EventSummary_InvitedSuppliers'] = 6
        awardedEvents.at[index, 'EventSummary_ParticipSuppliers'] = 6
        awardedEvents.at[index, 'EventSummary_ParticipationRate'] = 100.00
    
    awardedEvents.to_csv('./output/classification_awarded_events_data.csv', mode='w', header=True, encoding='utf-8', index=False)
    return awardedEvents

def updateUniqueKey(awardedEvents, colName, updateRows, startKeyIndex):
    colMap = {}
    colReverseMap = {}
    sequence = startKeyIndex

    for index in range(len(updateRows)):
        uniqueId = updateRows[index]
        colMap[uniqueId] = sequence
        colReverseMap[sequence] = uniqueId
        specificEvent = awardedEvents.query(colName + ' == "' + uniqueId + '"')
        for speicificIndex in specificEvent.index:
            awardedEvents.at[speicificIndex, colName] = sequence
        sequence += 1
    return (colMap, colReverseMap)

def encodeData(awardedEvents, colName, startKeyIndex):
    uniqueData = awardedEvents[colName].unique()
    results = updateUniqueKey(awardedEvents, colName, uniqueData, startKeyIndex)
    biddersKey = results[0]
    biddersReverseKey = results[1]
    return (awardedEvents, biddersKey, biddersReverseKey)

def balanceData(awardedEvents):
    X = awardedEvents.drop(['EventId', 'fraud'],axis=1)
    y = awardedEvents.fraud
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

    # Apply resampling to the training data only
    oversampler = SMOTE(random_state=0)
    X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)
    return(X, y, X_train, X_test, y_train, y_test, X_resampled, y_resampled)
    
def randomForest(X, y, X_train, X_test, y_train, y_test, X_resampled, y_resampled):
    model = RandomForestClassifier(random_state=0)
    model.fit(X_resampled, y_resampled)
    
    # Accuracy for training events
    y_train_pred = model.predict(X_train)
    trainingAccuracy = round(accuracy_score(y_train, y_train_pred) * 100, 2)
    print("Accuracy of training data = "+ str(trainingAccuracy))
    
    # Accuracy for resampled training events
    y_resampled_train_pred = model.predict(X_resampled)
    trainingUpSamplingAccuracy = round(accuracy_score(y_resampled, y_resampled_train_pred) * 100, 2)
    print("Accuracy of resampled training data = "+ str(trainingUpSamplingAccuracy))
    
    # Accuracy for test events
    y_test_pred = model.predict(X_test)
    testAccuracy = round(accuracy_score(y_test, y_test_pred) * 100, 2)
    print("Accuracy of test data = "+ str(testAccuracy))
    
    algoMap["RandomForest"] = (trainingAccuracy, trainingUpSamplingAccuracy, testAccuracy)
    
    return model

def persistModelPerformanceMetrics():
    # Persist model metrics and selection for UI
    row = 0
    modelP4Metrics = pd.DataFrame(columns=['Algorithm', 'Training Accuracy', 'UpSampling Training Accuracy', 'Test Accuracy'])
    for key in algoMap:
        algoTripple = algoMap[key]
        trainingAccuracy = round(algoTripple[0], 2)
        trainingUpSamplingAccuracy = round(algoTripple[1], 2)
        testAccuracy = round(algoTripple[1], 2)
        modelP4Metrics.loc[row] = [key, trainingAccuracy, trainingUpSamplingAccuracy, testAccuracy]
        row = row + 1
    modelP4Metrics.to_csv('./output/modelMetrics.csv', mode='w', header=True, encoding='utf-8', index=False)
    print("Model metrics file saved succesfully...")
    
def inference(inputDf):
    biddersMapFile = open('./output/bidders_map.dict', 'rb')
    biddersKey = pickle.load(biddersMapFile)
    biddersMapFile.close()
    
    supplierIdMapFile = open('./output/supplier_ids_map.dict', 'rb')
    suppliersIdKey = pickle.load(supplierIdMapFile)
    supplierIdMapFile.close()
    
    supplierCityMapFile = open('./output/supplier_city_map.dict', 'rb')
    supplierCitiesKey = pickle.load(supplierCityMapFile)
    supplierCityMapFile.close()
    
    supplierStateMapFile = open('./output/supplier_state_map.dict', 'rb')
    supplierStatesKey = pickle.load(supplierStateMapFile)
    supplierStateMapFile.close()
    
    supplierCountryMapFile = open('./output/supplier_country_map.dict', 'rb')
    supplierCountryKey = pickle.load(supplierCountryMapFile)
    supplierCountryMapFile.close()
    
    bidder = biddersKey[inputDf['EventParticipation_Bidder_UserId'][0]]
    supplier = suppliersIdKey[inputDf['EventParticipation_SupplierId'][0]]
    supplierCity = supplierCitiesKey[inputDf['EventParticipation_SupplierCity'][0]]
    supplierState = supplierStatesKey[inputDf['EventParticipation_SupplierState'][0]]
    supplierCountry = supplierCountryKey[inputDf['EventParticipation_SupplierCountry'][0]]
    awardedFlag = inputDf['EventParticipation_AwardedFlag'][0]
    invitedSuppliers = inputDf['EventSummary_InvitedSuppliers'][0]
    participSuppliers = inputDf['EventSummary_ParticipSuppliers'][0]
    participationRate = inputDf['EventSummary_ParticipationRate'][0]
    newModSample = [[bidder, supplier, supplierCity, supplierState, supplierCountry, awardedFlag, invitedSuppliers, participSuppliers, participationRate]]
    
    modelFile = open('./output/selected_ai_ml_model.mdl', 'rb')
    model = pickle.load(modelFile)
    modelFile.close()
    
    y_new_sample_pred = model.predict(newModSample)
    return y_new_sample_pred

def retrain():
    print("Work in progress...")
            

def main():
    data = load()
    awardedEvents = preprocess(data[0], data[1])
    
    # Encode bidders id
    results = encodeData(awardedEvents, 'EventParticipation_Bidder_UserId', 11)
    awardedEvents = results[0]
    biddersKey = results[1]
    biddersReverseKey = results[2]
    
    # Encode supplier id
    results = encodeData(awardedEvents, 'EventParticipation_SupplierId', 101)
    awardedEvents = results[0]
    suppliersIdKey = results[1]
    suppliersReverseIdKey = results[2]
    
    # Encode supplier city
    results = encodeData(awardedEvents, 'EventParticipation_SupplierCity', 201)
    awardedEvents = results[0]
    supplierCitiesKey = results[1]
    supplierCitiesReverseKey = results[2]
    
    # Encode supplier state
    results = encodeData(awardedEvents, 'EventParticipation_SupplierState', 301) 
    awardedEvents = results[0]
    supplierStatesKey = results[1]
    supplierStatesReverseKey = results[2]
    
     # Encode supplier country
    results = encodeData(awardedEvents, 'EventParticipation_SupplierCountry', 401)
    awardedEvents = results[0]
    supplierCountryKey = results[1]
    supplierCountryReverseKey = results[2]
    
    balData = balanceData(awardedEvents)
    
    X = balData[0]
    y = balData[1]
    X_train = balData[2]
    X_test = balData[3]
    y_train = balData[4]
    y_test = balData[5]
    X_resampled = balData[6]
    y_resampled = balData[7]
    
    model = randomForest(X, y, X_train, X_test, y_train, y_test, X_resampled, y_resampled)
    
    # Required during inference stage
    persistModelPerformanceMetrics()
    with open('./output/bidders_map.dict', 'wb') as fileHandle:
        pickle.dump(biddersKey, fileHandle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('./output/supplier_ids_map.dict', 'wb') as fileHandle:
        pickle.dump(suppliersIdKey, fileHandle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('./output/supplier_city_map.dict', 'wb') as fileHandle:
        pickle.dump(supplierCitiesKey, fileHandle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('./output/supplier_state_map.dict', 'wb') as fileHandle:
        pickle.dump(supplierStatesKey, fileHandle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('./output/supplier_country_map.dict', 'wb') as fileHandle:
        pickle.dump(supplierCountryKey, fileHandle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open('./output/selected_ai_ml_model.mdl', 'wb') as fileHandle:
        pickle.dump(model, fileHandle, protocol=pickle.HIGHEST_PROTOCOL)
    
    input = [['pseudotaggeds1', 'sid800', 'Chiyoda-ku', 'Tokyo', 'JP', True, 6.0, 6.0, 100.00]]
    inputDf = pd.DataFrame(input, columns=['EventParticipation_Bidder_UserId',
                                 'EventParticipation_SupplierId', 
                                 'EventParticipation_SupplierCity',
                                 'EventParticipation_SupplierState',
                                 'EventParticipation_SupplierCountry',	
                                 'EventParticipation_AwardedFlag',	
                                 'EventSummary_InvitedSuppliers',	
                                 'EventSummary_ParticipSuppliers',
                                 'EventSummary_ParticipationRate'])
    print(inputDf)
    print("Inference = "+ str(inference(inputDf)))


main()
    
    


    

