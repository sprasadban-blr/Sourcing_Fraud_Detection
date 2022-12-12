import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
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
    
    # Persist without encoding
    awardedEvents.to_csv('./output/classification_awarded_events_data.csv', mode='w', header=True, encoding='utf-8', index=False)
    
    # Encode bidders id
    results = encodeData(awardedEvents, 'EventParticipation_Bidder_UserId', 11)
    awardedEvents = results[0]
    biddersKey = results[1]
    biddersReverseKey = results[2]
    with open('./output/bidders_map.dict', 'wb') as fileHandle:
        pickle.dump(biddersKey, fileHandle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Encode supplier id
    results = encodeData(awardedEvents, 'EventParticipation_SupplierId', 101)
    awardedEvents = results[0]
    suppliersIdKey = results[1]
    suppliersReverseIdKey = results[2]
    with open('./output/supplier_ids_map.dict', 'wb') as fileHandle:
        pickle.dump(suppliersIdKey, fileHandle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Encode supplier city
    results = encodeData(awardedEvents, 'EventParticipation_SupplierCity', 201)
    awardedEvents = results[0]
    supplierCitiesKey = results[1]
    supplierCitiesReverseKey = results[2]
    with open('./output/supplier_city_map.dict', 'wb') as fileHandle:
        pickle.dump(supplierCitiesKey, fileHandle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Encode supplier state
    results = encodeData(awardedEvents, 'EventParticipation_SupplierState', 301) 
    awardedEvents = results[0]
    supplierStatesKey = results[1]
    supplierStatesReverseKey = results[2]
    with open('./output/supplier_state_map.dict', 'wb') as fileHandle:
        pickle.dump(supplierStatesKey, fileHandle, protocol=pickle.HIGHEST_PROTOCOL)
    
     # Encode supplier country
    results = encodeData(awardedEvents, 'EventParticipation_SupplierCountry', 401)
    awardedEvents = results[0]
    supplierCountryKey = results[1]
    supplierCountryReverseKey = results[2]
    with open('./output/supplier_country_map.dict', 'wb') as fileHandle:
        pickle.dump(supplierCountryKey, fileHandle, protocol=pickle.HIGHEST_PROTOCOL)
    
    '''
    awarded = awardedEvents.query('EventParticipation_AwardedFlag == True')
    for index in awarded.index:
        awardedEvents.at[index, 'EventParticipation_AwardedFlag'] = 1
    
    nonAwarded = awardedEvents.query('EventParticipation_AwardedFlag == False')
    for index in nonAwarded.index:
        awardedEvents.at[index, 'EventParticipation_AwardedFlag'] = 0
    '''
    
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
    print("Accuracy of training data (RandomForest) = "+ str(trainingAccuracy))
    
    # Accuracy for resampled training events
    y_resampled_train_pred = model.predict(X_resampled)
    trainingUpSamplingAccuracy = round(accuracy_score(y_resampled, y_resampled_train_pred) * 100, 2)
    print("Accuracy of upsampled training data (RandomForest) = "+ str(trainingUpSamplingAccuracy))
    
    # Accuracy for test events
    y_test_pred = model.predict(X_test)
    testAccuracy = round(accuracy_score(y_test, y_test_pred) * 100, 2)
    print("Accuracy of test data (RandomForest) = "+ str(testAccuracy))
    
    algoMap["RandomForest"] = (trainingAccuracy, trainingUpSamplingAccuracy, testAccuracy, model)
    
    return model

def KNN(X, y, X_train, X_test, y_train, y_test, X_resampled, y_resampled):
    #Create KNN Classifier
    model = KNeighborsClassifier()

    #Train the model using the training sets
    model.fit(X_train, y_train)
    
    # Accuracy for training events
    y_train_pred = model.predict(X_train)
    trainingAccuracy = round(accuracy_score(y_train, y_train_pred) * 100, 2)
    print("Accuracy of training data (KNN) = "+ str(trainingAccuracy))
    
    # Accuracy for resampled training events
    y_resampled_train_pred = model.predict(X_resampled)
    trainingUpSamplingAccuracy = round(accuracy_score(y_resampled, y_resampled_train_pred) * 100, 2)
    print("Accuracy of upsampled training data (KNN) = "+ str(trainingUpSamplingAccuracy))
    
    # Accuracy for test events
    y_test_pred = model.predict(X_test)
    testAccuracy = round(accuracy_score(y_test, y_test_pred) * 100, 2)
    print("Accuracy of test data = (KNN) = "+ str(testAccuracy))
    
    algoMap["KNN"] = (trainingAccuracy, trainingUpSamplingAccuracy, testAccuracy, model)
    
    return model

def logisticRegression(X, y, X_train, X_test, y_train, y_test, X_resampled, y_resampled):
    # Create LinearRegression classifier
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Accuracy for training events
    y_train_pred = model.predict(X_train)
    trainingAccuracy = round(accuracy_score(y_train, y_train_pred) * 100, 2)
    print("Accuracy of training data (Logistic Regression) = "+ str(trainingAccuracy))
    
    # Accuracy for resampled training events
    y_resampled_train_pred = model.predict(X_resampled)
    trainingUpSamplingAccuracy = round(accuracy_score(y_resampled, y_resampled_train_pred) * 100, 2)
    print("Accuracy of upsampled training data (Logistic Regression) = "+ str(trainingUpSamplingAccuracy))
    
    # Accuracy for test events
    y_test_pred = model.predict(X_test)
    testAccuracy = round(accuracy_score(y_test, y_test_pred) * 100, 2)
    print("Accuracy of test data = (Logistic Regression) = "+ str(testAccuracy))
    
    algoMap["Logistic Regression"] = (trainingAccuracy, trainingUpSamplingAccuracy, testAccuracy, model)
    
    return model

def printModelMetrics():
    print("---------------------------------------------------------------------------------------")
    print("Model Metrics...")
    for key in algoMap:
        print(key + " = Training Accuracy(" + str(algoMap[key][0]) + "), Upsampling Training Accuracy(" + str(algoMap[key][1]) + ")")
    print("---------------------------------------------------------------------------------------")
    
def compareAndStoreModel():
    upsamplingAccuracy = -9999
    selectedAlgo = ""
    selectedModel = None
    for key in algoMap:
        algoTripple = algoMap[key]
        algoUpsamplingAccuracy = algoTripple[1]
        if(upsamplingAccuracy < algoUpsamplingAccuracy):
            upsamplingAccuracy = algoUpsamplingAccuracy
            selectedAlgo = key
            selectedModel = algoTripple[3]
    
     # Required for inference
    with open('./output/selected_ai_ml_model.mdl', 'wb') as fileHandle:
        pickle.dump(selectedModel, fileHandle, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("Best algorithm selected = " +selectedAlgo) 


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

def getModelDictionaries():
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
    return (biddersKey, suppliersIdKey, supplierCitiesKey, supplierStatesKey, supplierCountryKey)
        
def inference(inputDf):
    dict = getModelDictionaries()
    biddersKey = dict[0]
    suppliersIdKey = dict[1]
    supplierCitiesKey = dict[2]
    supplierStatesKey = dict[3]
    supplierCountryKey = dict[4]
    
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

def retrainModel(selectedEvents):
    data = load()
    classificationData = data[0]
    clusterdEventsData = data[1]
    
    # Update selected events as fraud
    for index, eventRow in selectedEvents.iterrows():
        eventData = classificationData.query('EventId == "' +eventRow['EventId']+ '"')
        # Update label as fraud
        for row in eventData.index:
            classificationData.at[row, 'fraud'] = 1
    
    # Uodate the CSV file with new set of fraud entries
    #classificationData.to_csv('./output/classification_events_data.csv', mode='w', header=True, encoding='utf-8', index=False)
    #print('Classification data created succesfully..')
    
    # Retrain model
    trainModel(classificationData, clusterdEventsData, False)
    
            
def trainModel(classificationData, clusterdEventsData, trailMode):
    awardedEvents = preprocess(classificationData, clusterdEventsData)
    balData = balanceData(awardedEvents)
    
    X = balData[0]
    y = balData[1]
    X_train = balData[2]
    X_test = balData[3]
    y_train = balData[4]
    y_test = balData[5]
    X_resampled = balData[6]
    y_resampled = balData[7]
    
    # Compare algos only for first time and during retrain run best algo
    if(trailMode):
        logisticRegression(X, y, X_train, X_test, y_train, y_test, X_resampled, y_resampled)
        KNN(X, y, X_train, X_test, y_train, y_test, X_resampled, y_resampled)
        randomForest(X, y, X_train, X_test, y_train, y_test, X_resampled, y_resampled)
        # Required during inference stage
        persistModelPerformanceMetrics()
        compareAndStoreModel()
        printModelMetrics()
    else:
        # During retrain run best model
        model = randomForest(X, y, X_train, X_test, y_train, y_test, X_resampled, y_resampled)
        with open('./output/selected_ai_ml_model.mdl', 'wb') as fileHandle:
            pickle.dump(model, fileHandle, protocol=pickle.HIGHEST_PROTOCOL) 
       

def inferenceTest():
    # Inference test
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
    label = inference(inputDf)[0]
    if(label ==0):
         print("Inference = Non-Fraud")
    else:
        print("Inference = Fraud")
        
    
def main():
    data = load()
    trainModel(data[0], data[1], True)
    inferenceTest()

main()
    
    


    

