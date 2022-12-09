import pandas as pd

def createFraudEventsData(clusterNo):
    labeledEvents = pd.read_csv('./output/clustered_events_labels.csv')
    anamolyEvents = labeledEvents.query('labels == '+ str(clusterNo))['EventId'].unique()
    
    eventParticipation = pd.read_csv('./input/EventParticipation.csv')

    supplierId = ""
    supplierName = ""
    accepted = False
    declined = False
    intented = False
    declined_To_Respond = False
    participated = False
    awarded = False
    supplierMap = {}

    for anamolyEvent in anamolyEvents:
        print(anamolyEvent)
        participationDtls = eventParticipation.query('EventId == "' + anamolyEvent + '"')
        eventIndices = participationDtls.index
        
        for i in range(len(eventIndices)):
            supplierId = participationDtls.loc[eventIndices[i], 'EventParticipation_SupplierId']
            supplierName = participationDtls.loc[eventIndices[i], 'EventParticipation_SupplierName']
            accepted = participationDtls.loc[eventIndices[i], 'EventParticipation_AcceptedFlag']	
            declined = participationDtls.loc[eventIndices[i], 'EventParticipation_DeclinedFlag']
            intented = participationDtls.loc[eventIndices[i], 'EventParticipation_IntendToRespondFlag']
            declined_To_Respond = participationDtls.loc[eventIndices[i], 'EventParticipation_DeclinedToRespondFlag']
            participated = participationDtls.loc[eventIndices[i], 'EventParticipation_ParticipatedFlag']
            awarded = participationDtls.loc[eventIndices[i], 'EventParticipation_AwardedFlag']
            supplierMap[anamolyEvent + ":" + supplierId] = (anamolyEvent, supplierId, supplierName, accepted, declined, intented, declined_To_Respond, participated, awarded)
            print(anamolyEvent + ":" + supplierId)
        print("-------------------------------------------------------------")
    
    supplierParticipationDtls = pd.DataFrame(columns=['EventId', 'SupplierId', 'SupplierName', 'AcceptedFlag', 'DeclinedFlag', 'IntendToRespondFlag', 'DeclinedToRespondFlag', 
                                                  'ParticipatedFlag', 'AwardedFlag'])
    row = 0
    for eventSupplier in supplierMap:
        rowData = supplierMap[eventSupplier]
        supplierParticipationDtls.loc[row] = [rowData[0], rowData[1], rowData[2], rowData[3], rowData[4], rowData[5], rowData[6], rowData[7], rowData[8]]
        row = row + 1
    supplierParticipationDtls.to_csv('./output/clustered_events_data.csv', mode='w', header=True, encoding='utf-8', index=False)
    print('Fraud events data created successfully..')
    return supplierParticipationDtls

def main():
    createFraudEventsData(1)

main()