import pandas as pd

def createClassificationData():
    eventParticipation = pd.read_csv('./input/EventParticipation.csv')
    eventParticipationFinal = eventParticipation[[
        'EventParticipation_Id',
        'EventParticipation_Creator',
        'EventId',
        'EventParticipation_Bidder_UserName',
        'EventParticipation_Bidder_UserId',
        'EventParticipation_SupplierName',
        'EventParticipation_SupplierId',
        'EventParticipation_CommonSupplierName',
        'EventParticipation_CommonSupplierId',
        'EventParticipation_SupplierCity',
        'EventParticipation_SupplierState',
        'EventParticipation_SupplierCountry',
        'EventParticipation_SupplierType',
        'EventParticipation_Owner_UserName',
        'EventParticipation_Owner_UserId',
        'EventParticipation_AcceptedFlag',
        'EventParticipation_DeclinedFlag',
        'EventParticipation_IntendToRespondFlag',
        'EventParticipation_DeclinedToRespondFlag',
        'EventParticipation_ParticipatedFlag',
        'EventParticipation_AwardedFlag',
        'EventParticipation_BidsSubmitted'
    ]]
    
    # Fraud events for classification
    eventParticipationFinal['fraud'] = 0
    fraudEvents = eventParticipationFinal.query(
        'EventId in ("Doc6737990", "Doc7001940", "Doc6998816", "Doc6998796", "Doc7001865", "Doc7001832", "Doc6999613", "Doc6999590")')
    
    for row in fraudEvents.index:
        eventParticipationFinal.at[row, 'fraud'] = 1
        
    # Fraud events from clustering will be updated post clustering
    fraudEvents = eventParticipationFinal.query(
        "EventId in ('Doc6744368', 'Doc6762337', 'Doc6757695', 'Doc6853094', 'Doc6871841', 'Doc6972434', 'Doc6900882', 'Doc6972453')")
    
    for row in fraudEvents.index:
        eventParticipationFinal.at[row, 'fraud'] = 0
    
    eventParticipationFinal.to_csv('./output/classification_events_data.csv', mode='w', header=True, encoding='utf-8', index=False)
    print('Classification data created succesfully..')


def main():
    createClassificationData()

main()
    