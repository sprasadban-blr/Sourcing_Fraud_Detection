import pandas as pd

def load():
    print("Data loading in progress..")
    event = pd.read_csv('./input/Event.csv')
    eventParticipation = pd.read_csv('./input/EventParticipation.csv')
    eventSummary = pd.read_csv('./input/EventSummary.csv')
    supplierParticipation = pd.read_csv('./input/Event_SupplierParticipation.csv')
    rfxItemSummary = pd.read_csv('./input/Event_RfxItemSummary.csv')
    print("Data loading is completed.")
    return (event, eventParticipation, eventSummary, supplierParticipation, rfxItemSummary)

def preprocess(event_data):
    print("Preprocessing of data in progress..")
    event = event_data[0]
    eventParticipation = event_data[1]
    eventSummary = event_data[2]
    supplierParticipation = event_data[3]
    rfxItemSummary = event_data[4]
    
    event_supplier = supplierParticipation.merge(event, how='left', left_on=['EventId', 'ItemId'], 
                                                        right_on=['Event_EventId', 'Event_ItemId'])
    event_supplier_participation = event_supplier.merge(eventParticipation, how='left', left_on=['EventId', 'SupplierId'], 
                                                        right_on=['EventId', 'EventParticipation_SupplierId'])
    event_supplier_participation_summary = event_supplier_participation.merge(eventSummary, how='left', left_on=['EventId'], 
                                                        right_on=['EventId'])
    event_supplier_participation_summary_rfxItem = event_supplier_participation_summary.merge(rfxItemSummary, how='left', left_on=['EventId', 'ItemId'], 
                                                        right_on=['EventId', 'ItemId'])
    eventData = event_supplier_participation_summary_rfxItem
    [[
    'Event_EventId', 'Event_Creator', 'Event_TimeCreated', 'Event_TimeUpdated', 'Event_ItemTitle', 'Event_ItemId', 'Event_OwnRankingInfoRelease', 
    'Event_SupplierNamesInfoRelease','Event_CompetitorRankInfoRelease', 'Event_ParticipantSpecificValuesInfoRelease', 'Event_EventTitle', 'Event_ItemType', 
    'Event_ItemSubType', 'Event_EventType', 'Event_LotStatus', 'Event_LeadBidInfoRelease', 'Event_CompetitorBidsInfoRelease', 'Event_ImprovementType', 'Event_TieBidRule',
    'EventParticipation_TimeCreated', 'EventParticipation_TimeUpdated', 'EventParticipation_Bidder_UserName', 'EventParticipation_Bidder_UserId',
    'EventParticipation_SupplierId', 'EventParticipation_SupplierName', 'EventParticipation_CommonSupplierName', 'EventParticipation_CommonSupplierId', 
    'EventParticipation_SupplierCity', 'EventParticipation_SupplierState', 'EventParticipation_SupplierCountry', 'EventParticipation_EventStartDate', 
    'EventParticipation_EventEndDate', 'EventParticipation_EventCreateDate', 'EventParticipation_BiddingStartDate', 'EventParticipation_BiddingEndDate', 
    'EventParticipation_Owner_UserName', 'EventParticipation_Owner_UserId', 'EventParticipation_AcceptedFlag', 'EventParticipation_DeclinedFlag', 'EventParticipation_IntendToRespondFlag',
    'EventParticipation_DeclinedToRespondFlag', 'EventParticipation_ParticipatedFlag', 'EventParticipation_AwardedFlag', 'EventParticipation_NumEventAwarded',
    'EventParticipation_NumEventAccepted', 'EventParticipation_NumEventDeclined', 'EventParticipation_NumIntendToRespond', 'EventParticipation_NumDeclinedToRespond',
    'EventParticipation_NumEventBidOn', 'EventParticipation_BidsSubmitted', 'EventSummary_BidsSubmitted', 'EventSummary_BidsRemoved', 'EventSummary_SurrogateBids',
    'EventSummary_NumQuestions', 'EventSummary_InvitedSuppliers', 'EventSummary_AcceptedSuppliers', 'EventSummary_DeclinedSuppliers', 'EventSummary_ParticipSuppliers', 
    'IncumbentFlag', 'Surrogate_UserName', 'Surrogate_UserId', 'BidQuantity', 'BidTotalCost', 'AwardedAmount', 'PotentialSavings', 'SupplierAwardedHist', 
    'SupplierParticipation_NumItemAwarded', 'SupplierParticipation_NumItemAccepted', 'SupplierParticipation_NumItemDeclined', 'SupplierParticipation_NumItemBidOn', 'SupplierParticipation_BidsSubmitted', 
    'ItemQuantity', 'HistTotalCost', 'ResvTotalCost', 'IncumbentQuantity', 'IncumbentTotalCost', 'MktLeadQuantity', 'MktLeadTotalCost', 'InitialTotalCost', 
    'LeadPreBidTotalCost', 'AwardedQuantity', 'AwardedTotalCost', 'AwardedHistSpend', 'LeadingSavings', 'PendingSpend', 'PendingSavings', 'SubmitBidsForItem', 
    'SurrogtBidsForItem', 'RemvdBidsForItem', 'NumItemPending', 'NumItemClosed', 'TargetSavings'
    ]]
    print("Preprocessing of data is completed.")
    eventData.to_csv('./output/event_data.csv', mode='w', header=True, encoding='utf-8', index=False)
    print("Event data created succesfully.")
    
    
def main():
    event_data = load()
    preprocess(event_data)

main()