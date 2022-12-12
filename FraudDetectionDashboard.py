import streamlit as st
import altair as alt
import pandas as pd
from PIL import Image
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
import FraudClassification as fraudclf

# https://discuss.streamlit.io/t/streamlit-restful-app/409/2
# https://stackoverflow.com/questions/68273958/how-would-i-get-my-streamlit-application-to-use-a-flask-api-in-order-to-retrieve
# https://docs.streamlit.io/
# Editable Table: https://towardsdatascience.com/make-dataframes-interactive-in-streamlit-c3d0c4f84ccb
# Note: AgGrid is a commercial license (EULA)

@st.cache
def load_data():
    # Load data
    eventData = pd.read_csv('./input/Event.csv')
    eventParticipation = pd.read_csv('./input/EventParticipation.csv')
    eventSummary = pd.read_csv('./input/EventSummary.csv')
    eventSupplierParticipation =  pd.read_csv('./input/Event_SupplierParticipation.csv')
    eventRfxItemSummary =  pd.read_csv('./input/Event_RfxItemSummary.csv')
    classificationEventsData = pd.read_csv('./output/classification_awarded_events_data.csv')
    modelMetrics =  pd.read_csv('./output/modelMetrics.csv')
    modelMetrics = modelMetrics[['Algorithm','Training Accuracy','UpSampling Training Accuracy']]
    clusteredEventsData = pd.read_csv('./output/clustered_events_data.csv')
    return (eventData, eventParticipation, eventSummary, eventSupplierParticipation, eventRfxItemSummary, classificationEventsData, clusteredEventsData, modelMetrics)

def visualize_design():
    st.header("Fraud Detection Design")
    image = Image.open('./output/fraud_detection_steps.png')
    st.image(image, caption='High level design')
    
def visualize_outliers(clusteredEventsData):
    visualize_design()
    st.header("Preprocessed Non-Labelled Data")
    st.write(clusteredEventsData)
    
    st.header('Low Participation Clusters Using K-Prototype/K-Means')
    image = Image.open('./output/low_participation_clusters.png')
    st.image(image, caption='Low Participation Clusters')
    clusterNo = st.radio("Choose cluster to analyze", (0, 1))
    #clusterNo = st.number_input("Enter the cluster number", value=0)
    clusteredEventsLabelledData = pd.read_csv('./output/clustered_events_labels.csv')
    clusterDataSelection = clusteredEventsLabelledData.query('labels == '+str(clusterNo))

    gb = GridOptionsBuilder.from_dataframe(clusterDataSelection)
    gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
    gb.configure_side_bar() #Add a sidebar
    gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
    gridOptions = gb.build()

    grid_table  = AgGrid(
        clusterDataSelection,
        gridOptions=gridOptions,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        theme="streamlit"
    )
    
    sel_row = grid_table["selected_rows"]
    st.subheader("Selected Events")
    selectedEventsDf = pd.DataFrame(sel_row)
    
    
    if not selectedEventsDf.empty:
        selectedEventsDf = selectedEventsDf.drop(['_selectedRowNodeInfo'], axis=1)
        st.write(selectedEventsDf)
        
    if st.button('Re-Train Model'):
        st.write("Model retraining in progress...")
        fraudclf.retrainModel(selectedEventsDf)
        st.write("Retraining completed.")
            

def visualize_classification(classificationEventsData, modelMetrics, input, input1):
    visualize_design()
    st.header("Preprocessed Labelled Data")
    st.write(classificationEventsData)
    
    st.header("Prediction Model Metrics")
    st.write(modelMetrics)
    
    inputDf = pd.DataFrame(input1, columns=['EventParticipation_Bidder_UserId',
                                'EventParticipation_SupplierId', 
                                'EventParticipation_SupplierCity',
                                'EventParticipation_SupplierState',
                                'EventParticipation_SupplierCountry',	
                                'EventParticipation_AwardedFlag',	
                                'EventSummary_InvitedSuppliers',	
                                'EventSummary_ParticipSuppliers',
                                'EventSummary_ParticipationRate'])
    eventData1 = pd.read_csv('./output/classification_awarded_events_data.csv')
    eventDocDf = eventData1.query('EventId == "Doc7001940"')
    # Data having same address
    st.header("New Incoming Event..")
    st.write(eventDocDf)
    st.header("Predict event")
    st.write(inputDf)
    if st.button('Predict'):
        label = fraudclf.inference(inputDf)[0]
        if(label == 0):
            st.markdown("The given event details are classified as **'Non-Fraud'**")
        else:
            st.markdown("The given event details are classified as **'Fraud'**")
            st.markdown("**Supplier with same address.**")
    
def visualize_reclassification(input):
    visualize_design()
    inputDf = pd.DataFrame(input, columns=['EventParticipation_Bidder_UserId',
                                'EventParticipation_SupplierId', 
                                'EventParticipation_SupplierCity',
                                'EventParticipation_SupplierState',
                                'EventParticipation_SupplierCountry',	
                                'EventParticipation_AwardedFlag',	
                                'EventSummary_InvitedSuppliers',	
                                'EventSummary_ParticipSuppliers',
                                'EventSummary_ParticipationRate'])
    st.header("New Incoming Event..")
    st.write(inputDf)
    label = fraudclf.inference(inputDf)[0]
    if st.button('Predict'):
        label = fraudclf.inference(inputDf)[0]
        if(label == 0):
            st.markdown("The given event details are classified as **'Non-Fraud'**")
        else:
            st.markdown("The given event details are classified as **'Fraud'**")
            st.markdown("**Supplier with low participation rate.**")
    
def main(eventData, eventParticipation, eventSummary, eventSupplierParticipation, eventRfxItemSummary, classificationEventsData, clusteredEventsData, modelMetrics):
    
    page = st.sidebar.radio("Choose a page", ("Fraud Detection Design", "Event Data", "Event Participation", "Event Summary", "Event Supplier Participation", 
                                                  "Event Item Summary", "Classification View", "Outliers View",  "ReClassification View"))
    if page == "Fraud Detection Design":
        st.header("Fraud Detection Design")
        image = Image.open('./output/fraud_detection_steps.png')
        st.image(image, caption='High level design')
    elif page == "Event Data":
        st.header("About Event Data.")
        st.write("Please select a page on the left.")
        st.write(eventData) 
    elif page == "Event Participation":
        st.header("About Event Participation Data.")
        st.write(eventParticipation)
    elif page == "Event Summary":
        st.header("About Event Summary Data")
        st.write(eventSummary)
    elif page == "Event Supplier Participation":
        st.header("About Event Supplier Participation Data")
        st.write(eventSupplierParticipation)
    elif page == "Event Item Summary":
        st.header("About Event Item Summary Data")
        st.write(eventRfxItemSummary)
    elif page == "Classification View":
        input = [['dstaley', 'sid503', 'San Jose', 'CA', 'US', True, 26, 2, 7.6900]]
        input1 = [['Tom milton', 'ACM_44106', 'Bangalore', 'Karnataka', 'IN', True, 68.0000, 2.0000, 2.9400]]
        visualize_classification(classificationEventsData, modelMetrics, input, input1)
    elif page == "Outliers View":
        visualize_outliers(clusteredEventsData)
    elif page == "ReClassification View":
        input = [['dstaley', 'sid503', 'San Jose', 'CA', 'US', True, 26, 2, 7.6900]]
        visualize_reclassification(input)
                
#Load data and create ML model
data = load_data()
#Render Data and Predict model
main(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7])