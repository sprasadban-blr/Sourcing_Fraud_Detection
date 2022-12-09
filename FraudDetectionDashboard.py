import streamlit as st
import altair as alt
import pandas as pd
from PIL import Image
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode

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
    clusteredEventsData = pd.read_csv('./output/clustered_events_data.csv')
    return (eventData, eventParticipation, eventSummary, eventSupplierParticipation, eventRfxItemSummary, classificationEventsData, clusteredEventsData, modelMetrics)

def visualize_outliers(clusteredEventsData):
    st.title("Outliers View")
    st.write(clusteredEventsData)
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
            st.write("Model retraining...")

def main(eventData, eventParticipation, eventSummary, eventSupplierParticipation, eventRfxItemSummary, classificationEventsData, clusteredEventsData, modelMetrics):
    
    page = st.sidebar.radio("Choose a page", ("Event Data", "Event Participation", "Event Summary", "Event Supplier Participation", 
                                                  "Event Item Summary", "Classification View", "Outliers View"))
    if page == "Event Data":
        st.header("About Event Data.")
        st.write("Please select a page on the left.")
        st.write(eventData) 
    if page == "Event Participation":
        st.header("About Event Participation Data.")
        st.write(eventParticipation)
    if page == "Event Summary":
        st.header("About Event Summary Data")
        st.write(eventSummary)
    elif page == "Event Supplier Participation":
        st.header("About Event Supplier Participation Data")
        st.write(eventSupplierParticipation)
    elif page == "Event Item Summary":
        st.header("About Event Item Summary Data")
        st.write(eventRfxItemSummary)
    elif page == "Classification View":
        st.header("Classification View")
        st.write(classificationEventsData)
        st.header("Prediction Model Metrics")
        st.write(modelMetrics)
    elif page == "Outliers View":
        visualize_outliers(clusteredEventsData)
                
#Load data and create ML model
data = load_data()
#Render Data and Predict model
main(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7])