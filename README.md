# Fraud in sourcing, Finding anamolies during event awarding

**POC to check concept of doing fraud detection in sourcing using classification and clustering techniques**

### Highlights:
  * Fraud detection of labelled data containing suppliers having same address are trying to bid/getting awarded an event - **Build an classification model**
  * Anamolies detection to find an events containing supplier's with low participation rates - **Build an clustering model**
  * UI to Auditor for **tagging anamolies/fradulent events for retraining** the model
  * Support **realtime/near realtime** events to classify events are fraud/non-fraud

### Prerequisite:
  * **Install Python 3.6 *https://www.python.org/downloads/release/python-368/*** 
	* **Install needed python libraries**
		- python -m pip install --upgrade pip
		- pip install sklearn
		- pip install --upgrade streamlit
		- pip install pulp
		- pip install altair
		- pip install matplotlib
		- pip install pandas
		- pip install numpy
        - pip install st_aggrid

  * **Clone github**
    - $SRC_DIR>*git clone https://github.com/sprasadban-blr/Sourcing_Fraud_Detection*

* **Steps to extract Events data from reporting system**
    - Download **Event, EventParticipation, EventSummary, SupplierParticipation, RfxEventSummary** data from sourcing reporting DB (as CSV files from Inspector tool)
      * Persist the CSV files in *'./input/Event.csv'*, *'./input/EventParticipation.csv'*, *'./input/EventSummary.csv'*, *'./input/Event_SupplierParticipation.csv'* and *'./input/Event_RfxItemSummary.csv'*
      * Queries used to download the data can be referred from *'query.sql'* file
    - Parse the above CSV using *'EventsDataPreparation.py'* to create data required for the PoC
      * Persist the CSV files in *'./output/events_data.csv'*
    - **Note:** For now we have downloaded the CSV data from reporting system through ***'Inspector'*** tool.. In reality we need to build ***'knowledgebase'*** for bidding events/items/supplier details by extracting data directly from Reporting/Prism database

* **Steps to create AI/ML prediction model**
    - Create required data for classification by running *'ClassificationDataPreparation.py'*.. This will create a file in *'./output/classification_events_data.csv'*
	- Choose by running *'FraudClassification.py'* to select the best algorithm for our data using **LogisticRegression, KNN and Random Forest** algorithm
    - Persist the AI/ML model *'./output/selected_ai_ml_model.mdl'* and the dictionary used *'./output/bidders_map.dict, ./output/supplier_ids_map.dict, ./output/supplier_city_map.dict, ./output/supplier_state_map.dict'*, ./output/supplier_country_map.dict.. These are required during inference time
    - **Note:** For now we have tried out few prediction algorithms with *Training Accuracy and Upsampling Training Accuracy* as our prediction performance metrics.. We need to try other algorithms and performance metrics.

* **Steps to create AI/ML clustering model**
    - Create required data for clustering by running *'ClusteringDataPreparation.py'*.. This will create a file in *'./output/clustered_events_data.csv'*
	- Choose by running *'FraudClustering.py'* to select the best algorithm for our data using **K-Means and K_Prototype** algorithm.. This will create a file in *'./output/clustered_events_lables.csv and ./output/low_participation_clusters.png* containing events clustered groups.
	- **Note:** For now we have considered 2 clusters (Fraud Vs Non-Fraud) based on Elbow/Silhouette score technique.

### Run fraud detection dashboard application 
  ***$SRC_DIR>streamlit run FraudDetectionDashboard.py***

### Run UI application
  * *http://localhost:8501/*