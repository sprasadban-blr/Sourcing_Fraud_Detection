import pandas as pd
import numpy as np
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pacmap

def preprocessData():
    allEvents = pd.read_csv('./input/Event.csv')
    nonAwardedEvents = allEvents.query('Event_EventStatus != 7')['Event_EventId'].unique()
    # Remove non awarded events
    eventSummary = pd.read_csv('./input/EventSummary.csv')
    for nonAwardedEvent in nonAwardedEvents:
        events = eventSummary.query('EventId == "' + nonAwardedEvent + '"')
        eventSummary = eventSummary.drop(eventSummary.index[events.index])
    
    # After removing rows reset the index back
    eventSummary = eventSummary.reset_index(drop=True)
    
    # Remove events with non particpants suppliers
    nonParticipantSuppliers = eventSummary.query('EventSummary_ParticipSuppliers == 0')
    print(nonParticipantSuppliers)
    eventSummary = eventSummary.drop(eventSummary.index[nonParticipantSuppliers.index])
    
    # Compute event participation rate
    eventSummary['EventSummary_ParticipationRate'] = round((1-(eventSummary['EventSummary_InvitedSuppliers'] - eventSummary['EventSummary_ParticipSuppliers'])/
                                                      eventSummary['EventSummary_InvitedSuppliers'])*100, 2)
    #eventSummary = eventSummary.query('EventSummary_InvitedSuppliers > 25 and EventSummary_ParticipationRate > 3')
    
    eventSummaryCluster = eventSummary[[
        'EventId',
        'EventSummary_InvitedSuppliers',
        'EventSummary_ParticipSuppliers',
        'EventSummary_ParticipationRate'
    ]]
    # Invalid event
    eventSummaryCluster = eventSummaryCluster.drop(eventSummaryCluster.index[(eventSummaryCluster["EventId"] == "Doc975248")])
    # After removing rows reset the index back
    eventSummaryCluster = eventSummaryCluster.reset_index(drop=True)
    return eventSummaryCluster

def getFeatureList(data):
    colIndex = 0
    numericalColNames = []
    numericalColIndex = []
    categoricalColNames = []
    categoricalColIndex = []
    otherColNames = []
    otherColIndex = []
    
    for col in data.columns:
        if (is_string_dtype(data[col])):
            categoricalColNames.append(col)
            categoricalColIndex.append(colIndex)
        elif (is_numeric_dtype(data[col])):
            numericalColNames.append(col)
            numericalColIndex.append(colIndex)
        else:
            otherColNames.append(col)
            otherColIndex.append(colIndex)
        colIndex = colIndex + 1 
    print("Numerical Columns") 
    print(numericalColNames)
    print("Categorical Columns") 
    print(categoricalColNames)
    print("Other columns")
    print(otherColNames)
    return (numericalColNames, numericalColIndex, categoricalColNames, categoricalColIndex, otherColNames, otherColIndex)

def k_prototype(eventSummaryCluster, categoricalColIndex):
    kproto = KPrototypes(n_clusters=2, init='Cao')
    clusters = kproto.fit_predict(eventSummaryCluster, categorical=categoricalColIndex)
    #join data with labels 
    labels = pd.DataFrame(clusters)
    labeledEvents = pd.concat((eventSummaryCluster,labels),axis=1)
    labeledEvents = labeledEvents.rename({0:'labels'},axis=1)
    return labeledEvents

def k_means(eventSummaryCluster, noOfCluster):
    eventClusters_kmeans = eventSummaryCluster[[
        'EventSummary_InvitedSuppliers',
        'EventSummary_ParticipSuppliers',
        'EventSummary_ParticipationRate'
    ]]
    
    scaler = StandardScaler()
    eventClusters_kmeans_scaled = scaler.fit_transform(eventClusters_kmeans)
    # Since the fit_transform() strips the column headers
    # we add them after the transformation
    eventClusters_kmeans_std = pd.DataFrame(eventClusters_kmeans_scaled, columns=eventClusters_kmeans.columns)
    print(eventClusters_kmeans_std)
    km = KMeans(n_clusters=noOfCluster, 
            max_iter=300, 
            tol=1e-04, 
            init='k-means++', 
            n_init=10, 
            random_state=42, 
            algorithm='auto')

    km_fit = km.fit(eventClusters_kmeans_std)
    labels = pd.DataFrame(km_fit.labels_)
    labeledEvents = pd.concat((eventSummaryCluster, labels), axis=1)
    labeledEvents = labeledEvents.rename({0:'labels'}, axis=1)
    return (labeledEvents, eventClusters_kmeans_std, km_fit)

def drawNsaveCluster(eventClusters_kmeans_std, km_fit, y, noOfCluster):
    if (noOfCluster == 2):
        cluster_colors = ['#568f8b', '#1d4a60']
    else:
        cluster_colors = ['#568f8b', '#1d4a60', '#d15252']
        
   # PACMAP
    fig, ax = plt.subplots()

    embedding = pacmap.PaCMAP(random_state=42)
    X_std_pacmap = embedding.fit_transform(eventClusters_kmeans_std.to_numpy())

    for l, c, m in zip(range(0, noOfCluster), cluster_colors[0:km_fit.n_clusters], ('^', 's', 'o')):
        ax.scatter(X_std_pacmap[y == l, 0],
                    X_std_pacmap[y == l, 1],
                    color=c,
                    label='cluster %s' % l,
                    alpha=0.9,
                    marker=m
                    )
        
    ax.set_title("Anamolies of low participation rate")

    labels = np.unique(km_fit.labels_)
    labels = ["cluster "+str(l) for l in labels]
    fig.legend(labels, loc='lower center',ncol=len(labels), bbox_transform=(1,0), borderaxespad=-0.5)
    plt.tight_layout()
    plt.savefig('./output/low_participation_clusters.png')
    plt.close()
    
def main():
    print('Data preprocessing...')
    eventSummaryCluster = preprocessData()
    
    print('Checking numerical/categorical features...')
    featureList = getFeatureList(eventSummaryCluster)
    numericalColNames = featureList[0]
    numericalColIndex = featureList[1]
    categoricalColNames = featureList[2]
    categoricalColIndex = featureList[3]
    otherColNames = featureList[4]
    otherColIndex = featureList[5]
    
    print('Clustering events...')
    # K-Prototype cluster
    #labeledEvents = k_prototype(eventSummaryCluster, categoricalColIndex)
    
    # K-Means cluster
    noOfClusters = 2
    k_meansOutput = k_means(eventSummaryCluster, noOfClusters)
    labeledEvents = k_meansOutput[0]
    labeledEvents.to_csv('./output/clustered_events_labels.csv', mode='w', header=True, encoding='utf-8', index=False)
    
    y = labeledEvents['labels']
    eventClusters_kmeans_std = k_meansOutput[1]
    km_fit = k_meansOutput[2]
    drawNsaveCluster(eventClusters_kmeans_std, km_fit, y, noOfClusters)
    
    print('Clustered events saved succesfully..')

main()