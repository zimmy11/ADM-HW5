import networkx as nx
import matplotlib.pyplot as plt
import numpy as np 
from IPython.display import display, Markdown, IFrame
import seaborn as sns
import pandas as pd
import folium
from folium.plugins import MarkerCluster
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_airport_graph(df):
    """
    Create a directed graph from a DataFrame containing airport and flight data.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the data for the graph.

    Returns:
    nx.DiGraph: Directed graph representing the airport network.
    """
    G = nx.DiGraph() 

    for _, row in df.iterrows():
        
        origin = row['Origin_airport']
        destination = row['Destination_airport']
        
        G.add_node(origin, 
                    city=row['Origin_city'], 
                    population=int(row['Origin_population']), 
                    lat=float(row['Org_airport_lat']), 
                    long=float(row['Org_airport_long']))
        
        G.add_node(destination, 
                    city=row['Destination_city'], 
                    population=int(row['Destination_population']), 
                    lat=float(row['Dest_airport_lat']), 
                    long=float(row['Dest_airport_long']))
        
        G.add_edge(origin, destination, 
                    passengers=int(row['Passengers']), 
                    flights=int(row['Flights']), 
                    seats=int(row['Seats']), 
                    weight=int(row['Distance']), 
                    fly_date=row['Fly_date'])

    return G


def analyze_graph_features(flight_network):

    number_of_nodes = 0
    number_of_edges = len(flight_network.edges())
    dict_degrees_edges = dict()

    for node in flight_network.nodes:
        number_of_nodes = number_of_nodes + 1

        in_edges = 0
        out_edges = 0

        for _, _, attr in flight_network.edges(node, data = True):
            out_edges+=1

        for _, _, attr in flight_network.in_edges(node, data = True):
            in_edges+=1
            

        dict_degrees_edges[node] = [in_edges, out_edges]

    graph_density = (2 * number_of_edges) / (number_of_nodes* (number_of_nodes -1))
    

    in_degrees = [edge_degree[0] for edge_degree in dict_degrees_edges.values()]
    out_degrees = [edge_degree[1] for edge_degree in dict_degrees_edges.values()]
    


    # Create a subplot with 1 row, 2 columns
    fig = make_subplots(rows=1, cols=2, subplot_titles=('In-degree Histogram', 'Out-degree Histogram'))

    # Add in-degree histogram in the first subplot (left)
    fig.add_trace(
        go.Histogram(x=in_degrees, nbinsx=20, name='In-degree', marker=dict(color='steelblue')),
        row=1, col=1
    )

    # Add out-degree histogram in the second subplot (right)
    fig.add_trace(
        go.Histogram(x=out_degrees, nbinsx=20, name='Out-degree', marker=dict(color='darkorange')),
        row=1, col=2
    )

    # Update layout for better aesthetics
    fig.update_layout(
        title="In-degree vs Out-degree Histograms",
        xaxis_title="Degree",
        yaxis_title="Frequency",
        showlegend=True,
        height=500,  # adjust the height of the figure
        width=1000   # adjust the width of the figure
    )



    # Calculate 90th percentile for in-degrees and out-degrees

    # Firstly we build a dictionary to compute the total degree value for each airport(node)
    dict_degrees = dict()

    for node, degrees in dict_degrees_edges.items():
        dict_degrees[node] = degrees[0] + degrees[1]
    

    # wE use np.precentile to obtaine the percentile from the dictionary of the degrees for each node 
    degree_percentile = np.percentile(list(dict_degrees.values()), 90)
    
    # Identify airports that are "hubs" (in-degree or out-degree greater than 90th percentile)
    hubs = []
    
    # Check for hubs
    for node,degree in dict_degrees.items():
        if degree > degree_percentile:
            hubs.append((node, degree))

    threshold = 0.5
    # We check if the Graph is dense or sparse
    if graph_density > threshold:
        is_sparse = False
    else:
        is_sparse = True

    return number_of_nodes, number_of_edges, fig, hubs, is_sparse


def summarize_graph_features(flight_network):
    # Analyze graph features
    number_of_nodes, number_of_edges, degree_histogram, hubs, is_sparse = analyze_graph_features(flight_network)

    # Create a textual summary
    density_description = "dense" if not is_sparse else "sparse"
    summary_table = f"""
| Metric                  | Value                      |
|-------------------------|----------------------------|
| **Number of Airports (Nodes)**      | {number_of_nodes}          |
| **Number of Flights (Edges)**       | {number_of_edges}          |
| **Graph Density**           | {'{:.4f}'.format((2 * number_of_edges) / (number_of_nodes * (number_of_nodes - 1)))}|
| **Graph Classification**    | {density_description.capitalize()} |
"""

    row_labels = "| Hubs (Airports)          | " + " | ".join([hub[0] for hub in hubs]) + " |\n"
    separator_row = "|-----------------| " + " | ".join(["---"] * len(hubs)) + " |\n"
    # Create the degree row
    degree_row = "| **Degrees**          | " + " | ".join([str(hub[1]) for hub in hubs]) + " |\n"

    # Combine rows into the Markdown table
    hubs_table = row_labels + separator_row + degree_row

    display(Markdown("## **Graph Features Summary**"))

    # Display summary
    display(Markdown(summary_table))

    display(Markdown("### **Identified Hubs**"))
    # Display the hubs table
    display(Markdown(hubs_table))

    # Display the degree distribution histogram
    display(Markdown("### **Degree Distribution**"))
    degree_histogram.show()



def analysis_traffic_passengers(df, number_of_busiest_routes = 10):

    df = df[df["Distance"] > 0]

    df_grouped_airports = df.groupby(['Origin_airport', 'Destination_airport'])['Passengers'].sum().reset_index()

    df_sorted_passengers = df_grouped_airports.sort_values(by = ["Passengers"], ascending = False)[["Origin_airport", "Destination_airport", "Passengers"]].head(number_of_busiest_routes)

    df_sorted_passengers = df_sorted_passengers.rename(columns={"Passengers": "Total_Passengers"})
    
    df_sorted_passengers['Route'] = df_sorted_passengers['Origin_airport'] + " -> " + df_sorted_passengers['Destination_airport']



    fig = px.bar(df_sorted_passengers, 
             x='Route', 
             y='Total_Passengers', 
             color='Total_Passengers', 
             title='Busiest Routes', 
             labels={'Total_Passengers': 'Total Number of Passengers'},
             barmode='group')
    

    df_average_traffic = df.groupby(['Origin_airport', 'Destination_airport'])['Passengers'].mean().reset_index()
    df_average_traffic = df_average_traffic.rename(columns = {"Passengers": "Average_Passengers"})
    df_average_traffic["Average_Passengers"] = df_average_traffic["Average_Passengers"]

    df_most_traffic = df_average_traffic.sort_values(by = ["Average_Passengers"], ascending = False)[["Origin_airport", "Destination_airport", "Average_Passengers"]].head(number_of_busiest_routes)

    df_least_traffic = df_average_traffic[["Origin_airport", "Destination_airport", "Average_Passengers"]].sort_values(by = ["Average_Passengers"], ascending = True).head(number_of_busiest_routes)


    return df_sorted_passengers[['Origin_airport', 'Destination_airport',"Total_Passengers"]].style.hide(axis="index"), fig, df_most_traffic.style.format({'Average_Passengers': '{:.1f}'}).hide(axis="index"), df_least_traffic.style.format({'Average_Passengers': '{:.1f}'}).hide(axis="index")



def create_interactive_map(df):

    m = folium.Map(location=[39.8283, -98.5795], zoom_start=5)
    df = df.loc[df.groupby(['Origin_airport', 'Destination_airport'])['Passengers'].idxmax()].reset_index(drop=True)
    marker_cluster = MarkerCluster().add_to(m)

# Loop over the routes in the DataFrame
    for _, row in df.iterrows():
       
      
        # Create a PolyLine for each route with the dynamic weight based on passengers
        folium.PolyLine(
            locations=[(row['Org_airport_lat'], row['Org_airport_long']), 
                    (row['Dest_airport_lat'], row['Dest_airport_long'])],
            color='#4682B4',
            weight=0.1,  # Line thickness 
            opacity=0.3
        ).add_to(m)

    

    # Add markers for the airports
    airports = pd.concat([df[['Origin_airport','Org_airport_lat','Org_airport_long', 'Origin_city','Origin_population' ]].rename(columns={'Origin_airport': "Airport", 'Org_airport_lat': 'lat', 'Org_airport_long':'long', 'Origin_city':"City", 'Origin_population': 'Population'}), df[['Destination_airport','Dest_airport_lat','Dest_airport_long', 'Destination_city', 'Destination_population']].rename(columns={'Destination_airport': "Airport", 'Dest_airport_lat': 'lat', 'Dest_airport_long':'long', 'Destination_city': "City", "Destination_population": "Population"})],axis = 0 ).drop_duplicates(subset='Airport').reset_index(drop=True)
    for _, row in airports.iterrows():
        airport = row['Airport'] + ', ' + row["City"]
        folium.Marker(
            location=[row['lat'], row['long']],
            popup=f"{airport}",
            icon=folium.Icon(icon = 'plane', prefix = 'fa', color='red')
        ).add_to(marker_cluster)

    m.save("flight_network_map.html")
    print("Saved map as 'flight_network_map.html'.")
    

def generate_report(df, flight_network, number_of_busiest_routes=10):

    
    # Generate graph features summary
    summarize_graph_features(flight_network)

    # Generate traffic analysis (top routes)
    df_sorted_passengers, fig, df_most_traffic, df_least_traffic = analysis_traffic_passengers(df, number_of_busiest_routes)

    # Busiest Routes (Table and Chart)
    display(Markdown("### **Top Routes by Passenger Flow**"))
    display(df_sorted_passengers)
    display(fig)


    display(Markdown("### **Under-Utilized Routes**"))
    display(df_least_traffic)

    display(Markdown("### **Over-Utilized Routes**"))
    display(df_most_traffic)

    display(Markdown("### **Top Routes by Passenger Efficiency**"))

    df = df[df["Distance"] > 0]
    df_grouped = df.groupby(['Origin_airport', 'Destination_airport']).agg(
    mean_passengers=('Passengers', 'mean'),
    first_distance=('Distance', 'first')
).reset_index()

    df_grouped["Passenger_Efficiency"] = df_grouped["mean_passengers"]/ df_grouped["first_distance"]
    
    # Sort by Passenger_Efficiency in descending order
    df_sorted_efficiency = df_grouped.sort_values(by='Passenger_Efficiency', ascending=False)
    
    # Select the top routes
    df_top_routes = df_sorted_efficiency[['Origin_airport', 'Destination_airport', 'Passenger_Efficiency']].head(number_of_busiest_routes)
    df_top_routes['Route'] = df_top_routes['Origin_airport'] + " -> " + df_top_routes['Destination_airport']

    # Display the table
    display(df_top_routes[['Origin_airport','Destination_airport', 'Passenger_Efficiency']].style.format({'Passenger_Efficiency': '{:.1f}'}).hide(axis="index"))
    

    # Create a bar chart for the top routes by passenger efficiency
    fig = px.bar(df_top_routes, 
                 x='Route', 
                 y='Passenger_Efficiency', 
                 color='Passenger_Efficiency', 
                 title='Top Routes by Passenger Efficiency',
                 labels={'Passenger_Efficiency': 'Passenger Efficiency'},
                 barmode='group')
    fig.show()



    







