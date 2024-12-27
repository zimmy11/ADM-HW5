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
import heapq
from collections import deque


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
    # Initialize variables to store the number of nodes and edges
    number_of_nodes = 0
    number_of_edges = len(flight_network.edges())  # Count the number of edges in the graph
    dict_degrees_edges = dict()  # Create an empty dictionary to store in-degrees and out-degrees for each node

    # Iterate over each node in the flight network
    for node in flight_network.nodes:
        number_of_nodes = number_of_nodes + 1  # Increment the node count

        in_edges = 0  # Initialize in-degree counter
        out_edges = 0  # Initialize out-degree counter

        # Count the outgoing edges from the current node
        for _, _, attr in flight_network.edges(node, data=True):
            out_edges += 1

        # Count the incoming edges to the current node
        for _, _, attr in flight_network.in_edges(node, data=True):
            in_edges += 1

        # Store the in-degree and out-degree for the current node in the dictionary
        dict_degrees_edges[node] = [in_edges, out_edges]

    # Calculate graph density using the formula
    graph_density = (number_of_edges) / (number_of_nodes * (number_of_nodes - 1))

    # Extract the in-degrees and out-degrees from the dictionary for histogram plotting
    in_degrees = [edge_degree[0] for edge_degree in dict_degrees_edges.values()]
    out_degrees = [edge_degree[1] for edge_degree in dict_degrees_edges.values()]

    # Create a subplot with 1 row and 2 columns to display histograms for in-degrees and out-degrees
    fig = make_subplots(rows=1, cols=2, subplot_titles=('In-degree Histogram', 'Out-degree Histogram'))

    # Add in-degree histogram to the first subplot (left)
    fig.add_trace(
        go.Histogram(x=in_degrees, nbinsx=20, name='In-degree', marker=dict(color='steelblue')),
        row=1, col=1
    )

    # Add out-degree histogram to the second subplot (right)
    fig.add_trace(
        go.Histogram(x=out_degrees, nbinsx=20, name='Out-degree', marker=dict(color='darkorange')),
        row=1, col=2
    )

    # Update layout for better aesthetics
    fig.update_layout(
        title="In-degree vs Out-degree Histograms",  # Set the title of the plot
        xaxis_title="Degree",  # Label for x-axis
        yaxis_title="Frequency",  # Label for y-axis
        showlegend=True,  # Display legend
        height=500,  # Adjust the height of the figure
        width=1000   # Adjust the width of the figure
    )

    # Calculate the 90th percentile for the total degrees (in-degree + out-degree) of each node

    # Build a dictionary to compute the total degree value (in-degree + out-degree) for each node
    dict_degrees = dict()

    # Iterate over the nodes and calculate the total degree for each
    for node, degrees in dict_degrees_edges.items():
        dict_degrees[node] = degrees[0] + degrees[1]  # Sum in-degree and out-degree for each node

    # Use numpy's percentile function to get the 90th percentile of the degrees
    degree_percentile = np.percentile(list(dict_degrees.values()), 90)

    # Identify nodes (airports) that are "hubs", meaning their total degree is greater than the 90th percentile
    hubs = []

    # Iterate over the nodes and check if their total degree exceeds the 90th percentile
    for node, degree in dict_degrees.items():
        if degree > degree_percentile:
            hubs.append((node, degree))  # Add the node and its degree to the list of hubs

    threshold = 0.5  # Set a threshold to decide if the graph is dense or sparse

    # Check if the graph is dense or sparse based on the density calculated earlier
    if graph_density > threshold:
        is_sparse = False  # If the density is greater than the threshold, the graph is considered dense
    else:
        is_sparse = True  # Otherwise, the graph is considered sparse

    # Return the number of nodes, the number of edges, the figure with histograms, the list of hubs, and whether the graph is sparse
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



def analysis_traffic_passengers(df, number_of_busiest_routes=10):
    '''
    Input:

    df: input dataframe
    number_of_busiest_routes: number of values we wanna return in the sorted DataFrame
    
    Return:
    - A styled DataFrame for the busiest routes, with total passengers.
    - A Plotly bar plot for visualizing the busiest routes.
    - A styled DataFrame showing the routes with the most average passengers.
    - A styled DataFrame showing the routes with the least average passengers.
    
    '''

    # Group the data by 'Origin_airport' and 'Destination_airport', then sum the 'Passengers' for each route
    df_grouped_airports = df.groupby(['Origin_airport', 'Destination_airport'])['Passengers'].sum().reset_index()

    # Sort the grouped data by the 'Passengers' column in descending order and select the top routes
    df_sorted_passengers = df_grouped_airports.sort_values(by=["Passengers"], ascending=False)[["Origin_airport", "Destination_airport", "Passengers"]].head(number_of_busiest_routes)

    # Rename the 'Passengers' column to 'Total_Passengers' for clarity
    df_sorted_passengers = df_sorted_passengers.rename(columns={"Passengers": "Total_Passengers"})

    # Create a new 'Route' column combining the 'Origin_airport' and 'Destination_airport' for easy display
    df_sorted_passengers['Route'] = df_sorted_passengers['Origin_airport'] + " -> " + df_sorted_passengers['Destination_airport']

    # Create a bar plot using Plotly to visualize the busiest routes with total passengers
    fig = px.bar(df_sorted_passengers, 
                 x='Route', 
                 y='Total_Passengers', 
                 color='Total_Passengers', 
                 title='Busiest Routes', 
                 labels={'Total_Passengers': 'Total Number of Passengers'}, 
                 barmode='group')

    # Calculate the average number of passengers per route by grouping by 'Origin_airport' and 'Destination_airport'
    df_average_traffic = df.groupby(['Origin_airport', 'Destination_airport'])['Passengers'].mean().reset_index()

    # Rename the 'Passengers' column to 'Average_Passengers'
    df_average_traffic = df_average_traffic.rename(columns={"Passengers": "Average_Passengers"})

    # Sort the routes by average passengers in descending order and select the top routes
    df_most_traffic = df_average_traffic.sort_values(by=["Average_Passengers"], ascending=False)[["Origin_airport", "Destination_airport", "Average_Passengers"]].head(number_of_busiest_routes)

    # Sort the routes by average passengers in ascending order and select the bottom routes
    df_least_traffic = df_average_traffic[["Origin_airport", "Destination_airport", "Average_Passengers"]].sort_values(by=["Average_Passengers"], ascending=True).head(number_of_busiest_routes)


    return df_sorted_passengers[['Origin_airport', 'Destination_airport', "Total_Passengers"]].style.hide(axis="index"), fig, df_most_traffic.style.format({'Average_Passengers': '{:.1f}'}).hide(axis="index"), df_least_traffic.style.format({'Average_Passengers': '{:.1f}'}).hide(axis="index")



def create_interactive_map(df):
    """
    This function generates an interactive map to visualize flight routes and airports using Folium.

    Input:
    - A pandas DataFrame (`df`) containing flight and airport information.

    Output:
    - An interactive map saved as an HTML file, visualizing the busiest routes and airports, with airport markers 
      and flight routes displayed.
    """
    m = folium.Map(location=[39.8283, -98.5795], zoom_start=5)  # Initialize the map centered at the US
    df = df.loc[df.groupby(['Origin_airport', 'Destination_airport'])['Passengers'].idxmax()].reset_index(drop=True)  # Filter for the busiest routes
    marker_cluster = MarkerCluster().add_to(m)  # Add a MarkerCluster for the airport markers

    # Loop over the routes in the DataFrame and add PolyLines to represent the flight routes
    for _, row in df.iterrows():
        folium.PolyLine(
            locations=[(row['Org_airport_lat'], row['Org_airport_long']), 
                    (row['Dest_airport_lat'], row['Dest_airport_long'])],
            color='#4682B4',  # Line color (steel blue)
            weight=0.1,  # Line thickness
            opacity=0.3  # Line transparency
        ).add_to(m)  # Add the PolyLine to the map

    # Combine origin and destination airport data into a single DataFrame for markers
    airports = pd.concat([df[['Origin_airport','Org_airport_lat','Org_airport_long', 'Origin_city','Origin_population' ]].rename(columns={'Origin_airport': "Airport", 'Org_airport_lat': 'lat', 'Org_airport_long':'long', 'Origin_city':"City", 'Origin_population': 'Population'}),
                          df[['Destination_airport','Dest_airport_lat','Dest_airport_long', 'Destination_city', 'Destination_population']].rename(columns={'Destination_airport': "Airport", 'Dest_airport_lat': 'lat', 'Dest_airport_long':'long', 'Destination_city': "City", "Destination_population": "Population"})], axis=0)
    airports = airports.drop_duplicates(subset='Airport').reset_index(drop=True)  # Remove duplicate airports

    # Loop over the airport data and add markers for each airport
    for _, row in airports.iterrows():
        airport = row['Airport'] + ', ' + row["City"]
        folium.Marker(
            location=[row['lat'], row['long']],
            popup=f"{airport}",  # Show airport name and city in the popup
            icon=folium.Icon(icon='plane', prefix='fa', color='red')  # Red plane icon
        ).add_to(marker_cluster)  # Add the marker to the map

    # Save the map to an HTML file for interactive exploration
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

    # Under-Utilized Routes
    display(Markdown("### **Under-Utilized Routes**"))
    display(df_least_traffic)

    # Over-Utilized Routes
    display(Markdown("### **Over-Utilized Routes**"))
    display(df_most_traffic)

    # Top Routes by Passenger Efficiency
    display(Markdown("### **Top Routes by Passenger Efficiency**"))

    # We filter out all the routes that have distance = 0
    df = df[df["Distance"] > 0]
    df_grouped = df.groupby(['Origin_airport', 'Destination_airport']).agg(
    mean_passengers=('Passengers', 'mean'),
    first_distance=('Distance', 'first')
).reset_index()

    # We compute the Passenger Efficiency dividing the Average number of Passengers by the total KM of the flight
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


def filter_graph_by_date(G, attr, date):
    """
    Filters the input graph by the given date. This function removes edges that don't match the specified date 
    based on the provided attribute and returns a new graph containing only relevant edges and nodes.

    Parameters:
    G (nx.Graph): The original graph to be filtered.
    attr (str): The attribute in the edge data to filter on (e.g., "fly_date").
    date (str): The specific date to filter edges by.

    Returns:
    nx.Graph: A new graph containing only the edges that match the specified date.
    """
    G_filtered = nx.Graph()  # Create a new graph to store the filtered edges and nodes.
    
    # Loop over each edge in the original graph
    for u, v, data in G.edges(data=True):
        # If the edge matches the specified date, add it to the filtered graph
        if data.get(attr, 0) == date:
            G_filtered.add_edge(u, v, **data)  # Add the edge to the filtered graph
            G_filtered.add_node(u, **G.nodes[u])  # Add the origin node to the filtered graph
            G_filtered.add_node(v, **G.nodes[v])  # Add the destination node to the filtered graph
    
    return G_filtered  # Return the filtered graph.


def compute_Dijkstra(flight_network, source):
    """
    Computes the shortest paths from the source node to all other nodes in the graph using Dijkstra's algorithm.

    Parameters:
    flight_network (nx.Graph): The graph representing the flight network with weighted edges (flight routes).
    source (str): The starting node (airport) from which the shortest paths will be calculated.

    Returns:
    tuple: A tuple containing:
        - distances_dict (dict): A dictionary with the shortest distance from the source to each node.
        - prev (dict): A dictionary mapping each node to its previous node in the shortest path.
    """

    # Initialize the distance to the source as 0 and others as infinity.    
    distances_dict = {source: 0}  
    # Initialize the previous node dictionary to track the path.
    prev = {source: None}  
    # Priority queue to select the node with the smallest current distance.
    pq = [(0, source)]  
   
    # Set initial distances to infinity for all nodes except the source.
    for node, data in flight_network.nodes(data=True):

        if node != source:
             # Set all other nodes' distances to infinity.
            distances_dict[node] = np.inf 
        # Initialize previous nodes as None.
        prev[node] = None 
    
    # Main loop of Dijkstra's algorithm
    while pq:
         # Pop the node with the smallest distance.
        current_distance, current_node = heapq.heappop(pq) 

        # Loop through all neighboring nodes connected to the current node
        for next_node, data in flight_network[current_node].items():

            distance = current_distance + data["weight"]  # Calculate the distance to the next node.
            
            # If the new distance is smaller, update the shortest distance and previous node
            if distance < distances_dict[next_node]:
                # We update the distance_dict and we push the new instance in the Heap
                distances_dict[next_node] = distance
                prev[next_node] = current_node
                heapq.heappush(pq, (distance, next_node)) 
                
    return distances_dict, prev  # Return the distances and previous nodes dictionaries.


def reconstruct_path(prev, destination, source):
    """
    Reconstructs the shortest path from the source to the destination node based on the previous node dictionary.

    Parameters:
    prev (dict): A dictionary mapping each node to its previous node in the shortest path.
    destination (str): The destination node where the path ends.
    source (str): The source node where the path starts.

    Returns:
    list: A list of nodes representing the shortest path from source to destination.
    """
    path = []  # List to store the reconstructed path.
    current_node = destination  # Start from the destination node.
    
    # Trace back the path from destination to source using the 'prev' dictionary
    while current_node is not None:
        path.append(current_node)  # Append the current node to the path.
        current_node = prev.get(current_node, None)  # Move to the previous node.
    
    # If the path starts with the source, reverse the path to get the correct order
    if path and path[-1] == source:
        path.reverse()  # Reverse the path to get the correct order from source to destination.
        return path
    else:
        return []  # If no valid path exists, return an empty list.


def compute_best_route(graph, origin, destination, date):
    """
    Computes the best route (shortest path) between the origin and destination cities based on the specified date.
    

    Parameters:
    graph (nx.Graph): The graph representing the flight network.
    origin (str): The origin city or airport.
    destination (str): The destination city or airport.
    date (str): The date for which the best route is to be computed.

    Returns:
    pd.DataFrame: A pandas DataFrame containing the origin city, destination city, and the best route.
    """

    # Filter the graph by the specified date to only include the relevant edges.
    filtered_network = filter_graph_by_date(graph, "fly_date", date)
    
    # Find all airports in the origin and destination cities within the filtered network.
    origin_airports = [node for node, data in filtered_network.nodes(data=True) if origin in data.get("city", "")]
    destination_airports = [node for node, data in filtered_network.nodes(data=True) if destination in data.get("city", "")]
    
    best_distance = np.inf  # Initialize the best distance to infinity (for comparison).
    best_route = ''  # Initialize the best route as an empty string.

    # For each airport in the origin list, find the shortest route to the destination airports.
    for source in origin_airports:
        distances_dict, prev = compute_Dijkstra(filtered_network, source)  # Compute the shortest paths from the source.
        
        # For each destination airport, check if the path distance is shorter than the best known distance.
        for i in range(len(destination_airports)):
            distance = distances_dict[destination_airports[i]]
            
            # If a better route is found, update the best distance and route.
            if distance < best_distance:
                best_distance = distance
                best_route = "->".join(reconstruct_path(prev, destination_airports[i], source))  # Reconstruct the path.
    
    # If no valid route was found, return a message indicating no route.
    if best_route == '':
        best_route = 'No route found.'
    
    # Create a pandas DataFrame with the origin, destination, and best route information.
    data = {
        'Origin_city_airport': [origin],
        'Destination_city_airport': [destination],
        'Best_route': [best_route]    
    }
    
    # Return the result as a DataFrame.
    return pd.DataFrame(data)  

def degree_centrality(graph):
    n = len(graph)
    centrality = {}

    for node in graph:
        degree = len(graph[node])  # numero di vicini del nodo
        centrality[node] = degree / (n - 1) if n > 1 else 0.0

    return centrality


def closeness_centrality(graph):

    n = len(graph)
    centrality = {}

    def bfs_distances(start):
        
        dist = {node: None for node in graph}
        dist[start] = 0
        queue = deque([start])

        while queue:
            current = queue.popleft()
            for neighbor in graph[current]:
                if dist[neighbor] is None:
                    dist[neighbor] = dist[current] + 1
                    queue.append(neighbor)
        return dist

    for node in graph:
        dist_dict = bfs_distances(node)
        
        reachable_distances = [d for d in dist_dict.values() if d is not None and d > 0]
        
        if len(reachable_distances) == 0:
          
            centrality[node] = 0.0
        else:
            avg_distance = sum(reachable_distances) / len(reachable_distances)
            
            centrality[node] = (len(reachable_distances)) / sum(reachable_distances)
           

    return centrality

def betweenness_centrality(graph):
 
    n = len(graph)
    betweenness = {v: 0.0 for v in graph}  # inizializza a 0

    for s in graph:
        
        stack = []
        
        pred = {v: [] for v in graph}
       
        sigma = {v: 0.0 for v in graph}
        sigma[s] = 1.0
        
        dist = {v: -1 for v in graph}
        dist[s] = 0

        
        queue = [s]
        for v in queue:
            for w in graph[v]:
                
                if dist[w] < 0:
                    queue.append(w)
                    dist[w] = dist[v] + 1
                
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
            stack.append(v)

      
        delta = {v: 0.0 for v in graph}
        while stack:
            w = stack.pop()
            for v in pred[w]:
               
                c = (sigma[v] / sigma[w]) * (1 + delta[w])
                delta[v] += c
            
            if w != s:
                betweenness[w] += delta[w]
    for v in graph:
        betweenness[v] /= 2.0

    return betweenness

import numpy as np

def pagerank(graph, alpha=0.85, max_iter=100, tol=1e-6):
    """
    Computes the PageRank of each node in an unweighted graph 
    (if directed, adapt the transition matrix accordingly).
    graph: dict -> { node: set(neighbor_nodes), ... }
    alpha: damping factor (default 0.85)
    max_iter: maximum number of iterations (default 100)
    tol: convergence tolerance (default 1e-6)

    Returns a dict: {node: PR_value, ...}
    """
    # 1) Map nodes to indices (and vice versa)
    nodes = list(graph.keys())
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}

    # 2) Compute out-degrees
    outdegree = {node: len(graph[node]) for node in graph}

    # 3) Build the transition matrix T (n x n)
    #    T[i, j] = 1 / outdegree(j) if there is an edge j->i
    T = np.zeros((n, n), dtype=float)

    for node_j in graph:
        j = idx[node_j]
        if outdegree[node_j] == 0:
            # Dangling node -> distribute evenly
            for node_i in graph:
                i = idx[node_i]
                T[i, j] = 1.0 / n
        else:
            # Normalize among its neighbors
            for node_i in graph[node_j]:
                i = idx[node_i]
                T[i, j] = 1.0 / outdegree[node_j]

    # 4) Initialize PageRank vector
    PR = np.ones(n, dtype=float) / n

    # 5) Power iteration until convergence
    for _ in range(max_iter):
        PR_new = (1 - alpha) / n + alpha * T.dot(PR)
        if np.linalg.norm(PR_new - PR, ord=1) < tol:
            PR = PR_new
            break
        PR = PR_new

    # 6) Return results as a dictionary
    pagerank_dict = {}
    for node in graph:
        pagerank_dict[node] = PR[idx[node]]

    return pagerank_dict



    






