import networkx as nx
import csv

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
                    distance=int(row['Distance']), 
                    fly_date=row['Fly_date'])

    return G


