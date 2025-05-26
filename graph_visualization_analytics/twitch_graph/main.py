import pandas as pd
import networkx as nx
import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import eigs
from datetime import datetime
import matplotlib.pyplot as plt
import community as community_louvain # pip install python-louvain

# --- 0. Plotting Utilities ---
def plot_network_communities(G, partition, title="Network Communities", pos=None, save_path=None):
    """
    Plots the network with nodes colored by community.
    - G: NetworkX graph (preferably undirected for simple visualization)
    - partition: Dictionary mapping node to community ID
    - title: Title of the plot
    - pos: Optional layout for nodes (e.g., nx.spring_layout(G))
    - save_path: Optional path to save the plot image.
    """
    if G is None or not G.nodes():
        print("Graph is empty, cannot plot.")
        return

    if not partition:
        print("No partition provided, plotting without community colors.")
        node_colors = 'blue' # Default color
    else:
        # Create a list of colors for nodes based on their community
        # Using a colormap to get distinct colors
        cmap = plt.cm.get_cmap('viridis', max(partition.values()) + 1)
        node_colors = [cmap(partition[node]) if node in partition else 0.0 for node in G.nodes()]

    plt.figure(figsize=(12, 12))
    if pos is None:
        # For large graphs, spring_layout can be slow. Consider other layouts or subsampling for plotting.
        if G.number_of_nodes() > 1000:
            print(f"Warning: Graph has {G.number_of_nodes()} nodes. Spring layout might be very slow. Plotting a subgraph or using a faster layout is recommended for large graphs.")
            # Example: plot a subgraph of top K degree nodes if too large
            # top_k_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:200]
            # G_sub = G.subgraph([n[0] for n in top_k_nodes])
            # pos = nx.spring_layout(G_sub, k=0.15, iterations=20)
            # nx.draw_networkx(G_sub, pos, node_color=[cmap(partition[node]) if node in partition else 0.0 for node in G_sub.nodes()], with_labels=False, node_size=50, width=0.15)
            # For now, we'll attempt spring_layout but be mindful of performance.
            pos = nx.spring_layout(G, k=0.1, iterations=15) # Faster parameters
        else:
            pos = nx.spring_layout(G)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=50, alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5)
    
    # Consider drawing labels only for smaller graphs
    if G.number_of_nodes() < 100:
        nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title(title)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.show()

# --- 1. Data Loading and Preprocessing ---
def load_and_preprocess_data(csv_path="soc-sign-bitcoinalpha.csv"):
    """
    Loads the Bitcoin Alpha dataset and preprocesses it.
    """
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path, header=None, names=['SOURCE', 'TARGET', 'RATING', 'TIME'])
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return None

    print("Data loaded successfully.")
    df['DATETIME'] = pd.to_datetime(df['TIME'], unit='s') # More direct conversion
    print("Preprocessing complete.")
    return df

# --- 2. Network Construction ---
def build_network(df, time_weighting_scheme='absolute_rating', epsilon_days=None, current_time_dt=None):
    """
    Builds a NetworkX graph.
    - time_weighting_scheme:
        'absolute_rating': weight = abs(rating)
        'recency_scaled_rating': weight = abs(rating) * (timestamp / max_timestamp)
        'snapshot_window': Considers only edges within N days (epsilon_days) of current_time_dt. Weight = abs(rating).
        'time_decay_rating': weight = abs(rating) * exp(-decay_rate * age_in_days)
    - epsilon_days: Time window in days for snapshot or decay calculations.
    - current_time_dt: The reference datetime for recency/decay calculations. If None, uses max time in data.
    """
    if df is None:
        return None
        
    print(f"Building network with scheme: {time_weighting_scheme}...")
    G = nx.DiGraph()
    
    df_filtered = df.copy()

    if current_time_dt is None:
        current_time_dt = df_filtered['DATETIME'].max()
    
    current_time_ts = current_time_dt.timestamp() # For schemes needing numeric timestamps

    if time_weighting_scheme == 'snapshot_window' and epsilon_days is not None:
        start_time_dt = current_time_dt - pd.Timedelta(days=epsilon_days)
        df_filtered = df_filtered[(df_filtered['DATETIME'] >= start_time_dt) & (df_filtered['DATETIME'] <= current_time_dt)]

    if df_filtered.empty:
        print("Warning: No data to build the graph after time filtering.")
        return G

    max_timestamp_in_filtered_data = df_filtered['TIME'].max() # For recency scaling

    for _, row in df_filtered.iterrows():
        source, target, rating, timestamp = row['SOURCE'], row['TARGET'], row['RATING'], row['TIME']
        
        edge_attributes = {'rating': rating, 'timestamp': timestamp, 'datetime': row['DATETIME']}
        
        base_weight = abs(rating) # Most clustering algorithms prefer positive weights
        
        if time_weighting_scheme == 'recency_scaled_rating':
            # Normalize timestamp by max_timestamp in the current (potentially filtered) data
            # to get a recency factor between (approx) 0 and 1.
            recency_factor = timestamp / max_timestamp_in_filtered_data if max_timestamp_in_filtered_data > 0 else 1
            combined_weight = base_weight * recency_factor
        elif time_weighting_scheme == 'time_decay_rating' and epsilon_days is not None:
            # Example: exponential decay. epsilon_days defines the "half-life" or characteristic time.
            age_days = (current_time_dt - row['DATETIME']).total_seconds() / (24 * 60 * 60)
            decay_rate = np.log(2) / epsilon_days # Example: weight halves every epsilon_days
            decay_factor = np.exp(-decay_rate * age_days)
            combined_weight = base_weight * decay_factor
        elif time_weighting_scheme == 'snapshot_window': # Weight is just abs(rating) for snapshot
            combined_weight = base_weight
        else: # Default to 'absolute_rating'
            combined_weight = base_weight
            
        edge_attributes['weight'] = max(0.00001, combined_weight) # Ensure positive weight for Louvain

        G.add_edge(source, target, **edge_attributes)
        
    print(f"Network built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G

# --- 3.a. Modularity-based Clustering ---
def detect_communities_modularity(G, vary_expected_edges_method=None):
    if G is None or G.number_of_edges() == 0:
        print("Graph is empty, cannot perform community detection.")
        return None, None
        
    print("Detecting communities using modularity (Louvain)...")
    
    # Louvain works on undirected graphs with positive weights.
    H = G.to_undirected() if G.is_directed() else G.copy()
    
    # Ensure 'weight' attribute exists and is positive for all edges
    for u, v, data in H.edges(data=True):
        if 'weight' not in data or data['weight'] <= 0:
            # If weights are problematic, Louvain might behave unexpectedly or ignore them.
            # Assign a default small positive weight if missing or non-positive.
            # This is a choice that can affect results.
            # print(f"Warning: Edge ({u},{v}) has unsuitable weight {data.get('weight')}. Setting to 0.001 for Louvain.")
            H[u][v]['weight'] = 0.00001 # A very small positive weight

    if H.number_of_edges() > 0:
        partition = community_louvain.best_partition(H, weight='weight')
        modularity_score = community_louvain.modularity(partition, H, weight='weight')
        print(f"Louvain modularity: {modularity_score:.4f}")
        num_communities = len(set(partition.values()))
        print(f"Number of communities found: {num_communities}")
        
        # Add community attribute to nodes in the original graph G for other uses
        for node, comm_id in partition.items():
            if G.has_node(node): # Ensure node exists in G (might not if H was a component)
                 G.nodes[node]['community_louvain'] = comm_id
    else:
        partition = {}
        modularity_score = 0
        print("Graph has no edges; Louvain partition is empty.")

    # TODO (by user): Implement `vary_expected_edges_method` for custom modularity.
    # This would involve calculating Q = (1/2m) * sum_{ij} (A_ij - P_ij) * delta(c_i, c_j)
    # where P_ij is your custom expected number of edges.
    if vary_expected_edges_method:
        print(f"Custom modularity with '{vary_expected_edges_method}' is not yet implemented.")

    return partition, modularity_score

# --- 3.b. Hodge Laplacians and Eigenvector Analysis ---
def compute_incidence_matrix_B1(G):
    """Computes the node-to-edge incidence matrix B1."""
    if G is None or G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return None
    
    nodes = list(G.nodes())
    edges = list(G.edges()) # Using simple edges for B1 index
    
    node_map = {node: i for i, node in enumerate(nodes)}
    
    B1 = lil_matrix((len(nodes), len(edges)), dtype=np.float32)
    for edge_idx, (u, v) in enumerate(edges):
        B1[node_map[u], edge_idx] = -1 # Source
        B1[node_map[v], edge_idx] = 1  # Target
    return B1.tocsc(), nodes, edges, node_map

def compute_L0_laplacian(G, B1_csc=None):
    """Computes L0 (standard graph Laplacian) using B1 or NetworkX."""
    if G is None: return None
    print("Computing L0 (Node Laplacian)...")
    if B1_csc is not None:
        L0 = B1_csc @ B1_csc.transpose()
        print(f"L0 computed from B1. Shape: {L0.shape}")
        return L0
    else:
        # Fallback to NetworkX's Laplacian (typically for undirected)
        H = G.to_undirected() if G.is_directed() else G
        L0_nx = nx.laplacian_matrix(H)
        print(f"L0 computed using NetworkX. Shape: {L0_nx.shape}")
        return L0_nx

def get_fiedler_vector_L0(L0, H_nodes_list):
    """Computes the Fiedler vector from L0."""
    if L0 is None or L0.shape[0] < 2:
        print("L0 is too small for Fiedler vector computation.")
        return None, None, None
        
    print("Computing Fiedler vector from L0...")
    try:
        # k=2 for smallest algebraic eigenvalues. Second smallest is Fiedler.
        # Ensure k is less than matrix dimension.
        k_val = min(2, L0.shape[0] - 1) if L0.shape[0] > 1 else 1

        if L0.shape[0] >= 2 and k_val >=2 : # Need at least 2 eigenvalues for Fiedler
            eigenvalues, eigenvectors = eigs(L0, k=k_val, which='SM') # Smallest Magnitude
            
            # Sort eigenvalues and corresponding eigenvectors
            sorted_indices = np.argsort(eigenvalues.real)
            eigenvalues = eigenvalues.real[sorted_indices]
            eigenvectors = eigenvectors.real[:, sorted_indices]

            fiedler_value = eigenvalues[1]
            fiedler_vector_dense = eigenvectors[:, 1]
            
            # Create partition based on Fiedler vector sign
            fiedler_partition = {
                H_nodes_list[i]: (1 if fiedler_vector_dense[i] >= 0 else 0) 
                for i in range(len(H_nodes_list))
            }
            print(f"Fiedler vector computed. Eigenvalue: {fiedler_value:.4f}")
            return fiedler_vector_dense, fiedler_value, fiedler_partition
        else:
            print("Not enough nodes/dimensions for standard Fiedler vector computation.")
            return None, None, None
            
    except Exception as e:
        print(f"Error during L0 eigen-decomposition for Fiedler vector: {e}")
        return None, None, None

def compute_hodge_laplacian_L1_approx(G, B1_csc=None):
    """
    Computes an *approximate* L1 Hodge Laplacian (B1.T @ B1).
    The full L1 = B1.T @ B1 + B2 @ B2.T (boundary + coboundary).
    This version only computes the B1.T @ B1 part, which is related to edge connectivity
    and can detect gradient flows. True "holes" (cycles) are better captured by ker(L1_full).
    """
    if G is None or G.number_of_edges() == 0:
        print("Graph unsuitable for L1_approx computation (no edges).")
        return None, None, None
        
    if B1_csc is None:
        B1_csc, _, _, _ = compute_incidence_matrix_B1(G)
        if B1_csc is None: return None, None, None

    print("Computing L1_approx = B1.T @ B1...")
    L1_approx = B1_csc.transpose() @ B1_csc
    print(f"L1_approx (B1.T*B1) computed. Shape: {L1_approx.shape}")
    
    # Eigen-decomposition of L1_approx
    # Eigenvectors live on edges. Small eigenvalues are of interest.
    num_eigenvectors_l1 = min(5, L1_approx.shape[0] - 1) if L1_approx.shape[0] > 1 else 0
    
    if num_eigenvectors_l1 > 0:
        try:
            eigenvalues_l1, eigenvectors_l1 = eigs(L1_approx, k=num_eigenvectors_l1, which='SM')
            # Sort
            sorted_indices = np.argsort(eigenvalues_l1.real)
            eigenvalues_l1 = eigenvalues_l1.real[sorted_indices]
            eigenvectors_l1 = eigenvectors_l1.real[:, sorted_indices]
            print(f"Eigenvectors of L1_approx (shape: {eigenvectors_l1.shape}) computed.")
            return L1_approx, eigenvalues_l1, eigenvectors_l1
        except Exception as e:
            print(f"Error during L1_approx eigen-decomposition: {e}")
            return L1_approx, None, None
    else:
        print("Not enough edges for L1_approx eigenvector computation.")
        return L1_approx, None, None
    # Note: For true cycle detection / Betti_1, you'd need full L1 (with B2 for triangles)
    # and look at the kernel (eigenvalue 0). The eigenvectors of L1_approx relate more to
    # gradient and curl flows on edges.

# --- 3.c. Auto-clustering based on Hodge Laplacian similarity ---
# This remains highly conceptual as it requires defining "Hodge Laplacians"
# (e.g., L1 over different time windows), a similarity measure, and a clustering method.
def auto_cluster_hodge_laplacians(list_of_L_matrices_with_meta):
    """
    Placeholder for auto-clustering Hodge Laplacians.
    - list_of_L_matrices_with_meta: e.g., [(L1_matrix, {'time_window': 'T1'}), ...]
    """
    print("Conceptual: Auto-clustering Hodge Laplacians...")
    if not list_of_L_matrices_with_meta or len(list_of_L_matrices_with_meta) < 2:
        print("Not enough Hodge Laplacians provided to compare.")
        return {}

    # Example steps (user needs to implement):
    # 1. For each L_matrix in the list:
    #    - Potentially compute its spectrum (eigenvalues).
    # 2. Define a distance/similarity measure:
    #    - E.g., Euclidean distance between sorted spectra.
    #    - Frobenius norm of (L_i - L_j).
    # 3. Compute a pairwise distance matrix.
    # 4. Apply a clustering algorithm (e.g., Affinity Propagation, DBSCAN) on distances.
    # 5. Identify "top 3 most similar" or representative clusters.
    print("Auto-clustering for Hodge Laplacians is a research task and needs full implementation.")
    return {"status": "Not implemented"}


# --- 4. Validation ---
def validate_clusters(G, partition, partition_name="UnknownPartition", ground_truth_file=None):
    """
    Validates detected communities. Currently focuses on intrinsic and prints stats.
    Extrinsic validation needs a ground truth file.
    """
    print(f"\n--- Validating Clusters for {partition_name} ---")
    if not partition:
        print("No partition to validate.")
        return {}
        
    results = {'partition_name': partition_name}
    num_communities = len(set(partition.values()))
    results['num_communities'] = num_communities
    print(f"Number of communities: {num_communities}")

    if num_communities == 0 or num_communities == G.number_of_nodes():
        print("Partition is trivial (each node is its own community or all in one).")
        if G.number_of_nodes() >0 and num_communities == G.number_of_nodes() : results['modularity'] = -0.5 # Approx for all separate
        elif G.number_of_nodes() >0 and num_communities == 1 : results['modularity'] = 0.0 # Approx for all in one. Actual calc is better.


    # Modularity (if not already calculated, e.g., for Fiedler partition)
    if 'modularity' not in results and G.number_of_edges() > 0:
        H_val = G.to_undirected() if G.is_directed() else G
        # Ensure weights are positive for modularity calculation if G has 'weight'
        # This might need a temporary graph copy with adjusted weights if original G is kept.
        # For simplicity, assuming partition is valid for modularity calc.
        try:
            # Ensure H_val has weights suitable for modularity if partition was based on them
            # Temporary graph for modularity calculation if G's weights are problematic
            G_mod_calc = H_val.copy()
            valid_weights = True
            for u,v,d in G_mod_calc.edges(data=True):
                if d.get('weight', 1) <=0: # community_louvain.modularity needs positive weights
                    G_mod_calc[u][v]['weight'] = 0.00001 # Or some other strategy
            
            mod_score = community_louvain.modularity(partition, G_mod_calc, weight='weight')
            results['modularity'] = mod_score
            print(f"Calculated Modularity: {mod_score:.4f}")
        except Exception as e:
            print(f"Could not calculate modularity for {partition_name}: {e}")
            # This can happen if partition nodes don't match graph G_mod_calc nodes exactly.


    # TODO (by user): Implement extrinsic measures if ground_truth_file is provided.
    # This involves parsing ground_truth_file (mapping nodes to true communities)
    # and then using metrics like NMI, ARI from sklearn.metrics.
    # Example:
    # if ground_truth_file:
    #     true_labels_df = pd.read_csv(ground_truth_file) # node_id, true_community_id
    #     # Align true_labels and predicted_labels (partition) by nodes
    #     # common_nodes = ...
    #     # true_y = ... ; pred_y = ...
    #     # from sklearn.metrics import normalized_mutual_info_score
    #     # results['nmi'] = normalized_mutual_info_score(true_y, pred_y)
    #     print("Extrinsic validation (NMI, ARI) not implemented yet.")

    # For identifying specific entities (dark markets, mixers etc.):
    # This is the core research challenge. The soc-sign-bitcoinalpha dataset does not have these labels.
    # You would need to:
    # 1. Define characteristics of these entities.
    # 2. Analyze communities found by your methods (Louvain, Fiedler, Hodge-based).
    # 3. Look for communities whose structure/properties match these characteristics.
    # 4. Potentially use external data sources to verify if known entities fall into specific communities.
    print(f"Proving representation of specific entities (dark markets, etc.) for {partition_name} requires further research and likely external data/labeling.")
    
    return results

# --- 5. Interpretation and Comparative Analysis ---
def interpret_results(G, louvain_partition, fiedler_partition, l1_analysis, other_market_data_path=None):
    print("\n--- Interpretation of Results ---")
    
    if louvain_partition:
        print(f"Louvain found {len(set(louvain_partition.values()))} communities.")
        # TODO (by user): Analyze properties of Louvain communities (size, density, key nodes within).
        # For example, print sizes of top 5 communities:
        # from collections import Counter
        # comm_counts = Counter(louvain_partition.values())
        # print(f"Top 5 Louvain community sizes: {comm_counts.most_common(5)}")

    if fiedler_partition:
        print(f"Fiedler vector partitioned graph into {len(set(fiedler_partition.values()))} groups (typically 2).")
        # TODO (by user): Analyze these two partitions. Are they meaningful?

    if l1_analysis:
        L1_approx, eigenvalues_l1, eigenvectors_l1 = l1_analysis
        if eigenvalues_l1 is not None:
            print(f"L1_approx eigenvalues (smallest {len(eigenvalues_l1)}): {np.round(eigenvalues_l1, 4)}")
            # Interpretation: Eigenvectors of L1 (especially those for small eigenvalues if full L1 was used)
            # correspond to k-dimensional holes (cycles for L1).
            # For L1_approx = B1.T @ B1, small eigenvalues relate to "gradient flows" or paths of low resistance.
            # Eigenvectors live on edges. Clustering these edge eigenvectors could reveal important pathways or edge communities.
            print("Further analysis of L1 eigenvectors needed to interpret edge communities or cycles.")

    # Higher-dimensional holes (conceptual for now):
    # True interpretation of Betti numbers (Betti-0 components, Betti-1 cycles, Betti-2 voids)
    # would come from the kernel (null space) of the full Hodge Laplacians L0, L1, L2.
    # ker(L0) gives connected components.
    # ker(L1_full) gives independent cycles (1D holes).
    # This script only approximates L1.

    if other_market_data_path:
        print(f"\n--- Conceptual Comparative Analysis with {other_market_data_path} ---")
        # TODO (by user):
        # 1. Load data for the other market: df_other = load_and_preprocess_data(other_market_data_path)
        # 2. Build network: G_other = build_network(df_other, ...)
        # 3. Run same analyses: partition_other, fiedler_other, l1_other = ...
        # 4. Compare:
        #    - Graph metrics (density, clustering coefficient, diameter).
        #    - Community structure (number of communities, size distribution, modularity).
        #    - Hodge features (spectra of Laplacians, Betti numbers if computed).
        print("Comparative analysis not implemented. Would involve re-running pipeline on new data and comparing outputs.")

# --- Main Execution ---
if __name__ == "__main__":
    # Configuration
    DATA_PATH = "soc-sign-bitcoinalpha.csv"
    SAVE_PLOTS = True  # Set to True to save plots, False to only show
    PLOTS_DIR = "plots" # Directory to save plots

    if SAVE_PLOTS:
        import os
        if not os.path.exists(PLOTS_DIR):
            os.makedirs(PLOTS_DIR)
            print(f"Created directory: {PLOTS_DIR}")
    
    # --- Step 1: Load Data ---
    bitcoin_data_df = load_and_preprocess_data(DATA_PATH)
    
    if bitcoin_data_df is not None:
        # --- Step 2: Build Network ---
        # Choose a time weighting scheme for the main analyses
        # Options: 'absolute_rating', 'recency_scaled_rating', 'snapshot_window', 'time_decay_rating'
        # For snapshot or decay, current_time_dt and epsilon_days are relevant.
        # If None, current_time_dt will be max time in data.
        # Epsilon_days (e.g., 365 for a year, 30 for a month)
        
        # Example 1: Full graph, weights are absolute ratings
        # G = build_network(bitcoin_data_df, time_weighting_scheme='absolute_rating')

        # Example 2: Recency scaled ratings
        # G = build_network(bitcoin_data_df, time_weighting_scheme='recency_scaled_rating')
        
        # Example 3: Snapshot of last 365 days, weights are absolute ratings
        current_analysis_time = bitcoin_data_df['DATETIME'].max() # Or a specific date: pd.to_datetime('YYYY-MM-DD')
        days_window = 365 
        # G = build_network(bitcoin_data_df,
        #                   time_weighting_scheme='snapshot_window',
        #                   current_time_dt=current_analysis_time,
        #                   epsilon_days=days_window)
        
        # For now, let's use a simpler scheme for demonstration
        G = build_network(bitcoin_data_df, time_weighting_scheme='absolute_rating')

        # --- Step 3: Clustering & Analysis ---
        partition_louvain = None
        fiedler_results = {'vector': None, 'value': None, 'partition': None}
        l1_analysis_results = {'L1_approx': None, 'eigenvalues': None, 'eigenvectors': None}

        if G and G.number_of_nodes() > 0:
            # --- 3.a Modularity (Louvain) ---
            partition_louvain, score_louvain = detect_communities_modularity(G)
            if partition_louvain and SAVE_PLOTS:
                # For plotting, use an undirected version of G if it's directed.
                # Layout calculation for large graphs can be slow.
                H_plot = G.to_undirected() if G.is_directed() else G.copy()
                # Downsample graph for plotting if too large
                if H_plot.number_of_nodes() > 500:
                    nodes_to_plot = list(H_plot.nodes())[::int(H_plot.number_of_nodes()/500)] # simple subsample
                    # Or by degree: top_nodes = sorted(H_plot.degree, key=lambda item: item[1], reverse=True)[:500]
                    # plot_subgraph = H_plot.subgraph([n[0] for n in top_nodes])
                    plot_subgraph = H_plot.subgraph(nodes_to_plot)
                    print(f"Plotting a subsample of {plot_subgraph.number_of_nodes()} nodes for Louvain communities.")
                    plot_partition_louvain = {n: partition_louvain[n] for n in plot_subgraph.nodes() if n in partition_louvain}
                    plot_network_communities(plot_subgraph, plot_partition_louvain, 
                                             title=f"Louvain Communities (Modularity: {score_louvain:.3f}, Subgraph)",
                                             save_path=f"{PLOTS_DIR}/louvain_communities_subgraph.png")
                else:
                     plot_network_communities(H_plot, partition_louvain, 
                                             title=f"Louvain Communities (Modularity: {score_louvain:.3f})",
                                             save_path=f"{PLOTS_DIR}/louvain_communities.png")


            # --- 3.b L0 Fiedler Vector & L1 Approx ---
            B1, graph_nodes_list, graph_edges_list, node_to_idx_map = compute_incidence_matrix_B1(G)
            
            if B1 is not None:
                L0 = compute_L0_laplacian(G, B1_csc=B1) # Use B1 to ensure consistency if G is directed
                
                # Node list for Fiedler vector (ensure it matches L0's construction)
                # If L0 from B1: graph_nodes_list
                # If L0 from nx.laplacian_matrix(H): list(H.nodes())
                # Here B1 is from G's nodes, so graph_nodes_list should align
                fvec, fval, fpart = get_fiedler_vector_L0(L0, graph_nodes_list)
                fiedler_results = {'vector': fvec, 'value': fval, 'partition': fpart}
                if fpart and SAVE_PLOTS:
                    H_plot_fiedler = G.to_undirected() if G.is_directed() else G.copy()
                    # Similar subsampling for large graphs
                    if H_plot_fiedler.number_of_nodes() > 500:
                        nodes_to_plot_f = list(H_plot_fiedler.nodes())[::int(H_plot_fiedler.number_of_nodes()/500)]
                        plot_subgraph_f = H_plot_fiedler.subgraph(nodes_to_plot_f)
                        print(f"Plotting a subsample of {plot_subgraph_f.number_of_nodes()} nodes for Fiedler partition.")
                        plot_fpart = {n: fpart[n] for n in plot_subgraph_f.nodes() if n in fpart}
                        plot_network_communities(plot_subgraph_f, plot_fpart,
                                                 title=f"Fiedler Partition (Eigenvalue: {fval:.3f} approx, Subgraph)",
                                                 save_path=f"{PLOTS_DIR}/fiedler_partition_subgraph.png")
                    else:
                        plot_network_communities(H_plot_fiedler, fpart,
                                                 title=f"Fiedler Partition (Eigenvalue: {fval:.3f} approx)",
                                                 save_path=f"{PLOTS_DIR}/fiedler_partition.png")


                L1_approx, l1_vals, l1_vecs = compute_hodge_laplacian_L1_approx(G, B1_csc=B1)
                l1_analysis_results = {'L1_approx': L1_approx, 'eigenvalues': l1_vals, 'eigenvectors': l1_vecs}
                # Plotting L1 eigenvectors is more complex as they live on edges.
                # Could, for example, color edges by the value of a component of an L1 eigenvector.
                # This requires more specific visualization choices by the user.

            # --- 3.c Auto-clustering (Conceptual) ---
            # list_L_matrices = [] # Populate this with (L_matrix, metadata) from different graph snapshots/subgraphs
            # if list_L_matrices:
            #    auto_cluster_hodge_laplacians(list_L_matrices)
            
            # --- Step 4: Validation ---
            if partition_louvain:
                validate_clusters(G, partition_louvain, partition_name="Louvain")
            if fiedler_results['partition']:
                 validate_clusters(G, fiedler_results['partition'], partition_name="Fiedler_L0")
            
            # --- Step 5: Interpretation & Comparative Analysis ---
            interpret_results(G, partition_louvain, fiedler_results['partition'], 
                              (l1_analysis_results['L1_approx'], l1_analysis_results['eigenvalues'], l1_analysis_results['eigenvectors'])
                              # other_market_data_path="path_to_other_market_data.csv" # Example
                             )
        else:
            print("Graph could not be built or is empty. Skipping analysis steps.")

        print("\n--- Project Script Execution Finished ---")
        print("Remember: Many advanced parts (full Hodge L1/L2, custom modularity,")
        print("Hodge auto-clustering, specific entity identification) are conceptual")
        print("and require significant research and implementation.")
        if SAVE_PLOTS:
            print(f"Plots, if generated, are saved in the '{PLOTS_DIR}' directory.")
