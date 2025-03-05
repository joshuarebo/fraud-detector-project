"""
Fraud Detection System Architecture Visualization

This script generates a directed graph to represent the architecture of a fraud detection system.
The graph includes various components such as data ingestion, storage, model training, API deployment, 
and monitoring. The output is saved as an image.

Author: Joshua Rebo
Date: March 2025
"""

import os
import networkx as nx
import matplotlib.pyplot as plt


# ----------------------------- CONFIGURATION ----------------------------- #
# Define output directory
OUTPUT_DIR = "C:/Users/Hp/Fraud Detector/architecture/"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "system_architecture.png")

# ------------------------- FUNCTION DEFINITIONS ------------------------- #
def ensure_output_directory(directory: str):
    """
    Ensures that the output directory exists. If not, it creates it.
    
    :param directory: The path of the directory to check/create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"📁 Created directory: {directory}")
    else:
        print(f"✅ Directory already exists: {directory}")


def define_system_components():
    """
    Defines the core components and connections of the fraud detection system.

    :return: A tuple (components, edges) containing system components and their relationships.
    """
    components = {
        "User Input (Application)": (0, 4),
        "Data Ingestion Layer": (1, 3.5),
        "Data Storage (DB/CSV)": (2, 3),
        "Feature Engineering": (3, 2.5),
        "Model Training (ML/DL)": (4, 2),
        "Trained Model": (5, 1.5),
        "Fraud Detection API (FastAPI)": (6, 1),
        "Monitoring & MLOps (MLflow, Logging)": (7, 0.5),
        "Automated Retraining (CI/CD)": (8, 0),
    }

    edges = [
        ("User Input (Application)", "Data Ingestion Layer"),
        ("Data Ingestion Layer", "Data Storage (DB/CSV)"),
        ("Data Storage (DB/CSV)", "Feature Engineering"),
        ("Feature Engineering", "Model Training (ML/DL)"),
        ("Model Training (ML/DL)", "Trained Model"),
        ("Trained Model", "Fraud Detection API (FastAPI)"),
        ("Fraud Detection API (FastAPI)", "Monitoring & MLOps (MLflow, Logging)"),
        ("Monitoring & MLOps (MLflow, Logging)", "Automated Retraining (CI/CD)"),
        ("Automated Retraining (CI/CD)", "Model Training (ML/DL)"),
    ]

    return components, edges


def create_architecture_diagram(output_path: str):
    """
    Generates and saves a fraud detection system architecture diagram.

    :param output_path: The file path to save the diagram.
    """
    # Define system structure
    components, edges = define_system_components()

    # Create directed graph
    G = nx.DiGraph()
    G.add_nodes_from(components.keys())
    G.add_edges_from(edges)

    # Draw graph
    plt.figure(figsize=(12, 6))
    nx.draw(
        G,
        pos=components,
        with_labels=True,
        node_color="skyblue",
        edge_color="gray",
        node_size=3500,
        font_size=10,
        font_weight="bold",
        arrows=True,
        arrowsize=15
    )

    # Save the diagram
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Architecture diagram saved at: {output_path}")

    # Display the diagram
    plt.show()


# ------------------------- SCRIPT EXECUTION ------------------------- #
if __name__ == "__main__":
    print("\n🚀 Generating Fraud Detection System Architecture...\n")
    
    # Ensure the output directory exists
    ensure_output_directory(OUTPUT_DIR)
    
    # Generate and save architecture diagram
    create_architecture_diagram(OUTPUT_FILE)
    
    print("\n🎯 Process Complete! Your architecture diagram is ready.")
