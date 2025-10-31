import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import networkx as nx
from typing import List

class MarketBasketVisualizer:
    """
    Visualization tools for Market Basket Analysis
    """
    
    def __init__(self):
        self.colors = ['#21808D', '#32B8C6', '#5E5240', '#FCFCF9']
        sns.set_palette(sns.color_palette(self.colors))
        
    def plot_top_items(self, item_counts: pd.Series, n: int = 20, title: str = "Top Items"):
        """
        Plot top N items as bar chart
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        item_counts.head(n).plot(kind='barh', ax=ax, color='#21808D')
        ax.set_xlabel('Frequency', fontsize=12)
        ax.set_ylabel('Items', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_support_distribution(self, frequent_itemsets: pd.DataFrame):
        """
        Plot distribution of support values
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(frequent_itemsets['support'], bins=50, color='#32B8C6', edgecolor='black')
        ax.set_xlabel('Support', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Support Values', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_scatter_support_confidence(self, rules: pd.DataFrame):
        """
        Interactive scatter plot of support vs confidence colored by lift
        """
        fig = px.scatter(
            rules,
            x='support',
            y='confidence',
            size='lift',
            color='lift',
            hover_data=['antecedents', 'consequents'],
            title='Association Rules: Support vs Confidence (sized and colored by Lift)',
            labels={'support': 'Support', 'confidence': 'Confidence', 'lift': 'Lift'},
            color_continuous_scale='Teal'
        )
        fig.update_layout(height=600)
        return fig
    
    def plot_top_rules_bar(self, rules: pd.DataFrame, n: int = 10, metric: str = 'lift'):
        """
        Bar chart of top rules by a specific metric
        """
        top_rules = rules.nlargest(n, metric).copy()
        top_rules['rule'] = top_rules['antecedents'] + ' → ' + top_rules['consequents']
        
        fig = px.bar(
            top_rules,
            x=metric,
            y='rule',
            orientation='h',
            title=f'Top {n} Rules by {metric.capitalize()}',
            labels={metric: metric.capitalize(), 'rule': 'Association Rule'},
            color=metric,
            color_continuous_scale='Teal'
        )
        fig.update_layout(height=max(400, n * 40), yaxis={'categoryorder': 'total ascending'})
        return fig
    
    def plot_network_graph(self, rules: pd.DataFrame, top_n: int = 20):
        """
        Network graph visualization of association rules
        """
        # Take top N rules by lift
        top_rules = rules.nlargest(top_n, 'lift')
        
        # Create directed graph
        G = nx.DiGraph()
        
        for _, row in top_rules.iterrows():
            antecedents = str(row['antecedents']).split(', ')
            consequents = str(row['consequents']).split(', ')
            
            for ant in antecedents:
                for cons in consequents:
                    G.add_edge(ant, cons, weight=row['lift'])
        
        # Create layout
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # Create edge trace
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            hoverinfo='text',
            marker=dict(
                size=20,
                color='#21808D',
                line_width=2))
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=f'Association Rules Network (Top {top_n} by Lift)',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=0, l=0, r=0, t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           height=600
                       ))
        return fig
    
    def plot_heatmap(self, rules: pd.DataFrame, top_n: int = 20):
        """
        Heatmap of support, confidence, and lift for top rules
        """
        top_rules = rules.nlargest(top_n, 'lift').copy()
        top_rules['rule'] = top_rules['antecedents'] + ' → ' + top_rules['consequents']
        
        # Prepare data for heatmap
        heatmap_data = top_rules[['rule', 'support', 'confidence', 'lift']].set_index('rule')
        
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax, cbar_kws={'label': 'Value'})
        ax.set_title(f'Metrics Heatmap for Top {top_n} Rules', fontsize=14, fontweight='bold')
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Association Rules', fontsize=12)
        plt.tight_layout()
        return fig
