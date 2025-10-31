import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from typing import Tuple

class AprioriAnalyzer:
    """
    Performs Apriori algorithm and generates association rules
    """
    
    def __init__(self, basket_df: pd.DataFrame):
        self.basket = basket_df
        self.frequent_itemsets = None
        self.rules = None
        
    def find_frequent_itemsets(self, min_support: float = 0.01, use_colnames: bool = True) -> pd.DataFrame:
        """
        Find frequent itemsets using Apriori algorithm
        
        Parameters:
        -----------
        min_support : float
            Minimum support threshold (default: 0.01)
        use_colnames : bool
            Use column names instead of indices
            
        Returns:
        --------
        pd.DataFrame: Frequent itemsets with support values
        """
        self.frequent_itemsets = apriori(
            self.basket, 
            min_support=min_support, 
            use_colnames=use_colnames
        )
        
        # Add length column
        self.frequent_itemsets['length'] = self.frequent_itemsets['itemsets'].apply(lambda x: len(x))
        
        return self.frequent_itemsets
    
    def generate_rules(self, metric: str = "confidence", min_threshold: float = 0.5) -> pd.DataFrame:
        """
        Generate association rules from frequent itemsets
        
        Parameters:
        -----------
        metric : str
            Metric to use for filtering rules ('support', 'confidence', 'lift')
        min_threshold : float
            Minimum threshold for the metric
            
        Returns:
        --------
        pd.DataFrame: Association rules with metrics
        """
        if self.frequent_itemsets is None:
            raise ValueError("Run find_frequent_itemsets() first!")
        
        self.rules = association_rules(
            self.frequent_itemsets, 
            metric=metric, 
            min_threshold=min_threshold
        )
        
        # Convert frozensets to lists for better display
        self.rules['antecedents'] = self.rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        self.rules['consequents'] = self.rules['consequents'].apply(lambda x: ', '.join(list(x)))
        
        return self.rules
    
    def filter_rules(self, min_confidence: float = None, min_lift: float = None, 
                     min_support: float = None) -> pd.DataFrame:
        """
        Filter rules based on multiple criteria
        """
        if self.rules is None:
            raise ValueError("Generate rules first using generate_rules()!")
        
        filtered = self.rules.copy()
        
        if min_confidence is not None:
            filtered = filtered[filtered['confidence'] >= min_confidence]
        
        if min_lift is not None:
            filtered = filtered[filtered['lift'] >= min_lift]
        
        if min_support is not None:
            filtered = filtered[filtered['support'] >= min_support]
        
        return filtered
    
    def get_top_rules(self, n: int = 10, sort_by: str = 'lift') -> pd.DataFrame:
        """
        Get top N rules sorted by a specific metric
        """
        if self.rules is None:
            raise ValueError("Generate rules first!")
        
        return self.rules.nlargest(n, sort_by)
    
    def get_statistics(self) -> dict:
        """
        Get summary statistics
        """
        stats = {}
        
        if self.frequent_itemsets is not None:
            stats['total_frequent_itemsets'] = len(self.frequent_itemsets)
            stats['avg_support'] = round(self.frequent_itemsets['support'].mean(), 4)
            stats['max_support'] = round(self.frequent_itemsets['support'].max(), 4)
        
        if self.rules is not None:
            stats['total_rules'] = len(self.rules)
            stats['avg_confidence'] = round(self.rules['confidence'].mean(), 4)
            stats['avg_lift'] = round(self.rules['lift'].mean(), 4)
            stats['max_confidence'] = round(self.rules['confidence'].max(), 4)
            stats['max_lift'] = round(self.rules['lift'].max(), 4)
        
        return stats
