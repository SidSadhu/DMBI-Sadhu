import pandas as pd
import numpy as np
from typing import List, Tuple

class DataPreprocessor:
    """
    Handles data cleaning and preprocessing for Market Basket Analysis
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def clean_data(self) -> pd.DataFrame:
        """
        Perform comprehensive data cleaning
        """
        # Remove duplicates
        self.df.drop_duplicates(inplace=True)
        
        # Remove rows with missing invoice numbers
        self.df.dropna(subset=['InvoiceNo'], inplace=True)
        
        # Remove rows with missing descriptions
        self.df.dropna(subset=['Description'], inplace=True)
        
        # Remove canceled transactions (those starting with 'C')
        self.df = self.df[~self.df['InvoiceNo'].astype(str).str.startswith('C')]
        
        # Remove negative quantities
        if 'Quantity' in self.df.columns:
            self.df = self.df[self.df['Quantity'] > 0]
        
        # Strip whitespace and convert to lowercase
        self.df['Description'] = self.df['Description'].str.strip().str.lower()
        
        # Remove special characters from descriptions
        self.df['Description'] = self.df['Description'].str.replace('[^a-zA-Z0-9 ]', '', regex=True)
        
        return self.df
    
    def create_basket(self, country: str = None) -> pd.DataFrame:
        """
        Create transaction basket format for Apriori algorithm
        
        Parameters:
        -----------
        country : str, optional
            Filter transactions by country
        
        Returns:
        --------
        pd.DataFrame: Basket format with items as columns
        """
        df = self.df.copy()
        
        # Filter by country if specified
        if country and 'Country' in df.columns:
            df = df[df['Country'] == country]
        
        # Group by invoice and description
        basket = (df.groupby(['InvoiceNo', 'Description'])['Quantity']
                  .sum()
                  .unstack()
                  .reset_index()
                  .fillna(0)
                  .set_index('InvoiceNo'))
        
        # Convert to binary (0/1) format
        basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
        
        return basket_sets
    
    def get_cleaning_report(self) -> dict:
        """
        Generate data cleaning report
        """
        return {
            'original_rows': self.original_shape[0],
            'cleaned_rows': self.df.shape[0],
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'removal_percentage': round(((self.original_shape[0] - self.df.shape[0]) / self.original_shape[0]) * 100, 2),
            'unique_items': self.df['Description'].nunique() if 'Description' in self.df.columns else 0,
            'unique_invoices': self.df['InvoiceNo'].nunique() if 'InvoiceNo' in self.df.columns else 0
        }
    
    def get_top_items(self, n: int = 20) -> pd.Series:
        """
        Get top N most frequent items
        """
        return self.df['Description'].value_counts().head(n)
