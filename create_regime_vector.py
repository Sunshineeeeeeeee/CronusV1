"""
Create a combined regime vector representation from regime and sub-regime columns.

This script is used to enhance the results of the FinancialMapper by creating
a single vector representation that combines the regime and sub-regime information.

Usage:
    python create_regime_vector.py input_file.csv output_file.csv
    
Or import the function directly:
    from create_regime_vector import create_regime_vector
    results = create_regime_vector(dataframe)
"""

import pandas as pd
import sys
import numpy as np
from sklearn.preprocessing import LabelEncoder

def create_regime_vector(df):
    """
    Converts separate 'regime' and 'sub_regime' columns into a single tuple-based
    vector representation to better identify recurring regimes.
    
    Args:
        df: DataFrame with 'regime' and 'sub_regime' columns
        
    Returns:
        DataFrame with new columns:
            - 'regime_vector': Tuple of (regime, sub_regime)
            - 'regime_vector_encoded': Integer encoding of unique regime vectors
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Ensure the required columns exist
    if 'regime' not in result_df.columns or 'sub_regime' not in result_df.columns:
        raise ValueError("DataFrame must contain both 'regime' and 'sub_regime' columns")
    
    # Create the regime vector as a tuple
    result_df['regime_vector'] = list(zip(result_df['regime'], result_df['sub_regime']))
    
    # Count occurrences of each regime vector to identify common patterns
    vector_counts = result_df['regime_vector'].value_counts()
    print(f"Found {len(vector_counts)} unique regime vectors")
    
    # Show the most common regime vectors
    print("\nMost common regime vectors (regime, sub_regime):")
    print(vector_counts.head(10))
    
    # Create a numerical encoding for machine learning
    # Convert tuples to strings for encoding (tuples aren't hashable for sklearn)
    regime_vector_str = result_df['regime_vector'].astype(str)
    
    # Encode as integers
    encoder = LabelEncoder()
    result_df['regime_vector_encoded'] = encoder.fit_transform(regime_vector_str)
    
    # Create a mapping dictionary for reference
    vector_mapping = {encoded: vector for encoded, vector in 
                      zip(result_df['regime_vector_encoded'].unique(), 
                          result_df['regime_vector'].unique())}
    
    print("\nEncoding mapping (sample):")
    sample_size = min(10, len(vector_mapping))
    for i, (enc, vec) in enumerate(list(vector_mapping.items())[:sample_size]):
        print(f"{enc}: {vec}")
    
    return result_df

def main():
    """Command-line interface"""
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} input_file.csv output_file.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    print("Creating regime vector representation...")
    result_df = create_regime_vector(df)
    
    print(f"Writing results to {output_file}...")
    result_df.to_csv(output_file, index=False)
    print("Done!")

if __name__ == "__main__":
    main()