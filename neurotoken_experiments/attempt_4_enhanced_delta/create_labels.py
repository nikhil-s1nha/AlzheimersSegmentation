#!/usr/bin/env python3
"""
Create Subject Labels from OASIS Demographics
"""

import pandas as pd
import os

def main():
    # Load the Excel file
    excel_path = "../../../../AlzheimersSegmentation/AlzheimersSegmentation/Oasis Longitudinal Demographics.xlsx"
    output_path = "subject_labels.csv"
    
    print(f"Loading demographics from {excel_path}")
    
    try:
        df = pd.read_excel(excel_path)
        print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
        
        # Check for required columns
        if 'Subject ID' not in df.columns or 'CDR' not in df.columns:
            print("Error: Required columns 'Subject ID' and 'CDR' not found")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Clean data - drop rows with missing CDR
        df_clean = df.dropna(subset=['CDR'])
        print(f"After cleaning: {len(df_clean)} rows")
        
        # Group by Subject ID and get max CDR per subject
        subject_max_cdr = df_clean.groupby('Subject ID')['CDR'].max().reset_index()
        subject_max_cdr.columns = ['subject_id', 'max_cdr']
        
        print(f"Found {len(subject_max_cdr)} unique subjects")
        print("CDR distribution:")
        print(subject_max_cdr['max_cdr'].value_counts().sort_index())
        
        # Create binary labels (0 = normal, 1 = impaired)
        def create_binary_label(cdr):
            if cdr == 0.0:
                return 0  # Normal
            else:
                return 1  # MCI or AD
        
        subject_max_cdr['label'] = subject_max_cdr['max_cdr'].apply(create_binary_label)
        
        # Save to CSV
        subject_max_cdr.to_csv(output_path, index=False)
        print(f"Saved {len(subject_max_cdr)} subject labels to {output_path}")
        
        print("\nLabel distribution:")
        print(subject_max_cdr['label'].value_counts().sort_index())
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 