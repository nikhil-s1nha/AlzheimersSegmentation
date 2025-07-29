#!/usr/bin/env python3
"""
Debug script to test aseg.stats parsing
"""

import os

def debug_parse_aseg_stats(file_path):
    """Debug the aseg.stats parsing"""
    print(f"Debugging: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    stats_dict = {}
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            
            # Parse region volumes from table
            parts = line.split()
            if len(parts) >= 5:
                try:
                    # The region name starts after the numeric columns
                    # Find where the region name starts (after the last numeric column)
                    region_name = None
                    for i, part in enumerate(parts):
                        if part.replace('.', '').replace('-', '').isalpha():
                            region_name = ' '.join(parts[i:])
                            break
                    
                    if region_name:
                        # Get volume (4th column, index 3)
                        volume = float(parts[3])
                        stats_dict[region_name] = volume
                        
                        # Debug output for regions we care about
                        if any(keyword in region_name for keyword in ['Hippocampus', 'Amygdala', 'Lateral-Ventricle']):
                            print(f"Line {line_num}: Found region '{region_name}' = {volume}")
                            
                except (ValueError, IndexError) as e:
                    print(f"Line {line_num}: Error parsing line: {e}")
                    print(f"  Line: {line}")
                    continue
    
    print(f"\nExtracted {len(stats_dict)} regions")
    print("Regions we're looking for:")
    for region in ['Left-Hippocampus', 'Right-Hippocampus', 'Left-Amygdala', 'Right-Amygdala', 'Left-Lateral-Ventricle', 'Right-Lateral-Ventricle']:
        if region in stats_dict:
            print(f"  ✓ {region}: {stats_dict[region]}")
        else:
            print(f"  ✗ {region}: NOT FOUND")
    
    return stats_dict

if __name__ == "__main__":
    test_file = "/Volumes/SEAGATE_NIKHIL/subjects/OAS2_0001_session_1/stats/aseg.stats"
    debug_parse_aseg_stats(test_file) 