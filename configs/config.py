#!/usr/bin/env python3
"""
Configuration settings for the FreeSurfer parser
"""

# File paths and naming conventions
SUBJECT_PREFIX = "sub-"  # Prefix for subject directories
STATS_DIR = "stats"      # Name of the stats directory within each subject folder

# FreeSurfer output file names
ASEG_STATS_FILE = "aseg.stats"
LH_APARC_STATS_FILE = "lh.aparc.stats"
RH_APARC_STATS_FILE = "rh.aparc.stats"

# Output file names
NEUROTOKENS_JSON_FILE = "neurotokens.json"
NEUROTOKENS_CSV_FILE = "neurotokens.csv"
REGION_STATS_FILE = "region_statistics.json"

# File parsing settings
ASEG_COLUMNS = {
    'region_name_index': 4,  # Index of region name in aseg.stats
    'volume_index': 3,       # Index of volume in aseg.stats
}

APARC_COLUMNS = {
    'region_name_index': 0,  # Index of region name in aparc.stats
    'surface_area_index': 2, # Index of surface area in aparc.stats
    'thickness_index': 4,    # Index of thickness in aparc.stats
}

# Hemisphere naming
HEMISPHERE_NAMES = {
    'lh': 'Left',
    'rh': 'Right'
}

# NeuroToken formatting
NEUROTOKEN_FORMAT = {
    'value_precision': 1,    # Number of decimal places for values
    'z_score_precision': 1,  # Number of decimal places for z-scores
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S'
}

# Data validation settings
VALIDATION_CONFIG = {
    'min_volume': 0.0,       # Minimum valid volume (mm³)
    'max_volume': 100000.0,  # Maximum valid volume (mm³)
    'min_thickness': 0.5,    # Minimum valid thickness (mm)
    'max_thickness': 5.0,    # Maximum valid thickness (mm)
    'min_surface_area': 0.0, # Minimum valid surface area (mm²)
    'max_surface_area': 50000.0, # Maximum valid surface area (mm²)
}

# Sample data generation settings
SAMPLE_DATA_CONFIG = {
    'default_num_subjects': 5,
    'volume_mean': 2000.0,
    'volume_std': 500.0,
    'thickness_mean': 2.5,
    'thickness_std': 0.3,
    'surface_area_mean': 1500.0,
    'surface_area_std': 300.0,
}

# Region names for sample data generation
SAMPLE_ASEG_REGIONS = [
    "Left-Hippocampus", "Right-Hippocampus", "Left-Amygdala", "Right-Amygdala",
    "Left-Thalamus", "Right-Thalamus", "Left-Caudate", "Right-Caudate",
    "Left-Putamen", "Right-Putamen", "Left-Pallidum", "Right-Pallidum",
    "Left-Accumbens", "Right-Accumbens", "Left-VentralDC", "Right-VentralDC"
]

SAMPLE_APARC_REGIONS = [
    "bankssts", "caudal anterior cingulate", "caudal middle frontal",
    "cuneus", "entorhinal", "fusiform", "inferior parietal",
    "inferior temporal", "isthmus cingulate", "lateral occipital",
    "lateral orbitofrontal", "lingual", "medial orbitofrontal",
    "middle temporal", "parahippocampal", "paracentral",
    "pars opercularis", "pars orbitalis", "pars triangularis",
    "pericalcarine", "postcentral", "posterior cingulate",
    "precentral", "precuneus", "rostral anterior cingulate",
    "rostral middle frontal", "superior frontal", "superior parietal",
    "superior temporal", "supramarginal", "frontal pole", "temporal pole",
    "transverse temporal", "insula"
] 