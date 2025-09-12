# hcp-hmm-pipeline
Analysis pipeline for fMRI brain state dynamics using Hidden Markov Models on HCP data.

# HCP HMM Pipeline

This repository contains a reproducible pipeline for analyzing resting-state fMRI data with Hidden Markov Models (HMM).  
It is designed for use with the Human Connectome Project (HCP) dataset, but can be adapted to other fMRI datasets.

## Features
- Preprocessing and parcellation of fMRI time series
- HMM fitting and state inference
- Extraction of state metrics (Fractional Occupancy, Dwell Time, Switching Rate)
- Group-level statistical analysis
- Visualization of states and metrics

## Requirements
- Python 3.9+ (NumPy, Pandas, Scikit-learn, hmmlearn, nibabel, etc.)
- FSL & Connectome Workbench (for parcellation and dual regression)
- HCP dataset (not included)

## Usage
1. Clone the repository  
   ```bash
   git clone https://github.com/YourUser/hcp-hmm-pipeline.git
