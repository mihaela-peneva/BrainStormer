# EEG Data Analysis Project

This project is a Django web application for visualising EEG data. It uses the MNE Python library for processing and analysing the EEG data, and Plotly for creating interactive visualizations. The project's dependencies are managed by Anaconda.

## Requirements

- Python 3.11
- Django
- MNE
- pandas
- numpy
- matplotlib
- plotly
- Anaconda (for package management)

## Installation

1. Install [Anaconda](https://www.anaconda.com/products/distribution) if you haven't already.
2. Clone this repository:
```bash
git clone https://github.com/mihaela-peneva/BrainStormer.git
cd BrainStormer
```

3. Create a new Anaconda environment and activate it:
```bash
  conda create -n eeg-analysis python=3.11
  conda activate eeg-analysis
```

4. Install the required packages:
```bash
conda install django pandas numpy matplotlib plotly
conda install -c conda-forge mne
```

5. Run the Django development server:
```bash
cd BrainStormer
python manage.py runserver
```
or
```bash
cd BrainStormer
python3 manage.py runserver
```

6. Access the web application by navigating to `http://localhost:8000/` in your web browser.

## GUI Usage

Upload your EEG data in CSV format to the web application. The data should include timestamps and EEG channel measurements. The application will process the data using the MNE library, perform necessary calculations, and visualise the results using Plotly.




## Data Processing Pipeline Usage
1. Run example visualisation from MNE
```bash
conda activate eeg-analysis
python Controllers/miner.py
```
or
```bash
conda activate eeg-analysis
python3 Controllers/miner.py
```

2. Run project visualisation
- Our EEG data visualisation tool accepts three arguments:  
``file``: The CSV file containing EEG data (e.g., muse2.csv).  
``function``: The function to be called. Choose either visualise_default or visualise_filtered.  
``parameters``: An array of string parameters for the visualise_filtered function (e.g., ['eeg_1', 'eeg_2', 'eeg_3']). This argument is optional and only required for visualise_filtered.   
  
- Example run with visualise_default
```bash
conda activate eeg-analysis
python Controllers/transformer.py [file_name].csv visualise_default
```
or
```bash
conda activate eeg-analysis
python3 Controllers/transformer.py [file_name].csv visualise_default
```
  
- Example run with visualise_filtered
```bash
conda activate eeg-analysis
python Controllers/transformer.py [file_name].csv visualise_filtered eeg_1 eeg_2 eeg_3 eeg_4
```
or
```bash
conda activate eeg-analysis
python3 Controllers/transformer.py [file_name].csv visualise_filtered eeg_1 eeg_2 eeg_3 eeg_4
```

Note that for visualise_filtered, you need to provide the parameters (column names) without the square brackets and commas, just separated by spaces.


## Dependencies

The project imports the following packages:

- `django.shortcuts` for rendering views in the Django application.
- `argparse`, `csv` for file handling and CSV data manipulation.
- `pandas`, `numpy` for data manipulation and analysis.
- `mne` for EEG data processing and analysis.
- `matplotlib` for creating static plots.
- `plotly.graph_objects`, `plotly.subplots` for creating interactive plots.
- `mne.preprocessing.ICA` for independent component analysis (ICA) in MNE.




