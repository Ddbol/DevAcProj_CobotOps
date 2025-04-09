# UR3 CobotOps Automation Analysis 

# Dataset: 
UR3 CobotOps Dataset - time-series data from the UR3 cobot - offers insights into operational parameters and faults.

# Objective: 
Explore data and investigate feasibility of error prediction by machine learning

# Summary: 
Exploratory data analysis of UR3 CobotOps dataset. Investigated machine learning models (time series and classic supervised, initial screening with AutoGluon) to predict faults. Deployed Streamlit
application, via streamlit url and containarised in Docker, to generate a summary table and visualisations from uploaded run data.

# Streamlit application information: 
- Streamlit and docker files in streamlit_app_cobot folder using cobot_st_runsum_updated.py application code
- Two .xlsx and two .csv files all with same data as to run docker will only recognise file from directory (files will have to be renamed cobot_dataset.xlsx or .csv)
- Other files: .joblib - exported encoder, requirements.txt & dockerfile - docker deployment
- some generated png graphs as examples but all files can be downloaded from Streamlit_cobot_app_graphs_all_.zip
- Cobot_data_summ_app_vid.mp4 is video of application running 
- Docker - https://hub.docker.com/r/dboland717/cobot_run_summary
- https://cobotsummarygenerator.streamlit.app

- Second application code file uploaded cobot_st_runsum_updated_show_graph.py to show more graphs within the application. Only rolled out to streamlit url https://cobotdatasummarygeneratorwithgraphs.streamlit.app

# Tools used: 
Python, pandas, numpy, matplotlib, seaborn, scikitlearn, regularisation (scaling), autogluon, RandomForestClassifier, KNN, Docker, Streamlit, tensorflow, keras

# Files and folders:

x.PNG - Images to display in ipynb file

cobotops_DBolandMar25.ipynb - main code for exploratory data analysis and machine learning investigation

Cobot_data_dictionary.pdf - Data dictionary explaining meanings of features/column data original and generated

Cobot_analysis_DBMar2025_final.pdf or .pptx - Pdf or powerpoint (View raw to view online and interact with links) of presentation

label_encoder_grip_lost.joblib - label encoder saved for use with new data sets or to build streamlit and docker application

Cobot_dataset_02052023.xlsx - data input

Cobot_data_summ_app_vid.mp4, streamlit_app_cobot folder, Streamlit_cobot_app_graphs_all_.zip - refer to streamlit application information

autogluon folder - output of Autogluon timeseries modelling

output_x_ag...txt  - text output from codecell of Autogluon tabular modelling

ag-20250403_234001 - subset of one output of Autogluon tabular modelling, only with top performing RandomForestClassifier and KNN models uploaded (others available offline or run the code)
