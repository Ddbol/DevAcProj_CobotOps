'''
# run application in terminal via command : python -m streamlit run x.py
# ensure in correct directory (use cd - change directory command)
# fileupload size controlled by config file in .streamlit   - default 200MB, dropped to 5MB
# ctrl c to stop run in terminal before close browser
'''

#import libraries
import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import joblib
import datetime 
from datetime import datetime, time, timedelta
import pandas as pd
import os
from tensorflow import keras
from PIL import Image



# Set page configuration - this changes the browser tab title
st.set_page_config(
    page_title="Cobot run summary",
    page_icon="🤖",
    layout="wide"
)

# Set environment variables to disable GPU
## '3' means "only show errors" (suppress info, warnings, and debug messages)
## This disables GPU usage for TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

# Clear any existing TensorFlow session
## Releases memory used by TensorFlow
## Clears any stored states or cached operations
tf.keras.backend.clear_session()

def analyse_file(dataset):
    ### -------------- Analysis start - cleanup, extract, encode --------------###
    # rename cycle column, no spaces
    dataset = dataset.rename(columns={'cycle ': 'cycle'})

    # Function to clean and convert the timestamp
    def clean_timestamp(ts):
        # had to add replace " as was being added to number although not visible
        ts = ts.replace('T', ' ').replace('Z', '').replace('"','')
        return pd.to_datetime(ts, format= '%Y-%m-%d %H:%M:%S.%f')
       
    # Apply the clean_timestamp function to the DataFrame
    dataset['cleaned_timestamp'] = dataset['Timestamp'].apply(clean_timestamp)

    # Extract date and time 
    dataset['date'] = dataset['cleaned_timestamp'].dt.date
    dataset['time'] = dataset['cleaned_timestamp'].dt.time

    # loading the LabelEncoder from the file
    loaded_le = joblib.load('label_encoder_grip_lost.joblib')

    # Transforming grip_lost with the loaded LabelEncoder
    dataset["grip_lost_enc"] = loaded_le.transform(dataset["grip_lost"])

    # Verifying that transformations/encoding are identical
    transformations_equal = (dataset["grip_lost"]== dataset["grip_lost_enc"] ).all()

    st.write("False and True correctly converted to 0 and 1 respectively?", transformations_equal)

    ### --------------------------New cycle feature gneration -------------------###    
    # Prepare new cycle feature, where cycles are labelled in order run
    # Check if the value in 'column_name' is the same as the previous row
    dataset["new_cyc"] = dataset['cycle'] == dataset['cycle'].shift(1)


    # Initialize the result column with cycle 1
    dataset["cyc_rename"] = 1

    # Iterate through the DataFrame and apply the condition.
    for i in range(1, len(dataset)):
        #  if new_cyc not the same as previous row start new cycle number
        if not dataset.at[i, 'new_cyc']:
            dataset.at[i, 'cyc_rename'] = dataset.at[i-1, 'cyc_rename'] + 1
        else:
            dataset.at[i, 'cyc_rename'] = dataset.at[i-1, 'cyc_rename']
    
    # cycle length of time determination. set initial time
    init_time = dataset.loc[0, 'cleaned_timestamp']

    # Iterate through the DataFrame and apply the condition 
    for i in range(0, len(dataset)):
        # if value is false, the cycle has changed and that is new init_time
        if not dataset.at[i, 'new_cyc']:
            init_time = dataset.loc[i, 'cleaned_timestamp']
        else:
            init_time = init_time
        # subtract init_time from the timestamp of each row to find timepoint in cycle
        dataset.at[i, 'cyc_timepoint'] = dataset.at[i, 'cleaned_timestamp'] - init_time

    #cyc_timepoint is timedelta - set to float so can plot
    dataset['cyc_timepoint_sec'] = dataset['cyc_timepoint'].dt.total_seconds()

    ### -----------------Generating Timelapse vs. Cycles and faults plot --------###

    # Create a figure and a set of subplots
    fig, ax1 = plt.subplots(1, 1, figsize=(7,7))


    # Plot the first scatter plot
    ax1.scatter(dataset["cleaned_timestamp"], dataset["Robot_ProtectiveStop"], linestyle='', marker='o', color='b', alpha = 0.3, label='Protective stop')
    ax1.set_xlabel('Time: MM-DD HH')
    ax1.set_ylabel('Robot stop or grip loss (1 = true, 0 = false)')
    ax1.set_ylim(top=2)
    ax1.scatter(dataset["cleaned_timestamp"], dataset["grip_lost_enc"], linestyle='', marker='o', color='r', alpha = 0.3,label='Grip lost')
    plt.legend(loc="upper left")

    #Create a secondary y-axis
    ax2 = ax1.twinx()
    ax2.scatter(dataset["cleaned_timestamp"], dataset["cycle"], color='g', label='Original cycle number')
    ax2.scatter(dataset["cleaned_timestamp"], dataset["cyc_rename"], color='black', alpha =0.01, label='Revised cycle number')
    ax2.set_ylabel('Cycle number')

    # Add a title and show the plot
    plt.title('Timestamp vs. Cycles and Cobot stop or grip loss')
    plt.xticks(rotation=45)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig("Timelapse_cyc_fault.png")


    st.write("--------------------------------------- Generating Summary table . .  .  ---------------------------------")
    # Group by column 'cyc_rename'
    grouped = dataset.groupby('cyc_rename')

    # Create summary DataFrame
    summary_df = grouped.agg(
        cyc_rename =('cyc_rename', 'first'),  
        cyc_original =('cycle', 'first'),
        grip_lost=('grip_lost_enc', 'sum'),  
        protective_stop=('Robot_ProtectiveStop', 'sum'), 
        cyc_end = ('cleaned_timestamp', 'max'),
        cyc_start = ('cleaned_timestamp', 'min'),
        duration=('cyc_timepoint', 'max'),
        cyc_datapts = ('cyc_rename', 'count'), 
        temp_J0 = ('Temperature_T0','median'),
        temp_J1 = ('Temperature_J1','median'),  
        temp_J2 = ('Temperature_J2','median'), 
        temp_J3 = ('Temperature_J3','median'),
        temp_J4 = ('Temperature_J4','median'), 
        temp_J5 = ('Temperature_J5','median')         
    ).reset_index(drop=True)

    
    # Iterate through the DataFrame and apply the condition
    for i in range(0, len(summary_df)):
        # sum up errors
        summary_df.at[i, 'all_errors'] = summary_df.at[i, 'grip_lost'] + summary_df.at[i, 'protective_stop']
    # no start of next cycle for last row so -1 length
    for i in range(0, len(summary_df)-1):
        # calculate length of breaks after/between cycles
        summary_df.at[i, 'break_post_cyc'] = summary_df.at[i+1, 'cyc_start'] - summary_df.at[i, 'cyc_end']

    # set time of last row/index as 0 timedelta (same format other entries)
    summary_df.loc[len(summary_df)-1,'break_post_cyc'] = timedelta(0)

    #set timedelta variables to float so can plot or work with easier
    summary_df['duration_sec'] = summary_df['duration'].dt.total_seconds()
    summary_df['break_post_c_sec'] = summary_df['break_post_cyc'].dt.total_seconds()


    ### ------------Generating Summary plot - Cycle number vs Temp, errors and cycle length ------- ####
    # Create a figure and a set of subplots
    fig2, ax1 = plt.subplots(1, 1, figsize=(7,7))

    # Plot the first scatter plot
    ax1.plot(summary_df["cyc_rename"], summary_df["protective_stop"], color = "blue", alpha = 0.6, label='Protective stop')
    ax1.set_xlabel('Renamed Cycle number')
    ax1.set_ylabel('Protective stop/grip loss (count) or Temp (Celsius)')
    ax1.set_ylim(top=60)
    ax1.plot(summary_df["cyc_rename"], summary_df["grip_lost"], color = "red", alpha = 0.6,label='Grip lost')
    ax1.plot(summary_df["cyc_rename"], summary_df["temp_J0"], color = "black", alpha = 0.6,label='Temp J0')
    ax1.plot(summary_df["cyc_rename"], summary_df["temp_J1"], color = "grey", alpha = 0.6,label='Temp J1')
    ax1.plot(summary_df["cyc_rename"], summary_df["temp_J2"], color = "brown", alpha = 0.6,label='Temp J2')
    ax1.plot(summary_df["cyc_rename"], summary_df["temp_J3"], linestyle = "--", color = "black", alpha = 0.6,label='Temp J3')
    ax1.plot(summary_df["cyc_rename"], summary_df["temp_J4"], linestyle = "--",color = "grey", alpha = 0.6,label='Temp J4')
    ax1.plot(summary_df["cyc_rename"], summary_df["temp_J5"], linestyle = "--", color = "brown",alpha = 0.6,label='Temp J5')
    plt.legend(loc="upper left")


    #Create a secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(summary_df["cyc_rename"], summary_df["cyc_datapts"], color = "orange", alpha = 0.6,label='Cycle datapts/length')
    ax2.plot(summary_df["cyc_rename"], summary_df["all_errors"], color = "purple", alpha = 0.6, label='All errors')
    ax2.set_ylim(bottom=-40)
    ax2.set_ylabel('cycle points (count) or all errors (count)')

    # Add a title and show the plot
    plt.title('Cycle number vs Temp, errors and cycle length')
    plt.xticks(rotation=45)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.legend(loc="upper right")
    plt.tight_layout()
    fig2.savefig("Summary_plot_cyc_temp_fault_cyclength.png")


    ### ----------Create lists of cycles with errors/faults to displays -----------####
    # Initialize a list to store the applicable cycles
    cyc_names_grip_lost = []

    # Loop through each row in the DataFrame
    for i in range(len(summary_df)):
        if summary_df.grip_lost[i] >0:
            # Append the cycle name to the list
            cyc_names_grip_lost.append(summary_df.cyc_rename[i])
     

    # # Initialize a list to store the applicable cycles
    cyc_names_prot_stop = []

    # Loop through each row in the DataFrame

    for i in range(len(summary_df)):
        if summary_df.protective_stop[i] >0:
            # Append the cycle name to the list
            cyc_names_prot_stop.append(summary_df.cyc_rename[i])

    ##### ------------ Drop columns and display Summary Tables --------#####
    summary_df.drop(["duration","break_post_cyc"], axis='columns', inplace= True)
    st.write("Summary Table")
    # display summary table
    st.dataframe(summary_df) 
    #return  st.dataframe(summary_df) 
    st.write("These are the cycles where grip loss(es) occured:")
    st.dataframe(cyc_names_grip_lost)
    st.write("These are the cycles where protective stop(s) occured:")
    st.dataframe(cyc_names_prot_stop)

    ##### ------------ Generate Temperature, Current and Speed data for each cycle --------#####
    st.write("Generating Graphs for Temp, Current and Speed for individual cycles, saving directly into directory folder . . . Will take several mins")
    st.write("When file save completes some summary graphs of all cycles will be shown and cycle1's individual graph")
    # Iterate through the DataFrame and apply the condition 
    for i in range(1, len(summary_df)+1):
        df1 = dataset[dataset.cyc_rename==i]
        # Create a figure and a set of subplots
        fig = plt.figure(figsize=(21,7))

        # ----------Temperature graph ----------------------

        # set on figure
        ax1 = fig.add_subplot(1,3,1)
        # Plot the first scatter plot
        ax1.scatter(df1["cyc_timepoint_sec"], df1["Robot_ProtectiveStop"], marker='+', color='b', alpha = 0.7, label='Protective stop')
        ax1.set_xlabel('Cycle Time: sec')
        ax1.set_ylabel('Robot stop or grip loss (1 = true, 0 = false)')
        ax1.set_ylim(top=1.5)
        ax1.scatter(df1["cyc_timepoint_sec"], df1["grip_lost_enc"], marker='+', color='r', alpha = 0.7,label='Grip lost')
        plt.legend(loc="upper left")

        #Create a secondary y-axis
        ax2 = ax1.twinx()
        ax2.plot(df1["cyc_timepoint_sec"], df1["Temperature_T0"], label='Temperature_T0')
        ax2.plot(df1["cyc_timepoint_sec"], df1["Temperature_J1"],label='Temperature_J1')
        ax2.plot(df1["cyc_timepoint_sec"], df1["Temperature_J2"],label='Temperature_J2')
        ax2.plot(df1["cyc_timepoint_sec"], df1["Temperature_J3"], label='Temperature_J3') 
        ax2.plot(df1["cyc_timepoint_sec"], df1["Temperature_J4"],label='Temperature_J4')
        ax2.plot(df1["cyc_timepoint_sec"], df1["Temperature_J5"], label='Temperature_J5')
        # set limits of max and min values so all cycles scale the same for comparison
        ax2.set_ylim(top=50, bottom = 25)
        ax2.set_ylabel('Temperature')

        # Add a title and show the plot
        plt.title('Temperature vs. Cobot stop or grip loss')
        plt.xticks(rotation=45)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
        plt.legend(loc="upper right")

        # ----- Current graph -------

        # set on figure
        ax3 = fig.add_subplot(1,3,2)
        # Plot the first scatter plot
        ax3.scatter(df1["cyc_timepoint_sec"], df1["Robot_ProtectiveStop"], marker='+', color='b', alpha = 0.7, label='Protective stop')
        ax3.set_xlabel('Cycle Time: sec')
        ax3.set_ylabel('Robot stop or grip loss (1 = true, 0 = false)')
        ax3.set_ylim(top=1.5)
        ax3.scatter(df1["cyc_timepoint_sec"], df1["grip_lost_enc"], marker='+', color='r', alpha = 0.7,label='Grip lost')
        plt.legend(loc="upper left")

        #Create a secondary y-axis
        ax4 = ax3.twinx()
        ax4.plot(df1["cyc_timepoint_sec"], df1["Current_J0"], label='Current_J0')
        ax4.plot(df1["cyc_timepoint_sec"], df1["Current_J1"], label='Current_J1')
        ax4.plot(df1["cyc_timepoint_sec"], df1["Current_J2"], label='Current_J2')
        ax4.plot(df1["cyc_timepoint_sec"], df1["Current_J3"], label='Current_J3') 
        ax4.plot(df1["cyc_timepoint_sec"], df1["Current_J4"], label='Current_J4')
        ax4.plot(df1["cyc_timepoint_sec"], df1["Current_J5"], label='Current_J5')
        # set limits of max and min values so all cycles scale the same for comparison
        ax4.set_ylim(top=7, bottom = -7)
        ax4.set_ylabel('Current')

        # Add a title and show the plot
        plt.title('Current vs. Cobot stop or grip loss')
        plt.xticks(rotation=45)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
        plt.legend(loc="upper right")

        # ----- Speed graph -------
        # set on figure
        ax1 = fig.add_subplot(1,3,3)
        # Plot the first scatter plot
        ax1.scatter(df1["cyc_timepoint_sec"], df1["Robot_ProtectiveStop"], linestyle='', marker='+', color='b', alpha = 0.7, label='Protective stop')
        ax1.set_xlabel('Cycle Time: sec')
        ax1.set_ylabel('Robot stop or grip loss (1 = true, 0 = false)')
        ax1.set_ylim(top=1.5)
        ax1.scatter(df1["cyc_timepoint_sec"], df1["grip_lost_enc"], linestyle='', marker='+', color='r', alpha = 0.7,label='Grip lost')
        plt.legend(loc="upper left")

        #Create a secondary y-axis
        ax2 = ax1.twinx()
        ax2.plot(df1["cyc_timepoint_sec"], df1["Speed_J0"], label='Speed_J0')
        ax2.plot(df1["cyc_timepoint_sec"], df1["Speed_J1"], label='Speed_J1')
        ax2.plot(df1["cyc_timepoint_sec"], df1["Speed_J2"], label='Speed_J2')
        ax2.plot(df1["cyc_timepoint_sec"], df1["Speed_J3"], label='Speed_J3') 
        ax2.plot(df1["cyc_timepoint_sec"], df1["Speed_J4"], label='Speed_J4')
        ax2.plot(df1["cyc_timepoint_sec"], df1["Speed_J5"], label='Speed_J5')
        # set limits of max and min values so all cycles scale the same for comparison
        ax2.set_ylim(top=4, bottom = -4)
        ax2.set_ylabel('Speed')

        # Add a title and show the plot
        plt.title('Speed vs. Cobot stop or grip loss')
        plt.xticks(rotation=45)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
        plt.legend(loc="upper right")

        plt.tight_layout()
        #image_name = print("Cycle_",i,".png")
        fig.savefig(f'Cycle_{i}.png')



def main():
    # Title of the app
    st.title("Cobot UR3 Data Summary Generator")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

    # Check if a file has been uploaded
    if uploaded_file is not None:
        # analyse file
        st.write("Please wait while we check your file")

        # Display file details
        st.write("Filename:", uploaded_file.name)
        st.write("File type:", uploaded_file.type)
        st.write("File size:", uploaded_file.size, "bytes")

        # Read and display the content of the file (if it's a text file)
        if uploaded_file.type == "text/csv" or uploaded_file.type == "text/plain":
            content = uploaded_file.read().decode("utf-8")
            st.text_area("File content", content, height=300)
        else:
            st.write("File uploaded successfully!")
        
        st.write("-----------------Analysing file...--------------")
        st.write("Please wait while we analyse your file")
        dataset = pd.read_excel(uploaded_file.name)
        # run analyse_file function, which analyses file, generates and displays summary table and tables of cycles with faults
        analyse_file(dataset)

        # show graph 1
        st.image(Image.open("Timelapse_cyc_fault.png"))

        # show graph 2
        st.image(Image.open("Summary_plot_cyc_temp_fault_cyclength.png"))

        # show graph 3
        st.write("Temperature, Current and Speed graphs for cycle 1")
        st.write("Graphs for other individual cycles have been saved into directory folder")
        st.image(Image.open("Cycle_1.png"))
       

    else:
        st.write("Please upload a file.")

if __name__ == "__main__":
    main()
