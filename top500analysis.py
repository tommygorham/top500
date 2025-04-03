# Program:     top500analysis.py
# Description: Top500.org list analysis of the fastest supercomputers
# Author: Tommy Gorham

########################### required libs #####################################
import os 
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import numpy as np 
import math 
from statistics import mean, median, mode, pstdev, StatisticsError
import warnings 
import argparse
import datetime

current_path = os.getcwd()

########################## clean data functions ###############################
def get_data_file(): 
    parser = argparse.ArgumentParser(description="Analyze the top500 supercomputers.")
    parser.add_argument("file", nargs='?', default=current_path+"/datasets/november2024.xlsx", 
                        help="Path to the Excel file (default: ./datasets/november2024.xlsx)")
    args = parser.parse_args()
    return args.file

def strip_columns(df): 
    df.columns = df.columns.str.replace('[', '')
    df.columns = df.columns.str.replace(']', '')
    df.columns = df.columns.str.replace('/', '')
    df.columns = df.columns.str.replace(' ', '')
    df.columns = df.columns.str.replace('-', '')
    df.columns = df.columns.str.replace(' ', '')
    df.columns = df.columns.str.replace('(', '')
    df.columns = df.columns.str.replace(')', '')
    df.columns = df.columns.str.lower()
    return df

def drop_columns(df):  
    columns_to_remove = ['previousrank', 'firstappearance', 'firstrank', 'interconnect', 'site', 'siteid', 'systemid', 
                         'nhalf', 'country', 'continent', 'segment']
    for col in columns_to_remove:
        if col in df.columns:
            del df[col]
    return df
    
def check_and_clean_accelerator_column(df): 
    if 'acceleratorcoprocessor' in df.columns:
        df.rename(columns={'acceleratorcoprocessor': 'accelerator'}, inplace=True)
    if 'accelerator' in df.columns:
        df['accelerator'].replace([None, ''], np.nan, inplace=True)
    else:
        warnings.warn("The DataFrame does not contain an 'accelerator' or 'acceleratorcoprocessor' column.")
    return df

########################## pre-process functions ##############################
def append_cpu_column(df):
    try:
        df_cpunames = df[['processor']].copy()
        conditions = [
            df_cpunames['processor'].str.contains('AMD', na=False),
            df_cpunames['processor'].str.contains('Xeon', na=False),
            df_cpunames['processor'].str.contains('IBM', na=False)
        ]
        choices = ['AMD', 'Intel', 'IBM']
        df_cpunames['cpu'] = np.select(conditions, choices, default='OTHER')      
        df['cpu'] = df_cpunames['cpu']
        if df['cpu'].isnull().any():
            warnings.warn("Null values detected in the 'cpu' column.")
    except Exception as e:
        warnings.warn(f"An error occurred while appending the CPU column: {e}")
    return df

def add_system_architecture_column(df):
    try:
        if 'accelerator' not in df.columns:
            warnings.warn("The DataFrame does not contain an 'accelerator' column. Ensure it's added before calling this function.")
            return df
        df['sysarch'] = np.where(df['accelerator'].isna() | (df['accelerator'] == 'None'), 'CPU Only Machine', 'CPUGPU Machine')
    except Exception as e:
        warnings.warn(f"An error occurred while adding the system architecture column: {e}")
    return df

# Rename "Custom Interconnect" value to "Custom" in Interconnect Family column if it exists in the dataframe
def update_interconnect_family_custom(df):
    if 'interconnectfamily' in df.columns:
        df['interconnectfamily'] = df['interconnectfamily'].replace('Custom Interconnect', 'Custom')
    else:
        warnings.warn("The DataFrame does not contain an 'interconnectfamily' column.")
    return df

########################## plot data functions ################################
def plot_power_vs_performance(df):
    try:
        toscale = df[['powerkw', 'rmaxtflops', 'cpu', 'sysarch', 'nmax', 'name']].copy()
        toscale['powerlog2'] = np.log2(toscale['powerkw'])
        toscale['performancelog2'] = np.log2(toscale['rmaxtflops'])
        df_scaled = toscale.dropna(axis=0, how='any')
        df['log2power'] = df_scaled['powerlog2']
        df['log2performance'] = df_scaled['performancelog2']
        print("\n\nPower vs Performance, Hover for Machine Info\n")
        fig = px.scatter(
            df, 
            x="log2performance", 
            y="log2power", 
            color='cpu', 
            symbol='sysarch',  
            hover_data=['name'], 
            color_discrete_map={
                'AMD': 'red', 
                'Intel': '#0277bd',
                'IBM': 'blue', 
                'OTHER': 'orange'
            },
            labels={
                "log2performance": "Maximal LINPACK Performance (Tflops/sec Scaled to Log Base 2)", 
                "log2power": "Maximal Power (Kilowatts Scaled to Log Base 2)"
            },  
            width=1000,  
            height=600   
        )
        fig.write_html("InteractiveMachineInfo.html")
        fig.show()
    except KeyError as e:
        warnings.warn(f"Column not found in DataFrame: {e}")
    except Exception as e:
        warnings.warn(f"An error occurred while processing the data: {e}")
    return df

def process_cpus(df, plot=True, month=None, year=None):
    if month is None or year is None:
        month = month or "November"
        year = year or 2024
    title = f"CPU Share {month} {year}"
    counted = 0
    df_cpu_manuf = df['processor'] 
    print(f"\n################## {month} {year} CPU Distribution #####################\nUnique CPU Chip Manufacturers: {df_cpu_manuf.nunique()}")
    intel_count = df_cpu_manuf.str.contains('Xeon|Intel', na=False).sum()
    if intel_count > 0:
        print(f"\nINTEL Processors: {intel_count}")
        counted += intel_count
    amd_count = df_cpu_manuf.str.contains('AMD', na=False).sum()
    if amd_count > 0:
        print(f"\nAMD Processors: {amd_count}")
        counted += amd_count
    ibm_count = df_cpu_manuf.str.contains('IBM|Power', case=False, na=False).sum()
    if ibm_count > 0:
        print(f"\nIBM Processors: {ibm_count}")
        counted += ibm_count
    other_count = (500 - counted)
    print(f"\nOther Processors: {other_count}\n")
    if plot:
        data = [intel_count, amd_count, ibm_count, other_count]
        labels = ['Intel', 'AMD', 'IBM', 'Other']
        colors = sns.color_palette('hls')[0:5]
        plt.figure(figsize=(6, 6))
        plt.pie(data, labels=labels, colors=colors, autopct='%1.2f%%')
        plt.title(title)
        plt.gcf().set_facecolor('white')
        plt.savefig(f"{month}{year}CPUShare.png")
        plt.show()

def process_gpus(df, plot=True, month=None, year=None):
    if month is None or year is None:
        month = month or "November"
        year = year or 2024
    title = f"Accelerator Share {month} {year}"
    counted = 0
    df_acc_manuf = df['accelerator']
    df_acc_machines = df[df['accelerator'].notna()]  
    out_of = len(df_acc_machines.index) 
    print(f"\n\n################## {month} {year} GPU Distribution #####################\nUnique GPU/Accelerator Chip Manufacturers: {df_acc_manuf.nunique()}")
    nvidia_count = df_acc_manuf.str.contains('NVIDIA', case=False, na=False).sum()
    nvidia_ratio = nvidia_count / out_of
    print(f"\nNVIDIA GPU/ACCs: make up {nvidia_count} / {out_of} GPUs/Co-Processor Accelerators")
    print(f"or, {round(nvidia_ratio * 100, 3)}%\n")  
    if plot:
        other_count = out_of - nvidia_count
        data = [nvidia_count, other_count]
        labels = ['NVIDIA', 'Other']
        colors = sns.color_palette('Set2')[0:2]
        plt.figure(figsize=(6, 6))
        plt.pie(data, labels=labels, colors=colors, autopct='%1.2f%%')
        plt.title(title)
        plt.gcf().set_facecolor('white')
        plt.savefig(f"{month}{year}GPUShare.png")
        plt.show()

def process_heterogeneity(df, plot=True, month=None, year=None):
    if month is None or year is None:
        month = month or "November"
        year = year or 2024
    title = f"CPU vs CPU+GPU Machines {month} {year}"
    df['accelerator'].replace([None, ''], np.nan, inplace=True)
    cpuonly = df['accelerator'].isna().sum()
    if cpuonly > 0:
        print(f"\n################## {month} {year} HETEROGENEITY ##################\nCPU-Only Machines: {cpuonly}")
    cpu_acc = 500 - cpuonly
    print(f"\nHeterogeneous Machines: {cpu_acc}\n")
    data = [cpuonly, cpu_acc]
    labels = ['Homogeneous Machines', 'Heterogeneous Machines']
    colors = sns.color_palette('pastel')[0:5]
    if plot:
        plt.figure(figsize=(9, 6))
        plt.pie(data, labels=labels, colors=colors, autopct='%.0f%%')
        plt.title(title)
        plt.gcf().set_facecolor('white')
        plt.savefig(f"{month}{year}heterogeneity.png")
        plt.show()
        
def process_interconnect(df, plot=True, month=None, year=None):
    if month is None or year is None:
        month = month or "November"
        year = year or 2024
    title = f"{month} {year}: Interconnect Distribution"
    interconnects = df['interconnectfamily']  
    ic_ids = interconnects.nunique()  
    print(f"\n################# {month} {year} Interconnects ##################\nNumber of unique interconnects detected: {ic_ids}\n")
    value_counts = df['interconnectfamily'].value_counts()
    print(value_counts)
    if plot:
        print('\n')
        sns.barplot(x=value_counts.index, y=value_counts.values, palette="ch:.25").set(title=title)
        plt.gcf().set_size_inches(10, 5)
        plt.ylabel('Total')
        plt.xlabel('Interconnect Family')
        plt.gcf().set_facecolor('white')
        plt.savefig(f"{month}{year}interconnects.png")
        plt.show()

def main():
    # config 
    warnings.simplefilter(action='ignore', category=FutureWarning)  
    pd.options.mode.chained_assignment = None                       
    sns.set(font="Arial") 

    # get data  
    excel_file = get_data_file()
    df = pd.read_excel(excel_file, engine='openpyxl')

    # clean data
    df = strip_columns(df) 
    df = drop_columns(df) 
    df = check_and_clean_accelerator_column(df) 

    # pre-process data
    df = append_cpu_column(df) 
    df = add_system_architecture_column(df)
    update_interconnect_family_custom(df)
    
    # plot 
    plot_power_vs_performance(df) 
    process_cpus(df, plot=True)
    process_gpus(df, plot=True) 
    process_heterogeneity(df, plot=True)
    process_interconnect(df, plot=True)
    
if __name__ == "__main__":
    main()
