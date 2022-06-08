# Program: top500analysis.py
# Description: Top500.org June2022 list analysis
# Published: TBD
# Author: Tommy Gorham

# libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import warnings #to remove 'future warnings' from pandas output 

# load data into dataframes globally for use in jupyter notebook 
df_june22 = pd.read_excel("top500june22data.xlsx", engine='openpyxl')
df_nov21 = pd.read_excel("top500nov21data.xlsx", engine='openpyxl')
df_june11 = pd.read_excel("top500june11data.xls") 

########################### PREPROCESSING FUNCTIONS ###########################

# function to make cols easy to access  
def stripCols(df):
    # clean dataset for easy access to columns
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

# function to preserve memory by forcibly deleting what is not needed 
def dropCols(df):
    del df['previousrank']
    del df['interconnect']
    del df['site']
    del df['siteid']
    del df['systemid'] 
    del df['nhalf'] 
    return df
    
########################### END OF PREPROCESSING ###########################

########################### ANALYSIS FUNCTIONS ###########################

# count store and plot cpu manufacturers by passing df and name for pie chart
def processCpuManufacturers(df, m, y): 
    month = m
    year = y
    title = "CPU Share " + str(month) + " " + str(year)
    counted = 0
    df_cpu_manuf = df['processor'] 
    print("\n\nUnique Cpu Chip Manufacturers: ", df_cpu_manuf.nunique())
    ### Intel 
    intel_count = df_cpu_manuf.str.contains('Xeon|Intel').sum()
    if intel_count>0:
        print ("\nINTEL Processors: ", intel_count)
        counted += intel_count
    ### AMD
    amd_count=df_cpu_manuf.str.contains('AMD').sum()
    if amd_count>0:
        print ("\nAMD Processors: ", amd_count)
        counted += amd_count
    #IBM    
    ibm_count = df_cpu_manuf.str.contains('IBM|Power', case=False).sum()
    if ibm_count>0:
        print ("\nIBM Processors: ", ibm_count)
        counted += ibm_count
    other_count = (500 - counted)         
    print("\nOther Processors: ", other_count)
    # begin plot 
    data = [intel_count, amd_count, ibm_count, other_count]
    labels = ['Intel', 'AMD', 'IBM', 'Other'] 
    #define Seaborn color palette to use
    colors = sns.color_palette('hls')[0:5]
    #create pie chart
    plt.pie(data, labels = labels, colors = colors, autopct='%1.2f%%')
    plt.title(title)
    fig = plt.gcf()
    fig.set_facecolor('lightgrey')
    fig.set_size_inches(6,6) # or
    plt.savefig(str(month) + str(year) + "CPUShare.png")
    plt.show()
# end function

# function for cpu+gpu machines
def heterogeniety(df, m, y): 
    month = m
    year = y
    title = "Top500 CPU vs CPUGPU Machines " + str(month) + " " + str(year)
    df_acc = df['accelerator']
    cpuonly=df_acc.str.contains('None').sum()
    if cpuonly>0:
        print ("\n\nCPU-Only Machines: ", cpuonly)
    cpu_acc = 500 - cpuonly 
    print("\n\nHeterogeneous Machines: ", cpu_acc)
    sns.set_style("whitegrid")
    #define data
    data = [cpuonly, cpu_acc]
    labels = ['Homogenous Machines', 'Heterogenous Machines'] 
    #define Seaborn color palette to use
    colors = sns.color_palette('pastel')[0:5]
    #create pie chart
    plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
    plt.title(title)
    fig = plt.gcf()
    fig.set_size_inches(6,3) # or
    plt.savefig(str(month) + str(year) + "heterogeniety.png")
    plt.show()
    
def processAccManufacturers(df, m, y):
    month = m
    year = y
    title = "Accelerator Share " + str(month) + " " + str(year)
    counted = 0
    #drop all rows with 'None' for GPU/Accelerator
    df_acc_machines = df[['accelerator']].replace(to_replace='None', value=np.nan).dropna()
    df_acc_manuf = df['accelerator']
    out_of = len(df_acc_machines.index)
    print("\n\nUnique GPU/Accelerator Chip Manufacturers: ", df_acc_manuf.nunique())
    # Nvidia
    nvidia_count = df_acc_manuf.str.contains('NVIDIA', case=False).sum()
    nvidia_ratio = nvidia_count / out_of 
    print("\nNVIDIA GPU/ACCs: make up", nvidia_count, "/", out_of, "GPUs/Co-Processor Accelerators")
    print("\nOr, ", round(nvidia_ratio, 3), "%") 
        
 # df_june22.acceleratorcoprocessorcores.dropna()

########################### END OF ANALYSIS FUNCTIONS ###########################

def main():
    # remove future update warnings from jupyter output 
    warnings.simplefilter(action='ignore', category=FutureWarning) 
    global df_june22
    global df_nov21
    global df_june11
    
    # PREPROCESSING  
    df_june22 = stripCols(df_june22)
    df_nov21  = stripCols(df_nov21)
    df_june11 = stripCols(df_june11)   
    df_june22 = dropCols(df_june22)
    df_nov21  = dropCols(df_nov21) 
    # some column names were different in older datasets...doing a few things manually for now 
    # select explicitly what we want from older df's 
    df_june11 = df_june11[['rmax', 'rpeak', 'cores', 'nmax', 'power', 'processorfamily', 'processor', 'processorcores',        'systemfamily','systemmodel', 'operatingsystem', 'accelerator', 'architecture', 'interconnectfamily', 'interconnect']]
    #so gpu columns match older df 
    df_nov21.rename(columns={'acceleratorcoprocessor': 'accelerator'}, inplace=True)
    df_june22.rename(columns={'acceleratorcoprocessor': 'accelerator'}, inplace=True)
    
    # show trimmed cols 
    #print("\n June2022 Col names:\n", df_june22.columns)
    #print("\nNov2021 Col names:\n", df_nov21.columns)
    #print("\nJune2011 Col names:\n", df_june11.columns)
    
    #Process CPU Manufacturers
    #processCpuManufacturers(df_june11, "June", "2011")
    #processCpuManufacturers(df_nov21, "November", "2021") 
    #processCpuManufacturers(df_june22, "June", "2022")
    
    #heterogeniety(df_june11, "June", "2011")
    #heterogeniety(df_nov21, "November", "2021") 
    #heterogeniety(df_june22, "June", "2022")
    
    processAccManufacturers(df_june22, "June", "2022") 
    
    
    #powervseffeciency
    #performance_vs_cpu_cores_vs_gpu_cores(in machines that have gpus) 
    
    
    
if __name__ == "__main__":
    main()