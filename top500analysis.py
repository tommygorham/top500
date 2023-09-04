# Program:     top500analysis.py
# Description: Top500.org June2023 list analysis of the fastest supercomputers
#              based on their respective scores on the LINPACK Benchmark
# Author: Tommy Gorham

########################### libraries ###########################
import os 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import math 
from statistics import mean, median, mode, pstdev, StatisticsError
import warnings #to remove 'future warnings' from pandas output 
#################################################################

########################### datasets ###########################
current_path = os.getcwd()
#df_june22 = pd.read_excel(current_path+"/datasets/top500june22data.xlsx", engine='openpyxl')
df_latest =  pd.read_excel(current_path+"/datasets/TOP500JUNE2023.xlsx", engine='openpyxl')
#df_nov21 = pd.read_excel("top500nov21data.xlsx", engine='openpyxl')
#df_june11 = pd.read_excel(current_path+"/datasets/top500june11data.xls") 
################################################################

########################### pre-processing functions ###########################
def stripCols(df): # make cols easier to access from python df calls
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

def dropCols(df): # preserve memory by forcibly deleting what is not needed 
    del df['previousrank']
    del df['interconnect']
    del df['site']
    del df['siteid']
    del df['systemid'] 
    del df['nhalf'] 
    return df
###############################################################################

########################### basic analysis functions and plots ###########################
# INFO: All function names that begin with "process" take in 4 args: dataframe, month, year, bool for plotting (e.g., true == plot) 
# this function is to return the number of cpu chip manufacturers, if doPlot == true, it will produce a pie plot 
def processCpuManufacturers(df, m, y, doPlot): # count & plot cpu manufacturers (e.g., AMD, Intel, NVIDIA, IBM, etc...) 
    month = m
    year = y
    title = "CPU Share " + str(month) + " " + str(year)
    counted = 0
    df_cpu_manuf = df['processor']  #temp dataframe, scoped to this function
    # jupyter output 
    print("\n################## " + str(month) + str(year) + " CPU Distribution #####################\nUnique CPU Chip Manufacturers: ", df_cpu_manuf.nunique())
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
    ## IBM    
    ibm_count = df_cpu_manuf.str.contains('IBM|Power', case=False).sum()
    if ibm_count>0:
        print ("\nIBM Processors: ", ibm_count)
        counted += ibm_count
    other_count = (500 - counted)         
    print("\nOther Processors: ", other_count, "\n")
    # begin plot if bool var == true
    if doPlot is False: 
        return 
    else: 
        data = [intel_count, amd_count, ibm_count, other_count]
        labels = ['Intel', 'AMD', 'IBM', 'Other'] 
        #define Seaborn color palette to use
        colors = sns.color_palette('hls')[0:5]
        #create pie chart
        plt.pie(data, labels = labels, colors = colors, autopct='%1.2f%%')
        plt.title(title)
        fig = plt.gcf()
        fig.set_facecolor('white')
        fig.set_size_inches(6,6) # or
        plt.savefig(str(month) + str(year) + "CPUShare.png")
        plt.show()
# end function

# this function returns the number of gpu chip manufacturers, if doPlot == true, it will produce a pie plot 
def processAccManufacturers(df, m, y, doPlot):  
    month = m
    year = y
    title = "Accelerator Share " + str(month) + " " + str(year)
    counted = 0
    #drop all rows with 'None' for GPU/Accelerator
    df_acc_machines = df[['accelerator']].replace(to_replace='None', value=np.nan).dropna()
    df_acc_manuf = df['accelerator']
    out_of = len(df_acc_machines.index) #total systems with gpu acc's
    print("\n\n################## " + str(month) + str(year) + " GPU Distribution #####################\nUnique GPU/Accelerator Chip Manufacturers: ", df_acc_manuf.nunique())
    # Nvidia
    nvidia_count = df_acc_manuf.str.contains('NVIDIA', case=False).sum()
    nvidia_ratio = nvidia_count / out_of 
    print("\nNVIDIA GPU/ACCs: make up", nvidia_count, "/", out_of, "GPUs/Co-Processor Accelerators")
    print(" or, ", round(nvidia_ratio, 3), "%\n")  
    if doPlot is False: 
        return 
    else: 
        other_count = out_of - nvidia_count
        data = [nvidia_count, other_count]
        labels = ['NVIDIA', 'Other'] 
        #define Seaborn color palette to use
        colors = sns.color_palette('Set2')[0:2]
        #create pie chart
        plt.pie(data, labels = labels, colors = colors, autopct='%1.2f%%')
        plt.title(title)
        fig = plt.gcf()
        fig.set_facecolor('white')
        fig.set_size_inches(6,6) # or
        plt.savefig(str(month) + str(year) + "GPUShare.png")
        plt.show()
# end function

#this function returns the number of cpu-only machines vs cpu-gpu machines, if doPlot == true, it will produce a pie plot 
def processHeterogeneity(df, m, y, doPlot): # detect cpu vs cpu+gpu machines
    month = m
    year = y
    title = "CPU vs CPU+GPU Machines " + str(month) + " " + str(year)
    df_acc = df['accelerator']
    cpuonly=df_acc.str.contains('None').sum()
    if cpuonly>0:
        print ("\n################## " + str(month) + str(year) +" HETEROGENEITY  ##################\nCPU-Only Machines: ", cpuonly)
    cpu_acc = 500 - cpuonly 
    print("\nHeterogeneous Machines: ", cpu_acc, "\n")
    sns.set_style("whitegrid")
    #define data
    data = [cpuonly, cpu_acc]
    labels = ['Homogenous Machines', 'Heterogenous Machines'] 
    #define Seaborn color palette to use
    colors = sns.color_palette('pastel')[0:5]
    #create pie chart if bool true
    if doPlot is False: 
        return cpu_acc
    else: 
        plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
        plt.title(title)
        fig = plt.gcf()
        fig.set_size_inches(9,6) # or
        plt.savefig(str(month) + str(year) + "heterogeneity.png")
        plt.show()    
        return (cpu_acc / 500)# return percentage to calculate percent increase 
#end function 

# this function is for comparing the % change of two different datasets
def calcPercentIncrease(x, y): # y is initial val, x is new val 
     increase = ((x - y) / y)*100
     increase = math.trunc(increase) 
     percent = '\n' + str(increase) + "%"
     return percent

# this function returns GPU core stats from a dataset 
def printGpuAccCoreStats(df, m, y):
    month = m
    year = y
    title = "\n##################### GPU/ACC Core Stats " + str(month) + " " + str(year) + " #####################"
    print("\n"+title)
    df_cleaned = df.acceleratorcoprocessorcores.dropna()
    stats = [int(n) for n in df_cleaned] # list comprehension
    print('Minimum Cores: ' , "{:,}".format(min(stats)))
    print('Maximum Cores: ' , "{:,}".format(max(stats)))
    print('Range: ' , "{:,}".format(max(stats)-min(stats)))
    mean_stat = math.floor(mean(stats))  
    math.floor(mean_stat) 
    print('Mean: ' ,  "{:,}".format(mean_stat))
    #print('Mean: ' , "{:,}".format(mean(int(stats)))) 
    print('Median: ', "{:,}".format(median(stats)))
    print('Mode of Core Count: ', "{:,}".format(mode(stats)))   
#end function 

# this function plots the power in kW vs the performance in Linpack Tflops (rmax)  
def plotPowerVsPerformance(df, m, y):
    palette = {
    'AMD'  : 'tab:red', #e.g., AMD machines will produce red points 
    'Intel': 'tab:blue',
    'IBM'  : 'darkblue', 
    'OTHER' : 'tab:orange',
   } 
    month = m
    year = y
    title = str(month) + " " + str(year) + ": Power vs Performance via CPU Chip Manufacturer & System Architecture" 
    print("\n############## Power vs Performance " + str(month) + " " + str(year) + " ##################\n")  
    sns.set(rc={"figure.figsize":(10, 8)}) #width=3, #height=4
    sns.scatterplot(data=df, x="performancelog2", y="powerlog2", hue="cpu", style="sysarch", palette=palette).set(title=title)
    plt.xlabel('Maximal LINPACK Performance (Tflops/sec Scaled to Log Base 2)') #x label
    plt.ylabel('Maximal Power (Kilowatts Scaled to Log Base 2)') #y label
    plt.legend(loc='lower right', prop={'size':12}, markerscale=1.1) 
    plt.savefig(str(month)+str(year)+"powervperformance_cpu_and_arch.png")
    plt.show()
#end function 

#this function returns the respective count of interconnects, if doPLot == true, it produces a bar graph of the interconnects 
def processInterconnect(df, m, y, doPlot): 
    month = m
    year = y
    title =  str(month) + " " + str(year) + ": Interconnect Distribution"
    interconnects = df['interconnectfamily']  
    ic_ids = interconnects.nunique()
    print("\n################# "+ str(month) + str(year) + " Interconnects ##################\nNumber of unique interconnects detected: ", ic_ids, "\n")
    if doPlot is False: 
        print(df['interconnectfamily'].value_counts())
    else: 
        print(df['interconnectfamily'].value_counts())
        print('\n')
        sns.barplot(x=df_latest.interconnectfamily.value_counts().index, y=df_latest.interconnectfamily.value_counts(), palette="ch:.25").set(title=title)
        plt.gcf().set_size_inches(10,5)
        plt.ylabel('Total')
        plt.xlabel('InterconnectFamily')
        plt.savefig(str(month)+str(year)+"interconnects.png")
        plt.show()
#end function 

# this function returnsinfo on the top machine in the list 
# @cpu_cores_per-node: machine dependent, currently requires lookup 
# @gpus_per_node:      machine dependent, currently requires lookup 
# @gpu_cores_per_chip: machine dependent, currently requires lookup  
def showTopMachineSpecs(df, m, y): 
    month = m
    year = y
    topmachine = df.iloc[0]
    name = str(topmachine['name'])  
    # you will have to look the lines below up 
    cpu_cores_per_node = 64 # via: AMD Epyc Specs
    gpus_per_node = 4 # via the diagram of Frontier's nodes via: https://www.olcf.ornl.gov/wp-content/uploads/2020/02/frontier_node_diagram_lr.png
    gpu_cores_per_chip = 220 #via amd instinct: https://www.amd.com/system/files/documents/amd-instinct-mi200-datasheet.pdf
    gpu_cores_per_node = gpus_per_node*gpu_cores_per_chip
    
    # getting true cpu cores 
    tmcpucores = topmachine.totalcores - topmachine.acceleratorcoprocessorcores
    tmgpucores = topmachine.acceleratorcoprocessorcores
    gpu_to_cpu_core_ratio = math.floor(topmachine.acceleratorcoprocessorcores) / math.floor(tmcpucores) 
    # num nodes assuming 1 CPU per node, (e.g., Frontier) 
    numnodes =  math.floor(tmcpucores) / math.floor(topmachine.corespersocket)
    numnodes = math.floor(numnodes) 
    totalgpus = gpus_per_node * numnodes
    physical_device_ratio = numnodes/totalgpus
    
      ### FINAL PRINT ###
    print("\n################ " + str(month) + " " + str(year) + " Top Machine Info #####################\nCurrent fastest machine: " + name)
    print("Node Count: ", numnodes, 
          "nCPU Chip Count: ", numnodes, # 1 cpu per node via: https://www.olcf.ornl.gov/wp-content/uploads/2020/02/frontier_node_diagram_lr.png
          "\nCPU Cores Per Node: ", cpu_cores_per_node,
          "\nTotal CPU Cores: ", math.floor(tmcpucores),
          "\nGPUs Per Node: ", gpus_per_node,
          "\nGPU Cores Per Chip: ", gpu_cores_per_chip,
          "\nGPU Cores Per Node: ", gpu_cores_per_node,  
          "\nGPU Device Count: " , totalgpus, 
          "\nTotal GPU Cores: ", math.floor(tmgpucores), 
          "\nCPUs per GPU on each node: ", physical_device_ratio,
           "\nTotal GPU Cores to CPU Cores Ratio: ", gpu_to_cpu_core_ratio)
    
######################### END OF FUNCTIONS #######################################################

def main():
    ### SET OPTIONS 
    warnings.simplefilter(action='ignore', category=FutureWarning)  # remove future update warnings 
    pd.options.mode.chained_assignment = None  # default='warn'
    sns.set(font="Arial") # for plots
    
    ### GLOBAL VARS: declare dfs global if using jupyter (to facilitate notebook queries) 
    global df_latest
    #global df_june11
    
    ### CLEAN DATA 
    df_latest = stripCols(df_latest) # ensure any dfs that are created run stripCols 
    #df_june11 = stripCols(df_june11) # stripCols provides easier access of columns from python  
    df_latest = dropCols(df_latest)  # if dataset is ~10 years old, dropCols manually as below 
    # selecting the cols we want 
    #df_june11 = df_june11[['rmax', 'rpeak', 'cores', 'nmax', 'power', 'processorfamily', 'processor', 'processorcores',        'systemfamily','systemmodel', 'operatingsystem', 'accelerator', 'architecture', 'interconnectfamily', 'interconnect']]
    # end of data cleaning 
    
    ### PRE-PROCESSING
    # make sure col names match across datasets
    #df_june11.rename(columns={'cores': 'totalcores'}, inplace=True)
    df_latest.rename(columns={'acceleratorcoprocessor': 'accelerator'}, inplace=True)
    # append a CPU column to main df to contain a categorical var 
    df_cpunames = df_latest[[ "processor"]]
    df_cpunames.loc[df_cpunames['processor'].str.contains('AMD')] ='AMD'
    df_cpunames.loc[df_cpunames['processor'].str.contains('Xeon')] ='Intel'
    df_cpunames.loc[df_cpunames['processor'].str.contains('IBM')] ='IBM'
    df_cpunames.loc[df_cpunames['processor'].str.contains('GHz')] ='OTHER' # keep this line at the end of the above three reassignments or use the ~ operator here 
    df_latest['cpu']= df_cpunames # adding this back to main df (cpu chip manuf)
    # data validation, ensure everything reassigned without missing values
    if ((df_cpunames.isnull().values.any()) != False):
        print("\nNull Values Detected in df_cpunames.")     
    # df of scaled data for plotting power and performance    
     # add a heterogeneous column 
    df_latest['sysarch'] = np.where(df_latest['accelerator']=='None', 'CPU Only Machine', 'CPUGPU Machine') 
    toscale = df_latest[['powerkw', 'rmaxtflops', 'cpu', 'sysarch', 'nmax']] 
    toscale['powerlog2'] = np.log2(toscale['powerkw'])
    toscale.dropna()
    toscale['performancelog2'] = np.log2(toscale['rmaxtflops'])
    df_scaled = toscale.dropna(axis=0, how='any') 
    # add scaled data back to original df 
    df_latest['log2power'] = df_scaled['powerlog2'] 
    df_latest['log2performance'] = df_scaled['performancelog2']
    # end of preprocessing

    ### BEGIN ANALYSIS 
    processCpuManufacturers                       (df_latest, "June", "2023", True) # CPU 
    processAccManufacturers                       (df_latest, "June", "2023", True) # GPU 
    june23_heterogeneity = processHeterogeneity   (df_latest, "June", "2023", True) # CPU-only vs CPU+GPU machines
    #june11_heterogeneity = processHeterogeneity   (df_june11, "June", "2011", True) # CPU-only vs CPU+GPU machines in 2011 
    print("\nJune2023 Ratio of CPU-GPU Machines vs CPU only: ", june23_heterogeneity) 
    #print("\nJune2011 Ratio of CPU-GPU Machines vs CPU only: ", june11_heterogeneity) 
    #increase_in_heterogeneity =  calcPercentIncrease(june23_heterogeneity, june11_heterogeneity) 
    print('\nJune 2011 - June 2023 increase in heterogeneity: ' , end="")
    #print(increase_in_heterogeneity) 
    plotPowerVsPerformance                        (df_scaled, "June", "2023")       # kW by Rmax via CPU Chip Manufacturer
    # interactive/hover plot 
    import plotly.express as px
    print("\n\nPower vs Performance, Hover for Machine Info\n") 
    plt = px.scatter(df_latest, x="log2performance", y="log2power", color='cpu', symbol='sysarch',  hover_data=['name'], color_discrete_map={
    'AMD':'red', #e.g., AMD machines will produce red points 
    'Intel': '#0277bd',
    'IBM'  : 'blue', 
    'OTHER' : 'orange'}, labels={"log2performance":"Maximal LINPACK Performance (Tflops/sec Scaled to Log Base 2)", "log2power": "Maximal Power (Kilowatts Scaled to Log Base 2)"},  width=1200, height=800) 
    plt.write_html("InteractiveMachineInfo2023.html")
    plt.show()
    #end interactive hover plot 
    processInterconnect                           (df_latest, "June", "2023", True) # Gigabit ethernet, infiniband, etc. 
    printGpuAccCoreStats                          (df_latest, "June", "2023")       # GPU Core Stats
    showTopMachineSpecs                           (df_latest, "June", "2023")       # fastest machine 
    ###########################################################################################

    # if futher analysis necessary, divide data into two categories 
    df_quan = df_latest.select_dtypes(include=np.number) # numerics only df
    print("\n\nQuantitative Columns\n", df_quan.columns)
    
    df_qual = df_latest.select_dtypes(include=object) #categorical only df 
    print("\n\nQualitative Columns\n", df_qual.columns)
    
if __name__ == "__main__":
    main()
