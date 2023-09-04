# Description 
Using Python and Jupyter Notebook, I analyze [the latest top500 dataset](https://www.top500.org/lists/top500/).

# Main Updates from June 2022 -> June 2023 
### AMD has entered the chat
* 28 new machines with AMD CPUs replace 28 machines with Intel CPUs [see visualization of this](https://github.com/tommygorham/top500/blob/main/Visualizations/2023/June2023CPUShare.png)
### The number of heterogeneous (cpu + gpu) machines continues to increase 
* 17 new CPU + GPU Machines [see visualization of this](https://github.com/tommygorham/top500/blob/main/Visualizations/2023/June2023heterogeneity.png)
### NVIDIA continues to dominate 
* 90% of GPUs are NVIDIA [see visualization of this](https://github.com/tommygorham/top500/blob/main/Visualizations/2023/June2023GPUShare.png)


# Context 
The [top500](https://www.top500.org/project/top500_description/) is a list of the 500 most powerful commercially available computer systems known to us. 
The machines in the list are ordered via how fast they can solve a dense N by N system of linear equations Ax = b, which is a common task in engineering.
The performance, denoted as *Rmax*, represents each machines' maximal LINPACK performance achieved, as mesaured by the [LINPACK benchmark](http://www.netlib.org/utk/people/JackDongarra/PAPERS/hpl.pdf). The LINPACK benchmark is extensively used worldwide, and respective performance metrics are typically available for almost all capable systems.

Achieving high-performance with large-scale scientific applications ultimately depends on a large variety of factors. Given the realities of modern hardware architectures, and the challenges this puts on programmers of these machines, it's beneficial to analyze and understand the cardinal hardware characteristics that are nothing less than necessary for the exascale era. Ultimately, having a thorough and current interpretation of the hardware features that theoretically achieve such a high level of performance is ideal when designing approaches to exploit maximum parallelism in scientific applications. Below is a heatmap of some features of the top500 computers via their correlation with this metric. 

<p align="center">
<img src="https://github.com/tommygorham/top500/blob/main/Visualizations/Theoretical_Peak%2BPerformance_Heatmap_of_Corr_Spring22.png" height="600px" /> 
</p> 

```python
# to reproduce this 
df2 = df[[ 'rmaxtflops', 'totalcores', 'corespersocket', 'powerkw', 'processorspeedmhz', 'CPU', 'target' ]]
df2.dropna()
pd.set_option('max_columns', 8)
plt.figure (figsize = (12, 10))
cor = df_numeric.corr()
sns.heatmap (cor, annot=True, cmap=plt.cm.Reds)
plt.savefig("Heatmap.png")
plt.show()
```

# Data Visualizations
All of the latest data visualizations can be viewed [here](https://github.com/tommygorham/top500/blob/main/top500_notebook.ipynb). You can also view the visualizations from previous years in the [Visualizations folder](https://github.com/tommygorham/top500/tree/main/Visualizations). 



[Interactive Hover Plot](http://htmlpreview.github.io/?https://github.com/tommygorham/top500/blob/main/Visualizations/InteractiveMachineInfo2023.html)



