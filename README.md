# top500
The [top500](https://www.top500.org/project/top500_description/) is a list of the 500 most powerful commercially available computer systems known to us. 
The machines in the list are ordered via how fast they can solve a dense N by N system of linear equations Ax = b, which is a common task in engineering.
The performance, denoted as *Rmax*, represents each machines' maximal LINPACK performance achieved, as mesaured by the [LINPACK benchmark](http://www.netlib.org/utk/people/JackDongarra/PAPERS/hpl.pdf). The LINPACK benchmark is extensively used worldwide, and respective performance metrics are typically available for almost all capable systems.


# program description 
Using Python and Jupyter Notebook, I analyze [the latest top500 dataset](https://www.top500.org/lists/top500/2022/06/) released in June 2022, where a machine named [Frontier](https://www.olcf.ornl.gov/frontier/) from Oak Ridge National Laboratory achieved Exascale. This means for the first time in the history of humanity, 
a computer solved one quintillion (10^18) double precision (64-bit) calculations each second. This level of performance enables scientists from all domains to 
realistically simulate complex experiments and solve problems that would be impossible to solve without this computational power. In using this newly released dataset, I look to categorize the specifications of the chips and system architectures to identify trends in optimal hardware designs. 

Achieving high-performance with large-scale scientific applications ultimately depends on a large variety of factors. Given the realities of modern hardware architectures, and the challenges this puts on programmers of these machines, it's beneficial to analyze and understand the cardinal hardware characteristics that are nothing less than necessary for the exascale era. Ultimately, having a thorough and current interpretation of the hardware features that theoretically achieve such a high level of performance is ideal when designing approaches to exploit maximum parallelism in scientific applications. Below is a heatmap of some features of the top500 computers via their correlation with this metric. 

<p align="center">
<img src="https://github.com/tommygorham/top500/blob/main/Visualizations/Theoretical_Peak%2BPerformance_Heatmap_of_Corr_Spring22.png" height="600px" /> 
</p> 

# program updates
I plan to update this code with each new release of the top500 list (at the lastest) and eventually release an updated machine learning model that I made in Graduate School which can be used to identify the most effective hardware design specifications and the most optimal software/hardware combinations by training it on the latest datasets. 

# visualizations
All of the data visualizations from running [the program](https://github.com/tommygorham/top500/blob/main/top500analysis.py) in Jupyter Notebook can be viewed [here](https://github.com/tommygorham/top500/blob/main/top500_notebook.ipynb). You can also view the visualizations in the [Visualizations folder](https://github.com/tommygorham/top500/tree/main/Visualizations). I've included the scatterplot below as an example. This plot is produced by scaling the *powerkw* and *rmaxtflops* features, and renaming the *processor* to a more generalized category (e.g., AMD, NVIDIA, IBM, Other). The righmost datapoint (which is Red) is Frontier, AMD's heterogeneous machine that achieved exascale in June 2022. 

**To hover over data points and view machine info interactively,** [click here](http://htmlpreview.github.io/?https://github.com/tommygorham/top500/blob/main/Visualizations/InteractiveMachineInfo.html) 


<p align="center">
<img src="https://github.com/tommygorham/top500/blob/main/Visualizations/June2022powervperformance_cpu_and_arch.png" height="600px"  />
</p> 

[Interactive Hover Plot](http://htmlpreview.github.io/?https://github.com/tommygorham/top500/blob/main/Visualizations/InteractiveMachineInfo.html)

![InteractiveHover](https://user-images.githubusercontent.com/38857089/177635140-def4959e-1cf4-4806-99ec-f5d05377a479.png)


