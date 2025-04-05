# top500 
Data Visualization & Analysis of [the latest top500 dataset](https://www.top500.org/lists/top500/) using Python and Jupyter notebook. 

Latest Dataset: November 2024

## Machine Learning  
### Correlation between Theoretical Peak Performance and Computer Architecture 
![Theoretical_Peak+Performance_Heatmap_of_Corr_Spring22](Visualizations/2022/Theoretical_Peak+Performance_Heatmap_of_Corr_Spring22.png)

![ML Predictions](Visualizations/2022/ML_Model_Predictions_Spring2022.png)

### Key Findings and Observations

* **Power (kw)** and **total cores** show strong positive correlations indicating that power consumption and more cores generally lead 
to higher performance.

* **Cores per socket**, **Processor Speed (mhz)**  and **CPU Type** show relatively weaker positive correlations suggesting that cores 
per socket, CPU type, and processor clock speed are not significant drivers or predictors of supercomputer performance in modern systems.

## Top500 Visualizations & Charts 

### Power vs Performance 
[Interactive Hover
Plot](http://htmlpreview.github.io/?https://github.com/tommygorham/top500/blob/main/InteractiveMachineInfoNovember2024.html)

Non-interactive Plot  
<img src="https://github.com/tommygorham/top500/blob/main/Visualizations/2024/November2024powervsperformance.png" 
     alt="Power Vs Performance" width="765" />

<div style="display: flex; gap: 10px;">
  <img 
       src="https://github.com/tommygorham/top500/blob/main/Visualizations/2024/November2024CPUShare.png" 
       alt="CPU Share" width="400" />
  <img 
       src="https://github.com/tommygorham/top500/blob/main/Visualizations/2024/November2024GPUShare.png" 
       alt="GPU Share" width="400" />
</div>

<img src="https://github.com/tommygorham/top500/blob/main/Visualizations/2024/November2024heterogeneity.png" 
     alt="Heterogeneity" width="750" />

<img src="https://github.com/tommygorham/top500/blob/main/Visualizations/2024/November2024interconnects.png" 
     alt="Interconnects" width="750" />

## Trends 
### 2023 -> June 2024 
* AMD CPU Share now 31%, up from 24% last year (2023) 
* 2% increase in CPU+GPU machines from 2023  
* 88.6% of GPUs still NVIDIA 

### 2022 -> 2023 
* 28 new machines with AMD CPUs replace 28 machines with Intel CPUs
* 17 new CPU + GPU Machines
* 90% of GPUs are NVIDIA
