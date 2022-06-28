# top500
The [top500](https://www.top500.org/project/top500_description/) is a list of the 500 most powerful commercially available computer systems known to us. 
The machines in the list are ordered via how fast they can solve a dense N by N system of linear equations Ax = b, which is a common task in engineering.
The performance, denoted as *Rmax*, represents each machines' maximal LINPACK performance achieved, as mesaured by the [LINPACK benchmark](http://www.netlib.org/utk/people/JackDongarra/PAPERS/hpl.pdf). The LINPACK benchmark is extensively used worldwide, and respective performance metrics are typically available for almost all capable systems.

# program description 
Using Python and Jupyter Notebook, I analyze [the latest top500 dataset](https://www.top500.org/lists/top500/2022/06/) released in June 2022, where a machine named [Frontier](https://www.olcf.ornl.gov/frontier/) from Oak Ridge National Laboratory achieved Exascale. This means for the first time in the history of humanity, 
a computer solved one quintillion (10^18) double precision (64-bit) calculations each second. This level of performance enables scientists from all domains to 
realistically simulate complex experiments and solve problems that would be impossible to solve without this computational power. In using this newly released dataset, I look to categorize the specifications of the chips and system architectures to identify trends in optimal hardware designs. 


Achieving high-performance with large-scale scientific applications ultimately depends on a large variety of factors. Given the realities of modern hardware architectures, and the challenges this puts on programmers of these machines, it's beneficial to analyze and understand the cardinal hardware characteristics that are nothing less than necessary for the exascale era. Ultimately, having a thorough and current interpretation of the hardware features that theoretically achieve such a high level of performance is ideal when designing approaches to exploit maximum parallelism in scientific applications. 


# goal 
In analyzing this dataset, I hope to produce a meticulous understanding of the most effective hardware design specifications and the most optimal software/hardware combinations that are ultimately necessary for achieving such high performance efficiently. 
