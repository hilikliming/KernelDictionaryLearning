# KernelDictionaryLearning
This repo is dedicated to testing and experiments with the Kernel Dictionary Learning by H. V. Nguyen, V. M. Patel, N. M. Nasrabadi, and R. Chellappa

7/3/19- turns out the emnist set is a bit too big to host on here (probably not too wise anyway) so for the datasets that I said are in this repo but actually aren't go to this link: https://www.nist.gov/itl/products-and-services/emnist-dataset    and scroll to the bottom of the page which has the all datasets in matlab .mat files packed in a .zip. Sorry for the inconvenience.

7/3/19- Also, my usage of B matrix in these functions can be a little misleading. After reading the paper a little more thoroughly I realized there aren't a lot of options for B. due to the E_k matrix being square in the KKSVD formulation, B must have n samples in it's columns so its usually only makes sense to set B=Y, your training samples, as is done in most of the paper's formulation. Following the test scripts carefully should reveal that these are the B's I used as well... 

5/15/20 - Lastly, the implementation of KOMP in this repository is recursive, the pseudoinverse involved in the sparse least squares solution is recursively upated via the block matrix inversion formulas

Referenced Paper can be found here: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6288305

Journal Version Here: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.381.8160&rep=rep1&type=pdf
