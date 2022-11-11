# Assignment 2 &middot;

>

Directory for the `second` and final assignment of Machine Learning course (CS60050) offered in Autumn semester 2022, Department of CSE, IIT Kharagpur.

## Getting started

Read the assignment problem statement from [Assignment_2.pdf](/assignment_2/Assignment_2.pdf)

Dataset is provided here- [Dataset_A.csv](https://archive.ics.uci.edu/ml/datasets/Wine) and its description in [Dataset_A_Description.pdf](/assignment_2/Dataset_A_Description.pdf)

Python version information-  

```shell
Python 3.10.8
```

- Install required python packages-

```shell
pip install -r requirements.txt
```

(In case of an error, notice the required packages for running the files are- `numpy`, `matplotlib`, `pandas`, `sklearn`). Install them individually if the above command fails (version conflict)

## Solution

- [q1.py](/assignment_2/q1.py) is the submission file for the first question
- [q2.py](/assignment_2/q2.py) is the submission file for the second question

- Run these files individually to get the output on terminal and output files.

```shell
python q1.py
python q2.py
```

- [learning_rate_vs_accuracy](/assignment_2/learning_rate_vs_accuracy_.png) conatins the plot of learning rate vs accuracy of MLP model.
- [K_VS_NMI](/assignment_2/K_VS_NMI.png) conatins the plot of Normalized mutual information vs number of clusters.
- [PCA_visulation](/assignment_2/PCA_visulation.png) conatins the scatter plot of data with first two principal components
- [Variance_VS_Component_number](/assignment_2/Variance_VS_Component_number.png) contains the plot of cumulative variance of data vs number of components.
