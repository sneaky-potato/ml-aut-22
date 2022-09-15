# Assignment 1 &middot;

>

Directory for the `first` assignment of Machine Learning course (CS60050) offered in Autumn semester 2022, Department of CSE, IIT Kharagpur.

## Getting started

Read the assignment problem statement from [Assignment_1.pdf](/assignment_1/Assignment_1.pdf)

Dataset is provided in the file [Dataset_A.csv](/assignment_1/Dataset_A.csv) and its description in [Dataset_A_Description.pdf](/assignment_1/Dataset_A_Description.pdf)

Python version information-  

```shell
Python 3.10.5
```

- Install required python packages-

```shell
pip install -r requirements.txt
```

(In case of an error, notice the required packages for running the files are- `numpy`, `matplotlib`, `pandas`). Install them individually if the above command fails (version conflict)

- Run [data_cleaner.py](/assignment_1/data_cleaner.py) to clean and categorically encode the dataset and produce [cleaned_data.csv](/assignment_1/cleaned_data.csv)

```shell
python data_cleaner.py
```

## Solution

- [q1.py](/assignment_1/q1.py) is the submission file for the first question
- [q2.py](/assignment_1/q2.py) is the submission file for the second question

- Run these files individually to get the output on terminal and output files.

```shell
python q1.py
python q2.py
```

- [DecisionTree.txt](/assignment_1/DecisionTree.txt) contains the final decision tree.
- [accuracy_VS_depth.png](/assignment_1/accuracy_VS_depth.png) conatins the plot of accuracy vs depth for the decision tree.
- [NaiveBayes.txt](/assignment_1/NaiveBayes.txt) contains the results for Naive Bayes classifier.
