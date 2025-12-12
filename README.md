# shodh_ai

This repository contains my end-to-end solution for the Shodh AI Machine Learning assignment. The goal of the project is to design a system that can approve or deny loans in a way that maximizes long-term financial return, instead of simply predicting default. The project combines classical machine learning, financial reward modeling, and an offline reinforcement learning agent trained on historical loan performance.

Overview

The dataset used is the LendingClub 2007–2018 loan book, which contains detailed borrower information, loan characteristics, and repayment outcomes. The central idea is to understand loan risk, quantify profit or loss, and then learn an optimal approval policy using both supervised prediction models and an offline RL approach.

What This Project Includes

Data Loading and Preprocessing
The raw CSV is very large, so the data is loaded in chunks when needed. All major fields are cleaned, and a few important derived features are created. One of the most important features is the borrower’s credit history length, which is parsed from date strings of various formats. Categorical variables are one-hot encoded, and numerical variables are standardized using a scikit-learn preprocessing pipeline that is saved for reuse.

Supervised Learning Models
Three models were trained: Logistic Regression, Random Forest, and a custom MLP built in PyTorch. Rather than focusing only on classification metrics, these models also help in creating profit-aware threshold policies. Performance stabilizes around an AUC of about 0.70, which is consistent with what is typical on this dataset.

Reward Function and Economic Objective
The financial objective of a lender is very different from the classification objective.
A loan that defaults results in a loss equal to the loan principal.
A loan that is fully paid generates profit proportional to the interest rate.
A denied loan generates zero reward.
This transforms the task from a classification problem into a financial decision problem.

Profit-Maximizing Threshold Policies
Using the probabilities from the ML models, I search for a threshold that maximizes average profit on the test set. This demonstrates that even with simple classifiers, profit-aware decision rules can outperform standard accuracy-based thresholds.

Offline Reinforcement Learning (Discrete CQL)
The main RL part uses the Conservative Q-Learning algorithm (CQL) implemented through the d3rlpy library.
The environment is a single-step decision setup:
the state is the borrower’s feature vector,
the action is either approve or deny,
and the reward reflects the financial return.
Because we cannot interact with a live environment, the entire agent is trained on past data only, using an offline RL approach.
The final RL agent successfully trains using the discrete CQL implementation and produces a reward-maximizing approval strategy that is significantly different from the ML threshold policies.

Evaluation and Results
All metrics, including baseline model performance, threshold-optimized policies, and the RL agent’s financial return, are stored in the results folder. The RL policy shows the strongest performance in terms of accumulated reward, indicating that offline RL is capable of capturing profit opportunities missed by classical models.

Report
A detailed human-written report explaining the methodology, decisions, results, and limitations is included in this repository.

Repository Structure (Plain Description)

The data folder contains the raw dataset and a processed parquet version.
The models folder includes saved versions of all trained models: the preprocessing pipeline, classical ML models, the neural network classifier, and the offline RL agent.
The results folder contains summary metrics for all experiments.
The notebook file walks through the entire workflow from preprocessing to modeling and reinforcement learning.
The report file summarizes the final findings.

Key Takeaways

The main insight is that credit risk prediction alone is not enough.
To optimize lending, we need models that directly consider economic gain and loss.
Supervised models provide a useful baseline but are limited by their focus on classification objectives.
Adding a financial reward layer and tuning decision thresholds produces much better outcomes.
However, the offline RL agent goes a step further by learning a direct approximation of the optimal approval policy, using only past data and without requiring online exploration.

Future Improvements

Several enhancements could be made to extend this project:
adding uncertainty estimation to reduce over-confident approvals,
training deeper RL agents with tuned hyperparameters,
introducing temporal borrower behavior if available,
and eventually deploying the RL model behind an API for real-time decisioning.

