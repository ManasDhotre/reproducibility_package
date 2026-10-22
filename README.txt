Reproducibility package - "Making Confidence Scores Comparable in Multi-Agent Disease Screening"
1. pip install numpy pandas scipy scikit-learn matplotlib   (tested: Python 3.12.3, scikit-learn 1.8.0)
2. python reference_experiment.py   -> writes results_table.csv and results_summary.json (all numbers in Tables 2-3)
3. python figs.py                   -> writes fig_ece.png and fig_reliability.png (Figs. 2-3)
Data: pima.csv (PIMA Indians Diabetes, no header), heart.csv (Cleveland heart disease, 303 rows, 'target' column); the Wisconsin breast cancer data ship with scikit-learn.
Public datasets - cite UCI and the original dataset papers when you reuse them.
