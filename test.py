import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\ida\OneDrive\Skrivbord\TRA235\CSV_files\Autodoor - Troubleshooting - Haas Service Manual.csv', on_bad_lines='skip')
print(df)