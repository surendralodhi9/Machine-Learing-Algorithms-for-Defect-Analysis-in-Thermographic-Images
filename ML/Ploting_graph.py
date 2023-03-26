import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("plotoutputlogis2.csv")
#df
ordered_df = df.sort_values(by='Id')
my_range=range(1,len(df.index)+1)

plt.hlines(y=my_range, xmin=ordered_df['Actual'], xmax=ordered_df['Predicted'], color='grey', alpha=0.4)
plt.scatter(ordered_df['Actual'], my_range, color='navy', alpha=1, label='Actual')
plt.scatter(ordered_df['Predicted'], my_range, color='gold', alpha=0.8 , label='Predicted')
plt.legend()

# Add title and axis names
plt.yticks(my_range, ordered_df['Id'])
plt.title("Comparison of the Actual and Predicted values", loc='left')
plt.xlabel('Predicted value')
plt.ylabel('Id')
plt.show()
