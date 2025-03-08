from matplotlib import pyplot as plt
import pandas as pd
import os

PATH_CURRENT = os.path.abspath(os.getcwd())
PATH_ANNOTATED = f"{PATH_CURRENT}/annotated/"
PATH_IMG = f"{PATH_CURRENT}/img/"

def plot_equal_height_bars(model: str):
    file_path = PATH_ANNOTATED + f"{model}.csv"
    # Load the data
    df = pd.read_csv(file_path)
    
    # Drop the first column (financebench_id)
    df = df.drop(columns=["financebench_id"])
    
    # Rename columns
    df = df.replace({'C': 'Correct answer', 'I': 'Incorrect answer', 'R': 'Fail to answer'})
    
    # Plot the data
    df_counts = df.apply(pd.Series.value_counts).fillna(0)
    df_counts = df_counts.T
    
    # Convert counts to percentages
    df_percentages = df_counts.div(df_counts.sum(axis=1), axis=0)
    
    # Define colors for each category
    colors = {'Correct answer': '#69b57b', 'Incorrect answer': '#fc5251', 'Fail to answer': '#ea9d9d'}
    
    # Reorder columns to move "Fail to answer" to the top
    df_percentages = df_percentages[['Correct answer', 'Incorrect answer', 'Fail to answer']]
    
    ax = df_percentages.plot(kind='bar', stacked=True, color=[colors.get(x, '#333333') for x in df_percentages.columns], edgecolor='black')
    
    plt.title(model)
    plt.ylabel('Percentage of Data')
    plt.ylim(0, 1)
    plt.legend(bbox_to_anchor=(0.5, 1.25), loc='upper center', ncol=3)
    
    # Remove upper and right frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust x-labels to be flat and alternate if overlapping
    labels = ax.get_xticklabels()
    for i, label in enumerate(labels):
        label.set_rotation(0)
        label.set_verticalalignment('bottom')
        if i % 2 == 0:
            label.set_y(label.get_position()[1] - 0.05)  # Move labels lower
        else:
            label.set_y(label.get_position()[1] - 0.1)  # Move labels even lower
    
    plt.tight_layout()
    plt.savefig(PATH_IMG + f"{model}.png", dpi=300)
    plt.close()  # Close the plot to ensure it is saved properly

models = ["claude-3-5-haiku-20241022", "gpt-4o-mini"]
for model in models:
    plot_equal_height_bars(model)