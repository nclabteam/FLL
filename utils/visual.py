import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

def plot_accuracy_granularity(data, save_path='heatmap.png', figsize=(14, 8)):
    data = {row[0]: list(row[1:]) for row in data.rows()}

    # Create a Polars DataFrame from the dictionary
    df = pl.DataFrame(data)

    # Convert the Polars DataFrame to a long format for heatmap compatibility
    df_long = df.melt(id_vars=['accuracy'], variable_name='clients_and_servers', value_name='value')

    # Convert back to a wide format for seaborn heatmap compatibility
    df_pivot = df_long.pivot(index='clients_and_servers', columns='accuracy', values='value')

    # Convert to a format that seaborn can use directly
    df_pivot_pd = df_pivot.to_pandas()

    # Set the index to be the 'clients_and_servers' column
    df_pivot_pd.set_index('clients_and_servers', inplace=True)

    # Create the heatmap
    plt.figure(figsize=figsize)

    # Set the plot background color
    plt.rcParams['axes.facecolor'] = 'black'
    plt.rcParams['savefig.facecolor'] = 'black'
    plt.rcParams['figure.facecolor'] = 'black'

    # Set the font properties
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    # Create the heatmap with appropriate settings for a black background
    ax = sns.heatmap(
        df_pivot_pd, 
        annot=True, 
        cmap='viridis', 
        cbar=True, 
        fmt='.0f', 
        annot_kws={"size": 8, "color": "white"},
        linewidths=.5, 
        linecolor='black'
    )

    plt.title('Heatmap of Epochs to Reach Different Accuracy Levels', color='white')
    plt.xlabel('Accuracy (%)', color='white')
    plt.ylabel('Clients and Servers', color='white')

    # Set tick labels color
    plt.xticks(color='white')
    plt.yticks(color='white')

    # Change color bar (legend) label to white
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color('white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    # Save the plot as a PNG file with 300 DPI
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

def plot_participant_rate(data, figsize=(10, 5), save_path='participant.png'):
    # Extract keys and values
    client_ids = list(data.keys())
    iterations = list(data.values())

    # Create the plot
    plt.figure(figsize=figsize)

    # Set the background color
    plt.rcParams['axes.facecolor'] = 'black'
    plt.rcParams['savefig.facecolor'] = 'black'

    # Plot the data
    plt.bar(client_ids, iterations, color='skyblue')

    # Add titles and labels with larger font size and white color
    plt.title('Number of Iterations Each Client Participates In', fontsize=14, color='white')
    plt.xlabel('Client ID', fontsize=12, color='white')
    plt.ylabel('Number of Iterations', fontsize=12, color='white')

    # Set x-ticks and y-ticks to white color
    plt.xticks(client_ids, fontsize=10, color='white')
    plt.yticks(fontsize=10, color='white')

    # Remove border (spines)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    # Save the plot with higher DPI
    plt.savefig(save_path, dpi=300, bbox_inches='tight')