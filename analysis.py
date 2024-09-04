import os
import polars as pl
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for saving files
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
import pandas as pd

def setup_plot(figsize=(12, 8)):
    """
    Set up the plot with default settings.
    
    Args:
    figsize (tuple): The figure size (width, height) in inches.
    
    Returns:
    fig, ax: The figure and axis objects.
    """
    plt.clf()
    plt.cla()
    plt.close('all')
    
    fig, ax = plt.subplots(figsize=figsize, facecolor='black')
    ax.set_facecolor('black')
    
    # Remove borders
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Set default text color to white
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    
    return fig, ax

def set_titles(ax, title, xlabel, ylabel):
    """
    Set the title and labels for the plot with predefined styles.
    
    Args:
    ax: The axis object.
    title (str): The title of the plot.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    """
    ax.set_title(title, fontsize=24, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=20, labelpad=15)
    ax.set_ylabel(ylabel, fontsize=20, labelpad=15)

def save_and_show(fig, output_file):
    """
    Save the figure and display it.
    
    Args:
    fig: The figure object.
    output_file (str): The filename to save the figure.
    """
    plt.tight_layout()
    fig.savefig(output_file, facecolor='black', edgecolor='none', bbox_inches='tight', dpi=300)
    plt.close(fig)

def create_boxplot(acc, output_file='boxplot.png', interval=200):
    num_rounds = len(next(iter(acc.values())))
    num_intervals = num_rounds // interval
    num_cases = len(acc)

    fig_width = max(20, 8 + 1.5 * num_intervals)
    fig_height = max(10, 6 + 0.5 * num_cases)
    
    fig, ax = setup_plot((fig_width, fig_height))
    
    colors = plt.cm.Set3(np.linspace(0, 1, num_cases))
    color_dict = dict(zip(acc.keys(), colors))

    for i in range(num_intervals):
        start = i * interval
        end = (i + 1) * interval
        
        for j, (key, values) in enumerate(acc.items()):
            position = j + i * (num_cases + 1)
            box = ax.boxplot([values[start:end]], positions=[position], widths=0.6, 
                             patch_artist=True, showmeans=False)
            
            for patch in box['boxes']:
                patch.set_facecolor(color_dict[key])
            
            for element in ['whiskers', 'caps', 'medians']:
                plt.setp(box[element], color='white')
        
        middle_position = i * (num_cases + 1) + num_cases / 2
        ax.text(middle_position, ax.get_ylim()[0], f'{start+1}-{end}', 
                horizontalalignment='center', verticalalignment='top', color='white', fontsize=14)
        
        if i < num_intervals - 1:
            separator_position = (i + 1) * (num_cases + 1) - 0.5
            ax.axvline(x=separator_position, color='white', linestyle='--', alpha=0.5)

    set_titles(ax, f'Box Plots for All Cases (Every {interval} Rounds)', 'Cases and Round Intervals', 'Accuracy')
    ax.set_xticks([])

    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color_dict[key], edgecolor='white') for key in acc.keys()]
    ax.legend(legend_elements, acc.keys(), loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=12)

    ax.tick_params(axis='y', labelsize=14, pad=10)

    for text in ax.texts:
        text.set_y(text.get_position()[1] - 0.02)

    save_and_show(fig, output_file)

def create_accuracy_heatmap(acc, output_file='accuracy_heatmap.png', interval=5):
    fig_width = max(15, 8 + 0.5 * (100 // interval))  # Base width + additional width for each threshold
    fig_height = max(10, 5 + 0.5 * len(acc))  # Base height + additional height for each case
    fig, ax = setup_plot((fig_width, fig_height))

    accuracy_thresholds = np.arange(0, 101, interval)
    heatmap_data = np.full((len(acc), len(accuracy_thresholds)), np.nan)
    
    for i, (case, accuracies) in enumerate(acc.items()):
        accuracies_percent = [acc * 100 for acc in accuracies]  # Convert to percentage
        for j, threshold in enumerate(accuracy_thresholds):
            epoch = next((epoch for epoch, acc in enumerate(accuracies_percent) if acc >= threshold), None)
            if epoch is not None:
                heatmap_data[i, j] = epoch + 1  # +1 because epochs are typically 1-indexed
    
    sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd', cbar_kws={'label': 'Epoch'})
    
    set_titles(ax, 'Epoch at which Accuracy Threshold is Reached', 'Accuracy Threshold (%)', 'Cases')
    
    plt.xticks(np.arange(len(accuracy_thresholds)) + 0.5, accuracy_thresholds)
    plt.yticks(np.arange(len(acc)) + 0.5, list(acc.keys()), rotation=0)
    
    save_and_show(fig, output_file)

def create_learning_curves(acc, output_file='learning_curves.png'):
    num_cases = len(acc)
    num_epochs = len(next(iter(acc.values())))
    
    fig_width = 12
    fig_height = 8
    
    fig, ax = setup_plot((fig_width, fig_height))
    for case, accuracies in acc.items():
        ax.plot(range(1, len(accuracies) + 1), accuracies, label=case)
    
    set_titles(ax, 'Learning Curves for All Cases', 'Epoch', 'Accuracy')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    save_and_show(fig, output_file)

def create_accuracy_distribution(acc, output_file='accuracy_distribution.png'):
    num_cases = len(acc)
    
    fig_width = max(12, 8 + 0.5 * num_cases)
    fig_height = max(8, 6 + 0.2 * num_cases)
    
    fig, ax = setup_plot((fig_width, fig_height))
    data = []
    for case, accuracies in acc.items():
        data.extend([(case, acc) for acc in accuracies])
    df = pd.DataFrame(data, columns=['Case', 'Accuracy'])
    
    sns.violinplot(x='Case', y='Accuracy', data=df)
    set_titles(ax, 'Accuracy Distribution Across Epochs for All Cases', 'Case', 'Accuracy')
    plt.xticks(rotation=45)
    save_and_show(fig, output_file)

def create_accuracy_improvement_rate(acc, output_file='accuracy_improvement_rate.png'):
    num_cases = len(acc)
    num_epochs = len(next(iter(acc.values())))
    
    fig_width = 12
    fig_height = 8
    
    fig, ax = setup_plot((fig_width, fig_height))
    for case, accuracies in acc.items():
        improvement_rate = np.diff(accuracies)
        ax.plot(range(1, len(improvement_rate) + 1), improvement_rate, label=case)
    
    set_titles(ax, 'Accuracy Improvement Rate for All Cases', 'Epoch', 'Accuracy Improvement Rate')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    save_and_show(fig, output_file)

def create_moving_average_accuracy(acc, window=5, output_file='moving_average_accuracy.png'):
    num_cases = len(acc)
    num_epochs = len(next(iter(acc.values())))
    
    fig_width = 12
    fig_height = 8
    
    fig, ax = setup_plot((fig_width, fig_height))
    for case, accuracies in acc.items():
        moving_avg = np.convolve(accuracies, np.ones(window), 'valid') / window
        ax.plot(range(window, len(accuracies) + 1), moving_avg, label=case)
    
    set_titles(ax, f'Moving Average Accuracy (Window={window}) for All Cases', 'Epoch', 'Moving Average Accuracy')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    save_and_show(fig, output_file)

if __name__ == '__main__':
    root = 'runs'
    acc = {}
    for case in range(4, 7):
        case = f'exp{case}'
        directory = os.path.join(root, case)
        df = []
        for subdir in os.listdir(directory):
            path = os.path.join(directory, subdir)
            if not os.path.isdir(path): continue
            path = os.path.join(path, 'results', 'server.csv')
            if not os.path.exists(path): continue
            df.append(pl.read_csv(path))
        df = sum(df) / len(df)
        acc[case.replace('', '')] = df['test_global_accs'].to_list()
        for col in df.columns:
            if 'global_accs' in col and 'ResNet' in col:
                acc[case.replace('', '') + '' + col.replace('_global_accs', '')] = df[col].to_list()

    create_boxplot(acc, output_file='custom_boxplot.png', interval=200)
    create_accuracy_heatmap(acc, output_file='custom_heatmap.png', interval=5)
    create_accuracy_distribution(acc, 'accuracy_distribution.png')
    create_learning_curves(acc, 'learning_curves.png')
    create_accuracy_improvement_rate(acc, 'accuracy_improvement_rate.png')
    create_moving_average_accuracy(acc, window=3, output_file='moving_average_accuracy.png')