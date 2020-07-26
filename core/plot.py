import matplotlib.pyplot as plt
import pandas as pd

def plot_figures():
    df = pd.read_csv('./csv_log/training.log')
    
    plt.plot(df['accuracy'], label=('accuracy'))
    plt.plot(df['val_accuracy'], label=('validation accuracy'))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation accuracy')
    plt.legend()
    plt.savefig('./figures/accuracy.png')
    
    plt.plot(df['loss'], label=('loss'))
    plt.plot(df['val_loss'], label=('validation loss'))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation loss')
    plt.legend()
    plt.savefig('./figures/loss.png')

