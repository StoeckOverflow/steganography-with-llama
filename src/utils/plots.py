import numpy as np
import matplotlib.pyplot as plt
import json


def plot_DetectGPT_logprob_comparison(human_scores, llm_scores):
    '''
    Plot basic curve between 2 different scores. Not usable with DetectGPTSeeker main, but helpful for analyzing its internals.
    '''
    human_mean = np.mean(human_scores)
    llm_mean = np.mean(llm_scores)

    human_std = np.std(human_scores)
    llm_std = np.std(llm_scores)

    print("Human-written feeds: Mean =", human_mean, ", Std Dev =", human_std)
    print("LLM-written feeds: Mean =", llm_mean, ", Std Dev =", llm_std)

    plt.hist(human_scores, bins=10, alpha=0.5, label='Human')
    plt.hist(llm_scores, bins=10, alpha=0.5, label='LLM')
    plt.axvline(human_mean, color='blue', linestyle='dashed', linewidth=1)
    plt.axvline(llm_mean, color='orange', linestyle='dashed', linewidth=1)
    plt.legend()
    plt.title('Prediction Scores Distribution')
    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.savefig('testfig.png')