from jiwer import wer
import torch
import numpy as np

def calculate_wer(prediction, ground_truth):
    """
    Calculate Word Error Rate (WER) between the predicted and ground-truth sequences.
    
    Args:
        prediction (str): Predicted sequence.
        ground_truth (str): Ground-truth sequence.
    
    Returns:
        float: Computed WER.
    """

    return wer(ground_truth, prediction)

def best_epoch(validation_performances, weights):
    """
    Args:
    validation_performances (dict): Dictionary containing validation performances for each epoch
    weights (dict): Dictionary containing weights for each validation performance metric: wer (the lower the better), coe_audio, coe_video, cmr_audio, cmr_video
    Returns:
    best_epoch: int
    """
    scores = []
    for epoch_idx in range(len(validation_performances)):
        score = 0
        for key, weight in weights.items():
            score += validation_performances[epoch_idx][key] * weight
        scores.append(score)
    
    best_epoch = np.argmin(scores)
    
    if isinstance(best_epoch, np.ndarray):
        best_epoch = best_epoch[0]
    
    return best_epoch
