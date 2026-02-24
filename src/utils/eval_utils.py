import numpy as np
import torch

# Kullback-Leibler Divergence; measures how one probability distribution diverges from a second, expected probability distribution
def cal_kl(pred: np.ndarray, gt: np.ndarray, eps=1e-12) -> np.ndarray:
    """Calculate Kullback-Leibler Divergence between predicted and ground truth maps.
    This metric penalizes when the prediction has mass in different locations than ground truth.
    steps:
    1. Normalize pred and gt to get probability distributions (map1 and map2)
    2. Compute KL divergence using the formula: KL(P || Q) = sum(P(x) * log(P(x) / Q(x))) where P is the ground truth distribution and Q is the predicted distribution.
    3. Add eps to avoid division by zero and log of zero
    """
    map1, map2 = pred / (pred.sum() + eps), gt / (gt.sum() + eps)
    kld = np.sum(map2 * np.log(map2 / (map1 + eps) + eps))
    return kld

def cal_sim(pred: np.ndarray, gt: np.ndarray, eps=1e-12) -> np.ndarray:
    """Calculate Similarity metric between predicted and ground truth maps.
    This metric measures the overlap between the predicted and ground truth distributions.
    steps:
    1. Normalize pred and gt to get probability distributions (map1 and map2)
    2. Compute the element-wise minimum between the two distributions
    3. Sum the minimum values to get the similarity score
    """
    map1, map2 = pred / (pred.sum() + eps), gt / (gt.sum() + eps)
    intersection = np.minimum(map1, map2)

    return np.sum(intersection)

def image_binary(image, threshold):
    """Convert a grayscale image to binary based on a threshold.
    Pixels with values greater than the threshold are set to 1, others are set to 0.
    """
    output = np.zeros(image.size).reshape(image.shape)
    for xx in range(image.shape[0]):
        for yy in range(image.shape[1]):
            if (image[xx][yy] > threshold):
                output[xx][yy] = 1
    return output

def cal_nss(pred: np.ndarray, gt: np.ndarray, threshold=0.1) -> np.ndarray:
    """Calculate Normalized Scanpath Saliency (NSS) between predicted and ground truth maps.
    This metric measures how well the predicted saliency map aligns with human fixations.
    steps:
    1. Normalize the predicted map to have zero mean and unit standard deviation
    2. Normalize the ground truth map to have values between 0 and 1
    3. Binarize the ground truth map based on a threshold
    4. Compute the element-wise product of the normalized predicted map and the binary ground truth map
    5. Sum the result and divide by the number of fixations to get the NSS score
    """
    # pred = pred / 255.0
    # gt = gt / 255.0
    std = np.std(pred)
    u = np.mean(pred)

    smap = (pred - u) / std
    fixation_map = (gt - np.min(gt)) / (np.max(gt) - np.min(gt) + 1e-12)
    fixation_map = image_binary(fixation_map, threshold)

    nss = smap * fixation_map

    nss = np.sum(nss) / np.sum(fixation_map + 1e-12)

    return nss


def compute_cls_acc(preds, label):
    """Calculate classification accuracy between predicted and ground truth labels.
    steps:
    1. Get the predicted class by taking the argmax of the predictions
    2. Compare the predicted class with the ground truth labels
    3. Compute the number of correct predictions
    4. Divide by the total number of predictions to get the accuracy
    """
    pred = torch.max(preds, 1)[1]
    # label = torch.max(labels, 1)[1]
    num_correct = (pred == label).sum()
    return float(num_correct) / float(preds.size(0))

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0.0

    def update(self, val, n=1.0):
        self.val = val
        self.sum += val * n
        self.cnt += n
        if self.cnt == 0:
            self.avg = 1
        else:
            self.avg = self.sum / self.cnt