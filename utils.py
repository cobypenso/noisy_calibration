import torch
from torch import nn, optim
from torch.nn import functional as F
from scipy import optimize
import numpy as np
import csv
from torch import optim

class VectorScalingModel(nn.Module):
    def __init__(self, class_num=65):
        super(VectorScalingModel, self).__init__()
        self.W = Parameter(torch.ones(class_num))
        self.b = Parameter(torch.zeros(class_num))

    def forward(self, logits):
        out = logits * self.W + self.b
        return out

class MatrixScalingModel(nn.Module):
    def __init__(self, class_num=65):
        super(MatrixScalingModel, self).__init__()
        self.W = Parameter(torch.eye(class_num))
        self.b = Parameter(torch.zeros(class_num))

    def forward(self, logits):
        out = torch.matmul(logits, self.W) + self.b
        return out



def VectorOrMatrixScaling(logits, labels, outputs_target, labels_target, cal_method=None):
    nll_criterion = nn.CrossEntropyLoss().cuda()
    adaece_criterion = AdaptiveECELoss().cuda()
    ece_criterion = ECELoss().cuda()
    class_num = logits.shape[1]

    if cal_method == 'VectorScaling':
        cal_model = VectorScalingModel(class_num=class_num).cuda()
    elif cal_method == 'MatrixScaling':
        cal_model = MatrixScalingModel(class_num=class_num).cuda()
    optimizer = optim.SGD(cal_model.parameters(), lr=0.01, momentum=0.9)

    logits = logits.cuda().float()
    labels = labels.cuda().long()
    outputs_target = outputs_target.cuda().float()
    labels_target = labels_target.cuda().long()

    # Calculate NLL and ECE before vector scaling or matrix scaling
    before_calibration_nll = nll_criterion(outputs_target, labels_target).item()
    before_calibration_ece = ece_criterion(outputs_target, labels_target).item()

    max_iter = 5000 
    for _ in range(max_iter):
        optimizer.zero_grad()
        out = cal_model(logits)
        loss = nn.CrossEntropyLoss().cuda()(out, labels)
        loss.backward()
        optimizer.step()
    final_output = cal_model(outputs_target)

    # Calculate NLL and ECE after temperature scaling
    after_calibration_nll = nll_criterion(final_output, labels_target).item()
    after_calibration_adaece = adaece_criterion(final_output, labels_target).item()
    after_calibration_ece = ece_criterion(final_output, labels_target).item()

    return after_calibration_ece, after_calibration_adaece, after_calibration_nll


class TempScalingOnECE(nn.Module):
    def __init__(self, noisy_labels = False, epsilon = 0, transition_matrix = None):
        super(TempScalingOnECE, self).__init__()
        self.noisy_labels = noisy_labels
        self.epsilon = epsilon
        self.transition_matrix = transition_matrix
        self.temperature = 2.0

    def find_best_T(self, logits, labels, optimizer = 'fmin'):
        
        ece_criterion = ECELoss()
        def eval(x):
            "x ==> temperature T"
            x = torch.from_numpy(x)
            scaled_logits = logits.float() / x
            if self.noisy_labels:
                
                if self.transition_matrix is not None:
                    # If transition matrix is given
                    loss = ece_criterion.forward_with_noisy_labels_and_transition_matrix(scaled_logits, labels, self.transition_matrix)
                else:
                    # Use the epsilon provided
                    loss = ece_criterion.forward_with_noisy_labels(scaled_logits, labels, self.epsilon)
            else:   
                loss = ece_criterion.forward(scaled_logits, labels)
            return loss

        if optimizer == 'fmin':
            optimal_parameter = optimize.fmin(eval, torch.Tensor([2.0]), disp=False)
            self.temperature = optimal_parameter[0]
        elif optimizer == 'brute':
            rranges = (slice(1,10,0.05),)
            optimal_parameter = optimize.brute(eval, rranges, finish=optimize.fmin)
            self.temperature = optimal_parameter[0]
        else:
            raise Exception(f'{optimizer} not supported')
                
        return self.temperature.item()


class TempScalingOnAdaECE(nn.Module):
    def __init__(self, noisy_labels = False, epsilon = 0, transition_matrix = None):
        super(TempScalingOnAdaECE, self).__init__()
        self.noisy_labels = noisy_labels
        self.epsilon = epsilon
        self.transition_matrix = transition_matrix
        self.temperature = 2.0

    def find_best_T(self, logits, labels, optimizer = 'fmin'):
        
        ece_criterion = AdaptiveECELoss()
        def eval(x):
            "x ==> temperature T"
            x = torch.from_numpy(x)
            scaled_logits = logits.float() / x
            if self.noisy_labels:
                
                if self.transition_matrix is not None:
                    # If transition matrix is given
                    loss = ece_criterion.forward_with_noisy_labels_and_transition_matrix(scaled_logits, labels, self.transition_matrix)
                else:
                    # Use the epsilon provided
                    loss = ece_criterion.forward_with_noisy_labels(scaled_logits, labels, self.epsilon)
            else:   
                loss = ece_criterion.forward(scaled_logits, labels)
            return loss


        if optimizer == 'fmin':
            optimal_parameter = optimize.fmin(eval, torch.Tensor([2.0]), disp=False)
            self.temperature = optimal_parameter[0]
        elif optimizer == 'brute':
            rranges = (slice(1,10,0.05),)
            optimal_parameter = optimize.brute(eval, rranges, finish=optimize.fmin)
            self.temperature = optimal_parameter[0]
        else:
            raise Exception(f'{optimizer} not supported')

                
        return self.temperature.item()



class Oracle(nn.Module):
    def __init__(self):
        super(Oracle, self).__init__()

    def find_best_T(self, logits, labels):
        ece_criterion = ECELoss()
        def eval(x):
            "x ==> temperature T"
            x = torch.from_numpy(x)
            scaled_logits = logits.float() / x
            loss = ece_criterion(scaled_logits, labels)
            return loss
        optimal_parameter = optimize.fmin(eval, torch.Tensor([2.0]), disp=False)
        self.temperature = optimal_parameter[0]
        return self.temperature.item()




class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    """
    def __init__(self, n_bins=15, LOGIT = True):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.LOGIT = LOGIT

    def forward(self, logits, labels,):
        if self.LOGIT:
            softmaxes = F.softmax(logits, dim=1)
        else:
            softmaxes = logits
        confidences, predictions = torch.max(softmaxes, 1)
        correctness = predictions.eq(labels)
        confidences[confidences == 1] = 0.999999
        ece = torch.zeros(1, device=logits.device)

        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if (prop_in_bin.item() > 0) and (in_bin.sum() > 20):
                accuracy_in_bin = correctness[in_bin].float().mean()
                accuracy_in_bin = min(accuracy_in_bin, 0.99)
                accuracy_in_bin = max(accuracy_in_bin, 0.01)
                avg_confidence_in_bin = confidences[in_bin].mean().float()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    def forward_with_noisy_labels(self, logits, labels, epsilon, num_classes = 10):
        if self.LOGIT:
            softmaxes = F.softmax(logits, dim=1)
        else:
            softmaxes = logits
        confidences, predictions = torch.max(softmaxes, 1)
        correctness = predictions.eq(labels)
        confidences[confidences == 1] = 0.999999
        ece = torch.zeros(1, device=logits.device)

        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()

            if (prop_in_bin.item() > 0) and (in_bin.sum() > 20):
                accuracy_in_bin = correctness[in_bin].float().mean()
                accuracy_in_bin = min(accuracy_in_bin, 0.99)
                accuracy_in_bin = max(accuracy_in_bin, 0.01)

                # -- Fixing the noisy-accuracy -- #
                accuracy_in_bin = (accuracy_in_bin - (epsilon / (num_classes-1))) / (1 - epsilon - (epsilon / (num_classes - 1)))
                # ------------------------------- #

                avg_confidence_in_bin = confidences[in_bin].mean().float()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

    def forward_with_noisy_labels_and_transition_matrix(self, logits, labels, transition_matrix, num_classes = 10):
        from sklearn.metrics import confusion_matrix
        if self.LOGIT:
            softmaxes = F.softmax(logits, dim=1)
        else:
            softmaxes = logits
        confidences, predictions = torch.max(softmaxes, 1)
        correctness = predictions.eq(labels)
        confidences[confidences == 1] = 0.999999
        ece = torch.zeros(1, device=logits.device)
        
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())

            prop_in_bin = in_bin.float().mean()
            if (prop_in_bin.item() > 0) and (in_bin.sum() > 20):
                predictions_in_bin = predictions[in_bin]
                labels_in_bin = labels[in_bin]
                # Compute confusion matrix in bin
                M = confusion_matrix(labels_in_bin, predictions_in_bin, labels = np.arange(num_classes), normalize='all')

                # Compute "fixed" accuracy in bin -  A = M * inv(P)
                accuracy_in_bin = np.trace(M * np.linalg.inv(transition_matrix))
                accuracy_in_bin = min(accuracy_in_bin, 0.99)
                accuracy_in_bin = max(accuracy_in_bin, 0.01)

                avg_confidence_in_bin = confidences[in_bin].mean().float()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece


class AdaptiveECELoss(nn.Module):
    '''
    Compute Adaptive ECE
    '''
    def __init__(self, n_bins=15, LOGIT=True):
        super(AdaptiveECELoss, self).__init__()
        self.nbins = n_bins
        self.LOGIT = LOGIT

    def histedges_equalN(self, x):
        npt = len(x)
        return np.interp(np.linspace(0, npt, self.nbins + 1),
                     np.arange(npt),
                     np.sort(x))
    def forward(self, logits, labels):
        if self.LOGIT:
            softmaxes = F.softmax(logits, dim=1)
            confidences, predictions = torch.max(softmaxes, 1)
        else:
            confidences, predictions = torch.max(logits, 1)
        correctness = predictions.eq(labels)
        confidences[confidences == 1] = 0.999999
        n, bin_boundaries = np.histogram(confidences.cpu().detach(), self.histedges_equalN(confidences.cpu().detach()))
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        ece = torch.zeros(1, device=logits.device)

        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = correctness[in_bin].float().mean()
                accuracy_in_bin = min(accuracy_in_bin, 0.99)
                accuracy_in_bin = max(accuracy_in_bin, 0.01)
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

    def forward_with_noisy_labels(self, logits, labels, epsilon, num_classes = 10):
        if self.LOGIT:
            softmaxes = F.softmax(logits, dim=1)
            confidences, predictions = torch.max(softmaxes, 1)
        else:
            confidences, predictions = torch.max(logits, 1)
        confidences[confidences == 1] = 0.999999
        correctness = predictions.eq(labels)
        n, bin_boundaries = np.histogram(confidences.cpu().detach(), self.histedges_equalN(confidences.cpu().detach()))
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        ece = torch.zeros(1, device=logits.device)

        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = correctness[in_bin].float().mean()
                accuracy_in_bin = min(accuracy_in_bin, 0.99)
                accuracy_in_bin = max(accuracy_in_bin, 0.01)

                # -- Fixing the noisy-accuracy -- #
                accuracy_in_bin = (accuracy_in_bin - (epsilon / (num_classes-1))) / (1 - epsilon - (epsilon / (num_classes - 1)))
                # ------------------------------- #

                avg_confidence_in_bin = confidences[in_bin].mean().float()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

    def forward_with_noisy_labels_and_transition_matrix(self, logits, labels, transition_matrix, num_classes = 10):
        from sklearn.metrics import confusion_matrix
        if self.LOGIT:
            softmaxes = F.softmax(logits, dim=1)
            confidences, predictions = torch.max(softmaxes, 1)
        else:
            confidences, predictions = torch.max(logits, 1)

        confidences[confidences == 1] = 0.999999
        correctness = predictions.eq(labels)
        n, bin_boundaries = np.histogram(confidences.cpu().detach(), self.histedges_equalN(confidences.cpu().detach()))
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        ece = torch.zeros(1, device=logits.device)

        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())

            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                predictions_in_bin = predictions[in_bin]
                labels_in_bin = labels[in_bin]
                # Compute confusion matrix in bin
                M = confusion_matrix(labels_in_bin, predictions_in_bin, labels = np.arange(num_classes), normalize='all')

                # Compute "fixed" accuracy in bin -  A = M * inv(P)
                accuracy_in_bin = np.trace(M * np.linalg.inv(transition_matrix))
                accuracy_in_bin = min(accuracy_in_bin, 0.99)
                accuracy_in_bin = max(accuracy_in_bin, 0.01)

                avg_confidence_in_bin = confidences[in_bin].mean().float()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

    

def get_accuracy_of_bins(logits, labels, bins):
    bin_boundaries = torch.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    accuracy_list = []
    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    correctness = predictions.eq(labels)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = correctness[in_bin].float().mean()
            accuracy_list.append(accuracy_in_bin)
        else:
            accuracy_list.append(None)
    
    return accuracy_list

def get_confidence_of_bins(logits, bins):
    bin_boundaries = torch.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    confidence_list = []
    prop_in_bin_list = []

    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            avg_confidence_in_bin = confidences[in_bin].mean().float()
            confidence_list.append(avg_confidence_in_bin)
        else:
            confidence_list.append(None)

        
        prop_in_bin_list.append(prop_in_bin.item())

    return confidence_list, prop_in_bin_list