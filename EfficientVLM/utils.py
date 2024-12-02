import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def topk_accuracy(predicted_classes, labels, k):
    probs, classes = torch.topk(predicted_classes, k, dim=1)
    topk_predictions = torch.zeros_like(labels).scatter_(1, classes, 1.0)
    correct_topk = ((topk_predictions * labels) > 0).sum().item()
    return  correct_topk


class HiddenStateProjector(nn.Module):
    def __init__(self, teacher_dim, student_dim):
        super(HiddenStateProjector, self).__init__()
        self.projection = nn.Linear(teacher_dim, student_dim)

    def forward(self, teacher_hidden_states):
        return self.projection(teacher_hidden_states)


def hidden_state_distillation_loss(student_hidden_states, teacher_hidden_states, projector):
    projected_teacher_hidden_states = projector(teacher_hidden_states)
    loss = nn.functional.mse_loss(student_hidden_states, projected_teacher_hidden_states)

    return loss

def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    # Soften the student and teacher logits using temperature scaling
    student_soft = F.log_softmax(student_logits / temperature, dim=1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=1)

    # Calculate the KL divergence loss
    loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)

    return loss

def attention_distillation_loss(student_attentions, teacher_attentions, mapping_strategy='direct'):
    
    if mapping_strategy == 'direct':
        # Match the first `len(student_attentions)` layers of the teacher with the student
        selected_teacher_attentions = teacher_attentions[:len(student_attentions)]
    elif mapping_strategy == 'average':
        # Average teacher's attention maps in pairs to align with student layers
        num_student_layers = len(student_attentions)
        selected_teacher_attentions = [
            sum(teacher_attentions[2 * i:2 * i + 2]) / 2 for i in range(num_student_layers)
        ]
    else:
        raise ValueError("Unsupported mapping strategy. Use 'direct' or 'average'.")

    loss = 0
    for student_att, teacher_att in zip(student_attentions, selected_teacher_attentions):
        loss += F.mse_loss(student_att, teacher_att)

    # Average the loss across all layers
    loss /= len(student_attentions)
    return loss
