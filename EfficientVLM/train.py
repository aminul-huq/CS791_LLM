import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import *


def train_and_validate(model, train_dataloader, val_dataloader, optimizer, epochs, save_path, device):
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")

        # Training Phase
        model.train()
        total_train_loss = 0.0
        correct_train_top5 = 0
        total_train_topk = 0
        correct_train_all = 0
        total_train = 0

        for batch in tqdm(train_dataloader, desc="Training"):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            # Forward pass
            outputs = model(**batch, output_hidden_states=True, output_attentions=True)
            loss = outputs.loss
            total_train_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Predictions and true labels
            logits = outputs.logits
            labels = batch["labels"]
            labels = (labels > 0).int()
            predicted_classes = torch.sigmoid(logits)

            # Top-5 accuracy
            correct_top5 = topk_accuracy(predicted_classes, labels, k=5)
            correct_train_top5 += correct_top5

            # Overall accuracy (Top-1)
            correct_top1 = topk_accuracy(predicted_classes, labels, k=1)
            correct_train_all += correct_top1

            total_train += labels.sum().item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_topk_accuracy = correct_train_top5 / total_train
        avg_train_all_accuracy = correct_train_all / total_train

        print(
            f"Training - Loss: {avg_train_loss:.4f}, Top-5 Accuracy: {avg_train_topk_accuracy:.4f}, Overall Accuracy: {avg_train_all_accuracy:.4f}")

        # Validation Phase
        model.eval()
        total_val_loss = 0.0
        correct_val_top5 = 0
        total_val_topk = 0
        correct_val_all = 0
        total_val = 0

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                batch = {k: v.to(device) for k, v in batch.items()}

                # Forward pass
                outputs = model(**batch, output_hidden_states=True, output_attentions=True)
                loss = outputs.loss
                total_val_loss += loss.item()

                # Predictions and true labels
                logits = outputs.logits
                labels = batch["labels"]
                labels = (labels > 0).int()
                predicted_classes = torch.sigmoid(logits)

                # Top-5 accuracy
                correct_top5 = topk_accuracy(predicted_classes, labels, k=5)
                correct_val_top5 += correct_top5
                # Overall accuracy (Top-1)
                correct_top1 = topk_accuracy(predicted_classes, labels, k=1)
                correct_val_all += correct_top1
                total_val += labels.sum().item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_val_topk_accuracy = correct_val_top5 / total_val
        avg_val_all_accuracy = correct_val_all / total_val

        print(
            f"Validation - Loss: {avg_val_loss:.4f}, Top-5 Accuracy: {avg_val_topk_accuracy:.4f}, Overall Accuracy: {avg_val_all_accuracy:.4f}")

        if epoch % 10 == 0:
            os.makedirs(save_path, exist_ok=True)
            model_path = f"{save_path}teacher_{epoch}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Teacher model saved to {save_path}")


    if (save_path == 'LLM/models/teacher_models/'):
        os.makedirs(save_path, exist_ok=True)
        model_path = f"{save_path}teacher.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Teacher model saved to {save_path}")
    elif (save_path == '/home/mpervin/LLM/models/student_models/'):
        os.makedirs(save_path, exist_ok=True)
        model_path = f"{save_path}student.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Student model saved to {save_path}")
    return {
        'train_loss': avg_train_loss,
        'train_topk_accuracy': avg_train_topk_accuracy,
        'train_all_accuracy': avg_train_all_accuracy,
        'val_loss': avg_val_loss,
        'val_topk_accuracy': avg_val_topk_accuracy,
        'val_all_accuracy': avg_val_all_accuracy
    }


def test(model, test_dataloader, optimizer, epochs, device):
    # Validation Phase
    model.eval()
    total_val_loss = 0.0
    correct_val_top5 = 0
    total_val_topk = 0
    correct_val_all = 0
    total_val = 0

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Validation"):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch, output_hidden_states=True, output_attentions=True)
            loss = outputs.loss
            total_val_loss += loss.item()

            # Predictions and true labels
            logits = outputs.logits
            labels = batch["labels"]
            labels = (labels > 0).int()
            predicted_classes = torch.sigmoid(logits)

            # Top-5 accuracy
            correct_top5 = topk_accuracy(predicted_classes, labels, k=5)
            correct_test_top5 += correct_top5
            # Overall accuracy (Top-1)
            correct_top1 = topk_accuracy(predicted_classes, labels, k=1)
            correct_test_all += correct_top1
            total_test += labels.sum().item()

    avg_test_loss = total_test_loss / len(test_dataloader)
    avg_test_topk_accuracy = correct_test_top5 / total_test
    avg_test_all_accuracy = correct_test_all / total_test

    print(
        f"Validation - Loss: {avg_test_loss:.4f}, Top-5 Accuracy: {avg_test_topk_accuracy:.4f}, Overall Accuracy: {avg_test_all_accuracy:.4f}")

    return {
        'val_loss': avg_test_loss,
        'val_topk_accuracy': avg_test_topk_accuracy,
        'val_all_accuracy': avg_test_all_accuracy
    }


def KD_train_based_logits(teacher_model, student_model, data_loader, alpha, optimizer, epochs, save_path, device):
    teacher_model.eval()
    student_model.train()
    train_accuracy_list = []  # To store training accuracy per epoch

    for i in tqdm(range(epochs)):  # Loop over the dataset multiple times
        print(f"Epoch: {i}")
        running_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        correct_train_top5 = 0
        correct_train_all = 0
        total_train = 0
        total_train_loss = 0

        for batch in data_loader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch['labels']  # Adjust if your key is different

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            with torch.no_grad():
                teacher_outputs = teacher_model(**batch, output_hidden_states=True, output_attentions=True)
            student_outputs = student_model(**batch, output_hidden_states=True, output_attentions=True)

            distill_loss = distillation_loss(student_outputs.logits, teacher_outputs.logits, temperature=3.0)
            loss = (1 - alpha) * student_outputs.loss + alpha * distill_loss
            total_train_loss += loss.item()

            # Predictions and true labels
            logits = student_outputs.logits
            labels = (labels > 0).int()
            predicted_classes = torch.sigmoid(logits)

            # Top-5 accuracy
            correct_top5 = topk_accuracy(predicted_classes, labels, k=5)
            correct_train_top5 += correct_top5
            # Top-1 accuracy
            correct_top1 = topk_accuracy(predicted_classes, labels, k=1)
            correct_train_all += correct_top1

            total_train += labels.sum().item()

            # Backpropagation
            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(data_loader)
        avg_train_topk_accuracy = correct_train_top5 / total_train
        avg_train_all_accuracy = correct_train_all / total_train

        print(
            f"Validation - Loss: {avg_train_loss:.4f}, Top-5 Accuracy: {avg_train_topk_accuracy:.4f}, Overall Accuracy: {avg_train_all_accuracy:.4f}")

    model_path = f"{save_path}student_logit.pth"
    torch.save(student_model.state_dict(), model_path)
    print(f"Logits states based Student model saved to {save_path}")
    return {
        'train_loss': avg_train_loss,
        'train_topk_accuracy': avg_train_topk_accuracy,
        'train_all_accuracy': avg_train_all_accuracy,
    }


def KD_train_based_hidden_states(teacher_model, student_model, data_loader, projector, alpha, optimizer, epochs,
                                 save_path, device):
    teacher_model.eval()
    student_model.train()
    train_accuracy_list = []  # To store training accuracy per epoch

    for i in tqdm(range(epochs)):  # Loop over the dataset multiple times
        print(f"Epoch: {i}")
        running_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        correct_train_top5 = 0
        correct_train_all = 0
        total_train = 0
        total_train_loss = 0

        for batch in data_loader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch['labels']  # Adjust if your key is different

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            with torch.no_grad():
                teacher_outputs = teacher_model(**batch, output_hidden_states=True, output_attentions=True)
            student_outputs = student_model(**batch, output_hidden_states=True, output_attentions=True)

            distill_loss = hidden_state_distillation_loss(student_outputs.hidden_states[0].to('cpu'),
                                                          teacher_outputs.hidden_states[0].to('cpu'), projector)
            loss = (1 - alpha) * student_outputs.loss + alpha * distill_loss
            total_train_loss += loss.item()

            # Predictions and true labels
            logits = student_outputs.logits
            labels = (labels > 0).int()
            predicted_classes = torch.sigmoid(logits)

            # Top-5 accuracy
            correct_top5 = topk_accuracy(predicted_classes, labels, k=5)
            correct_train_top5 += correct_top5
            # Top-1 accuracy
            correct_top1 = topk_accuracy(predicted_classes, labels, k=1)
            correct_train_all += correct_top1

            total_train += labels.sum().item()

            # Backpropagation
            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(data_loader)
        avg_train_topk_accuracy = correct_train_top5 / total_train
        avg_train_all_accuracy = correct_train_all / total_train

        print(
            f"Validation - Loss: {avg_train_loss:.4f}, Top-5 Accuracy: {avg_train_topk_accuracy:.4f}, Overall Accuracy: {avg_train_all_accuracy:.4f}")
        # Save the student model
    model_path = f"{save_path}student_hidden.pth"
    torch.save(student_model.state_dict(), model_path)
    print(f"Hidden states based Student model saved to {save_path}")

    return {
        'train_loss': avg_train_loss,
        'train_topk_accuracy': avg_train_topk_accuracy,
        'train_all_accuracy': avg_train_all_accuracy,
    }


def KD_train_based_attentions(teacher_model, student_model, data_loader, projector, alpha, mapping_strategy, optimizer,
                              epochs, save_path, device):
    teacher_model.eval()
    student_model.train()
    train_accuracy_list = []  # To store training accuracy per epoch

    for i in tqdm(range(epochs)):  # Loop over the dataset multiple times
        print(f"Epoch: {i}")
        running_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        correct_train_top5 = 0
        correct_train_all = 0
        total_train = 0
        total_train_loss = 0

        for batch in data_loader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch['labels']  # Adjust if your key is different

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            with torch.no_grad():
                teacher_outputs = teacher_model(**batch, output_hidden_states=True, output_attentions=True)
            student_outputs = student_model(**batch, output_hidden_states=True, output_attentions=True)

            distill_loss = attention_distillation_loss(student_outputs.attentions, teacher_outputs.attentions,
                                                       mapping_strategy)
            loss = (1 - alpha) * student_outputs.loss + alpha * distill_loss
            total_train_loss += loss.item()

            # Predictions and true labels
            logits = student_outputs.logits
            labels = (labels > 0).int()
            predicted_classes = torch.sigmoid(logits)

            # Top-5 accuracy
            correct_top5 = topk_accuracy(predicted_classes, labels, k=5)
            correct_train_top5 += correct_top5
            # Top-1 accuracy
            correct_top1 = topk_accuracy(predicted_classes, labels, k=1)
            correct_train_all += correct_top1

            total_train += labels.sum().item()

            # Backpropagation
            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(data_loader)
        avg_train_topk_accuracy = correct_train_top5 / total_train
        avg_train_all_accuracy = correct_train_all / total_train

        print(
            f"Validation - Loss: {avg_train_loss:.4f}, Top-5 Accuracy: {avg_train_topk_accuracy:.4f}, Overall Accuracy: {avg_train_all_accuracy:.4f}")
        # Save the student model
    model_path = f"{save_path}student_attention.pth"
    torch.save(student_model.state_dict(), model_path)
    print(f"Attention based Student model saved to {save_path}")

    return {
        'train_loss': avg_train_loss,
        'train_topk_accuracy': avg_train_topk_accuracy,
        'train_all_accuracy': avg_train_all_accuracy,
    }


def KD_train_based_hybrid(teacher_model, student_model, data_loader, projector, alpha, mapping_strategy, optimizer,
                          epochs, save_path, device):
    teacher_model.eval()
    student_model.train()
    train_accuracy_list = []  # To store training accuracy per epoch

    for i in tqdm(range(epochs)):  # Loop over the dataset multiple times
        print(f"Epoch: {i}")
        running_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        correct_train_top5 = 0
        correct_train_all = 0
        total_train = 0
        total_train_loss = 0

        for batch in data_loader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch['labels']  # Adjust if your key is different

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            with torch.no_grad():
                teacher_outputs = teacher_model(**batch, output_hidden_states=True, output_attentions=True)
            student_outputs = student_model(**batch, output_hidden_states=True, output_attentions=True)

            logits_loss = distillation_loss(student_outputs.logits, teacher_outputs.logits, temperature=3.0)

            hidden_loss = hidden_state_distillation_loss(student_outputs.hidden_states[0].to('cpu'),
                                                         teacher_outputs.hidden_states[0].to('cpu'), projector)

            attention_loss = attention_distillation_loss(student_outputs.attentions,
                                                         teacher_outputs.attentions, mapping_strategy)

            loss = (1 - alpha) * student_outputs.loss + alpha * (attention_loss + hidden_loss + logits_loss)

            total_train_loss += loss.item()

            # Predictions and true labels
            logits = student_outputs.logits
            labels = (labels > 0).int()
            predicted_classes = torch.sigmoid(logits)

            # Top-5 accuracy
            correct_top5 = topk_accuracy(predicted_classes, labels, k=5)
            correct_train_top5 += correct_top5
            # Top-1 accuracy
            correct_top1 = topk_accuracy(predicted_classes, labels, k=1)
            correct_train_all += correct_top1

            total_train += labels.sum().item()

            # Backpropagation
            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(data_loader)
        avg_train_topk_accuracy = correct_train_top5 / total_train
        avg_train_all_accuracy = correct_train_all / total_train

        print(
            f"Validation - Loss: {avg_train_loss:.4f}, Top-5 Accuracy: {avg_train_topk_accuracy:.4f}, Overall Accuracy: {avg_train_all_accuracy:.4f}")
        # Save the student model
    model_path = f"{save_path}student_hybrid.pth"
    torch.save(student_model.state_dict(), model_path)
    print(f"Hybrid Student model saved to {save_path}")

    return {
        'train_loss': avg_train_loss,
        'train_topk_accuracy': avg_train_topk_accuracy,
        'train_all_accuracy': avg_train_all_accuracy,
    }


import torch.nn.utils.prune as prune

def apply_pruning(module, sparsity):
    """
    Apply unstructured pruning to the specified module.
    """
    for name, param in module.named_parameters():
        if 'weight' in name:
            prune.l1_unstructured(module, name='weight', amount=sparsity)

def remove_pruning(module):
    """
    Remove pruning reparameterization from the module.
    """
    for name, param in module.named_parameters():
        if 'weight_orig' in name:
            prune.remove(module, 'weight')


def apply_pruning_to_layer(layer, sparsity):
    """
    Recursively applies pruning to all linear submodules in a composite layer.
    """
    for name, module in layer.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=sparsity)
            # print(f"Pruned")

def remove_pruning_from_layer(layer):
    """
    Recursively removes pruning reparameterization from all linear submodules in a composite layer,
    only if the layer has been pruned.
    """
    for name, module in layer.named_modules():
        if isinstance(module, nn.Linear):
            if hasattr(module, "weight_orig"):  # Check if the layer is pruned
                prune.remove(module, "weight")
                # print(f"Removed pruning ")
            else:
                pass
                # print(f"Skipping removal")

def KD_train_efficientvlm(teacher_model, student_model, data_loader, projector, alpha, beta, gamma, optimizer, epochs,
                          save_path, device, sparsity_vision=0.3, sparsity_text=0.3, sparsity_crossmodal=0.2):
    """
    Train the student model using EfficientVLM-based knowledge distillation with Modal-Adaptive Pruning.
    """
    import os
    os.makedirs(save_path, exist_ok=True)

    teacher_model.eval()
    student_model.train()

    # Locate layers for pruning
    vision_layers = student_model.vilt.encoder.layer  # Update this based on the actual architecture
    text_layers = student_model.vilt.encoder.layer
    cross_modal_layers = student_model.vilt.encoder.layer

    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")
        running_loss = 0.0
        total_correct_top5 = 0
        total_correct_top1 = 0
        total_samples = 0

        # Apply pruning before each epoch
        for layer in vision_layers:
            apply_pruning_to_layer(layer, sparsity_vision)
        for layer in text_layers:
            apply_pruning_to_layer(layer, sparsity_text)
        for layer in cross_modal_layers:
            apply_pruning_to_layer(layer, sparsity_crossmodal)

        for batch in tqdm(data_loader, desc="Training"):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch['labels']

            optimizer.zero_grad()

            # Forward pass
            with torch.no_grad():
                teacher_outputs = teacher_model(**batch, output_hidden_states=True, output_attentions=True)

            student_outputs = student_model(**batch, output_hidden_states=True, output_attentions=True)

            # Compute distillation losses
            logits_loss = distillation_loss(student_outputs.logits, teacher_outputs.logits, temperature=3.0)
            hidden_loss = hidden_state_distillation_loss(
                student_outputs.hidden_states[0].to('cpu'),
                teacher_outputs.hidden_states[0].to('cpu'),
                projector
            )
            attention_loss = attention_distillation_loss(
                student_outputs.attentions,
                teacher_outputs.attentions
            )

            loss = (1 - alpha) * student_outputs.loss + alpha * (
                    beta * logits_loss + gamma * (hidden_loss + attention_loss))
            running_loss += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Calculate top-5 and top-1 accuracy
            logits = student_outputs.logits

            # Ensure labels are class indices
            if labels.dim() > 1:  # One-hot encoded case
                labels = torch.argmax(labels, dim=1)

            # Get top-5 and top-1 predictions
            _, top5_preds = logits.topk(5, dim=1)  # Shape: [batch_size, 5]
            _, top1_preds = logits.topk(1, dim=1)  # Shape: [batch_size, 1]

            # Compare predictions with labels
            total_correct_top5 += (top5_preds == labels.view(-1, 1)).sum().item()
            total_correct_top1 += (top1_preds.view(-1) == labels).sum().item()
            total_samples += labels.size(0)

        # Calculate epoch metrics
        avg_loss = running_loss / len(data_loader)
        top5_accuracy = total_correct_top5 / total_samples
        top1_accuracy = total_correct_top1 / total_samples

        # Remove pruning reparameterization
        for layer in vision_layers:
            remove_pruning_from_layer(layer)
        for layer in text_layers:
            remove_pruning_from_layer(layer)
        for layer in cross_modal_layers:
            remove_pruning_from_layer(layer)

        # Print epoch metrics
        print(f"Epoch {epoch + 1} Loss: {avg_loss:.4f}, Top-5 Accuracy: {top5_accuracy:.4f}, Top-1 Accuracy: {top1_accuracy:.4f}")

        # Ensure the save_path exists
        os.makedirs(save_path, exist_ok=True)

        # Save the model at the end of the epoch
        model_path = f"{save_path}efficientvlm_student_epoch_{epoch + 1}.pth"
        torch.save(student_model.state_dict(), model_path)
        print(f"Saved model after epoch {epoch + 1} to {model_path}")

    # Final model save
    final_model_path = f"{save_path}efficientvlm_student_pruned.pth"
    torch.save(student_model.state_dict(), final_model_path)
    print(f"Final pruned EfficientVLM Student model saved to {final_model_path}")


