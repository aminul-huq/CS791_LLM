import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import *

def train_and_validate(model, train_dataloader, val_dataloader, optimizer, epochs, save_path, model_name, device):
    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}/{epochs}")
        
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
            correct_top5 =  topk_accuracy(predicted_classes, labels, k=5)
            correct_train_top5 += correct_top5

            # Overall accuracy (Top-1)
            correct_top1 =  topk_accuracy( predicted_classes, labels, k=1)
            correct_train_all += correct_top1
            
            total_train += labels.sum().item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_topk_accuracy = correct_train_top5 / total_train
        avg_train_all_accuracy = correct_train_all / total_train

        print(f"Training - Loss: {avg_train_loss:.4f}, Top-5 Accuracy: {avg_train_topk_accuracy:.4f}, Overall Accuracy: {avg_train_all_accuracy:.4f}")

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
                correct_top5 = topk_accuracy(predicted_classes, labels,k = 5)
                correct_val_top5 += correct_top5
                # Overall accuracy (Top-1)
                correct_top1 = topk_accuracy(predicted_classes, labels, k = 1)
                correct_val_all += correct_top1
                total_val += labels.sum().item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_val_topk_accuracy = correct_val_top5 / total_val
        avg_val_all_accuracy = correct_val_all / total_val

        print(f"Validation - Loss: {avg_val_loss:.4f}, Top-5 Accuracy: {avg_val_topk_accuracy:.4f}, Overall Accuracy: {avg_val_all_accuracy:.4f}")
        
    model_path = save_path + model_name +  ".pth"
    torch.save(model.state_dict(), model_path)
    print(f"Teacher model saved to {save_path}")
        
    return {
        'train_loss': avg_train_loss,
        'train_topk_accuracy': avg_train_topk_accuracy,
        'train_all_accuracy': avg_train_all_accuracy,
        'val_loss': avg_val_loss,
        'val_topk_accuracy': avg_val_topk_accuracy,
        'val_all_accuracy': avg_val_all_accuracy
    }

def test(model, test_dataloader, device):
    # Validation Phase
    model.eval()
    total_test_loss = 0.0
    correct_test_top5 = 0
    total_test_topk = 0
    correct_test_all = 0
    total_test = 0

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Validation"):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch, output_hidden_states=True, output_attentions=True)
            loss = outputs.loss
            total_test_loss += loss.item()

            # Predictions and true labels
            logits = outputs.logits
            labels = batch["labels"]
            labels = (labels > 0).int()
            predicted_classes = torch.sigmoid(logits)

            # Top-5 accuracy
            correct_top5 = topk_accuracy(predicted_classes, labels,k = 5)
            correct_test_top5 += correct_top5
            # Overall accuracy (Top-1)
            correct_top1 = topk_accuracy(predicted_classes, labels, k = 1)
            correct_test_all += correct_top1
            total_test += labels.sum().item()

    avg_test_loss = total_test_loss / len(test_dataloader)
    avg_test_topk_accuracy = correct_test_top5 / total_test
    avg_test_all_accuracy = correct_test_all / total_test

    print(f"Validation - Loss: {avg_test_loss:.4f}, Top-5 Accuracy: {avg_test_topk_accuracy:.4f}, Overall Accuracy: {avg_test_all_accuracy:.4f}")

    return {
    'val_loss': avg_test_loss,
    'val_topk_accuracy': avg_test_topk_accuracy,
    'val_all_accuracy': avg_test_all_accuracy
}



def KD_train_based_logits(teacher_model, student_model, data_loader, alpha, optimizer,epochs,save_path,device):
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
        total_train_loss =0
    
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

            distill_loss =  distillation_loss(student_outputs.logits, teacher_outputs.logits, temperature=3.0)
            loss = (1 - alpha) * student_outputs.loss + alpha * distill_loss
            total_train_loss += loss.item()
            
            # Predictions and true labels
            logits = student_outputs.logits
            labels = (labels > 0).int()
            predicted_classes = torch.sigmoid(logits)

            # Top-5 accuracy
            correct_top5 =  topk_accuracy(predicted_classes, labels, k=5)
            correct_train_top5 += correct_top5
            # Top-1 accuracy
            correct_top1 =  topk_accuracy(predicted_classes, labels, k=1)
            correct_train_all += correct_top1
            
            total_train += labels.sum().item()
    
            # Backpropagation
            loss.backward()
            optimizer.step()
    
        avg_train_loss = total_train_loss / len(data_loader)
        avg_train_topk_accuracy = correct_train_top5 / total_train
        avg_train_all_accuracy = correct_train_all / total_train

        print(f"Validation - Loss: {avg_train_loss:.4f}, Top-5 Accuracy: {avg_train_topk_accuracy:.4f}, Overall Accuracy: {avg_train_all_accuracy:.4f}")
        
    model_path = f"{save_path}student_logit.pth"
    torch.save(student_model.state_dict(), model_path)
    print(f"Logits states based Student model saved to {save_path}")
    return {
        'train_loss': avg_train_loss,
        'train_topk_accuracy': avg_train_topk_accuracy,
        'train_all_accuracy': avg_train_all_accuracy,
    }
        
def KD_train_based_hidden_states(teacher_model, student_model, data_loader, projector, alpha, optimizer, epochs,save_path,device):
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
        total_train_loss =0
    
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

            distill_loss =  hidden_state_distillation_loss(student_outputs.hidden_states[0].to('cpu'), 
                                                           teacher_outputs.hidden_states[0].to('cpu'), projector)
            loss = (1 - alpha) * student_outputs.loss + alpha * distill_loss
            total_train_loss += loss.item()
            
            # Predictions and true labels
            logits = student_outputs.logits
            labels = (labels > 0).int()
            predicted_classes = torch.sigmoid(logits)

            # Top-5 accuracy
            correct_top5 =  topk_accuracy(predicted_classes, labels, k=5)
            correct_train_top5 += correct_top5
            # Top-1 accuracy
            correct_top1 =  topk_accuracy(predicted_classes, labels, k=1)
            correct_train_all += correct_top1
            
            total_train += labels.sum().item()
    
            # Backpropagation
            loss.backward()
            optimizer.step()
    
        avg_train_loss = total_train_loss / len(data_loader)
        avg_train_topk_accuracy = correct_train_top5 / total_train
        avg_train_all_accuracy = correct_train_all / total_train

        print(f"Validation - Loss: {avg_train_loss:.4f}, Top-5 Accuracy: {avg_train_topk_accuracy:.4f}, Overall Accuracy: {avg_train_all_accuracy:.4f}")
        # Save the student model
    model_path = f"{save_path}student_hidden.pth"
    torch.save(student_model.state_dict(), model_path)
    print(f"Hidden states based Student model saved to {save_path}")

    return {
        'train_loss': avg_train_loss,
        'train_topk_accuracy': avg_train_topk_accuracy,
        'train_all_accuracy': avg_train_all_accuracy,
    }
        
def KD_train_based_attentions(teacher_model, student_model, data_loader, projector, alpha, mapping_strategy, optimizer, epochs, save_path, device):
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
        total_train_loss =0
    
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

            distill_loss =  attention_distillation_loss(student_outputs.attentions, teacher_outputs.attentions, mapping_strategy)
            loss = (1 - alpha) * student_outputs.loss + alpha * distill_loss
            total_train_loss += loss.item()
            
            # Predictions and true labels
            logits = student_outputs.logits
            labels = (labels > 0).int()
            predicted_classes = torch.sigmoid(logits)

            # Top-5 accuracy
            correct_top5 =  topk_accuracy(predicted_classes, labels, k=5)
            correct_train_top5 += correct_top5
            # Top-1 accuracy
            correct_top1 =  topk_accuracy(predicted_classes, labels, k=1)
            correct_train_all += correct_top1
            
            total_train += labels.sum().item()
    
            # Backpropagation
            loss.backward()
            optimizer.step()
    
        avg_train_loss = total_train_loss / len(data_loader)
        avg_train_topk_accuracy = correct_train_top5 / total_train
        avg_train_all_accuracy = correct_train_all / total_train

        print(f"Validation - Loss: {avg_train_loss:.4f}, Top-5 Accuracy: {avg_train_topk_accuracy:.4f}, Overall Accuracy: {avg_train_all_accuracy:.4f}")
        # Save the student model
    model_path = f"{save_path}student_attention.pth"
    torch.save(student_model.state_dict(), model_path)
    print(f"Attention based Student model saved to {save_path}")

    return {
        'train_loss': avg_train_loss,
        'train_topk_accuracy': avg_train_topk_accuracy,
        'train_all_accuracy': avg_train_all_accuracy,
    }
        
def KD_train_based_hybrid(teacher_model, student_model, data_loader, projector, alpha, mapping_strategy, optimizer, epochs, save_path, device):
    teacher_model.eval()
    student_model.train()
    train_accuracy_list = []  # To store training accuracy per epoch

    
    for i in tqdm(range(epochs)):  # Loop over the dataset multiple times
        # tracker.epoch_start()
        print(f"Epoch: {i}")
        running_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        correct_train_top5 = 0
        correct_train_all = 0
        total_train = 0
        total_train_loss =0
    
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

            logits_loss =  distillation_loss(student_outputs.logits, teacher_outputs.logits, temperature=3.0)
            
            hidden_loss =  hidden_state_distillation_loss(student_outputs.hidden_states[0].to('cpu'), 
                                                           teacher_outputs.hidden_states[0].to('cpu'), projector)
            
            attention_loss =  attention_distillation_loss(student_outputs.attentions, 
                                                           teacher_outputs.attentions, mapping_strategy)
            
            loss = (1 - alpha) * student_outputs.loss + alpha * (attention_loss + hidden_loss + logits_loss)
            
            total_train_loss += loss.item()
            
            # Predictions and true labels
            logits = student_outputs.logits
            labels = (labels > 0).int()
            predicted_classes = torch.sigmoid(logits)

            # Top-5 accuracy
            correct_top5 =  topk_accuracy(predicted_classes, labels, k=5)
            correct_train_top5 += correct_top5
            # Top-1 accuracy
            correct_top1 =  topk_accuracy(predicted_classes, labels, k=1)
            correct_train_all += correct_top1
            
            total_train += labels.sum().item()
    
            # Backpropagation
            loss.backward()
            optimizer.step()


        # tracker.epoch_end()
    
        avg_train_loss = total_train_loss / len(data_loader)
        avg_train_topk_accuracy = correct_train_top5 / total_train
        avg_train_all_accuracy = correct_train_all / total_train

        print(f"Validation - Loss: {avg_train_loss:.4f}, Top-5 Accuracy: {avg_train_topk_accuracy:.4f}, Overall Accuracy: {avg_train_all_accuracy:.4f}")
        # Save the student model
    model_path = f"{save_path}student_hybrid.pth"
    torch.save(student_model.state_dict(), model_path)
    print(f"Hybrid Student model saved to {save_path}")

    return {
        'train_loss': avg_train_loss,
        'train_topk_accuracy': avg_train_topk_accuracy,
        'train_all_accuracy': avg_train_all_accuracy,
    }
        