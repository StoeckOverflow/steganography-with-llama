import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import LeaveOneOut
import copy

class BahdanauAttention(nn.Module):
    def __init__(self, n_hidden_enc, n_hidden_dec):
        
        super(BahdanauAttention, self).__init__()
        
        self.h_hidden_enc = n_hidden_enc
        self.h_hidden_dec = n_hidden_dec
        
        self.W = nn.Linear(n_hidden_enc + n_hidden_dec, n_hidden_dec, bias=False) 
        self.V = nn.Parameter(torch.rand(n_hidden_dec))

    def forward(self, hidden_dec, last_layer_enc):
        ''' 
            PARAMS:           
                hidden_dec:     [b, n_layers, n_hidden_dec]    (1st hidden_dec = encoder's last_h's last layer)                 
                last_layer_enc: [b, seq_len, n_hidden_enc * 2] 
            
            RETURN:
                att_weights:    [b, src_seq_len] 
        '''
        batch_size = last_layer_enc.size(0)
        src_seq_len = last_layer_enc.size(1)

        hidden_dec = hidden_dec[:, -1, :].unsqueeze(1).repeat(1, src_seq_len, 1)         #[b, src_seq_len, n_hidden_dec]

        tanh_W_s_h = torch.tanh(self.W(torch.cat((hidden_dec, last_layer_enc), dim=2)))  #[b, src_seq_len, n_hidden_dec]
        tanh_W_s_h = tanh_W_s_h.permute(0, 2, 1)       #[b, n_hidde_dec, seq_len]
        
        V = self.V.repeat(batch_size, 1).unsqueeze(1)  #[b, 1, n_hidden_dec]
        e = torch.bmm(V, tanh_W_s_h).squeeze(1)        #[b, seq_len]
        
        att_weights = F.softmax(e, dim=1)              #[b, src_seq_len]
        
        return att_weights

class Classifier(nn.Module):
    def __init__(self, statistical_dim, semantic_dim, num_classes=2):
        super(Classifier, self).__init__()
        self.attention = BahdanauAttention(statistical_dim, semantic_dim)
        self.dense_layer = nn.Linear(statistical_dim, semantic_dim)
        self.classification_layer = nn.Linear(semantic_dim, num_classes)

    def forward(self, fused_features, original_features):
        attention_weights = self.attention(fused_features, original_features)
        attention_weights = attention_weights.unsqueeze(1)
        fused_vector = torch.bmm(attention_weights, original_features).squeeze(1)
        aggregate_layer =  self.dense_layer(fused_vector)
        output_vector = self.classification_layer(aggregate_layer)
        return output_vector

class Classifier_Trainer():
    def __init__(self, classifier: Classifier):
        self.classifier = classifier

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
    def train_classifier_with_cross_validation(self, original_encoder_features, fused_features, train_labels, num_epochs=100, learning_rate=0.0001, batch_size=64, k_folds=10):        
        numeric_train_labels = [1 if int(label) == -1 else 0 for label in train_labels]
        train_labels = torch.tensor(numeric_train_labels, dtype=torch.long)
        y_numpy = train_labels.numpy()
        
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        
        best_model = None
        best_f1_score = -float('inf')

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_numpy)), y_numpy)):
            print(f"Training on fold {fold+1}/{k_folds}...")
            
            best_fold_f1_score = -float('inf')
            patience = 10
            patience_counter = 0
            
            self.classifier.apply(self.init_weights)
            
            train_idx_tensor = torch.tensor(train_idx, dtype=torch.long)
            val_idx_tensor = torch.tensor(val_idx, dtype=torch.long)

            train_dataset = TensorDataset(original_encoder_features[train_idx_tensor], fused_features[train_idx_tensor], train_labels[train_idx_tensor])
            val_dataset = TensorDataset(original_encoder_features[val_idx_tensor], fused_features[val_idx_tensor], train_labels[val_idx_tensor])

            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.classifier.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in self.classifier.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            
            optimizer = torch.optim.Adam(optimizer_grouped_parameters , lr=learning_rate)
            criterion = nn.CrossEntropyLoss()
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
            
            for epoch in range(num_epochs):
                self.classifier.train()
                total_loss = 0.0
                all_train_predictions = []
                all_train_targets = []

                # Training Loop
                for batch_original_encoder_features, batch_fused_features, batch_labels in train_loader:
                    outputs = self.classifier(batch_fused_features, batch_original_encoder_features)
                    loss = criterion(outputs, batch_labels)
                    
                    probability_vector = torch.softmax(outputs, dim=1)
                    train_predicted_labels = torch.argmax(probability_vector, dim=1)
                    all_train_predictions.extend(train_predicted_labels.detach().numpy())
                    all_train_targets.extend(batch_labels.cpu().numpy())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    
                f1_train = f1_score(all_train_targets, all_train_predictions, average='binary')
                acc_train = accuracy_score(all_train_targets, all_train_predictions)
                rec_train = recall_score(all_train_targets, all_train_predictions, average='binary')
                pre_train = precision_score(all_train_targets, all_train_predictions, average='binary', zero_division=0)

                print(f'Fold {fold+1}, Epoch [{epoch+1}/{num_epochs}], Training Loss: {total_loss/len(train_loader)}, F1 Score: {f1_train:.4f}, Precision: {pre_train:.4f}, Recall: {rec_train:.4f}, Accuracy: {acc_train:.4f}')

                # Validation Loop
                self.classifier.eval()
                val_loss = 0.0
                all_predictions = []
                all_targets = []

                with torch.no_grad():
                    for batch_original_encoder_features, batch_fused_features, batch_labels in val_loader:
                        outputs = self.classifier(batch_fused_features, batch_original_encoder_features)
                        loss = criterion(outputs, batch_labels)
                        val_loss += loss.item()

                        probabilities_vector = torch.softmax(outputs, dim=1)
                        predicted_labels = torch.argmax(probabilities_vector, dim=1)
                        all_predictions.extend(predicted_labels.cpu().numpy())
                        all_targets.extend(batch_labels.cpu().numpy())
                
                avg_val_loss = val_loss / len(val_loader)
                scheduler.step(avg_val_loss)

                f1 = f1_score(all_targets, all_predictions, average='binary')
                acc = accuracy_score(all_targets, all_predictions)
                rec = recall_score(all_targets, all_predictions, average='binary')
                pre = precision_score(all_targets, all_predictions, average='binary', zero_division=0)
                
                metrics['accuracy'].append(acc)
                metrics['precision'].append(pre)
                metrics['recall'].append(rec)
                metrics['f1'].append(f1)
                
                print(f'Fold {fold+1}, Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss/len(val_loader)}, F1 Score: {f1:.4f}, Precision: {pre:.4f}, Recall: {rec:.4f}, Accuracy: {acc:.4f}')
                
                if f1 > best_fold_f1_score:
                    best_fold_f1_score = f1
                    best_fold_model = copy.deepcopy(self.classifier.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Stopping early at epoch {epoch+1} due to no improvement.")
                    break
            
            if best_fold_f1_score > best_f1_score:
                best_f1_score = best_fold_f1_score
                best_model = best_fold_model

            for key in metrics:
                metrics[key] = np.mean(metrics[key])
                avg_metric = np.mean(metrics[key])
                print(f"Fold: {fold+1}, Average {key}: {avg_metric:.4f}")
        
        if best_model is not None:
            self.classifier.load_state_dict(best_model)
            
        return metrics, best_f1_score, best_model

    def train_classifier_with_loocv(self, original_encoder_features, fused_features, train_labels, num_epochs=100, learning_rate=0.0001):
        numeric_train_labels = [1 if int(label) == -1 else 0 for label in train_labels]
        train_labels_tensor = torch.tensor(numeric_train_labels, dtype=torch.long)
        loo = LeaveOneOut()
        dataset = TensorDataset(original_encoder_features, fused_features, train_labels_tensor)
        metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

        for train_index, test_index in loo.split(original_encoder_features.numpy()):
            print(f"Training on sample {test_index[0]+1} as test set...")

            train_subset = Subset(dataset, train_index)
            test_subset = Subset(dataset, test_index)

            train_loader = DataLoader(train_subset, batch_size=len(train_subset))
            test_loader = DataLoader(test_subset, batch_size=1)

            self.classifier.apply(self.init_weights)
            optimizer = torch.optim.Adam(self.classifier.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()

            # Training Loop
            self.classifier.train()
            for epoch in range(num_epochs):
                for batch in train_loader:
                    inputs, fused, labels = batch
                    optimizer.zero_grad()
                    outputs = self.classifier(fused, inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

            # Evaluation phase
            self.classifier.eval()
            with torch.no_grad():
                for inputs, fused, labels in test_loader:
                    outputs = self.classifier(fused, inputs)
                    predicted = outputs.argmax(1)
                    metrics['accuracy'].append(accuracy_score(labels.numpy(), predicted.numpy()))
                    metrics['precision'].append(precision_score(labels.numpy(), predicted.numpy(), zero_division=0))
                    metrics['recall'].append(recall_score(labels.numpy(), predicted.numpy()))
                    metrics['f1'].append(f1_score(labels.numpy(), predicted.numpy()))

        for key in metrics:
            metrics[key] = np.mean(metrics[key])
            print(f"Average {key}: {metrics[key]:.4f}")
        
        return metrics
    
    def evaluate_classifier(self, original_encoder_features, fused_features, test_labels, batch_size=64):
        self.classifier.eval()
        
        numeric_test_labels = [1 if int(label) == -1 else 0 for label in test_labels]
        test_labels = torch.tensor(numeric_test_labels, dtype=torch.long)
        test_dataset = TensorDataset(original_encoder_features, fused_features, test_labels)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_original_encoder_features, batch_fused_features, batch_labels in test_loader:
                outputs = self.classifier(batch_fused_features, batch_original_encoder_features)
                outputs = torch.squeeze(outputs, dim=1)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_labels = torch.argmax(probabilities, dim=1)
                all_predictions.extend(predicted_labels.cpu().numpy())
                all_targets.extend(batch_labels.cpu().numpy())

        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='binary', zero_division=0)
        recall = recall_score(all_targets, all_predictions, average='binary')
        f1 = f1_score(all_targets, all_predictions, average='binary')
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        all_ones = all(prediction == 1 for prediction in all_predictions)
        if all_ones:
            print("All predictions are 1")
        else:
            print("There are some predictions that are not 1")