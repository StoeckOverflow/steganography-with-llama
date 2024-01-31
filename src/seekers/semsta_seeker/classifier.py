import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

'''DEPRECATED
class Classifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super(Classifier, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, fused_features):
        attention_weights = F.softmax(self.attention(fused_features), dim=1)
        attention_applied = fused_features * attention_weights.expand_as(fused_features)
        
        aggregated_features = attention_applied.sum(1)
        
        logits = self.classifier(aggregated_features)
        return logits
'''

class BahdanauAttention(nn.Module):
    def __init__(self, key_size, query_size, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.key_layer = nn.Linear(key_size, hidden_size)
        self.query_layer = nn.Linear(query_size, hidden_size)
        self.energy_layer = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        # query = [batch_size, query_size]
        # keys = [batch_size, seq_len, key_size]

        query = query.unsqueeze(1).repeat(1, keys.size(1), 1)  # [batch_size, seq_len, query_size]
        keys = self.key_layer(keys)  # [batch_size, seq_len, hidden_size]

        energy = torch.tanh(keys + query)  # [batch_size, seq_len, hidden_size]
        energy = self.energy_layer(energy).squeeze(2)  # [batch_size, seq_len]

        attention = F.softmax(energy, dim=1)  # [batch_size, seq_len]
        return attention

class Classifier(nn.Module):
    def __init__(self, hidden_size, dense_units, num_classes=2):
        super(Classifier, self).__init__()
        
        self.attention = BahdanauAttention(hidden_size, hidden_size, hidden_size)
        self.dense_layer = nn.Linear(hidden_size, dense_units)
        self.output_layer = nn.Linear(dense_units, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, fused_features, original_features):
        attention_weights = self.attention(fused_features, original_features)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), original_features).squeeze(1)

        dense_output = self.relu(self.dense_layer(context_vector))
        dense_output = self.dropout(dense_output)
        output = self.output_layer(dense_output)

        return output

    def train_classifier(self, original_encoder_features, fused_features, train_labels, num_epochs=100, learning_rate=0.001, batch_size=128):
        self.train()
        numeric_train_labels = [(int(float(label)) + 1) // 2 for label in train_labels]
        train_labels = torch.tensor(numeric_train_labels, dtype=torch.long)

        train_dataset = TensorDataset(original_encoder_features, fused_features, train_labels)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch_original_encoder_features, batch_fused_features, batch_labels in train_loader:

                outputs = self(batch_fused_features, batch_original_encoder_features)
                loss = criterion(outputs, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader)}')
            
        torch.save(self.state_dict(), 'resources/models/classifier.pth')
        
    def evaluate_classifier(self, original_encoder_features, fused_features, test_labels, batch_size=128):
        self.eval()
        
        numeric_test_labels = [(int(float(label)) + 1) // 2 for label in test_labels]
        test_labels = torch.tensor(numeric_test_labels, dtype=torch.long)
        
        test_dataset = TensorDataset(original_encoder_features, fused_features, test_labels)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_original_encoder_features, batch_fused_features, batch_labels in test_loader:
                
                outputs = self(batch_fused_features, batch_original_encoder_features)
                probabilities = torch.softmax(outputs, dim=1)[:, 1]
                predicted_labels = torch.argmax(probabilities, dim=1).tolist()

                all_predictions.extend(predicted_labels.cpu().numpy())
                all_targets.extend(batch_labels.cpu().numpy())

        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions)
        recall = recall_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        all_ones = all(prediction == 1 for prediction in all_predictions)
        if all_ones:
            print("All predictions are 1")
        else:
            print("There are some predictions that are not 1")