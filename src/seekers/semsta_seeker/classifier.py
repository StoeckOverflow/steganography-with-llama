import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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

    def train_classifier(self, fused_features, train_labels, num_epochs=10, learning_rate=0.001, batch_size=32):
       
        train_labels = torch.tensor(train_labels, dtype=torch.float32)

        train_dataset = TensorDataset(fused_features, train_labels)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch_features, batch_labels in train_loader:

                outputs = self(batch_features)
                loss = criterion(outputs, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader)}')
            
        torch.save(self.state_dict(), 'resources/model/classifier.pth')
