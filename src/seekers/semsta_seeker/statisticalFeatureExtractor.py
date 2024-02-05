import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
import os

class AutoEncoder2(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(AutoEncoder2, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(True),
            nn.Linear(hidden_dims[0], output_dim),
            nn.ReLU(True)  # You can use sigmoid or another activation function depending on your choice
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dims[0]),
            nn.ReLU(True),
            nn.Linear(hidden_dims[0], input_dim),
            nn.Sigmoid()  # Assuming the input data is normalized between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(AutoEncoder, self).__init__()
        # Encoder
        encoder_layers = []
        for i, hidden_dim in enumerate(hidden_dims):
            encoder_layers.append(nn.Linear(input_dim if i == 0 else hidden_dims[i - 1], hidden_dim))
            encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Linear(hidden_dims[-1], output_dim))
        encoder_layers.append(nn.Sigmoid())  # Sigmoid activation for the last layer to match the paper
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        reversed_hidden_dims = list(reversed(hidden_dims))
        for i, hidden_dim in enumerate(reversed_hidden_dims):
            decoder_layers.append(nn.Linear(output_dim if i == 0 else reversed_hidden_dims[i - 1], hidden_dim))
            decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(reversed_hidden_dims[-1], input_dim))
        decoder_layers.append(nn.Sigmoid())  # Assuming input data is normalized
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
   
class StatisticalFeatureExtractor():
    def __init__(self, hidden_dim=1024, output_dim=1024, disable_tqdm=False):
        self.input_dim = 1000
        self.hidden_dim = hidden_dim
        self.tfidf_vectorizer = TfidfVectorizer(max_features=self.input_dim)
        self.auto_encoder = AutoEncoder(input_dim=self.input_dim, hidden_dims=[hidden_dim, int(hidden_dim/2)], output_dim=output_dim)
        self.disable_tqdm = disable_tqdm  
     
    def train_autoencoder(self, train_newsfeeds, num_epochs=50, learning_rate=0.001, batch_size=32, validation_split=0.3):
        flat_train_newsfeeds = [' '.join(sublist) for sublist in train_newsfeeds]
        self.tfidf_vectorizer.fit(flat_train_newsfeeds)
        tfidf_matrix = self.tfidf_vectorizer.transform(flat_train_newsfeeds).toarray()

        # Split data into training and validation sets
        split_index = int(np.floor((1 - validation_split) * len(tfidf_matrix)))
        train_data, val_data = tfidf_matrix[:split_index], tfidf_matrix[split_index:]

        train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(val_data, dtype=torch.float32))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        self.auto_encoder.train()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.auto_encoder.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            total_train_loss = 0.0
            for inputs in train_loader:
                inputs = inputs[0]
                optimizer.zero_grad()
                _, decoded = self.auto_encoder(inputs)
                loss = criterion(decoded, inputs)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            # Validate the model
            self.auto_encoder.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for inputs in val_loader:
                    inputs = inputs[0]
                    _, decoded = self.auto_encoder(inputs)
                    val_loss = criterion(decoded, inputs)
                    total_val_loss += val_loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

        model_dir = 'resources/models/'

        # Check if the directory exists, and if not, create it
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        tfidf_vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
        autoencoder_model_path = os.path.join(model_dir, 'autoencoder_model.pth')

        # Save the model and vectorizer
        with open(tfidf_vectorizer_path, 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)

        torch.save(self.auto_encoder.state_dict(), autoencoder_model_path)
    
    def load_resources(self):
        try:
            self.auto_encoder = AutoEncoder(self.input_dim, [self.hidden_dim, int(self.hidden_dim/2)], self.output_dim)
            self.auto_encoder.load_state_dict(torch.load('resources/models/autoencoder_model.pth', map_location=torch.device('cpu')))
            self.auto_encoder.eval()
            with open('resources/models/tfidf_vectorizer.pkl', 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError("Model or vectorizer file not found. Please train the AutoEncoder first.") from e

    def get_statistical_features(self, newsfeed):
        # Assuming that self.auto_encoder and self.tfidf_vectorizer have been loaded previously
        tfidf_vectors = self.tfidf_vectorizer.transform(newsfeed)  # Assuming newsfeed is a single string of text
        tfidf_tensor = torch.tensor(tfidf_vectors.toarray(), dtype=torch.float32)
        with torch.no_grad():
            latent_representation, _ = self.auto_encoder(tfidf_tensor)
        return latent_representation # Remove batch dimension if batch size is 1