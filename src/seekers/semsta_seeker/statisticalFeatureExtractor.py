import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
   
class StatisticalFeatureExtractor():
    def __init__(self, hidden_dim=128, disable_tqdm=False):
        self.input_dim = 1000
        self.hidden_dim = hidden_dim
        self.tfidf_vectorizer = TfidfVectorizer(max_features=self.input_dim)
        self.auto_encoder = None
        self.disable_tqdm = disable_tqdm  
    
    def train_autoencoder(self, train_newsfeeds, num_epochs=50, learning_rate=0.001, batch_size=128):
        flat_train_newsfeeds = [' '.join(sublist) for sublist in train_newsfeeds]
        self.tfidf_vectorizer.fit(flat_train_newsfeeds)

        self.auto_encoder = AutoEncoder(input_dim=self.input_dim, hidden_dim=self.hidden_dim)
        self.auto_encoder.train()
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.auto_encoder.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            total_loss = 0.0

            np.random.shuffle(train_newsfeeds)
            num_batches = len(train_newsfeeds) // batch_size
            tfidf_matrix = self.tfidf_vectorizer.transform(flat_train_newsfeeds).toarray()
            dataset = TensorDataset(torch.tensor(tfidf_matrix, dtype=torch.float32))
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            for inputs in data_loader:
                optimizer.zero_grad()
                encoded, decoded = self.auto_encoder(inputs[0])
                
                loss = criterion(decoded, inputs[0])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / num_batches
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

        with open('resources/models/tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)

        torch.save(self.auto_encoder.state_dict(), 'resources/models/autoencoder_model.pth')

    def get_statistical_features(self, newsfeed):
        if self.auto_encoder is None:
            try:
                self.auto_encoder = AutoEncoder(self.input_dim, self.hidden_dim)
                self.auto_encoder.load_state_dict(torch.load('resources/models/autoencoder_model.pth'))
                self.auto_encoder.eval()
                with open('resources/models/tfidf_vectorizer.pkl', 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
            except FileNotFoundError:
                print("Model file not found. Starting training process.")
                self.train_autoencoder(newsfeed_directory_path='resources/feeds/doctored_feeds_articles')
                self.auto_encoder.eval()

        with torch.no_grad():
            tfidf_vectors = self.tfidf_vectorizer.transform(newsfeed)
            tfidf_tensor = torch.tensor(tfidf_vectors.toarray(), dtype=torch.float32)
            latent_representation, _ = self.auto_encoder(tfidf_tensor)

        return latent_representation