import torch
from torch import nn
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from tqdm import tqdm
import glob
import json

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
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class StatisticalFeatureExtractor():
    
    def __init__(self, disable_tqdm, hidden_dim=128):
        self.input_dim = 1000
        self.hidden_dim = hidden_dim
        self.tfidf_vectorizer = TfidfVectorizer(max_features=self.input_dim)
        self.auto_encoder = None
        self.disable_tqdm = disable_tqdm      
    
    def train_autoencoder(self, newsfeed_directory_path, num_epochs=10, learning_rate=0.001):
        train_newsfeeds = self.load_newsfeeds(newsfeed_directory_path)
        flat_train_newsfeeds = [item for sublist in train_newsfeeds for item in sublist]
        self.tfidf_vectorizer.fit(flat_train_newsfeeds)
        
        self.auto_encoder = AutoEncoder(input_dim=self.input_dim, hidden_dim=self.hidden_dim)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.auto_encoder.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 4
                
            for i in range(num_batches):    
                newsfeed = train_newsfeeds[i]
                    
                tfidf_vectors = self.tfidf_vectorizer.transform(newsfeed)
                tfidf_vectors_dense = tfidf_vectors.todense()
                tfidf_tensor = torch.tensor(tfidf_vectors_dense, dtype=torch.float32)
                    
                input_dim = tfidf_vectors.shape[1]
                    
                outputs = self.auto_encoder(tfidf_tensor)
                loss = criterion(outputs, tfidf_tensor)
                    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                    
                total_loss += loss.item()
            
            avg_loss = total_loss / num_batches
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
        
        torch.save(self.auto_encoder.state_dict(), 'resources/models/autoencoder_model.pth')

    def load_newsfeeds(self, newsfeeds_directory_path):
        newsfeeds_files_pattern = os.path.join(newsfeeds_directory_path,'*.json')
        feed_paths = glob.glob(newsfeeds_files_pattern)
        feed_paths = sorted(feed_paths)
        all_newsfeeds = []
        i = 0
        for feed_path in feed_paths:
            if (i <= 2) or (i >= 6 and i <= 8) or (i >= 12 and i <= 14): # manual train testsplit of different classes
                with open(feed_path, 'r') as file:
                    parsed_feed = json.load(file)
                feed_array = parsed_feed['feed']
                all_newsfeeds.append(feed_array)
            i+=1
        
        return all_newsfeeds

    def get_statistical_features(self, newsfeed):
        tfidf_vectors = self.tfidf_vectorizer.fit_transform(newsfeed)
        tfidf_vectors_dense = tfidf_vectors.todense()
        tfidf_tensor = torch.tensor(tfidf_vectors_dense, dtype=torch.float32)

        if self.auto_encoder is None:
            try:
                self.auto_encoder = AutoEncoder(self.input_dim, self.hidden_dim)
                self.auto_encoder.load_state_dict(torch.load('resources/models/autoencoder_model.pth'))
                self.auto_encoder.eval()
            except FileNotFoundError:
                print("Model file not found. Starting training process.")
                self.train_autoencoder(newsfeed_directory_path='resources/feeds/doctored_feeds_articles')
                self.auto_encoder.eval()

        with torch.no_grad():
            encoded_features = self.auto_encoder.encoder(tfidf_tensor)
        
        return encoded_features