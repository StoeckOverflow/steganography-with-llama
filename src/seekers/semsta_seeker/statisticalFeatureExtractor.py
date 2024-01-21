import torch
from torch import nn
from sklearn.feature_extraction.text import TfidfVectorizer

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class StatisticalFeatureExtractor(object):
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer()
        self.trained = False
        self.auto_encoder = None
        self.input_dim = None
        self.hidden_dim = 128 # chose suitable_value
    
    def train_autoencoder(self, newsfeed, save_flag=True):
        tfidf_vectors  = self.tfidf_vectorizer.fit_transform(newsfeed)
        tfidf_vectors_dense = tfidf_vectors.todense()
        tfidf_tensor = torch.tensor(tfidf_vectors_dense, dtype=torch.float32)
       
        self.input_dim = tfidf_vectors.shape[1]
        autoencoder = self.auto_encoder(self.input_dim, self.hidden_dim)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

        num_epochs = 10
        for epoch in range(num_epochs):
            # Forward pass
            outputs = autoencoder(tfidf_tensor)
            loss = criterion(outputs, tfidf_tensor)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if save_flag:
            torch.save(autoencoder.state_dict(), 'autoencoder_model.pth')
    
    def load_autoencoder(self):
        if self.auto_encoder is None:
            self.autoencoder = AutoEncoder(self.input_dim, self.hidden_dim)
        self.auto_encoder.load_state_dict(torch.load('autoencoder_model.pth'))
        self.auto_encoder.eval()
        
    def get_statistical_features(self, newsfeed):
        tfidf_vectors = self.tfidf_vectorizer.transform(newsfeed)
        tfidf_vectors_dense = tfidf_vectors.todense()
        tfidf_tensor = torch.tensor(tfidf_vectors_dense, dtype=torch.float32)
        self.input_dim = tfidf_vectors.shape[1]
        if self.autoencoder is None:
            raise RuntimeError('Autoencoder must be trained and loaded first')
        
        with torch.no_grad():
            encoded_features = self.autoencoder.encoder(tfidf_tensor)
        
        return encoded_features