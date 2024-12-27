import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class HVAJourneyPredictor:
    def __init__(self, seq_length=10):
        self.seq_length = seq_length
        self.model = None
        self.label_encoder = LabelEncoder()

    def prepare_sequences(self, journey_data):
        """
        Prepare sequences for LSTM model.
        
        :param journey_data: DataFrame with columns ['customer_id', 'timestamp', 'hva']
        :return: X (input sequences), y (target HVAs)
        """
        encoded_hvas = self.label_encoder.fit_transform(journey_data['hva'])
        sequences = []
        targets = []

        for customer_id in journey_data['customer_id'].unique():
            customer_hvas = encoded_hvas[journey_data['customer_id'] == customer_id]
            for i in range(len(customer_hvas) - self.seq_length):
                sequences.append(customer_hvas[i:i+self.seq_length])
                targets.append(customer_hvas[i+self.seq_length])

        return np.array(sequences), np.array(targets)

    def fit(self, journey_data):
        """
        Fit the HVA journey predictor model.
        
        :param journey_data: DataFrame with columns ['customer_id', 'timestamp', 'hva']
        """
        X, y = self.prepare_sequences(journey_data)
        n_hvas = len(self.label_encoder.classes_)

        self.model = Sequential([
            LSTM(50, activation='relu', input_shape=(self.seq_length, 1)),
            Dense(n_hvas, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(X, y, epochs=10, batch_size=32)

    def predict_next_hva(self, hva_sequence):
        """
        Predict the next HVA given a sequence of HVAs.
        
        :param hva_sequence: List of HVAs
        :return: Predicted next HVA
        """
        encoded_sequence = self.label_encoder.transform(hva_sequence)
        X = np.array([encoded_sequence[-self.seq_length:]])
        predicted_encoded = self.model.predict(X).argmax()
        return self.label_encoder.inverse_transform([predicted_encoded])[0]
