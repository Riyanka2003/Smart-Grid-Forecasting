import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --- CONFIGURATION ---
SEQ_LENGTH = 24  # Look back 24 hours to predict the next hour
EPOCHS = 20      # How many times to practice (Training loops)

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# 1. Load Data
print("Loading data...")
df = pd.read_csv('smart_grid_data.csv')
raw_data = df['Energy_Consumption_MW'].values.reshape(-1, 1)

# 2. Normalize (Scale data between 0 and 1)
# Neural Networks hate big numbers (like 80 MW). They like 0.0 to 1.0.
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(raw_data)

# 3. Create Sequences (The "Sliding Window")
X, y = create_sequences(scaled_data, SEQ_LENGTH)

# 4. Split into Train (80%) and Test (20%)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Training on {len(X_train)} hours of data...")
print(f"Testing on {len(X_test)} hours of data...")

# 5. Build the LSTM Brain
model = Sequential([
    LSTM(50, return_sequences=False, input_shape=(SEQ_LENGTH, 1)), # Memory Layer
    Dense(1) # Output Layer (Prediction)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# 6. Train the Model
print("Starting training (this might take a minute)...")
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=32, validation_split=0.1, verbose=1)

# 7. Test the Model
predictions = model.predict(X_test)

# 8. Un-Normalize (Convert back to Megawatts)
predictions_mw = scaler.inverse_transform(predictions)
actual_mw = scaler.inverse_transform(y_test)

# 9. Visualize Results
plt.figure(figsize=(12, 6))
plt.plot(actual_mw, label='Actual Grid Load', color='blue', alpha=0.6)
plt.plot(predictions_mw, label='AI Prediction', color='red', linestyle='dashed')
plt.title('Smart Grid Forecast: Actual vs Predicted')
plt.xlabel('Hours (Test Set)')
plt.ylabel('Energy (MW)')
plt.legend()
plt.show()

# 10. Save the model for later
model.save('grid_model.h5')
print("Model saved as grid_model.h5")