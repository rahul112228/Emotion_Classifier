import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization, Dropout
from tensorflow.keras.models import Model
import dotenv
import os
dotenv.load_dotenv() 
#---------------------------------------------------#
qubits = os.getenv("DATASET_DIM")+1
#---------------------------------------------------#
def create_dense_transformer_model():
    inputs = Input(shape=(len(qubits) * 3,))
    
    # Multi-Head Self-Attention Layer
    for _ in range(20):
        attention = MultiHeadAttention(num_heads=2, key_dim=64)(inputs, inputs)
        attention = Dropout(0.1)(attention)
        attention = LayerNormalization(epsilon=1e-6)(inputs + attention)
    
    # Feed-Forward Neural Network
    ffnn = Dense(64, activation='relu')(attention)
    ffnn = Dense(32, activation='relu')(ffnn)
    
    outputs = Dense(1)(ffnn)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model
