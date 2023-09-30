from Model.model import create_dense_transformer_model
import tensorflow as tf
import dotenv
import os

from modules.Quantum_encoding import QuantumEncodedData
dotenv.load_dotenv() 
#---------------------------------------------------#
N_TRAIN =  os.getenv("N_TRAIN")
N_TEST = os.getenv("N_TEST")
DATASETLINK= os.getenv("DATASET_PATH")
#---------------------------------------------------#
x_train, y_train,x_test,y_test=DATASETLINK
pqk_model = create_dense_transformer_model()
pqk_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),optimizer=tf.keras.optimizers.Adam(learning_rate=0.003), metrics=['accuracy'])
x_train_pqk,x_test_pqk,y_train_new,y_test_new=QuantumEncodedData(x_train, y_train,x_test,y_test)

pqk_history = pqk_model.fit(tf.reshape(x_train_pqk, [N_TRAIN, -1]),
    y_train_new,
    batch_size=32,
    epochs=1000,
    verbose=0,
    validation_data=(tf.reshape(x_test_pqk, [N_TEST, -1]), y_test_new))