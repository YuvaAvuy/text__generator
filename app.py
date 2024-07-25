import streamlit as st
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# Load the Tiny Shakespeare dataset
dataset, info = tfds.load('tiny_shakespeare', with_info=True, as_supervised=False)
text = next(iter(dataset['train']))['text'].numpy().decode('utf-8')

# Create a mapping from unique characters to indices
vocab = sorted(set(text))
char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = np.array(vocab)

# Numerically represent the characters
text_as_int = np.array([char2idx[c] for c in text])

# Define the model architecture
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

# Set the model parameters
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

# Build the model
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint('./training_checkpoints'))  # Update with the path to your checkpoint
model.build(tf.TensorShape([1, None]))

# Define the text generation function
def generate_text(model, start_string):
    num_generate = 1000
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    model.reset_states()
    for _ in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])
    return start_string + ''.join(text_generated)

# Streamlit app
st.title("Text Generation with Tiny Shakespeare Dataset")
start_string = st.text_input("Enter a seed string:", "QUEEN: So, let's end this")
if st.button("Generate Text"):
    generated_text = generate_text(model, start_string)
    st.write(generated_text)
