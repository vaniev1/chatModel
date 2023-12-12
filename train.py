import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from data import questions, answers 

# Инициализация токенизатора
tokenizer = Tokenizer()

# Проверка наличия файла с состоянием токенизатора
tokenizer_file = 'tokenizer.json'
if os.path.exists(tokenizer_file):
    with open(tokenizer_file, 'r') as f:
        tokenizer_config = f.read()
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_config)
else:
    # Если файл отсутствует, проводим инициализацию
    tokenizer.fit_on_texts(questions + answers)

# Преобразование текста в последовательности чисел
questions_sequences = tokenizer.texts_to_sequences(questions)
answers_sequences = tokenizer.texts_to_sequences(answers)

# Добавление заполнения для одинаковой длины последовательностей
max_len = max(max(len(seq) for seq in questions_sequences), max(len(seq) for seq in answers_sequences))
questions_padded = pad_sequences(questions_sequences, maxlen=max_len, padding='post')
answers_padded = pad_sequences(answers_sequences, maxlen=max_len, padding='post')

# Создание LSTM модели
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=50, input_length=max_len),
    tf.keras.layers.LSTM(50, return_sequences=True),
    tf.keras.layers.Dense(len(tokenizer.word_index) + 1, activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(questions_padded, answers_padded, epochs=50000)

# Сохранение модели в файл
model.save('chatbot_model.h5')

# Сохранение состояния токенизатора в файл
tokenizer_config = tokenizer.to_json()
with open('tokenizer.json', 'w') as f:
    f.write(tokenizer_config)