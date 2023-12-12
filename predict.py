import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Загрузка токенизатора из файла
tokenizer_file = 'tokenizer.json'
if os.path.exists(tokenizer_file):
    with open(tokenizer_file, 'r') as f:
        tokenizer_config = f.read()
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_config)
else:
    print("Токенизатор не найден. Запустите сначала train.py.")
    exit()

# Загрузка обученной модели
model = tf.keras.models.load_model('chatbot_model.h5')

# Инициализация токенизатора
tokenizer = Tokenizer()

# Максимальная длина последовательности
max_len = 4  # Используйте такое же значение, как в файле train.py

# Пример использования
while True:
    user_question = input("Цы фарстата дам ис? (науад 'выход' рахизынма): ")

    if user_question.lower() == 'выход':
        print("Дзабах у!")
        break

    # Обновление токенизатора с текущим вопросом
    tokenizer.fit_on_texts([user_question])

    question_sequence = tokenizer.texts_to_sequences([user_question])
    question_padded = pad_sequences(question_sequence, maxlen=max_len, padding='post')
    predicted_answer_sequence = model.predict(question_padded)

    # Извлекаем индексы слов с наибольшей вероятностью
    predicted_indices = tf.argmax(predicted_answer_sequence, axis=-1).numpy()[0]

    # Извлекаем слова из индексов
    predicted_words = [word for word, index in tokenizer.word_index.items() if index in predicted_indices]

    # Собираем предложение из слов
    predicted_answer = ' '.join(predicted_words)

    print(f"{predicted_answer}")