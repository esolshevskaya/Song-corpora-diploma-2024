import os
import pandas as pd
import pymorphy2
import json
import string
import openpyxl
import nltk
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import cmudict
from nltk.tokenize import word_tokenize
from transformers import pipeline

#nltk.download('cmudict')
#nltk.download('punkt')

# Инициализация пайплайна для анализа эмоциональной окраски
emotion_pipeline = pipeline("text-classification", model="Djacon/rubert-tiny2-russian-emotion-detection")

banned_words = {}

banned_words_files = ["banwords_ru.txt"]

#cmu_dict = cmudict.dict()

tag_names = {
    'nomn': 'Nominative',
    'gent': 'Genitive',
    'datv': 'Dative',
    'accs': 'Accusative',
    'ablt': 'Instrumental',
    'loct': 'Locative',
    'gen2': 'Second Genitive',
    'acc2': 'Second Accusative',
    'sing': 'Singular',
    'plur': 'Plural',
    None: 'Undefined',
    '1per': 'First Person',
    '2per': 'Second Person',
    '3per': 'Third Person',
    'pres': 'Present Tense',
    'past': 'Past Tense',
    'futr': 'Future Tense',
    'indc': 'Indicative Mood',
    'impr': 'Imperative Mood',
    'ADJF': 'Adjective (full form)',
    'ADJS': 'Adjective (short form)',
    'ADVB': 'Adverb',
    'COMP': 'Comparative',
    'CONJ': 'Conjunction',
    'GRND': 'Gerund',
    'INFN': 'Infinitive',
    'INTJ': 'Interjection',
    'NOUN': 'Noun',
    'NPRO': 'Noun Pronoun',
    'NUMR': 'Numeral',
    'PRED': 'Predicate',
    'PREP': 'Preposition',
    'PRTF': 'Participle (full form)',
    'PRTS': 'Participle (short form)',
    'PRCL': 'Particle',
    'VERB': 'Verb',
}

excluded_pos = {'CONJ'}   
excluded_pos_normal = {'PRCL', 'CONJ', 'INTJ', 'PRON', 'PREP', 'NPRO', 'PRTS'}


def is_valid_data(lines, data_dict):
    # Проверяем, соответствуют ли данные типу данных в словаре
    if len(lines) < len(data_dict):
        return False
    
    for i, line in enumerate(lines):
        if i >= len(data_dict):  # Прекращаем проверку, если закончились столбцы в словаре
            break
        data_type = data_dict[i]
        if data_type == 'int':
            try:
                int(line.strip())
            except ValueError:
                return False
        elif data_type == 'str':
            pass  # Не нужно проверять, строка всегда подходит
        elif data_type == 'float':
            try:
                float(line.strip())
            except ValueError:
                return False
        else:
            return False  # Неподдерживаемый тип данных
    return True

def collect_data(directory='./songs', banned_words_files=None, data_dict=None):
    if banned_words_files is None:
        banned_words_files = []
    if data_dict is None:
        data_dict = ['str', 'str', 'int', 'str']  # По умолчанию ожидаем строки, кроме года, который ожидаем как целое число

    banned_files_set = set(os.path.abspath(file) for file in banned_words_files)  # Преобразуем в множество для быстрого поиска

    data = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                
                if os.path.abspath(file_path) in banned_files_set:
                    continue

                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if not is_valid_data(lines, data_dict):
                        continue  # Пропускаем файлы, не соответствующие требованиям словаря
                    artist = lines[0].strip()
                    song = lines[1].strip()
                    year = lines[2].strip()
                    genre = lines[3].strip()
                    data.append({'artist': artist, 'song': song, 'year': int(year), 'genre': genre, 'path': file_path})
    df = pd.DataFrame(data)
    df['year'] = df['year'].astype(int)
    return df

def is_rhyme(word1, word2):
    return word1.endswith(word2[-2:])

def main_analysis(period_df, banned_words_files, genre_or_artist, period):
    banned_words = set()
    for filename in banned_words_files:
        with open(filename, 'r', encoding='utf-8') as f:
            banned_words.update(line.strip() for line in f)

    morph = pymorphy2.MorphAnalyzer()

    total_lines = 0
    total_words = 0
    total_unique_words = set()
    part_of_speech_counts = Counter()
    case_counts = Counter()
    number_counts = Counter()
    person_counts = Counter()
    tense_counts = Counter()
    voice_counts = Counter()
    mood_counts = Counter()
    ngram_counts = Counter()
    slang_count = 0
    total_repeated_lines = 0
    unique_rhymed_words = set()
    rhymed_lines_count = 0
    rhymes = []
    all_words = []
    unique_lines = set()
    genre_unique_words = defaultdict(set)
    song_sentiments = {}

    for _, row in period_df.iterrows():
        genre = row['genre']
        song_name = row['song'] 
        with open(row['path'], 'r', encoding='utf-8') as f:
            lines = f.readlines()[6:]
            lines = [line.strip() for line in lines if line.strip()]
            total_lines += len(lines)
            
            # Удаляем повторяющиеся строки, оставляя по одной
            unique_lines.update(lines)

            words = [word.strip(string.punctuation + "—") for line in lines for word in line.split()]
            total_words += len(words)
            total_unique_words.update(words)
            all_words.extend(words)

            line_counts = Counter(lines)
            total_repeated_lines += sum(count for count in line_counts.values() if count > 1)

            line_endings = [line.split()[-1] for line in lines if line.split()]
            for i in range(0, len(line_endings) - 1, 2):
                if is_rhyme(line_endings[i], line_endings[i+1]):
                    unique_rhymed_words.update([line_endings[i], line_endings[i+1]])
                    rhymed_lines_count += 2
                    rhymes.append((line_endings[i], line_endings[i+1]))

            for word in words:
                parsed_word = morph.parse(word)[0]
                part_of_speech_counts[parsed_word.tag.POS] += 1
                if parsed_word.tag.POS == 'NOUN':
                    case_counts[parsed_word.tag.case] += 1
                    number_counts[parsed_word.tag.number] += 1
                elif parsed_word.tag.POS == 'VERB':
                    person_counts[parsed_word.tag.person] += 1
                    number_counts[parsed_word.tag.number] += 1
                    tense_counts[parsed_word.tag.tense] += 1
                    voice_counts[parsed_word.tag.voice] += 1
                    mood_counts[parsed_word.tag.mood] += 1
                if word.lower() in banned_words:
                    slang_count += 1

            vectorizer = CountVectorizer(ngram_range=(2, 3), analyzer='word')
            ngrams = vectorizer.fit_transform([' '.join(words)])
            ngram_counts.update(dict(zip(vectorizer.get_feature_names_out(), ngrams.toarray().sum(axis=0))))

            # Собираем уникальные слова для каждого жанра
            lemmatized_words = [morph.parse(word)[0].normal_form for word in words]
            genre_unique_words[genre].update(lemmatized_words)

            # Анализ эмоциональной окраски песни
            song_text = ' '.join(lines)
            sentiment_scores = emotion_pipeline(song_text)
            song_sentiments[song_name] = sentiment_scores

    songs_with_years = [(row['song'], row['year'], song_sentiments[row['song']]) for _, row in period_df.iterrows() if row['song'] in song_sentiments]
    songs_with_years.sort(key=lambda x: x[1]) 

    total_songs = len(period_df)
    avg_lines_per_song = total_lines / total_songs if total_songs else 0
    avg_words_per_line = total_words / total_lines if total_lines else 0
    avg_word_length = sum(len(word) for word in total_unique_words) / len(total_unique_words) if total_unique_words else 0
    avg_parts_of_speech_per_song = {pos: count / total_songs for pos, count in part_of_speech_counts.items()}

    rhyme_percentage = (rhymed_lines_count / total_lines) * 100 if total_lines else 0
    unique_rhymed_words_count = len(unique_rhymed_words)

    # Подсчеты для частотных слов
    lemmatized_all_words = [morph.parse(word)[0].normal_form for word in all_words if morph.parse(word)[0].tag.POS not in excluded_pos and word not in string.punctuation and word not in banned_words]
    word_counts = Counter(lemmatized_all_words)
    top_words = word_counts.most_common(20)

    # Фильтрация для нормальных слов
    normal_words = [word for word in lemmatized_all_words if morph.parse(word)[0].tag.POS not in excluded_pos_normal and word not in string.punctuation and word not in banned_words]
    normal_word_counts = Counter(normal_words)
    top_normal_words = normal_word_counts.most_common(20)

    # Подсчеты для наименее частотных слов
    least_frequent_words = word_counts.most_common()[:-31:-1]

    # Индекс уникальности слов
    unique_word_indices = []
    for _, row in period_df.iterrows():
        song_name = row['song']
        with open(row['path'], 'r', encoding='utf-8') as f:
            lines = f.readlines()[6:]
            lines = [line.strip() for line in lines if line.strip()]
            words = [word for line in lines for word in line.split()]
            lemmatized = [morph.parse(word)[0].normal_form for word in words]
            unique_words = set(lemmatized)
            index = len(unique_words) / len(lemmatized) if lemmatized else 0
            unique_word_indices.append((song_name, index))

    unique_word_indices_with_years = [(row['song'], row['year'], index) for (_, row), (song_name, index) in zip(period_df.iterrows(), unique_word_indices)]
    unique_word_indices_with_years.sort(key=lambda x: x[1])

    print(f"\nStatistics for the given time period:")

    print(f"\nTotal number of lines in songs: {total_lines}")

    print(f"\nAverage number of lines per song: {avg_lines_per_song:.2f}")

    print(f"\nTotal number of repeated lines in songs: {total_repeated_lines}")

    print(f"\nAverage number of words per line: {avg_words_per_line:.2f}")

    print(f"\nAverage word length: {avg_word_length:.2f}")

    print("\nNumber of cases:")

    case_output = "\n".join([f"{tag_names.get(case, 'Undefined')}: {count:.2f}" for case, count in case_counts.items()])

    print(case_output)

    print("\nNumber of numbers:")

    number_output = "\n".join([f"{tag_names.get(number, 'Undefined')}: {count:.2f}" for number, count in number_counts.items()])

    print(number_output)

    print("\nNumber of persons:")

    person_output = "\n".join([f"{tag_names.get(person, 'Undefined')}: {count:.2f}" for person, count in person_counts.items()])

    print(person_output)

    print("\nNumber of tenses:")

    tense_output = "\n".join([f"{tag_names.get(tense, 'Undefined')}: {count:.2f}" for tense, count in tense_counts.items()])

    print(tense_output)

    print("\nNumber of voices:")

    voice_output = "\n".join([f"{tag_names.get(voice, 'Undefined')}: {count:.2f}" for voice, count in voice_counts.items()])

    print(voice_output)

    print("\nNumber of moods:")

    mood_output = "\n".join([f"{tag_names.get(mood, 'Undefined')}: {count:.2f}" for mood, count in mood_counts.items()])

    print(mood_output)

    print("\nAverage number of different parts of speech per song:")

    for pos, count in avg_parts_of_speech_per_song.items():
        tag_name = tag_names.get(pos, pos)
        print(f"{tag_name}: {count:.2f}")

    print(f"\nNumber of profanities and slang: {slang_count}")

    print("\nFrequency of bigrams:")

    top_bigrams = {ngram: count for ngram, count in ngram_counts.items() if len(ngram.split()) == 2}
    top_bigrams = Counter(top_bigrams).most_common(30)

    for bigram, count in top_bigrams:
        print(f"{bigram}: {count}")

    print("\nFrequency of trigrams:")

    top_trigrams = {ngram: count for ngram, count in ngram_counts.items() if len(ngram.split()) == 3}
    top_trigrams = Counter(top_trigrams).most_common(30)

    for trigram, count in top_trigrams:
        print(f"{trigram}: {count}")

    print(f"\nPercentage of rhymed lines: {rhyme_percentage:.2f}%")
    print(f"Number of unique rhymed words: {unique_rhymed_words_count}")

    print("\nList of found rhymes:")

    unique_rhymes = set()

    for rhyme_pair in rhymes:
        if rhyme_pair not in unique_rhymes:
            print(f"{rhyme_pair[0]} - {rhyme_pair[1]}")
            unique_rhymes.add(rhyme_pair)

    print("\nTop most frequent words (20):")

    for word, count in top_words:
        print(f"{word}: {count}")

    print("\nTop most frequent normalized words (20):")

    for word, count in top_normal_words:
        print(f"{word}: {count}")

    print("\nTop least frequent words (30):")

    for word, count in least_frequent_words:
        print(f"{word}: {count}")

    print("\nWord uniqueness index for each song:")

    for song_name, year, index in unique_word_indices_with_years:
        print(f"Song: {song_name}, Year: {year}, Uniqueness Index: {index:.2f}")

    print("\nUnique words by genres:")

    for genre, words in genre_unique_words.items():
        print(f"Genre: {genre}, Number of unique words: {len(words)}")

    print("\nEmotional tone of songs:")

    for song, year, scores in songs_with_years:
        print(f"\nSong: {song}")
        print(f"Year: {year}")
        for score in scores:
            print(f"{score['label']}: {score['score']:.4f}")

    df_summary = pd.DataFrame({
        'Metric': ['Total number of lines', 'Average number of lines per song', 'Total number of repeated lines', 'Average number of words per line', 'Average word length'],
        'Value': [total_lines, avg_lines_per_song, total_repeated_lines, avg_words_per_line, avg_word_length]
    })

    df_cases = pd.DataFrame(list(case_counts.items()), columns=['Case', 'Count'])
    df_numbers = pd.DataFrame(list(number_counts.items()), columns=['Number', 'Count'])
    df_persons = pd.DataFrame(list(person_counts.items()), columns=['Person', 'Count'])
    df_tenses = pd.DataFrame(list(tense_counts.items()), columns=['Tense', 'Count'])
    df_voices = pd.DataFrame(list(voice_counts.items()), columns=['Voice', 'Count'])
    df_moods = pd.DataFrame(list(mood_counts.items()), columns=['Mood', 'Count'])
    df_pos = pd.DataFrame(list(avg_parts_of_speech_per_song.items()), columns=['Part of Speech', 'Average Count'])
    df_slang = pd.DataFrame({'Metric': ['Number of profanities and slang'], 'Value': [slang_count]})

    # Создаем датафреймы для биграмм и триграмм
    df_bigrams = pd.DataFrame(Counter({ngram: count for ngram, count in ngram_counts.items() if len(ngram.split()) == 2}).most_common(30), columns=['Bigram', 'Count'])
    df_trigrams = pd.DataFrame(Counter({ngram: count for ngram, count in ngram_counts.items() if len(ngram.split()) == 3}).most_common(30), columns=['Trigram', 'Count'])

    # Создаем датафрейм для рифм
    df_rhymes = pd.DataFrame(list(set(rhymes)), columns=['Rhyme 1', 'Rhyme 2'])

    # Создаем датафреймы для частотности слов
    df_top_words = pd.DataFrame(top_words, columns=['Word', 'Count'])
    df_top_normal_words = pd.DataFrame(top_normal_words, columns=['Word', 'Count'])
    df_least_frequent_words = pd.DataFrame(least_frequent_words, columns=['Word', 'Count'])

    # Создаем датафрейм для уникальности слов
    df_uniqueness = pd.DataFrame(unique_word_indices_with_years, columns=['Song', 'Year', 'Uniqueness Index'])

    # Создаем датафрейм для уникальных слов по жанрам
    df_genre_unique_words = pd.DataFrame([(genre, len(words)) for genre, words in genre_unique_words.items()], columns=['Genre', 'Number of Unique Words'])

    # Обрабатываем данные о тональности песен
    emotional_data = []
    for song, year, scores in songs_with_years:
        for score in scores:
            emotional_data.append({'Song': song, 'Year': year, 'Label': score['label'], 'Score': score['score']})

    df_emotional_tone = pd.DataFrame(emotional_data)

    # Сохранение в файл Excel
    with pd.ExcelWriter('song_analysis_output.xlsx') as writer:
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
        df_cases.to_excel(writer, sheet_name='Cases', index=False)
        df_numbers.to_excel(writer, sheet_name='Numbers', index=False)
        df_persons.to_excel(writer, sheet_name='Persons', index=False)
        df_tenses.to_excel(writer, sheet_name='Tenses', index=False)
        df_voices.to_excel(writer, sheet_name='Voices', index=False)
        df_moods.to_excel(writer, sheet_name='Moods', index=False)
        df_pos.to_excel(writer, sheet_name='Parts of Speech', index=False)
        df_slang.to_excel(writer, sheet_name='Slang', index=False)
        df_bigrams.to_excel(writer, sheet_name='Bigrams', index=False)
        df_trigrams.to_excel(writer, sheet_name='Trigrams', index=False)
        df_rhymes.to_excel(writer, sheet_name='Rhymes', index=False)
        df_top_words.to_excel(writer, sheet_name='Top Words', index=False)
        df_top_normal_words.to_excel(writer, sheet_name='Top Normal Words', index=False)
        df_least_frequent_words.to_excel(writer, sheet_name='Least Frequent Words', index=False)
        df_uniqueness.to_excel(writer, sheet_name='Uniqueness', index=False)
        df_genre_unique_words.to_excel(writer, sheet_name='Genre Unique Words', index=False)
        df_emotional_tone.to_excel(writer, sheet_name='Emotional Tone', index=False)

def analysis_by_time(df):
    def is_valid_period(period):
        parts = period.split('-')
        return len(parts) == 2 and all(part.isdigit() for part in parts) and int(parts[0]) <= int(parts[1])

    while True:
        period = input("Enter the time period (e.g., 1990-2000): ")
        if not is_valid_period(period):
            print("Invalid format. Example input: 1990-2000")
            continue

        start_year, end_year = map(int, period.split('-'))
        period_df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
        if period_df.empty:
            print(f"No songs found for the time period {start_year}-{end_year}")
            continue

        main_analysis(period_df, banned_words_files, "time_period", f"{start_year}-{end_year}")
        break

def analysis_genre(df):
    available_genres = df['genre'].unique()
    print("Available genres:")
    for genre in available_genres:
        print(genre)
    
    chosen_genre = input("Enter the genre for analysis: ")
    if chosen_genre not in available_genres:
        print("Error: Chosen genre is not available.")
        return

    def is_valid_period(period):
        parts = period.split('-')
        return len(parts) == 2 and all(part.isdigit() for part in parts) and int(parts[0]) <= int(parts[1])

    while True:
        period = input("Enter the time period (e.g., 1990-2000): ")
        if not is_valid_period(period):
            print("Invalid format. Example input: 1990-2000")
            continue

        start_year, end_year = map(int, period.split('-'))
        period_df = df[(df['year'] >= start_year) & (df['year'] <= end_year) & (df['genre'] == chosen_genre)]
        if period_df.empty:
            print(f"No '{chosen_genre}' genre songs found for the time period {start_year}-{end_year}")
            continue

        main_analysis(period_df, banned_words_files, chosen_genre, f"{start_year}-{end_year}")
        break

def analysis_artist(df):
    available_artists = sorted(df['artist'].unique())
    print("Available artists:")
    for artist in available_artists:
        print(artist)

    chosen_artist = input("Enter the artist's name for analysis: ")
    if chosen_artist not in available_artists:
        print("Error: Chosen artist is not available.")
        return

    def is_valid_period(period):
        parts = period.split('-')
        return len(parts) == 2 and all(part.isdigit() for part in parts) and int(parts[0]) <= int(parts[1])

    while True:
        period = input("Enter the time period (e.g., 1990-2000): ")
        if not is_valid_period(period):
            print("Invalid format. Example input: 1990-2000")
            continue

        start_year, end_year = map(int, period.split('-'))
        period_df = df[(df['year'] >= start_year) & (df['year'] <= end_year) & (df['artist'] == chosen_artist)]
        if period_df.empty:
            print(f"No songs by '{chosen_artist}' found for the time period {start_year}-{end_year}")
            continue

        main_analysis(period_df, banned_words_files, chosen_artist, f"{start_year}-{end_year}")
        break


def general():
    df = collect_data(directory='.', banned_words_files=banned_words_files)
    while True:
        print("\nGeneral analysis menu:")
        print("1. Analysis by time period.")
        print("2. Analysis by genre.")
        print("3. Analysis by artist.")
        print("4. Back.")

        choice = input("Choose an option: ")

        if choice == '1':
            analysis_by_time(df)
        elif choice == '2':
            analysis_genre(df)
        elif choice == '3':
            analysis_artist(df)
        elif choice == '4':
            break
        else:
            print("Invalid choice. Please try again.")
