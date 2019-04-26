import pandas

FILENAME = './hate-speech-and-offensive-language/data/labeled_data.csv'

data = pandas.read_csv(FILENAME)
# Preview the first 5 lines of the loaded data
# training, validation, test
TRAINING_FILENAME = "./training.csv"
VALIDATION_FILENAME = "./validation.csv"
TEST_FILENAME = "./test.csv"

TRAINING_OFFENSIVE_FILENAME = "./training_offensive.csv"
TRAINING_REGULAR_FILENAME = "./training_regular.csv"
TEST_OFFENSIVE_FILENAME = "./test_offensive.csv"
TEST_REGULAR_FILENAME = "./test_regular.csv"

dataframe_column = ["", "offensive", "tweet"]

row_count = len(data)
training_row_count = round(row_count * 0.9)
test_row_count = round(row_count * 0.1)
assert training_row_count + test_row_count == row_count

print(f"Total row count: {row_count}")
print(f"Training set: {training_row_count}")
print(f"Test set: {test_row_count}")

def find_max_word_count():
    max_count = 0
    max_tweet = ""
    for i, row in data.iterrows():
        tweet = row.get("tweet")
        curr_count = len(tweet.split(" "))
        if curr_count > max_count:
            max_count = curr_count
            max_tweet = tweet
    print(max_count)
    print(max_tweet)

def generate_training_data():
    # Generate training data
    rows = list(data.iterrows())
    rows = [_[1] for _ in rows]
    training_data_offensive = []
    training_data_regular = []
    offset = 0
    for index, row in enumerate(rows[offset:training_row_count]):
        # 0 = hate speech, 1 = offensive language, 2 = neither
        offensive = 1 if row.get("class") < 2 else 0
        if offensive:
            training_data_offensive.append([index, offensive, row.get("tweet")])
        else:
            training_data_regular.append([index, offensive, row.get("tweet")])

    print("training data")
    # training_df = pandas.DataFrame(data=training_data, columns=dataframe_column)
    # training_df.to_csv(TRAINING_FILENAME, encoding='utf-8')
    training_offensive = pandas.DataFrame(data=training_data_offensive, columns=dataframe_column)
    training_offensive.to_csv(TRAINING_OFFENSIVE_FILENAME, encoding='utf-8')
    training_regular = pandas.DataFrame(data=training_data_regular, columns=dataframe_column)
    training_regular.to_csv(TRAINING_REGULAR_FILENAME, encoding='utf-8')

    # Generate test data
    test_data_offensive = []
    test_data_regular = []
    offset += training_row_count
    for index, row in enumerate(rows[offset:offset + test_row_count]):
        offensive = 1 if row.get("class") < 2 else 0
        if offensive:
            test_data_offensive.append([index, offensive, row.get("tweet")])
        else:
            test_data_regular.append([index, offensive, row.get("tweet")])

    test_offensive = pandas.DataFrame(data=test_data_offensive, columns=dataframe_column)
    test_offensive.to_csv(TEST_OFFENSIVE_FILENAME, encoding='utf-8')
    test_regular = pandas.DataFrame(data=test_data_regular, columns=dataframe_column)
    test_regular.to_csv(TEST_REGULAR_FILENAME, encoding='utf-8')

generate_training_data()