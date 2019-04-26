import pandas

FILENAME = './hate-speech-and-offensive-language/data/labeled_data.csv'

data = pandas.read_csv(FILENAME)
# Preview the first 5 lines of the loaded data
# training, validation, test
TRAINING_FILENAME = "./training.csv"
VALIDATION_FILENAME = "./validation.csv"
TEST_FILENAME = "./test.csv"

dataframe_column = ["", "offensive", "tweet"]

row_count = len(data)
training_row_count = round(row_count * 0.8)
validation_row_count = round(row_count * 0.1)
test_row_count = row_count - (training_row_count + validation_row_count)
assert training_row_count + validation_row_count + test_row_count == row_count

print(f"Total row count: {row_count}")
print(f"Training set: {training_row_count}")
print(f"Validation set: {validation_row_count}")
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
    training_data = []
    offset = 0
    for index, row in enumerate(rows[offset:training_row_count]):
        # 0 = hate speech, 1 = offensive language, 2 = neither
        offensive = 1 if row.get("class") < 2 else 0
        training_data.append([index, offensive, row.get("tweet")])

    print("training data")
    print(f"From {training_data[0]} to {training_data[-1]}")
    training_df = pandas.DataFrame(data=training_data, columns=dataframe_column)
    training_df.to_csv(TRAINING_FILENAME, encoding='utf-8')

    # Generate validation data
    validation_data = []
    offset += training_row_count
    for index, row in enumerate(rows[offset:offset + validation_row_count]):
        offensive = 1 if row.get("class") < 2 else 0
        validation_data.append([index, offensive, row.get("tweet")])

    print("validation data")
    print(f"From {validation_data[0]} to {validation_data[-1]}")

    validation_df = pandas.DataFrame(data=validation_data, columns=dataframe_column)
    validation_df.to_csv(VALIDATION_FILENAME, encoding="utf-8")

    # Generate test data
    test_data = []
    offset += validation_row_count
    for index, row in enumerate(rows[offset:offset + test_row_count]):
        offensive = 1 if row.get("class") < 2 else 0
        test_data.append([index, offensive, row.get("tweet")])

    print("test data")
    print(f"From {test_data[0]} to {test_data[-1]}")

    test_df = pandas.DataFrame(data=test_data, columns=dataframe_column)
    test_df.to_csv(TEST_FILENAME, encoding="utf-8")

find_max_word_count()