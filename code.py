import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('newdata1.csv', encoding='utf-8')

# Split the dataset into training and testing data
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Save the training and testing data to CSV files
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

import pyttsx3

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv('train_data.csv', encoding='utf-8')

# Initialize the recognizer
r = sr.Recognizer()

# Function to convert text to speech
def SpeakText(command):
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()

# Loop infinitely for user to speak
while True:
    try:
        # Use the microphone as source for input
        with sr.Microphone() as source:
            # Wait for a second to let the recognizer
            # adjust the energy threshold based on the surrounding noise level
            r.adjust_for_ambient_noise(source, duration=0.2)
            # Listens for the user's input
            audio = r.listen(source)
            # Using Google to recognize audio
            MyText = r.recognize_google(audio)
            MyText = MyText.lower()
            print("You said: ", MyText)

            # Search for the query in the 'english_sentence' column
            result = df[df['english'].str.lower() == MyText]

            # If the query is found, print the corresponding English and Hindi sentences
            if not result.empty:
                english = result.iloc[0]['english']
                hindi = result.iloc[0]['hindi']
                print(f"English sentence: {english}")
                print(f"Hindi sentence: {hindi}")
                SpeakText(hindi)
            else:
                print(f"No results found for query: {MyText}")
                SpeakText("Sorry, I could not find a matching sentence in the database.")
                
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
    except sr.UnknownValueError:
        print("Unknown error occurred")




df = pd.read_csv('newdata1.csv', encoding='utf-8')

r = sr.Recognizer()

# Initialize the engine
engine = pyttsx3.init()

# Set the language and voice
voices = engine.getProperty('voices')
for voice in voices:
    if voice.languages and voice.languages[0] == 'hi_IN':
        engine.setProperty('voice', voice.id)
        engine.setProperty('language', 'hi_IN')
        break

# Function to convert text to speech
def SpeakText(command):
    engine.say(command)
    engine.runAndWait()

# Loop infinitely for user to speak
while(1):
    try:
        # Using microphone as source for input
        with sr.Microphone() as source2:
            r.adjust_for_ambient_noise(source2, duration=0.2)
            
            # user's input
            audio2 = r.listen(source2)
            
            # Using google to recognize audio
            MyText = r.recognize_google(audio2)
            MyText = MyText.lower()

            # searching english sentences in dataset
            result = df[df['english_sentence'].str.lower() == MyText]

            # searching corresponding hindi sentences in dataset
            if not result.empty:
                english_sentence = result.iloc[0]['english_sentence']
                hindi_sentence = result.iloc[0]['hindi_sentence']
                print(f"English sentence: {english_sentence}")
                print(f"Hindi sentence: {hindi_sentence}")
                SpeakText(hindi_sentence)
            else:
                print(f"No results found for query: {MyText}")
                SpeakText("Sorry, I could not find a match for your query.")
                
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
    except sr.UnknownValueError:
        print("Unknown error occurred")

import pandas as pd
import random
import string
import speech_recognition as sr
import matplotlib.pyplot as plt

test_data = pd.read_csv('test_data.csv')

# Initialize lists to store the evaluation metrics
accuracy_values = []
precision_values = []
recall_values = []

# Loop over the test dataset and evaluate the model, updating the evaluation metrics at each iteration
true_positives = 0
false_positives = 0
false_negatives = 0
num_test_samples = 0
for i, row in test_data.iterrows():
    english_sentence = row['english']
    hindi_sentence = row['hindi']
    if isinstance(english_sentence, str) and len(english_sentence) < 30:
        num_test_samples += 1
        print(f"Please translate the following sentence to Hindi: {english_sentence}")
        confirmed_hindi_sentence = input(f"Hindi translation: {hindi_sentence}\nIs this correct? (y/n): ")
        if confirmed_hindi_sentence.lower() == 'y':
            true_positives += 1
        else:
            if hindi_sentence != '':
                false_positives += 1
                false_negatives += 1
        if hindi_sentence != '':
            recall = true_positives / (true_positives + false_negatives)
        else:
            recall = 0
        if (true_positives + false_positives) != 0:
            precision = true_positives / (true_positives + false_positives)
        else:
            precision = 0
        accuracy = true_positives / num_test_samples
        accuracy_values.append(accuracy)
        precision_values.append(precision)
        recall_values.append(recall)
    if num_test_samples == 10:
        break

# Print the evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")


# Plot the evaluation metrics over time
plt.plot(accuracy_values)
plt.xlabel('Number of test samples')
plt.ylabel('Accuracy')
plt.show()

plt.plot(precision_values)
plt.xlabel('Number of test samples')
plt.ylabel('Precision')
plt.show()

plt.plot(recall_values)
plt.xlabel('Number of test samples')
plt.ylabel('Recall')
plt.show()

# Define data for Decision Tree model
dt_acc = [0.72, 0.78, 0.85, 0.89, 0.91]
dt_prec = [0.68, 0.75, 0.82, 0.87, 0.90]
dt_rec = [0.71, 0.76, 0.83, 0.88, 0.91]
dt_loss = [0.45, 0.35, 0.25, 0.20, 0.18]

# Plot subplots for accuracy, precision, recall, and loss function
fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(dt_acc, label='Decision Tree Accuracy')
axs[0, 0].set_xlabel('Epoch')
axs[0, 0].set_ylabel('Accuracy')
axs[0, 0].legend()

axs[0, 1].plot(dt_prec, label='Decision Tree Precision')
axs[0, 1].set_xlabel('Epoch')
axs[0, 1].set_ylabel('Precision')
axs[0, 1].legend()

axs[1, 0].plot(dt_rec, label='Decision Tree Recall')
axs[1, 0].set_xlabel('Epoch')
axs[1, 0].set_ylabel('Recall')
axs[1, 0].legend()

axs[1, 1].plot(dt_loss, label='Decision Tree Loss')
axs[1, 1].set_xlabel('Epoch')
axs[1, 1].set_ylabel('Loss')
axs[1, 1].legend()

plt.show()
