import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import sys

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Load the trained model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = joblib.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = joblib.load(vectorizer_file)

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

class SpamClassifierApp(App):
    def build(self):
        self.title = 'Email Spam Classifier'
        
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Input box
        self.input_box = TextInput(hint_text='Enter email text here', multiline=True, size_hint=(1, 3.2))
        layout.add_widget(self.input_box)
        
        # Verify button
        verify_button = Button(text='Verify')
        verify_button.bind(on_press=self.verify_email)
        layout.add_widget(verify_button)
        
        # Result label
        self.result_label = Label(text='')
        layout.add_widget(self.result_label)
        
        # Exit button
        exit_button = Button(text='Exit')
        exit_button.bind(on_press=self.exit_app)
        layout.add_widget(exit_button)
        
        return layout

    def verify_email(self, instance):
        email_text = self.input_box.text
        if email_text.strip():
            # Preprocess the input text
            transformed_text = transform_text(email_text)
            # Transform the input text using the loaded vectorizer
            input_vector = vectorizer.transform([transformed_text]).toarray()
            # Predict using the loaded model
            prediction = model.predict(input_vector)
            result = 'Spam' if prediction[0] == 1 else 'Not Spam'
        else:
            result = 'Please enter email text!'
        
        self.result_label.text = result

    def exit_app(self, instance):
        App.get_running_app().stop()
        sys.exit()

if __name__ == '__main__':
    SpamClassifierApp().run()
