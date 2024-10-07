# MG-Chatbot

This repository contains a simple retrieval-based chatbot implemented in Python using the NLTK library and Scikit-learn. The chatbot reads a text file document to create a bag of words and responds to user queries based on the cosine similarity between the input and the preprocessed document content.

## Features

- **Pre-processing**:
  - Converts document content to lowercase.
  - Tokenizes content at both word and sentence levels.
  - Removes punctuation and stopwords.
  - Lemmatizes the corpus to avoid introducing non-words.
  
- **Vectorization**:
  - Uses Term Frequency-Inverse Document Frequency (TF-IDF) to rescale the frequency of words.
  - TF-IDF helps to counteract the dominance of highly frequent words that may not contain much informational content.

- **Vector Comparison**:
  - Uses cosine similarity to measure the similarity between the user's input and the document content.

## How to Use

1. **Installation**:
   Ensure you have the required libraries installed:
   ```bash
   pip install nltk scikit-learn
   ```

2. **Set Up Environment**:
   Change the working directory to the location of the dataset:
   ```python
   import os
   os.chdir('C:\\Users\\MichaelGore\\Documents\\')
   ```

3. **Download NLTK Data**:
   Ensure NLTK popular datasets are downloaded:
   ```python
   import nltk
   nltk.download('popular', quiet=True)
   ```

4. **Run the Script**:
   Execute the script:
   ```bash
   python chatbot.py
   ```

5. **Interact with the Chatbot**:
   - The chatbot will greet you and prompt you to enter your queries.
   - Type your queries and get responses based on the preprocessed document content.
   - To exit, type "Bye!".

## Example

When you run the script, you will see:
```
ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!
```

You can then interact with the chatbot by typing your queries and receiving appropriate responses.

## Recent Changes

The recent changes for [chatbot.py](https://github.com/MoneKEE/Python-Code-Samples/blob/master/chatbot.py) are:
- [40e38ba](https://github.com/MoneKEE/Python-Code-Samples/commit/40e38bab452cba935872c8eb3d039cb7aaa41af5): "chatbot.py updated chatbot.py to work with my apex laptop filesystem."
- [87f923b](https://github.com/MoneKEE/Python-Code-Samples/commit/87f923bd2cdfce43faaed2aa403bbe14db17b4fb): "fix greeting sentence."
- [f845505](https://github.com/MoneKEE/Python-Code-Samples/commit/f8455054e2ced6d984d23668cd59799039b451f0): "Add files via upload"

In summary, the script was initially uploaded, then updated to fix the greeting sentence, and later adjusted to work with a specific file system.
