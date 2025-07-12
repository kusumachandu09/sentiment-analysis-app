from textblob import TextBlob

def check_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "Positive 😀"
    elif polarity < 0:
        return "Negative 😞"
    else:
        return "Neutral 😐"

if __name__ == "__main__":
    user_text = input("Enter your text to analyze: ")
    result = check_sentiment(user_text)
    print("Sentiment:", result)
