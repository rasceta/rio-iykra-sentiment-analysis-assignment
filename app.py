import pickle

from os import environ
from os.path import join, dirname
from dotenv import load_dotenv

from flask import Flask, request, jsonify
from config import FEATURE_EXTRACTOR_FILEPATH, CLASSIFIER_FILEPATH, LABELS, DATA_FILEPATH
from preprocess import preprocess

app = Flask(__name__)

with open(FEATURE_EXTRACTOR_FILEPATH, 'rb') as infile:
    app.feature_extractor = pickle.load(infile)

with open(CLASSIFIER_FILEPATH, 'rb') as infile:
    app.classifier = pickle.load(infile)

def reply_success(data):
    response = jsonify({
        "data": data
    })

    response.headers['Access-Control-Allow-Origin'] = '*'

    return response

def reply_error(code, message):
    response = jsonify({
        "error": {
            "code": code,
            "message": message
        }
    })

    response.headers['Access-Control-Allow-Origin'] = '*'

    return response

@app.route("/")
def index():
    return "<h1>Sentiment Analysis API using Flask</h1>"

@app.route("/classify", methods=["GET", "POST"])
def classify():
    if request.method == "GET":
        text = request.args.get("text", None)
    elif request.method == "POST":
        json_req = request.get_json()
        text = json_req["text"]
    else:
        return reply_error(code=400, message="Supported method is 'GET' and 'POST'")

    if text:
        # IMPORTANT: Use [text] because sklearn vectorizer expects an iterable as the input
        # IMPORTANT: classifier.predict returns an array, so get the first element
        label = app.classifier.predict(app.feature_extractor.transform([text]))[0]

        return reply_success(data={
            'text': text,
            'sentiment': LABELS[label]
        })

    return reply_error(code=400, message="Text is not specified")

@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    if request.method == "GET":
        text = request.args.get("text", None)
    elif request.method == "POST":
        json_req = request.get_json()
        text = json_req["text"]
    else:
        return reply_error(code=400, message="Supported method is 'GET' and 'POST'")

    # Open positive.txt and preprocess it
    with open(DATA_FILEPATH + "/positive.txt", "r", encoding='utf-8') as infile:
        positive_tweets = infile.readlines()
        pre_positive_tweets = [preprocess(tweet,do_stem=False) for tweet in positive_tweets]

    # Do the same with negative tweets
    with open(DATA_FILEPATH + "/negative.txt", "r", encoding='utf-8') as infile:
        negative_tweets = infile.readlines()
        pre_negative_tweets = [preprocess(tweet,do_stem=False) for tweet in negative_tweets]

    if text:
        # IMPORTANT: Use [text] because sklearn vectorizer expects an iterable as the input
        # IMPORTANT: classifier.predict returns an array, so get the first element
        text_list = text
        # Changing text into a list
        if not isinstance(text, list):
            text_list = [text]
        text_labels = [app.classifier.predict(app.feature_extractor.transform([line]))[0] for line in text_list]
        text_labels = [LABELS[text_label] for text_label in text_labels]
        is_existed_list = [True if line.lower() in pre_positive_tweets + pre_negative_tweets else False for line in text_list]
        msg_list = ["We have it already!" if item else "Your feedback is well received!" for item in is_existed_list]

        for idx, val in enumerate(text_list):
            if is_existed_list[idx]:
                continue
            # Append new text to negative/positive.txt
            if text_labels[idx] == "negative":
                negative_tweets.append(val + "\n")
            elif text_labels[idx] == "positive":
                positive_tweets.append(val + "\n")

        # Write new negative/positive.txt lines
        if len(negative_tweets) > len(pre_negative_tweets):
            with open(DATA_FILEPATH + "/negative.txt", "w", encoding='utf-8') as outfile:
                outfile.write(''.join(negative_tweets))
        if len(positive_tweets) > len(pre_positive_tweets):
            with open(DATA_FILEPATH + "/positive.txt", "w", encoding='utf-8') as outfile:
                outfile.write(''.join(positive_tweets))

        # Retrain model when new 10 new data is added
        if len(positive_tweets + negative_tweets) % 10 == 0 and len(positive_tweets + negative_tweets) > len(pre_positive_tweets + pre_negative_tweets):
            return reply_success(data={'text': text_list,'sentiment': text_labels,'msg': msg_list,}), train.main()
        else:
            return reply_success(data={
                'text': text_list,
                'sentiment': text_labels,
                'msg': msg_list,
            })

    return reply_error(code=400, message="Text is not specified")

if __name__ == "__main__":
    dotenv_path = join(dirname(__file__), '.env')
    load_dotenv(dotenv_path)

    port = int(environ.get("PORT"))
    debug = environ.get("DEBUG")

    if debug == "True":
        app.run(threaded=True, port=port, debug=True)
    else:
        app.run(threaded=True, port=port, debug=False)
