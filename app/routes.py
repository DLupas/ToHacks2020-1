from app import app, chatbot
from app.chatbot import returnPhrase #get the chat function
from flask import render_template, url_for, redirect, request, jsonify

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    return render_template('index.html')

@app.route('/bot', methods=['POST'])
def return_phrase():
    return jsonify({'phrase': returnPhrase(request.form['message'])})