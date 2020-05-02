from app import app
from flask import render_template, url_for, redirect, request

@app.route('/')
def hello_world():
    return render_template('base.html')
