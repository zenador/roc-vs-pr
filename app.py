from flask import Flask, render_template, request

from charts import show_charts

app = Flask(__name__)

@app.route("/")
def home():
	return render_template('form.html')

@app.route("/chart", methods=['POST'])
def chart():
	return show_charts(data=request.form)
