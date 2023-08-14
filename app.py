from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def Home():
    return render_template('base.html')


@app.route("/about owner")
def about():
    return "<p>about page</p>"

@app.route("/resource")
def resource():
    return "<p>resource page</p>"


if __name__ == '__main__':
    app.run(debug=True)