from flask import Flask, render_template

app = Flask()

@app.route("/")
def launch():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug = True)