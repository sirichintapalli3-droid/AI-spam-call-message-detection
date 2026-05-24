from flask import Flask, render_template, request, redirect, session, jsonify
import sqlite3
import pandas as pd
import pickle
import webbrowser

app = Flask(__name__)
app.secret_key = "secret123"

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# =========================
# DATABASE INIT
# =========================
def init_db():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fullname TEXT,
        username TEXT,
        email TEXT,
        phone TEXT,
        password TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message TEXT,
        result TEXT,
        probability REAL
    )
    """)

    conn.commit()
    conn.close()

init_db()

# =========================
# HOME PAGE
# =========================
@app.route("/")
def index():
    return render_template("index.html")

# =========================
# REGISTER
# =========================
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        conn = sqlite3.connect("database.db")
        c = conn.cursor()

        c.execute("""
            INSERT INTO users (fullname, username, email, phone, password)
            VALUES (?, ?, ?, ?, ?)
        """, (
            request.form["fullname"],
            request.form["username"],
            request.form["email"],
            request.form["phone"],
            request.form["password"]
        ))

        conn.commit()
        conn.close()

        return redirect("/login")

    return render_template("register.html")

# =========================
# LOGIN
# =========================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = request.form["username"]
        p = request.form["password"]

        conn = sqlite3.connect("database.db")
        c = conn.cursor()

        c.execute("SELECT * FROM users WHERE username=? AND password=?", (u, p))
        user = c.fetchone()

        conn.close()

        if user:
            session["user"] = u
            return redirect("/home")

        return "Invalid Login"

    return render_template("login.html")

# =========================
# HOME
# =========================
@app.route("/home")
def home():
    if "user" not in session:
        return redirect("/login")

    return render_template("home.html", user=session["user"])

# =========================
# CSV UPLOAD + PREDICTION
# =========================
@app.route("/upload_csv", methods=["POST"])
def upload_csv():

    file = request.files.get("csvfile")

    if not file:
        return "No file uploaded"

    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower()

    if "message" not in df.columns:
        return f"CSV must contain 'message' column. Found: {list(df.columns)}"

    texts = df["message"].astype(str).fillna("").tolist()

    vecs = vectorizer.transform(texts)
    preds = model.predict(vecs)
    probs = model.predict_proba(vecs)[:, 1] * 100

    spam = 0
    normal = 0

    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    for msg, p, pr in zip(texts, preds, probs):

        result = "SPAM" if p == 1 else "NORMAL"

        if p == 1:
            spam += 1
        else:
            normal += 1

        c.execute("""
            INSERT INTO history (message, result, probability)
            VALUES (?, ?, ?)
        """, (msg, result, float(pr)))

    conn.commit()
    conn.close()

    return render_template(
        "dashboard.html",
        spam=spam,
        normal=normal
    )
# =========================
# DASHBOARD (FIXED)
# =========================
@app.route("/dashboard")
def dashboard():

    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("SELECT * FROM history ORDER BY id DESC")
    data = c.fetchall()

    c.execute("SELECT COUNT(*) FROM history WHERE result='SPAM'")
    spam = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM history WHERE result='NORMAL'")
    normal = c.fetchone()[0]

    conn.close()

    return render_template(
        "dashboard.html",
        data=data,
        spam=spam,
        normal=normal
    )

# =========================
# HISTORY
# =========================
@app.route("/history")
def history():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("SELECT * FROM history ORDER BY id DESC")
    data = c.fetchall()

    conn.close()

    return render_template("history.html", data=data)

# =========================
# CHART API
# =========================
@app.route("/chart-data")
def chart_data():

    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("SELECT result, COUNT(*) FROM history GROUP BY result")
    rows = c.fetchall()

    conn.close()

    spam = 0
    normal = 0

    for r in rows:
        if r[0] == "SPAM":
            spam = r[1]
        elif r[0] == "NORMAL":
            normal = r[1]

    return jsonify({
        "spam": spam,
        "normal": normal
    })

# =========================
# LOGOUT
# =========================
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000/")
    app.run(debug=True)