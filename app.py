from flask import Flask, render_template, request, jsonify
import mysql.connector
import pandas as pd

app = Flask(__name__)

@app.route("/")
def launch():
    cursor.execute("select name from song_list")
    name = []
    for i in cursor:
        name.append(i[0])
    
    cursor.execute("select duration from song_list")
    duration = []
    for i in cursor:
        duration_ms = i[0]
        duration_min = duration_ms // 60000
        duration_sec = (duration_ms % 60000) // 1000
        duration.append(f"{duration_min}:{duration_sec:02}")
    
    cursor.execute("select id from song_list")
    id = []
    for i in cursor:
        id.append(i)

    artists=[]
    for i in id:
        cursor.execute(f"select artists from artist_mapped_id where id = '{i[0]}'")
        artist_one = []
        for j in cursor:
            artist_one.append(j[0])
        artists.append(artist_one)

    length = range(len(name))

    return render_template('index.html', zipped = zip(name, artists, duration, length))



@app.route("/history", methods = ['POST'])
def update_history():
    data = request.json
    print(data)
    # print(type(data))
    # print(data['data'])

    cursor.execute(f"Select id from song_list where name = '{data['data']}'")
    for i in cursor:
        id = i[0]
    
    try:
        cursor.execute(f"insert into history(id) values ('{id}')")
    except:
        cursor.execute(f"delete from history where id = '{id}'")
        print('deleted')
        cursor.execute(f"insert into history(id) values ('{id}')")
    mydb.commit()
    return jsonify(success=True)

@app.route("/show_history")
def show_hist():
    id = []
    name = []
    duration = []
    artists=[]
    cursor.execute("select id from history order by no desc")
    for i in cursor:
        id.append(i[0])
        
    for i in id:
        cursor.execute(f"select name from song_list where id = '{i}'")
        for j in cursor:
            name.append(j[0])
        
        cursor.execute(f"select duration from song_list where id = '{i}'")
        for j in cursor:
            duration_ms = j[0]
            duration_min = duration_ms // 60000
            duration_sec = (duration_ms % 60000) // 1000
            duration.append(f"{duration_min}:{duration_sec:02}")

        cursor.execute(f"select artists from artist_mapped_id where id = '{i}'")
        artist_one = []
        for j in cursor:
            artist_one.append(j[0])
        artists.append(artist_one)


    length = range(len(name))
    
    return render_template('history.html', zipped = zip(name, zip(artists), duration, length))

        
if __name__ == "__main__":
    mydb = mysql.connector.connect(
        host='localhost',
        user='root',
        password='Shreyash2003'
    )
    cursor = mydb.cursor()
    cursor.execute("use music_db")

    app.run(debug = True)
    
    
    


    
    
    