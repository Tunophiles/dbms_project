from flask import Flask, render_template
import mysql.connector
import pandas as pd

app = Flask(__name__)

@app.route("/")
def launch():
    mydb = mysql.connector.connect(
        host='localhost',
        user='root',
        password='Pavan@123'
    )

    cursor = mydb.cursor()
    cursor.execute("use music_db1")

    cursor.execute("select name from song_list")
    name = []
    for i in cursor:
        name.append(i[0])
    
    cursor.execute("select duration_ms from song_list")
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
            artist_one.append(j)
        artists.append(artist_one)
    
    # for i in cursor:
    #     artist.append(i)
    
    # df = pd.DataFrame()
    # df_list = []
    # for i in cursor:
    #     df_list.append(list(i))
    # df = pd.DataFrame(df_list)

    
    return render_template('index.html', zipped = zip(name, artists, duration))








if __name__ == "__main__":
    app.run(debug = True)
    
    
    


    
    
    