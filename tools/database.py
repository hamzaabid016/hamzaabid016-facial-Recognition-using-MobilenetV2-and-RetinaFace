import sqlite3
import os
import uuid
import datetime
import base64
import numpy as np
import cv2

# SQLite database file path
db_file = 'database_test.db'

def create_table():
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Create face_encodings_table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_encodings_table (
            uu_id TEXT PRIMARY KEY,
            name TEXT,
            ref_no TEXT,
            summary TEXT,
            image_bytes BLOB,
            image_encodes BLOB
        )
    ''')

    # Add missing columns if necessary
    required_columns = {
        'ref_no': 'TEXT',
        'summary': 'TEXT'
    }
    
    cursor.execute('PRAGMA table_info(face_encodings_table)')
    columns = {info[1] for info in cursor.fetchall()}
    for column, data_type in required_columns.items():
        if column not in columns:
            cursor.execute(f"ALTER TABLE face_encodings_table ADD COLUMN {column} {data_type}")

    # Create log table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS log_table (
            log_id TEXT PRIMARY KEY,
            uu_id TEXT,
            timestamp TEXT
        )
    ''')

    conn.commit()
    cursor.close()
    conn.close()


def get_all_data():
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute('SELECT uu_id, name, ref_no, summary, image_bytes, image_encodes FROM face_encodings_table')
    all_data = cursor.fetchall()
    cursor.close()
    conn.close()
    return all_data


def insert_data(uu_id, name, ref_no, summary, image_bytes, image_encodes):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    image_encodes_bytes = image_encodes.tobytes()
    cursor.execute('''
        INSERT INTO face_encodings_table (uu_id, name, ref_no, summary, image_bytes, image_encodes)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (uu_id, name, ref_no, summary, image_bytes, image_encodes_bytes))
    conn.commit()
    cursor.close()
    conn.close()

def log_entry(uu_id):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    log_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute('''
        INSERT INTO log_table (log_id, uu_id, timestamp)
        VALUES (?, ?, ?)
    ''', (log_id, uu_id, timestamp))
    conn.commit()
    cursor.close()
    conn.close()

def get_all_logs():
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute('SELECT log_id, uu_id, timestamp FROM log_table')
    all_logs = cursor.fetchall()
    cursor.close()
    conn.close()
    return all_logs

def data_setting(data):
    data_final = []
    for re in data:
        uu_id, name, ref_no, summary, image_bytes, encodes_bytes = re
        image_encodes = np.frombuffer(encodes_bytes, dtype=np.float32)
        data_final.append((uu_id, name, ref_no, summary, image_bytes, image_encodes))
    return data_final

# Ensure tables are created when the module is imported
create_table()
