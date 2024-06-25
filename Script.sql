CREATE TABLE names_info (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    address TEXT,
    sex TEXT,
    member_since DATE
);
CREATE TABLE faces_biometrics (
    face_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name_id INTEGER,
    cropped_image BLOB,
    embedding BLOB,
    FOREIGN KEY (name_id) REFERENCES names_info (id)
);
CREATE TABLE logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id INTEGER,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    camera_id TEXT,
    FOREIGN KEY (face_id) REFERENCES faces_biometrics (face_id)
);
