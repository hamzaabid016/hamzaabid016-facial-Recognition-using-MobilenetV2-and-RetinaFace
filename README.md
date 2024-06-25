Facial Recognition using MobilenetV2 and RetinaFace
This project implements facial recognition using MobilenetV2 for feature extraction and RetinaFace for face detection. It is designed to recognize faces in images using pre-trained models and databases.

Features
Face detection using RetinaFace.
Facial feature extraction using MobilenetV2.
Database integration for storing recognized faces.
Example usage with sample images.
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/hamzaabid016/hamzaabid016-facial-Recognition-using-MobilenetV2-and-RetinaFace.git
cd hamzaabid016-facial-Recognition-using-MobilenetV2-and-RetinaFace
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Download pre-trained models (if necessary) and place them in the appropriate directories.

Usage
Prepare your images for recognition and place them in the recognized_images directory.

Run the application:

bash
Copy code
python app.py
Follow the instructions on the console to initiate facial recognition.

File Structure
app.py: Main application script.
requirements.txt: List of Python dependencies.
Dockerfile: Docker configuration for containerization.
Script.sql: SQL script for database setup.
models/: Directory for storing pre-trained models.
recognized_images/: Directory for storing images to be recognized.
tools/: Directory containing utility scripts (e.g., architecture.py, database.py).
Contributing
Contributions are welcome! If you'd like to contribute to this project, please fork the repository and submit a pull request with your proposed changes.

License
MIT License

Contact
For questions or support, please contact Hamza Abid at hamzaabid677@gmail.com

