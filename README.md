# Gesture-Snake-Game
Gesture Snake Game A hand-tracking Snake game that uses your webcam. Control the snake by moving your index finger in real time.

Features Real-time hand tracking using MediaPipe Smooth finger-based movement control Peace sign gesture to restart the game Professional interface with live camera feed Score tracking with snake growth mechanics

How to Play Keep your hand inside the yellow box shown on the camera screen Move your index finger to guide the snake Eat the red apples to grow and increase your score Avoid hitting the walls or the snake’s own tail Show a peace sign using your index and middle finger to restart the game

Quick Setup (Windows)

Create virtual environment
python -m venv venv venv\Scripts\activate

Install requirements
pip install -r requirements.txt

Run the game
python snake.py

Controls Hand movement: Steer snake Peace sign: Restart game ESC: Quit

Troubleshooting Permission Error? → Run PowerShell as Administrator NumPy Error? → Delete venv folder and recreate Camera not working? → Check webcam privacy settings

Requirements Webcam Python 3.8-3.10 Windows/Mac/Linux

