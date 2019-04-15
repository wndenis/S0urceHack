# **S0urceHack**


Convolutional neural network powered automation script for s0urce.io game

**Features:**
* Fully automated  
* 0.999 network accuracy  
* Emulates direct input  
* Leaves custom message to targets   

**Requirements:**  
Python 3.5.6  
py-opencv 3.4.2  
pillow 5.4.1  
Tensorflow 1.10.0  
Pynput  
Keras 2.1.6  

## To run:  
`python main.py`  
You will have about 3 seconds to switch to browser with game.
_Target list_, _Target_info_ and _cdm_ windows must be opened and not occlude each other.
To stop script, press <kbd>esc</kbd>.

## To train:  
`python brain.py`  
Process will take about 1 hour.

## To collect dataset from game:  
`python record_dataset.py`  
You will have about 3 seconds to switch to browser with game.  
_cdm_ window must be opened and visible.  
To collect data, just start playing the game, and the script will store keys and images from your screen.
Script support backspace key, which means you can correct your errors.  
Also, script will validate input by compare lengths of detected word and your word.  
Collected data will be stored in subfolder of _dataset_ folder with name corresponding to char represented.  
You must rerun script each time after finishing hacking the target.  
NOTE: do not move the cursor using arrows - this will break all up.
_ _ _
Script was not tested in multi-monitor environment
