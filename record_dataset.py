import cv2
import time
from os import listdir, mkdir
from os.path import isfile, join, exists
from pynput import keyboard
from main import find_input_field, get_screen, find_chars
from Config import path_to_dataset


def get_chars():
    screen = get_screen()
    x, y, w, h = find_input_field(screen)
    screen = screen[y:y + h, x:x + w]
    chars = find_chars(screen)
    return chars


def add_to_dataset(char, img):
    if not exists(path_to_dataset):
        mkdir(path_to_dataset)
    subdir = char + "/"
    local_dir = path_to_dataset + subdir
    if not exists(local_dir):
        mkdir(local_dir)
    files = sorted([f for f in listdir(local_dir) if isfile(join(local_dir, f))])
    if len(files):
        names = [int(elem[:-4]) for elem in files]
        file_name = str(max(names) + 1) + ".png"
    else:
        file_name = "0.png"
    cv2.imwrite(local_dir + file_name, img)


def on_press(key):
    global screen, word, dir, chars
    try:
        print('{0} pressed'.format(
            key.char))
        word.append(key.char)
    except AttributeError:
        print('{0} pressed'.format(
            key))
        if key == keyboard.Key.backspace:
            if len(word):
                word.pop()
        elif key == keyboard.Key.enter:
            if len(word) == len(chars) or True:
                for c, cimg in zip(word, chars):
                    add_to_dataset(c, cimg)
            else:
                print("Wrong word!")
            time.sleep(0.75)
            chars = get_chars()
            word.clear()

    print(word)


if __name__ == "__main__":
    from dpi_fix import dpi_fix
    dpi_fix()
    keyboard_cr = keyboard.Controller()
    time.sleep(3)
    word = []
    chars = get_chars()

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
