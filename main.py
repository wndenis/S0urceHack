from dpi_fix import dpi_fix
dpi_fix()

import numpy as np
from PIL import ImageGrab
import os
import cv2
import time
import random

from pynput import mouse, keyboard

import Config
from brain import Thinker

keyboard_cr = keyboard.Controller()
mouse_cr = mouse.Controller()


def get_screen():
    oldpos = mouse_cr.position
    mouse_cr.position = (0, 0)
    time.sleep(0.01)
    img = ImageGrab.grab()
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    mouse_cr.position = oldpos
    return img


def find_target_button(img):
    targets = set()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for template_num in range(6):
        name = 'icons/' + str(template_num) + '.png'
        template = cv2.imread(name, 0)
        if template is None:
            continue

        template = cv2.resize(template, (img.shape[1] // 96, img.shape[0] // 54))  # todo: % of screen??
        # w, h = template.shape[::-1]

        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.9
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            targets.add(pt)
            # cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    targets = sorted(targets, key=lambda item: item[1])
    targets = targets[4:]
    # for target in targets:
    #     cv2.rectangle(img, target, target, (0, 255, 0), 5)
    # cv2.imshow('', img)
    # cv2.waitKey()
    target = random.choice(targets)
    target = (target[0], target[1] + 5)
    return target


def find_hack_button(img):
    lower = np.array(Config.red_low)
    upper = np.array(Config.red_high)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower, lower)
    mask2 = cv2.inRange(hsv, upper, upper)

    mask = cv2.bitwise_or(mask1, mask2)
    # res = cv2.bitwise_and(img, img, mask=mask)

    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    # cv2.drawContours(img, [cnt], 0, (0, 0, 255), 3)
    x, y, w, h = cv2.boundingRect(cnt)
    center = (x + w//2, y + h//2)
    return center


def find_port(img, midpoint):
    x0, y0 = midpoint[0] - 25 * 6, midpoint[1] - 10
    x1, y1 = midpoint[0] + 25 * 6, midpoint[1] + 40
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray[y0:y1, x0:x1]
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)[0:3]
    # if True:
    #     for cnt in cnts:
    #         cv2.drawContours(gray, [cnt], 0, (255, 255, 255), 3)
    #     cv2.imshow("", gray)
    #     cv2.waitKey()
    global_points = []
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        center = (x + w // 2, y + h // 2)
        global_points.append((center[0] + x0, center[1] + y0))
    # for p in global_points:
    #     cv2.rectangle(img, p, p, (0, 255, 0), 5)
    # cv2.imshow("", img)
    return random.choice(global_points)

def find_input_field(img):
    lower = np.array(Config.green_low)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, lower)

    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    # cv2.drawContours(img, [cnt], 0, (0, 0, 255), 3)
    # x, y, w, h = cv2.boundingRect(cnt)
    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
    # cv2.imshow("", img)
    return cv2.boundingRect(cnt)

def find_chars(img):
    # denoise
    img = cv2.fastNlMeansDenoisingColored(img, h=16)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.blur(img, (2, 2))
    _, img = cv2.threshold(img, 119, 255, cv2.THRESH_BINARY)

    # crop left and right
    bg = int(img[1, 1])
    cropped = False
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            if abs(int(img[y, x]) - bg) > 2:
                img = img[:, x:]
                cropped = True
                break
        if cropped:
            break

    cropped = False
    for x in reversed(range(img.shape[1])):
        for y in range(img.shape[0]):
            if abs(int(img[y, x]) - bg) > 5:
                img = img[:, :x]
                cropped = True
                break
        if cropped:
            break

    # fill bar
    bar = np.zeros((1, img.shape[1]), np.uint8)
    bg = int(img[1, 1])
    black_count = 0
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            if img[y, x] != bg:
                black_count += 1
        if black_count > 1:
            bar[0, x] = 255
        else:
            bar[0, x] = 128
        black_count = 0
    bar_mask = cv2.resize(bar.copy(), (img.shape[1], img.shape[0]))

    # Compute regions
    bar = list(bar[0])
    # print(bar)
    regions = []
    prev_elem = None
    count = 0
    for elem in bar:
        if elem != prev_elem:
            if prev_elem == 128:
                regions.append([count])
            elif prev_elem == 255:
                regions[-1].append(count)
            elif prev_elem is None:
                regions.append([count])
            prev_elem = elem
        elif count == len(bar) - 1:
            regions[-1].append(count)
        count += 1
    regions = [elem for elem in regions if len(elem) == 2]
    # print(regions)

    min_width = 7
    stitched_regions = []
    i = 0
    while i < len(regions):
        elem = regions[i]
        if elem[1] - elem[0] >= min_width:
            stitched_regions.append(elem)
        elif i < len(regions) - 1:
            i += 1
            elem[1] = regions[i][1]
            stitched_regions.append(elem)
        else:
            stitched_regions.append(elem)
        i += 1
    # print(stitched_regions)

    max_width = 16  # todo:need to workaround with this
    teared_regions = []
    i = 0
    while i < len(stitched_regions):
        elem = stitched_regions[i]
        if elem[1] - elem[0] <= max_width:
            teared_regions.append(elem)
        else:
            w = elem[1]
            d = elem[0] + (elem[1] - elem[0]) // 2
            elem1 = [elem[0], d]
            elem2 = [d, elem[1]]
            teared_regions.append(elem1)
            teared_regions.append(elem2)
        i += 1
    # print(teared_regions)

    chars = []
    for rg in teared_regions:
        # print(rg[1] - rg[0], end=", ")
        char = img[0:img.shape[1], rg[0]:rg[1]]
        char = cv2.resize(char, (16, 32))
        chars.append(char)
        # cv2.imshow("", char)
        # cv2.waitKey()

    # for rg in teared_regions:
    #     cv2.rectangle(img, (rg[0], 0), (rg[1], img.shape[1] - 1), 127+127//2, 1)
    return chars


def stop(key):
    if key is keyboard.Key.esc:
        print("Esc pressed, hacking stopped.")
        os._exit(1)


def type_chars(chars):
    for cc in chars:
        keyboard_cr.type(cc)
        time.sleep(0.001 + random.random() * 0.01)


if __name__ == "__main__":

    listener = keyboard.Listener(
        on_press=stop)
    listener.start()
    print("Started hacking.")

    from record_dataset import add_to_dataset
    time.sleep(0.5)
    thinker = Thinker()
    while True:
        time.sleep(0.55)
        try:
            target_btn = find_target_button(get_screen())
            mouse_cr.position = target_btn
            mouse_cr.click(mouse.Button.left)
            time.sleep(0.35)

            hack_btn = find_hack_button(get_screen())
            mouse_cr.position = hack_btn
            mouse_cr.click(mouse.Button.left)
            time.sleep(0.35)

            port_btn = find_port(get_screen(), hack_btn)
            mouse_cr.position = port_btn
            mouse_cr.click(mouse.Button.left)
            time.sleep(0.35)

        except:
            continue
        # hack him
        misspell_prob = 0
        prev_word = ""
        definitely_mistake = False
        while True:
            screen = get_screen()
            x, y, w, h = find_input_field(screen)
            screen = screen[y:y + h, x:x + w]

            char_imgs = find_chars(screen)
            if len(char_imgs) < 4:
                break
            print("="*80)
            predicted_chars = ""
            predictions = []
            for char_img in char_imgs:
                prediction = thinker.think(char_img)
                predictions.append(prediction)
                predicted_chars += prediction[0]
                print(prediction)
            print(predicted_chars)

            # HARDCODE ZONE!
            # Due to issue with word splitting (which I'll maybe fix later)
            # We need to redefine words with some known mistakes
            if predicted_chars in {"unpackbmfile", "unpackhpfile"}:
                predicted_chars = "unpacktmpfile"
            elif predicted_chars in {"getmysgldomain"}:
                predicted_chars = "getmysqldomain"
            elif predicted_chars in {"create3axievector"}:
                predicted_chars = "create3axisvector"
            elif predicted_chars in {"accountuame"}:
                predicted_chars = "accountname"
            # HARDCODE ZONE!

            if prev_word == predicted_chars:
                if misspell_prob > 4:
                    if not definitely_mistake:
                        time.sleep(random.random() * 5 + 5)
                        definitely_mistake = True
                        continue
                    combinations = []
                    print("Can not recognize, ACTIVATE BRUTE FORCE:")
                    for x in range(1, 2**len(predicted_chars)):
                        x = str(bin(x))[2:]
                        x = '0' * (len(predicted_chars) - len(x)) + x
                        x = list(map(int, x))
                        word = "".join([predictions[i][j] for i, j in zip(range(len(predicted_chars)), x)])
                        print("Trying %s" % word)
                        prev_word = word
                        type_chars(word)
                        keyboard_cr.press(keyboard.Key.enter)
                        keyboard_cr.release(keyboard.Key.enter)
                        time.sleep(0.1)

                        # screen_check = get_screen()
                        # x1, y1, w1, h1 = find_input_field(screen)
                        # screen_check = screen_check[y1:y1 + h1, x1:x1 + w1]
                        # if screen_check != screen:
                        #     break

                    misspell_prob = 0
                    print()
                    time.sleep(4)
                    continue
                else:
                    misspell_prob += 1
            else:
                misspell_prob = False
            prev_word = predicted_chars
            type_chars(predicted_chars)
            keyboard_cr.press(keyboard.Key.enter)
            keyboard_cr.release(keyboard.Key.enter)
            time.sleep(0.4 + random.random() * 0.4)
        print("###HACKED###\n")
        time.sleep(0.5)
        for x in range(50):
            keyboard_cr.press(keyboard.Key.backspace)
            time.sleep(0.01)
        keyboard_cr.release(keyboard.Key.backspace)
        type_chars("PWND by CNN")
        time.sleep(0.1)
        keyboard_cr.press(keyboard.Key.enter)
        time.sleep(0.05)
        keyboard_cr.release(keyboard.Key.enter)

        # validation = input("If prediction is correct, press enter, else print the correct word: ")
        # if len(validation) != 0:
        #     for i in range(len(validation)):
        #         if validation[i] != predicted_chars[i]:
        #             add_to_dataset(validation[i], char_imgs[i])
        #             print("Added %s to retrain later" % validation[i])
