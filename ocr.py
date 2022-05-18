import os
import sys
from PIL import Image
import time




async def read_image(image_path,ocr):

    try:
        tic = time.time()
        result = ocr.readtext(image_path)
        result_processed = []
        for each in result:
            result_processed.append(each[-2])
        toc = time.time()
        return result_processed, toc-tic

    except:
        return "[ERROR] Unable to process file: {0}".format(image_path)



async def read_image_3(image, ocr):

    info = []
    try:
        ocr = PaddleOCR(use_angle_cls=True, lang='en')
        result = ocr.ocr(image, cls=True)
        for line in result:
            info.append(line[1])
        info.append(info)
        return info
    except:
        return "[ERROR] Unable to process file: {0}".format(image)

    