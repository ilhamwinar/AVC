# The TensorRT engine runs inference in the following workflow:

# Allocate buffers for inputs and outputs in the GPU.
# Copy data from the host to the allocated input buffers in the GPU.
# Run inference in the GPU.
# Copy results from the GPU to the host.
# Reshape the results as necessary.

import re
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import pycuda.autoinit
import cv2
import os
import time
from queue import Empty, Queue
from threading import Thread
from websocket import create_connection
import json
import datetime
import multiprocessing
import logging
import websocket
import socket
from decouple import config
import requests

os.chdir("/workspace")
CWD = os.getcwd()
# =============================================================
# Load Environment Variables
# =============================================================
IP = config("IP")
TELEGRAM_BOT_TOKEN = config("TELEGRAM_BOT_TOKEN")
CHAT_ID = config("CHAT_ID")
GERBANG = config("GERBANG")
GARDU = config("GARDU")
MASK_CAM1 = config("MASK_CAM1")
MASK_CAM2 = config("MASK_CAM2")
MASK_CAM3 = config("MASK_CAM3")
MODEL_2CAM = config("MODEL_2CAM")
MODEL_3CAM = config("MODEL_3CAM")
RTSP_CAM1 = config("RTSP_CAM1")
RTSP_CAM2 = config("RTSP_CAM2")
RTSP_CAM3 = config("RTSP_CAM3")
# =============================================================
# Logging Config
# =============================================================
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
# =============================================================
# Masking Camera Image
# =============================================================
maskpath1 = os.path.join(CWD, "data", MASK_CAM1)
maskpath2 = os.path.join(CWD, "data", MASK_CAM2)
maskpath3 = os.path.join(CWD, "data", MASK_CAM3)

mask1 = cv2.imread(maskpath1, 0)
mask1 = cv2.resize(mask1, (100, 100), interpolation=cv2.INTER_CUBIC)

mask2 = cv2.imread(maskpath2, 0)
mask2 = cv2.resize(mask2, (270, 270), interpolation=cv2.INTER_CUBIC)

mask3 = cv2.imread(maskpath3, 0)
mask3 = cv2.resize(mask3, (225, 225), interpolation=cv2.INTER_CUBIC)

# =============================================================
# Engine Path
# =============================================================
engine_path = os.path.join(CWD, "model", "7class-v1.plan")
engine_3cam_path = os.path.join(CWD, "model", "6class-3cam-d1-bn-v2.plan")

# =============================================================
# AI Params
# =============================================================
batch_size = 1
data_type = trt.float32

# =============================================================
# Multiprocessing Params
# =============================================================
manager = multiprocessing.Manager()
qTrig1 = manager.Queue()
qTrig2 = manager.Queue()
qTrig3 = manager.Queue()
qImage1 = manager.Queue()
qImage2 = manager.Queue()
qImage3 = manager.Queue()

# =============================================================
# Image Path
# =============================================================
datapath = "../data"
global counter

# =============================================================
# Image Path
# =============================================================
def telegram_bot_sendtext(bot_message):
   message = "AI Error di "+GERBANG+" "+GARDU + " "
   bot_token = TELEGRAM_BOT_TOKEN
   bot_chatID = CHAT_ID
   send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + message + bot_message

   response = requests.get(send_text)

   return response.json()

# =============================================================
# Web Socket Function
# =============================================================

def on_message(ws, message):
    data = json.loads(message)
    detection = data["data"]["detection"]
    flag = data["data"]["flag"]
    if detection is True:
        size1 = qImage1.qsize()
        size2 = qImage2.qsize()
        size3 = qImage3.qsize()
        if flag == 1 or size1 != size2 or size1 != size3 or size2 != size3:
            logging.info("CLEARING QUEUE")
            if size1 > 0:
                for i in range(0, size1):
                    try:
                        qImage1.get(timeout=0.1)
                    except Exception as e:
                        logging.error(f"Exception in clearing queue im1 : {e}")
                        telegram_bot_sendtext(f"Exception in clearing queue im1 : {e}")
            if size2 > 0:
                for i in range(0, size2):
                    try:
                        qImage2.get(timeout=0.1)
                    except Exception as e:
                        logging.error(f"Exception in clearing queue im2 : {e}")
                        telegram_bot_sendtext(f"Exception in clearing queue im2 : {e}")
            if size3 > 0:
                for i in range(0, size3):
                    try:
                        qImage3.get(timeout=0.1)
                    except Exception as e:
                        logging.error(f"Exception in clearing queue im3 : {e}")
                        telegram_bot_sendtext(f"Exception in clearing queue im3 : {e}")

        logging.info("RECEIVING SIGNAL FROM LIDAR")
        # print("RECEIVING SIGNAL FROM LIDAR ", flush=True)
        qTrig1.put(1)
        qTrig2.put(1)
        qTrig3.put(1)
        # time.sleep(0.08)
        # qPredict.put(1)
        # timeNow = datetime.datetime.now().strftime('%m/%d/%Y %H:%M:%S.%f')[:-3]
        logging.info("SEND SIGNAL CAMERA TO CAPTURE")
        # print ("SEND SIGNAL CAMERA TO CAPTURE ",, flush=True)


def listen(url):
    global ws, counter
    counter = 0
    # websocket.setdefaulttimeout(1800)
    while True:
        try:
            ws = websocket.WebSocketApp(
                url, on_message=on_message, on_close=on_close, on_open=on_open
            )
            ws.run_forever(skip_utf8_validation=True)
        except Exception as e:
            # logging.error("Websocket connection Error  : {}".format(e))
            logging.error(f"Websocket connection Error: {e}")
        if counter == 1:
            logging.debug("Closed..reconnecting websocket")


def on_close(ws):
    global counter
    counter = counter + 1
    if counter == 4:
        counter = 2
    # logging.debug("closed")


def on_open(ws):
    global counter
    counter = 0
    logging.debug("connection established")


def on_error(ws, error):
    logging.error(f"Error ws : {error}")
    telegram_bot_sendtext(f"Error ws : {error}")

# =============================================================
# AI Image Preparation Function
# =============================================================


def allocate_buffers(engine, batch_size, data_type):
    print("engine", engine)
    """
    This is the function to allocate buffers for input and output in the device
    Args:
    engine : The path to the TensorRT engine.
    batch_size : The batch size for execution time.
    data_type: The type of the data for input and output, for example trt.float32
    Output:
    h_input_1: Input in the host.
    d_input_1: Input in the device.
    h_output_1: Output in the host.
    d_output_1: Output in the device.
    stream: CUDA stream.
    """

    # Determine dimensions and create page-locked memory buffers (which won't be swapped to disk) to hold host inputs/outputs.
    h_input_1 = cuda.pagelocked_empty(
        batch_size * trt.volume(engine.get_binding_shape(0)),
        dtype=trt.nptype(data_type),
    )
    h_output = cuda.pagelocked_empty(
        batch_size * trt.volume(engine.get_binding_shape(1)),
        dtype=trt.nptype(data_type),
    )
    # Allocate device memory for inputs and outputs.
    d_input_1 = cuda.mem_alloc(h_input_1.nbytes)

    d_output = cuda.mem_alloc(h_output.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    return h_input_1, d_input_1, h_output, d_output, stream


def load_images_to_buffer(cam1, cam2, pagelocked_buffer):
    # preprocessed = np.asarray(pics).ravel()
    # img1 = cv2.imread(cam1)
    # print(series1.iloc[i])
    # Graysclae
    img1 = cv2.cvtColor(cam1, cv2.COLOR_BGR2GRAY)
    # img1 = cv2.resize(img1, (480, 270), interpolation=cv2.INTER_CUBIC)
    # Crop to square
    img1 = img1[0:270, 210:480]
    # resize
    img1 = cv2.resize(img1, (100, 100), interpolation=cv2.INTER_CUBIC)
    # with mask
    img1 = cv2.bitwise_and(img1, img1, mask=mask1)
    dst1 = cv2.equalizeHist(img1)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    cl1 = clahe.apply(dst1)
    appended = np.zeros((270, 270))
    appended[:100, :100] = cl1[:100, :100]
    # print(cl1)
    # img2 = cv2.imread(cam2)
    # print(series2.iloc[i])
    # Graysclae
    img2 = cv2.cvtColor(cam2, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.resize(img2, (480, 270), interpolation=cv2.INTER_CUBIC)
    # Crop to square
    img2 = img2[0:270, 210:480]
    # resize
    img2 = cv2.resize(img2, (270, 270), interpolation=cv2.INTER_CUBIC)
    # with mask
    img2 = cv2.bitwise_and(img2, img2, mask=mask2)
    dst2 = cv2.equalizeHist(img2)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    cl2 = clahe.apply(dst2)

    concatenated = np.vstack((appended, cl2))

    concatenated = concatenated.astype(np.float32)
    concatenated *= 1.0 / 255
    concatenated = np.expand_dims(concatenated, axis=0)

    # X = (img_to_array(load_img(FILE, color_mode='grayscale', target_size=(100, 100), interpolation='bicubic')))
    # X = np.array(X).astype('float32') / 255
    # X = np.expand_dims(X, axis=0)
    preprocessed = concatenated.ravel()
    np.copyto(pagelocked_buffer, preprocessed)


def load_3cam_to_buffer(cam1, cam2, cam3, pagelocked_buffer2):
    # Graysclae
    img1 = cv2.cvtColor(cam1, cv2.COLOR_BGR2GRAY)
    # img1 = cv2.resize(img1, (480, 270), interpolation=cv2.INTER_CUBIC)
    # Crop to square
    img1 = img1[0:270, 210:480]
    # resize
    img1 = cv2.resize(img1, (100, 100), interpolation=cv2.INTER_CUBIC)
    # with mask
    img1 = cv2.bitwise_and(img1, img1, mask=mask1)
    dst1 = cv2.equalizeHist(img1)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    cl1 = clahe.apply(dst1)
    app1 = np.zeros((270, 270))
    app1[:100, :100] = cl1[:100, :100]

    img2 = cv2.cvtColor(cam2, cv2.COLOR_BGR2GRAY)
    img2 = img2[0:270, 210:480]
    # resize
    img2 = cv2.resize(img2, (270, 270), interpolation=cv2.INTER_CUBIC)
    # with mask
    img2 = cv2.bitwise_and(img2, img2, mask=mask2)
    dst2 = cv2.equalizeHist(img2)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    cl2 = clahe.apply(dst2)

    img3 = cv2.cvtColor(cam3, cv2.COLOR_BGR2GRAY)
    img3 = img3[45:270, 107:332]
    img3 = cv2.bitwise_and(img3, img3, mask=mask3)
    dst3 = cv2.equalizeHist(img3)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    cl3 = clahe.apply(dst3)
    app2 = np.zeros((270, 270))
    app2[:225, :225] = cl3[:225, :225]

    concatenated = np.vstack((app1, cl2))
    concatenated = np.vstack((concatenated, app2))
    concatenated = concatenated.astype(np.float32)
    concatenated *= 1.0 / 255
    concatenated = np.expand_dims(concatenated, axis=0)
    preprocessed = concatenated.ravel()
    np.copyto(pagelocked_buffer2, preprocessed)

# =============================================================
# Run AI Model
# =============================================================

def do_inference(
    engine, cam1, cam2, h_input_1, d_input_1, h_output, d_output, stream, batch_size
):

    """
    This is the function to run the inference
    Args:
      engine : Path to the TensorRT engine
      pics_1 : Input images to the model.
      h_input_1: Input in the host
      d_input_1: Input in the device
      h_output_1: Output in the host
      d_output_1: Output in the device
      stream: CUDA stream
      batch_size : Batch size for execution time
      height: Height of the output image
      width: Width of the output image
    Output:
      The list of output images
    """
    load_images_to_buffer(cam1, cam2, h_input_1)

    with engine.create_execution_context() as context:
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(d_input_1, h_input_1, stream)

        # Run inference.
        context.profiler = trt.Profiler()
        context.execute(batch_size=1, bindings=[int(d_input_1), int(d_output)])

        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        # Synchronize the stream
        stream.synchronize()
        # Return the host output.
        out = h_output
        return out


def do_inference_3cam(
    engine,
    cam1,
    cam2,
    cam3,
    h_input_1,
    d_input_1,
    h_output,
    d_output,
    stream,
    batch_size,
):
    load_3cam_to_buffer(cam1, cam2, cam3, h_input_1)

    with engine.create_execution_context() as context:
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(d_input_1, h_input_1, stream)

        # Run inference.

        context.profiler = trt.Profiler()
        context.execute(batch_size=1, bindings=[int(d_input_1), int(d_output)])

        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        # Synchronize the stream
        stream.synchronize()
        # Return the host output.
        out = h_output
        return out

# =============================================================
# Get Camera From CCTV
# =============================================================

def getCamera(src, qListen, qImage, att):
    logging.info(f"{att} IS STARTING")
    # print ("{} IS STARTING".format(att), flush=True)
    try:
        cap = cv2.VideoCapture(src)
    except cv2.error as e:
        logging.error(f"Error in opening {att} : {e}")
        telegram_bot_sendtext(f"Error in opening {att} : {e}")

    counterFalse = 0
    # filepath = "/media/1610C00B10BFF03B/dataset-avc-2020/170221/"
    height = 270
    width = 480
    baseUrl = re.findall("(?<=\@)(.*?)(?=\/)", src)[0]
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2)
            s.connect((baseUrl, 554))
            if cap.isOpened():
                # print ("executing")
                ret, image = cap.read()
                if ret:
                    image = cv2.resize(
                        image, (width, height), interpolation=cv2.INTER_CUBIC
                    )
                    try:
                        qListen.get(block=False)
                        logging.info(f"TRIGGER {att} RECEIVED")
                        # print("TRIGGER {} RECEIVED".format(att), flush=True)
                        # lock.acquire()
                        qImage.put(image)
                        # logging.info(f'DONE PUT {att}')
                        # qListen.task_done()
                    except Empty:
                        continue
                    except Exception as e:
                        logging.error(f"Exception in {att} thread : {e}")
                        telegram_bot_sendtext(f"Exception in {att} thread : {e}")
                        break
                        # qListen.close()
                        # qImage.close()
                else:
                    logging.error(f"{att} not returning image")
                    telegram_bot_sendtext(f"{att} not returning image")
                    cap.release()
                    try:
                        cap = cv2.VideoCapture(src)
                    except cv2.error as e:
                        logging.error(f"Error in opening {att}")
                        telegram_bot_sendtext(f"Error in opening {att}")
                        continue

            else:
                logging.error(f"{att} cannot be open, trying to reconnect")
                telegram_bot_sendtext(f"{att} cannot be open, trying to reconnect")
                # print ("{} cannot be open, trying to reconnect".format(att))
                time.sleep(0.5)
                try:
                    cap = cv2.VideoCapture(src)
                except cv2.error as e:
                    logging.error(f"Error in opening {att} : {e}")
                    telegram_bot_sendtext(f"Error in opening {att} : {e}")
                    continue
        except Exception as e:
            logging.error(f"Exception in {att} in connecting : {e}")
            telegram_bot_sendtext(f"Exception in {att} in connecting : {e}")


if __name__ == "__main__":
    # vivotek cam
    ipcamFront = RTSP_CAM1
    # dahua cam
    ipcamBack = RTSP_CAM2
    ipcamBack2 = RTSP_CAM3
    t = time.time()
    logging.info("STARTING THE MACHINE")
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    with open(engine_3cam_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine_3cam = runtime.deserialize_cuda_engine(f.read())

    logging.info(f"time taken to start the machine {time.time() - t}")
    # ws = websocket.WebSocketApp("ws://192.168.88.103:3008/test", on_message=on_message, on_close=on_close)
    # wst = Thread(target=ws.run_forever)
    # wst.daemon = True

    url = "ws://"+IP+":3008/test"
    detection = Thread(target=listen, args=(url,), daemon=True)
    # detection.name = 'Detect'
    im1 = multiprocessing.Process(
        target=getCamera, args=(ipcamFront, qTrig1, qImage1, "CAM1")
    )
    im1.name = "Cam1"
    im2 = multiprocessing.Process(
        target=getCamera, args=(ipcamBack, qTrig2, qImage2, "CAM2")
    )
    im2.name = "Cam2"
    im3 = multiprocessing.Process(
        target=getCamera, args=(ipcamBack2, qTrig3, qImage3, "CAM3")
    )
    im3.name = "Cam3"

    im1.start()
    im2.start()
    im3.start()
    time.sleep(0.5)
    detection.start()
    while True:
        try:
            # t = time.time()
            # t2 = datetime.datetime.now().strftime("%d%m%y-%H%M%S")
            image1 = qImage1.get(timeout=0.5)
            logging.info("GET CAM 1")
            image2 = qImage2.get(timeout=1)
            logging.info("GET CAM 2")
            image3 = qImage3.get(timeout=1)
            logging.info("GET CAM 3")
            # print("GET ALL CAM", flush=True)
            t = time.time()
            t2 = datetime.datetime.now().strftime("%d%m%y-%H%M%S")
            h_input_1, d_input_1, h_output, d_output, stream = allocate_buffers(
                engine, batch_size, data_type
            )
            out = do_inference(
                engine,
                image1,
                image2,
                h_input_1,
                d_input_1,
                h_output,
                d_output,
                stream,
                batch_size,
            )
            # print("PREDICTION : {}, time elapsed: {} ".format(np.argmax(out), time.time() - t), flush=True)
            vtype = int(np.argmax(out))
            conf = float(out[vtype])
            if vtype >= 4 and vtype < 6:
                logging.info("DONE FIRST INFERENCE")
                # print ("DONE FIRST INFERENCE", flush = True)
                h_input_1, d_input_1, h_output, d_output, stream = allocate_buffers(
                    engine_3cam, batch_size, data_type
                )
                out2 = do_inference_3cam(
                    engine_3cam,
                    image1,
                    image2,
                    image3,
                    h_input_1,
                    d_input_1,
                    h_output,
                    d_output,
                    stream,
                    batch_size,
                )
                out2 = out2[0]
                # logging.info(f'test out2 : {out2}')
                if out2 >= 0.55:
                    vtype = 5
                else:
                    vtype = 4
                conf = out2
                # vtype = int(np.argmax(out2)) + 4
            # print ("VTYPE", vtype)
            if vtype == 6:
                vtype = 7
            # clocknow = datetime.datetime.now().strftime('%m/%d/%Y %H:%M:%S')
            time_elapsed = time.time() - t
            logging.info(
                f"PREDICTION : {vtype}, CONFIDENCE : {conf}, time elapsed : {time_elapsed}"
            )
            # print("{} [INFO] PREDICTION : {}, CONFIDENCE : {}, time elapsed: {} ".format(clocknow, vtype, conf, time.time() - t), flush=True)
            f1 = (
                datapath
                + "/"
                + str(vtype)
                + "/"
                + t2
                + "-"
                + str(vtype)
                + "-"
                + str(vtype)
                + "-"
                + str(conf)
                + "-"
                + "cam1.jpg"
            )
            f2 = (
                datapath
                + "/"
                + str(vtype)
                + "/"
                + t2
                + "-"
                + str(vtype)
                + "-"
                + str(vtype)
                + "-"
                + str(conf)
                + "-"
                + "cam2.jpg"
            )
            f3 = (
                datapath
                + "/"
                + str(vtype)
                + "/"
                + t2
                + "-"
                + str(vtype)
                + "-"
                + str(vtype)
                + "-"
                + str(conf)
                + "-"
                + "cam3.jpg"
            )
            cv2.imwrite(f1, image1)
            cv2.imwrite(f2, image2)
            cv2.imwrite(f3, image3)
            try:
                ws.send(
                    json.dumps(
                        {
                            "vehicle_type": vtype,
                            "confidence_type_0": float(out[0]),
                            "confidence_type_1": float(out[1]),
                            "confidence_type_2": float(out[2]),
                            "confidence_type_3": float(out[3]),
                            "confidence_type_4": float(out[4]),
                            "confidence_type_5": float(out[5]),
                            "cam1_name": f1,
                            "cam2_name": f2,
                            "cam3_name": f3,
                        }
                    )
                )
                logging.info("SEND SUCCESS")
            except Exception as e:
                logging.error(f"Exception occurred in sending: {e}")
                telegram_bot_sendtext(f"Exception occurred in sending: {e}")
            # f1 = datapath + "/" + str(vtype) + "/" + t2 + "-" + str(vtype) + "-" + str(conf) + "-" + 'cam1.jpg'
            # f2 = datapath + "/" + str(vtype) + "/" + t2 + "-" + str(vtype) + "-" + str(conf) + "-" + 'cam2.jpg'
            # f3 = datapath + "/" + str(vtype) + "/" + t2 + "-" + str(vtype) + "-" + str(conf) + "-" + 'cam3.jpg'
            # cv2.imwrite(f1, image1)
            # cv2.imwrite(f2, image2)
            # cv2.imwrite(f3, image3)
            # print (f1)
        except Empty:
            continue

    # detection.terminate()
    # detection.join()
    im1.terminate()
    im1.join()
    im2.terminate()
    im2.join()
    im3.terminate()
    im3.join()
