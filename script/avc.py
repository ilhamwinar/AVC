import re
import ctypes
import pycuda.autoinit
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
from itertools import chain
import json
import datetime
import multiprocessing
import logging
import websocket
import threading
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
MODEL_OBJECT_DETECTION_CAM12 = config("MODEL_OBJECT_DETECTION_CAM12")
MODEL_OBJECT_DETECTION_CAM3 = config("MODEL_OBJECT_DETECTION_CAM3")
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
# Engine Params
# =============================================================
PLUGIN_LIBRARY = os.path.join(CWD, "model", "libmyplugins.so")
engine_file_path_cam12 = os.path.join(CWD, "model", MODEL_OBJECT_DETECTION_CAM12)
engine_file_path_cam3 = os.path.join(CWD, "model", MODEL_OBJECT_DETECTION_CAM3)
ctypes.CDLL(PLUGIN_LIBRARY)

# =============================================================
# AI Params
# =============================================================
CONF_THRESH = 0.5
IOU_THRESHOLD = 0.1

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
# Yolov5 TensortRT Class
# =============================================================


class YoLov5TRT(object):
    """
    description: A YOLOv5 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path):
        # Create a Context on this device,
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            # print('bingding:', binding, engine.get_binding_shape(binding))
            size = trt.volume(engine.get_binding_shape(
                binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size

    def infer(self, raw_image):
        threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # Do image preprocess
        batch_image_raw = []
        batch_origin_h = []
        batch_origin_w = []
        batch_input_image = np.empty(
            shape=[self.batch_size, 3, self.input_h, self.input_w])
        input_image, image_raw, origin_h, origin_w = self.preprocess_image(
            raw_image)
        batch_image_raw.append(image_raw)
        batch_origin_h.append(origin_h)
        batch_origin_w.append(origin_w)
        np.copyto(batch_input_image[0], input_image)
        batch_input_image = np.ascontiguousarray(batch_input_image)

        # Copy input image to host buffer
        np.copyto(host_inputs[0], batch_input_image.ravel())
        start = time.time()
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(batch_size=self.batch_size,
                              bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()
        end = time.time()
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        # Here we use the first row of output in that batch_size = 1
        output = host_outputs[0]
        # Do postprocess
        result_boxes, result_scores, result_classid = self.post_process(
            output, origin_h, origin_w
        )
        return result_classid, end - start

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()

    def get_raw_image(self, image_path_batch):
        """
        description: Read an image from image path
        """
        for img_path in image_path_batch:
            yield cv2.imread(img_path)

    def get_raw_image_zeros(self, image_path_batch=None):
        """
        description: Ready data for warmup
        """
        return np.zeros([self.input_h, self.input_w, 3], dtype=np.uint8)

    def preprocess_image(self, raw_bgr_image):
        """
        description: Convert BGR image to RGB,
                     resize and pad it to target size, normalize to [0,1],
                     transform to NCHW format.
        param:
            input_image_path: str, image path
        return:
            image:  the processed image
            image_raw: the original image
            h: original height
            w: original width
        """
        image_raw = raw_bgr_image
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        # Calculate widht and height and paddings
        r_w = self.input_w / w
        r_h = self.input_h / h
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
        )
        image = image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    def xywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes numpy, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes numpy, each row is a box [x1, y1, x2, y2]
        """
        y = np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - \
                (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - \
                (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - \
                (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - \
                (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h

        return y

    def post_process(self, output, origin_h, origin_w):
        """
        description: postprocess the prediction
        param:
            output:     A numpy likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...]
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes numpy, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a numpy, each element is the score correspoing to box
            result_classid: finally classid, a numpy, each element is the classid correspoing to box
        """
        # Get the num of boxes detected
        num = int(output[0])
        # Reshape to a two dimentional ndarray
        pred = np.reshape(output[1:], (-1, 6))[:num, :]
        # Do nms
        boxes = self.non_max_suppression(
            pred, origin_h, origin_w, conf_thres=CONF_THRESH, nms_thres=IOU_THRESHOLD)
        result_boxes = boxes[:, :4] if len(boxes) else np.array([])
        result_scores = boxes[:, 4] if len(boxes) else np.array([])
        result_classid = boxes[:, 5] if len(boxes) else np.array([])
        return result_boxes, result_scores, result_classid

    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        """
        description: compute the IoU of two bounding boxes
        param:
            box1: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
            box2: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
            x1y1x2y2: select the coordinate format
        return:
            iou: computed iou
        """
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / \
                2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / \
                2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / \
                2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / \
                2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,
                                              0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,
                                              0], box2[:, 1], box2[:, 2], box2[:, 3]

        # Get the coordinates of the intersection rectangle
        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        # Intersection area
        inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
            np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou

    def non_max_suppression(self, prediction, origin_h, origin_w, conf_thres=0.5, nms_thres=0.4):
        """
        description: Removes detections with lower object confidence score than 'conf_thres' and performs
        Non-Maximum Suppression to further filter detections.
        param:
            prediction: detections, (x1, y1, x2, y2, conf, cls_id)
            origin_h: original image height
            origin_w: original image width
            conf_thres: a confidence threshold to filter detections
            nms_thres: a iou threshold to filter detections
        return:
            boxes: output after nms with the shape (x1, y1, x2, y2, conf, cls_id)
        """
        # Get the boxes that score > CONF_THRESH
        boxes = prediction[prediction[:, 4] >= conf_thres]
        # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes[:, :4] = self.xywh2xyxy(origin_h, origin_w, boxes[:, :4])
        # clip the coordinates
        boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h - 1)
        # Object confidence
        confs = boxes[:, 4]
        # Sort by the confs
        boxes = boxes[np.argsort(-confs)]
        # Perform non-maximum suppression
        keep_boxes = []
        while boxes.shape[0]:
            large_overlap = self.bbox_iou(np.expand_dims(
                boxes[0, :4], 0), boxes[:, :4]) > nms_thres
            label_match = boxes[0, -1] == boxes[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            keep_boxes += [boxes[0]]
            boxes = boxes[~invalid]
        boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
        return boxes

# =============================================================
# Thread Warm Up Class
# =============================================================


class warmUpThread(threading.Thread):
    def __init__(self, yolov5_wrapper):
        threading.Thread.__init__(self)
        self.yolov5_wrapper = yolov5_wrapper

    def run(self):
        results_classid, use_time = self.yolov5_wrapper.infer(
            self.yolov5_wrapper.get_raw_image_zeros())
        print(
            'warm_up time->{:.2f}ms'.format(use_time * 1000))

# =============================================================
# Thread Inference Class
# =============================================================


class inferThread(threading.Thread):
    def __init__(self, yolov5_wrapper, image_path_batch):
        threading.Thread.__init__(self)
        self.yolov5_wrapper = yolov5_wrapper
        self.image_path_batch = image_path_batch
        self._return = None

    def run(self):
        results_classid, use_time = self.yolov5_wrapper.infer(
            self.image_path_batch)
        print('result->{}, time->{:.2f}ms'.format(results_classid, use_time * 1000))
        self._return = results_classid

    def join(self):
        threading.Thread.join(self)
        return self._return


# =============================================================
# Telegram Bot
# =============================================================
def telegram_bot_sendtext(bot_message):
    message = "AI Error di "+GERBANG+" "+GARDU + " "
    bot_token = TELEGRAM_BOT_TOKEN
    bot_chatID = CHAT_ID
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + \
        bot_chatID + '&parse_mode=Markdown&text=' + message + bot_message

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
                        telegram_bot_sendtext(
                            f"Exception in clearing queue im1 : {e}")
            if size2 > 0:
                for i in range(0, size2):
                    try:
                        qImage2.get(timeout=0.1)
                    except Exception as e:
                        logging.error(f"Exception in clearing queue im2 : {e}")
                        telegram_bot_sendtext(
                            f"Exception in clearing queue im2 : {e}")
            if size3 > 0:
                for i in range(0, size3):
                    try:
                        qImage3.get(timeout=0.1)
                    except Exception as e:
                        logging.error(f"Exception in clearing queue im3 : {e}")
                        telegram_bot_sendtext(
                            f"Exception in clearing queue im3 : {e}")

        logging.info("RECEIVING SIGNAL FROM LIDAR")
        qTrig1.put(1)
        qTrig2.put(1)
        qTrig3.put(1)
        logging.info("SEND SIGNAL CAMERA TO CAPTURE")


def listen(url):
    global ws, counter
    counter = 0
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


def on_open(ws):
    global counter
    counter = 0
    logging.debug("connection established")


def on_error(ws, error):
    logging.error(f"Error ws : {error}")
    telegram_bot_sendtext(f"Error ws : {error}")

# =============================================================
# Get Camera From CCTV
# =============================================================


def getCamera(src, qListen, qImage, att):
    logging.info(f"{att} IS STARTING")

    try:
        cap = cv2.VideoCapture(src)
    except cv2.error as e:
        logging.error(f"Error in opening {att} : {e}")
        telegram_bot_sendtext(f"Error in opening {att} : {e}")

    height = 270
    width = 480
    baseUrl = re.findall("(?<=\@)(.*?)(?=\/)", src)[0]
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2)
            s.connect((baseUrl, 554))
            if cap.isOpened():
                ret, image = cap.read()
                if ret:
                    image = cv2.resize(
                        image, (width, height), interpolation=cv2.INTER_CUBIC
                    )
                    try:
                        qListen.get(block=False)
                        logging.info(f"TRIGGER {att} RECEIVED")
                        qImage.put(image)
                    except Empty:
                        continue
                    except Exception as e:
                        logging.error(f"Exception in {att} thread : {e}")
                        telegram_bot_sendtext(
                            f"Exception in {att} thread : {e}")
                        break
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
                telegram_bot_sendtext(
                    f"{att} cannot be open, trying to reconnect")
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
    buffer_list = [7, 7, 7]
    conf = 0

    # Websocket Parameter
    url = "ws://"+IP+":3008/test"
    detection = Thread(target=listen, args=(url,), daemon=True)

    # Multiprocessing Camera
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
    # YoLov5TRT instance
    yolov5_wrapper_cam12 = YoLov5TRT(engine_file_path_cam12)
    yolov5_wrapper_cam3 = YoLov5TRT(engine_file_path_cam3)
    # Create a new thread to do warm_up
    for i in range(5):
        thread1 = warmUpThread(yolov5_wrapper_cam12)
        thread1.start()
        thread1.join()

    # Thread Camera Start
    im1.start()
    im2.start()
    im3.start()
    time.sleep(0.5)

    # Thread Websocker Start
    detection.start()

    while True:
        try:
            # Queue for Image from camera
            image1 = qImage1.get(timeout=0.2)
            logging.info("GET CAM 1")
            image2 = qImage2.get(timeout=0.5)
            logging.info("GET CAM 2")
            image3 = qImage3.get(timeout=0.5)
            logging.info("GET CAM 3")
            time_image = datetime.datetime.now().strftime("%d%m%y-%H%M%S")
            vtype = 8
            # Thread Image Detection
            # ====================================
            # Model Categories Cam 1 and Cam 2
            # 0 = Bus
            # 1 = Car
            # 2 = One Tire
            # 3 = Two Tire
            # 4 = Truck Large
            # 5 = Truck Small
            # ====================================
            # Model Categories Cam 3
            # 0 = One Tire
            # 1 = Three Tire
            # 2 = Two Tire
            thread1 = inferThread(yolov5_wrapper_cam12, image1)
            thread1.start()
            raw_result1 = list(chain(thread1.join(), buffer_list))
            raw_result1.sort()
            # Filter Remove One Tire, Two Tire,and Three Tire Detection
            result1 = [x for x in raw_result1 if x != 2 and x != 3]
            if result1[0] == 0:
                # Golongan 1 Bus
                vtype = 1
            elif result1[0] == 1:
                # Golongan 1
                vtype = 0
            else:
                thread2 = inferThread(yolov5_wrapper_cam12, image2)
                thread2.start()
                raw_result2 = list(chain(thread2.join(), buffer_list))
                raw_result2.sort()
                result2 = [x for x in raw_result2 if x !=
                           0 and x != 1 and x != 4 and x != 5]
                # Truck L and Double Two Tire
                if (result1[0] == 4 and result2[0] == 3 and result2[1] == 3):
                    # Golongan 5
                    vtype = 5
                # Truck L and Double One Tire
                elif (result1[0] == 4 and result2[0] == 2 and result2[1] == 2):
                    # Check Cam 3
                    thread3 = inferThread(yolov5_wrapper_cam3, image3)
                    thread3.start()
                    raw_result3 = list(chain(thread3.join(), buffer_list))
                    raw_result3.sort()
                    if (result3[0] == 0 and result3[1] == 0):
                        # Golongan 4
                        golongan_prediksi = 4
                    else:
                       # Golongan 3
                        golongan_prediksi = 3
                    # Golongan 4
                    vtype = 4
                # Truck L and One Tire and Double Tire
                elif (result1[0] == 4 and result2[0] == 2 and result2[1] == 3):
                    # Check Cam 3
                    thread3 = inferThread(yolov5_wrapper_cam3, image3)
                    thread3.start()
                    result3 = thread3.join()
                    if 1 in result3:
                        # Golongan 5
                        vtype = 5
                    else:
                        # Golongan 4
                        vtype = 4
                # Truck L and Two Tire
                elif (result1[0] == 4 and result2[0] == 3):
                    # Golongan 3
                    vtype = 3
                # Truck L or Truck S
                elif result1[0] == 5 or result1[0] == 4:
                    # Golongan 2
                    vtype = 2

            # print("{} [INFO] PREDICTION : {}, CONFIDENCE : {}, time elapsed: {} ".format(clocknow, vtype, conf, time.time() - t), flush=True)
            f1 = (
                datapath
                + "/"
                + str(vtype)
                + "/"
                + time_image
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
                + time_image
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
                + time_image
                + "-"
                + str(vtype)
                + "-"
                + str(vtype)
                + "-"
                + str(conf)
                + "-"
                + "cam3.jpg"
            )
            try:
                if vtype == 8:
                    vtype = 7
                ws.send(
                    json.dumps(
                        {
                            "vehicle_type": vtype,
                            "confidence_type_0": float(0),
                            "confidence_type_1": float(0),
                            "confidence_type_2": float(0),
                            "confidence_type_3": float(0),
                            "confidence_type_4": float(0),
                            "confidence_type_5": float(0),
                            "cam1_name": f1,
                            "cam2_name": f2,
                            "cam3_name": f3,
                        }
                    )
                )
                logging.info("GOLONGAN : %d", vtype)
                logging.info("SEND SUCCESS")
            except Exception as e:
                logging.error(f"Exception occurred in sending: {e}")
                telegram_bot_sendtext(f"Exception occurred in sending: {e}")
            cv2.imwrite(f1, image1)
            cv2.imwrite(f2, image2)
            cv2.imwrite(f3, image3)
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
