from flask import Flask, render_template, Response, request, redirect, url_for, session, flash, jsonify ,make_response
from flask_socketio import SocketIO, emit

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

from util.detector import detect_and_predict_mask
from imutils.video import VideoStream
import threading
import numpy as np
import imutils
import time
import cv2
import os
import json

from config import *
import base64
from datetime import datetime
import pymysql
import pytz
import secrets
import threading

app=Flask(__name__)
app.config['SECRET_KEY'] = 'mysecret'
app.config['JSON_AS_ASCII'] = False
socketIo = SocketIO(app, cors_allowed_origins="*")
# socketIo = SocketIO(app)

def generate_frames():
	#region main face detec
	global state_thread_save_db
	global trick
	global jpg_as_text
	global frame
	global vs

	old_data = None
	data = None
	
	# RTSP_URL = 'rtsp://admin:T0UCHics@161.82.233.182:14432/LiveMedia/ch1/Media1'
	# RTSP_URL = 'http://192.168.11.110:8080/video'
	vs = VideoStream(src=1).start()
	# Load our serialized face detector model from disk
	prototxtPath = os.path.join(os.getcwd(),'models','deploy.prototxt')
	weightsPath = os.path.join(os.getcwd(),'models','res10_300x300_ssd_iter_140000.caffemodel')
	lodemodelPath = os.path.join(os.getcwd(),'models','mask_detector.model')
	faceNet = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)
	maskNet = load_model(lodemodelPath)
	socketIo.emit('datas-image',data = readData())	
	# Loop over the frames from the video stream
	print("in-----")
	while True:
		# Grab the frame from the threaded video stream and resize it to have a maximum width of 600 pixels
		state_thread_save_db = True
		frame = vs.read()
		
		if frame is None:
			#streaming image Camera-not-working
			image = cv2.imread('no-came.png')
			width, height = image.shape[1],image.shape[0]
			crop_image = center_crop(image, (height,width))
			frameImg = cv2.imencode('.jpg', crop_image)[1].tobytes()
			yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frameImg + b'\r\n')

			vs = VideoStream(src=RTSP_URL).start()	
			# print("Can't receive frame (stream end?). Exiting ...")
			# time.sleep(2)

		else :
			frame = imutils.resize(frame,width = 600)
			frame_2 = frame.copy()
			(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
			#Active face detec
			data = len(locs)
			if (data != old_data) :
				if (data==1) :
					trick = True 
				else :
					trick = False
				old_data = data
			else :
				trick=False

			for (box, pred) in zip(locs, preds):
				(startX, startY, endX, endY) = box
				(mask, withoutMask) = pred
				label = "Mask" if mask > withoutMask else "No Mask"
				label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
				cv2.putText(frame, label, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (250, 0, 0), 2)

				cv2.rectangle(frame, (startX, startY), (endX, endY), (250, 0, 0), 2)
				if (trick):
					crop = frame_2[startY-80:endY+80,startX-50:endX+50]

					try:
						retval, buffer = cv2.imencode('.jpg', crop)
						jpg_as_text = base64.b64encode(buffer)
					except:
						print("An exception occurred")

			width, height = frame.shape[1],frame.shape[0]
			crop_frame = center_crop(frame, (height,width))
			frameWeb = cv2.imencode('.jpg', crop_frame)[1].tobytes()
			yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frameWeb + b'\r\n')
	#endregion

def center_crop(img, dim):
	width, height = img.shape[1], img.shape[0]
	# process crop width and height for max available dimension
	crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
	crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
	mid_x, mid_y = int(width/2), int(height/2)
	cw2, ch2 = int(crop_width/2), int(crop_height/2) 
	crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
	return crop_img		

@app.route('/video',methods = ['GET'])
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/face')
def showData():
	return make_response(jsonify(readData()),200)
				
def readData():
	conn = connect_db()
	datecurentFix = []
	rows=[]
	with conn:
		cur = conn.cursor(pymysql.cursors.DictCursor)
		cur.execute("select * from  user")
		rows = cur.fetchall()
		cur.close()
		if rows:
			rows.reverse()
			if len(rows) < 3:
				return rows
			else :
				for count in range(3):
					datecurentFix.append(rows[count])

	return datecurentFix
		
def connect_db():
    return pymysql.connect(host=DBHOST, user=DBUSER, port=DBPORT, password=DBPASS, db=DBNAME)

def dateThai():
    tz = pytz.timezone('Asia/Bangkok')
    now1 = datetime.now(tz)
    month_name = 'x มกราคม กุมภาพันธ์ มีนาคม เมษายน พฤษภาคม มิถุนายน กรกฎาคม สิงหาคม กันยายน ตุลาคม พฤศจิกายน ธันวาคม'.split()[now1.month]
    thai_year = now1.year + 543
    time_str = now1.strftime('%H:%M:%S')
    format_date = f'{now1.day} {month_name} {thai_year}'
    return format_date # 30 ตุลาคม 2560 20:45:30

def insertDB():
	global jpg_as_text
	vaccines=['Sinovac', 'Sinopharm', 'Aztrazeneca', 'Pfizer', 'Moderna']
	vasccine1=secrets.choice(vaccines)
	vasccine2=secrets.choice(vaccines)
	now = datetime.now() 
	date= dateThai()
	timeT = now.strftime("%H:%M:%S")
	location = "Thailand 5G EIC"

	conn = connect_db()
	with conn.cursor() as cursor :
		sql_select_query = """SELECT * FROM user WHERE date =%s """
		cursor.execute(sql_select_query,(date,))
		rows = cursor.fetchall()
		total = len(rows)
		sql = "INSERT INTO user (date,time,location,vasccine1,vasccine2,image,count_total) values(%s, %s, %s, %s ,%s,%s,%s)"
		cursor.execute(sql,(date,timeT,location,vasccine1,vasccine2,jpg_as_text,total))
		conn.commit()
		cursor.close()
		# print("insert db")
		time.sleep(0.1)

def thread_db():
	global state_thread_save_db
	global trick
	while True :
		if (trick) :
			insertDB()
			socketIo.emit('datas-image',data = readData())	
		time.sleep(0.05)

if __name__=="__main__":
	state_thread_save_db =False
	trick= False
	frame = None
	jpg_as_text =''
	vs=None
	threading.Thread(target=thread_db, daemon=True).start()
	socketIo.run(app,host="0.0.0.0", port=8000,debug=True)
	
