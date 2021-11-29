import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import matplotlib
import exercises

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

font = ImageFont.truetype("arial.ttf")

edges = {
	"nose": "nose", "left_eye": "nose", "right_eye": "nose", "left_ear": "left_eye", "right_ear": "right_eye",
	"left_shoulder": "left_elbow", "right_shoulder": "right_elbow", "left_elbow": "left_wrist", 
	"right_elbow": "right_wrist", "left_wrist": "left_wrist", "right_wrist": "right_wrist", 
	"left_hip": "left_knee", "right_hip": "right_knee", "left_knee": "left_ankle", "right_knee": "right_ankle",
	"left_ankle": "left_ankle", "right_ankle": "right_ankle"
}

body_joints = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
			   "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
			   "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", 
			   "right_knee", "left_ankle", "right_ankle"]
joint_by_idx = {i: joint for i, joint in enumerate(body_joints)}
idx_by_joint = {v: k for k, v in joint_by_idx.items()}


def find_keypoints(input_image: np.array) -> dict:
	""" функция для определения ключевых точек на изображении.
		Принимает на вход: input_image, np.array, (H x W x RGB), входное изображение
		Возвращает: keypoints, dict словарь с координатами каждой ключевой точки """
	img_shape = [input_image.shape[1], input_image.shape[0]]
	img = cv2.resize(input_image, (256, 256))
	ascept = [img_shape[0] / 256, img_shape[1] / 256]

	img = tf.cast([img], dtype=tf.uint8)
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
	interpreter.set_tensor(input_details[0]['index'], img.numpy())
	interpreter.invoke()
	keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

	keypoints = {}
	for i, k in enumerate(keypoints_with_scores[0][0]):
		if k[2] > 0.25:
			keypoints[body_joints[i]] = (k[1] * 256 * ascept[0], k[0] * 256 * ascept[1])
	if len(keypoints) == 0:
		return {}
	return keypoints


def draw_joints(keypoints, image, exercise, report=None):
	"""
	Функция для отрисовки обнаруженных частей тела и отчёта о выполнении упражнения
	Принимает на вход:
		-keypoints: координаты ключевых точек частей тела, полученных через find_keypoints;
		-image: входное изображение, аналогичное по формату с изображением для find_keypoints;
		-exercise: упражнение, которое выполняется в данный момент (также можно передать CustomExercisesSet)
		-report: если None, то заново генерирует отчёт о выполнении упражнения.
	"""
	if len(keypoints) == 0:
		return image

	if report is None:
		name, is_started, log, repeats_count = exercise.get_report(keypoints)
	else:
		name, is_started, log, repeats_count = report

	cv2.putText(image, str(repeats_count), (40, 40),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

	image_pil = Image.fromarray(image)
	draw = ImageDraw.Draw(image_pil)
	draw.text((100, 20), name, font=font, fill=(0, 255, 0, 255))
	image = np.array(image_pil)

	if is_started:
		joints_for_exercise = exercise.get_used_joints()
		for i, joint in enumerate(joints_for_exercise):
			cv2.line(image, (int(keypoints[joint][0]), int(keypoints[joint][1])),
					(int(keypoints[edges[joint]][0]), int(keypoints[edges[joint]][1])),
					tuple(matplotlib.colors.hsv_to_rgb([i / float(len(edges)), 1.0, 1.0]) * 255), 
					2, lineType=cv2.LINE_AA)
	return image