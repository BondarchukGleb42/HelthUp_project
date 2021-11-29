import math
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def distance(x0, x1, y0, y1):
	return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5


class ExerciseController:
	"""
	Основной класс, реализующий базовую логику для всех упражнений,
	такую как инициализация, хранение и предсказание координат частей тела,
	формирование отчёта о состоянии выполнения упражнения.
	"""
	def __init__(self):
		self.is_started = False
		self.setup = False
		self.repeats_count = 0
		self.used_joints = []

	def get_used_joints(self):
		return self.used_joints

	def set_coords_log(self, k):
		self.coords_log = {joint: [[float(k[joint][0])]*60, [float(k[joint][1])]*60] for joint in self.get_used_joints()}

	def update_coords_log(self, k):
		for joint in self.get_used_joints():
			self.coords_log[joint][0].append(k[joint][0])
			self.coords_log[joint][1].append(k[joint][1])
			self.coords_log[joint][0] = self.coords_log[joint][0][1:]
			self.coords_log[joint][1] = self.coords_log[joint][1][1:]

	def get_coords_log(self):
		return self.coords_log

	def predict_coords(self):
		new_joints = {}
		for joint in self.get_used_joints():
			poly_model = PolynomialFeatures(degree=3)
			linear_model = LinearRegression()

			joint_x, joint_y = self.get_coords_log()[joint]
			joint_x, joint_y = np.array(joint_x).reshape(-1, 1), np.array(joint_y).reshape(-1, 1)
			x = joint_x[-1] + sum([joint_x[i] - joint_x[i-1] for i in range(1, len(joint_x))]) / len(joint_x)

			X_poly = poly_model.fit_transform(joint_x, joint_y)
			linear_model.fit(X_poly, joint_y)
			y = linear_model.predict(poly_model.transform(x.reshape(-1, 1))) 
			new_joints[joint] = [int(x), int(y)]
		return new_joints

	def get_report(self, k):
		all_joints_detected = all([joint in list(k.keys()) for joint in self.get_used_joints()])
		if not self.is_started and not all_joints_detected:
			return (self.get_name(), self.is_started, None, self.repeats_count)
		elif not self.is_started:
			self.set_coords_log(k)
			self.is_started = True
		elif not all_joints_detected:
			return (self.get_name(), False, None, self.repeats_count)

		self.update_coords_log(k)
		self.exercise(k)
		return (self.get_name(), self.is_started, None, self.repeats_count)





class NeckRotationController(ExerciseController):
	def __init__(self): 
		ExerciseController.__init__(self)
		self.used_joints = ["right_eye", "left_eye", "nose"]

	def get_name(self):
		return "повороты шеи в сторону"

	def exercise(self, k):
		left = k["left_eye"][0] - k["nose"][0]
		right = k["nose"][0] - k["right_eye"][0]
		if left <= 0:
			if not self.setup:
				self.repeats_count += 1
				self.setup = True
		elif right <= 0:
			if self.repeats_count == 0:
				self.setup = True
			if self.setup:
				self.repeats_count += 1
				self.setup = False


class NeckTiltController(ExerciseController):
	def __init__(self):
		ExerciseController.__init__(self)
		self.used_joints = ["right_eye", "left_eye", "nose"]

	def get_name(self):
		return "наклоны шеи в сторону"

	def exercise(self, k):
		tg = (k["left_eye"][1] - k["right_eye"][1]) / (k["left_eye"][0] - k["right_eye"][0])
		if tg < -0.6:
			if not self.setup:
				self.repeats_count += 1
				self.setup = True
		elif tg > 0.6:
			if self.repeats_count == 0:
				self.setup = True
			if self.setup:
				self.repeats_count += 1
				self.setup = False


class CustomExercisesSet(ExerciseController):
	def __init__(self, exercises):
		self.exercises = exercises
		self.target_exercise = 0
		self.used_joints = set(sum([exc[0].used_joints for exc in exercises], []))

	def get_report(self, k):
		if self.target_exercise >= len(self.exercises):
			return ("", False, "Все упражнения выполнены", 0)

		report = self.exercises[self.target_exercise][0].get_report(k)
		if report[3] >= self.exercises[self.target_exercise][1]:
			self.target_exercise += 1
		return report
 