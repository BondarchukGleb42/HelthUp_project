from processor import find_keypoints, draw_joints
from exercises import NeckRotationController, NeckTiltController, CustomExercisesSet
import cv2

video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# составим свой кастомный комплекс упражнений
custom_exercise_set = CustomExercisesSet([(NeckRotationController(), 10), (NeckTiltController(), 10)])

while True:
	ret, frame = video_capture.read() # считаем изображение с веб-камеры

	frame = cv2.flip(frame, 1)
	keypoints = find_keypoints(frame) # найдём ключевые точки на изображении
	# сформируем отчёт и отрисуем его на изображении
	output_image = draw_joints(keypoints, frame, custom_exercise_set)

	cv2.imshow('Keypoint image', output_image) # выведем изображение
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

video_capture.release()
cv2.destroyAllWindows()