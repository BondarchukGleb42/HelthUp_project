# HelthUp_project

Программа для мониторинга за качеством выполнения зарядки для шеи на основе алгоритмов компьютерного зрения.  

Необходимые библиотеки:  
  - pip install tensorflow
  - pip install opencv-python


Функционал определения ключевых точек и мониторинга состояния выполнения упражнения вынесены в отдельный модули exercise.py и processor.py, благодаря чему их легко интегрировать в код.  

Пример использования:  

```python
from processor import find_keypoints, draw_joints
from exercises import NeckRotationController, NeckTiltController, CustomExercisesSet
import cv2
video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# создадим свой комплекс упражнений, включающий 5 поворотов шеи, 5 наклонов шеи и ещё 5 поворотов.
custom_exercise_set = CustomExercisesSet([(NeckRotationController(), 5), (NeckTiltController(), 5), (NeckRotationController(), 5)])

while True:
	ret, frame = video_capture.read()
	frame = cv2.flip(frame, 1)
	keypoints = find_keypoints(frame) # найдём ключевые точки
  # отследим состояния выполнения зарядки в данный момент и отрисуем это
	output_image = draw_joints(keypoints, frame, custom_exercise_set)

	cv2.imshow('Keypoint image', output_image) # выведем изображение
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
```
