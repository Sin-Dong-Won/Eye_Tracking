import cv2
import dlib

# 얼굴 검출기와 랜드마크 검출기 초기화
face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 비디오 파일 또는 웹캠에서 프레임 읽기
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    # 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 얼굴 검출
    faces = face_detector(gray)
    
    # 검출된 모든 얼굴에 대해 랜드마크 검출
    for face in faces:
        landmarks = landmark_detector(gray, face)
        
        # 왼쪽 눈과 오른쪽 눈의 좌표 추출
        left_eye = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)
        
        # 눈을 둘러싼 사각형 그리기
        cv2.rectangle(frame, (left_eye[0]-10, left_eye[1]-10), (left_eye[0]+10, left_eye[1]+10), (0, 255, 0), 2)
        cv2.rectangle(frame, (right_eye[0]-10, right_eye[1]-10), (right_eye[0]+10, right_eye[1]+10), (0, 255, 0), 2)
        
    # 결과 출력
    cv2.imshow("Eye Tracking", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()