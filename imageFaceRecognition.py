import cv2
import face_recognition

# load the image
imgSundar = face_recognition.load_image_file("images/sundar.jpg")
imgSundar = cv2.cvtColor(imgSundar, cv2.COLOR_BGR2RGB)#changing bgr to rgb
imgTest = face_recognition.load_image_file("images/bill gates.jpg") #load the test image
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgSundar)[0]
encodeSundar = face_recognition.face_encodings(imgSundar)[0] #encode the image
cv2.rectangle(imgSundar,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,0),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,0),2)

#compare faces
results = face_recognition.compare_faces([encodeSundar], encodeTest)
#print the result
if results[0]:
	print('Faces matched both are same person')
else:
	print('Faces not matched both are different person')

cv2.imshow("Sundar", imgSundar)
cv2.imshow("Sundar test", imgTest)
cv2.waitKey(0)
