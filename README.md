# dnn_face_recognition_ex

## start  
Place the photo of the person you want to authenticate in **images**.  

## step1  
Execute **step1_step3.bat**  
A photo list (imagelist.txt) to be authenticated is created.  

## step2  
Execute **step2.bat**  
The feature vector of each target person is generated from the photo list (imagelist.txt) to be authenticated, and the data is generated in user_shape.  
The face image recognized from the photo is saved in user_images, but it will not be used anymore.  

## step3  
Execute **step1_step3.bat**  
A list of feature vectors (shapelist.txt) for each subject to be authenticated is generated.  

## step4
Use **step4_face_recognition_cpu.bat** or **step4_face_recognition_cuda.bat** to identify the person on your webcam or USB camera.  