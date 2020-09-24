del /Q user_images\*.*
del /Q user_shape\*.*
del /Q tmp\*.*
call face_recognition_mkl.bat --face_chk 1 --one_person 1  %1 %2 %3 %4 %5 %6 %7 %8 %9  --m 
