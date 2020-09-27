mkdir user_shape
mkdir user_images
mkdir tmp
mkdir capture

dir /B user_shape\*.txt > shapelist.txt
dir /B images\*.png > imagelist.txt
dir /B images\*.jpg >> imagelist.txt
