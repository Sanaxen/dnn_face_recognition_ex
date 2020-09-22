# 準備

## imagesホルダーに画像を登録。  
画像ファイルは 登録者名_数字列 でjpegかpngの画像  

imagelist.txt に画像ファイルをリストアップ  
※step1_step3.batを叩くと作成されます。  

## 顔特徴ベクトルにエンコード  
cuda10.2が利用できる場合  step2_cuda.bat  
MKLが利用できる場合       step2_mkl.bat  
上記以外の場合           step2_cpu.bat  

imagelist.txt から画像ファイルを読み込んで顔特徴ベクトルにエンコードして
user_shapeホルダーに 登録者名_数字列.txt で生成されます。  

## 登録者リストの作成  
shapelist.txt にuser_shapeホルダーにある登録者の顔特徴ベクトルにエンコードしたファイルをリストアップ  
※step1_step3.batを叩くと作成されます。  

#### 以上で登録は完了  

# 顔認証
cuda10.2が利用できる場合  step4_face_recognition_cuda.bat  
MKLが利用できる場合       step4_face_recognition_mkl.bat  
上記以外の場合            step4_face_recognition_cpu.bat  


## コマンドラインオプション  
dnn_face_recognition_ex.exe [パラメータ] [コマンド]  

### パラメータ  
- `--t` value  
 - - value=認証閾値(default 0.2)  

- `--one_person` [0|1]  
   - - 0:同時複数人  
   - - 1:一人のみ  

- `--face_chek` [0|1]  
	- - 0: 顔の向きを正面に限定しない  
	- - 1: 顔の向きを正面に限定  

- `--dnn_face_detect` [0|1]  
	- - 0:defaultの顔認識  
	- - 1:CNN basedの顔認識  
	- - 2:Resnet basedの顔認識  
	
- `--video` moving_image_file  
	- - 入力を動画ファイルにします。  
	 
- `--no_show` [0|1]  
	- - 0: 画像・動画を表示する  
	- - 1: 画像・動画を表示しない 　
  
### コマンド  
- `--cap` [username]  
	- - 正面顔をカメラまたは画像からキャプチャしてcaptureフォルダに１０枚生成します。  
	
- `--m`  
	- - shapelist.txt からuser_shapeホルダーに登録者の顔特徴ベクトルにエンコードしたファイルをリストアップ  

- `--recog`  
	- - カメラまたは動画ファイルから顔認証  

- `--image` imagefile[.png|.jpg]  
	- - 画像ファイルから顔認証  

#### コマンドラインオプションが画像ファイルだけの場合  
- user_shapeホルダーに登録者の顔特徴ベクトルにエンコードしたファイルを生成  
