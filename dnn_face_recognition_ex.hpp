#pragma ones

#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d/features2d.hpp"

#include "putText_Jpn/putText_Jpn.h"
using namespace dlib;
using namespace std;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
	alevel0<
	alevel1<
	alevel2<
	alevel3<
	alevel4<
	max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,
	input_rgb_image_sized<150>
	>>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------

inline std::string getFilename(const char* name, std::string& pathname, std::string& extname)
{
	std::string fullpath = std::string(name);
	int path_i = fullpath.find_last_of("\\") + 1;//7
	int ext_i = fullpath.find_last_of(".");//10
	pathname = fullpath.substr(0, path_i + 1);//0文字目から７文字切り出す "C:\\aaa\\"
	extname = fullpath.substr(ext_i, fullpath.size() - ext_i); // 10文字目から４文字切り出す ".txt"
	std::string filename = fullpath.substr(path_i, ext_i - path_i);// ７文字目から３文字切り出す　"bbb"

	return filename;
}
inline std::string getFilename(std::string& name, std::string& pathname, std::string& extname)
{
	return getFilename(name.c_str(), pathname, extname);
}
inline void _putText(cv::Mat& img, const cv::String& text, const cv::Point& org, const char* fontname, double fontScale, cv::Scalar color)
{
	int fontSize = (int)(10 * fontScale); // 10は適当
	int width = img.cols;
	int height = fontSize * 3 / 2; // 高さはフォントサイズの1.5倍

	HDC hdc = ::CreateCompatibleDC(NULL);

	BITMAPINFOHEADER header;
	::ZeroMemory(&header, sizeof(BITMAPINFOHEADER));
	header.biSize = sizeof(BITMAPINFOHEADER);
	header.biWidth = width;
	header.biHeight = height;
	header.biPlanes = 1;
	header.biBitCount = 24;
	BITMAPINFO bitmapInfo;
	bitmapInfo.bmiHeader = header;
	HBITMAP hbmp = ::CreateDIBSection(NULL, (LPBITMAPINFO)&bitmapInfo, DIB_RGB_COLORS, NULL, NULL, 0);
	::SelectObject(hdc, hbmp);

	BITMAP  bitmap;
	::GetObject(hbmp, sizeof(BITMAP), &bitmap);

	int back_color = 0x99;
	int memSize = (((bitmap.bmBitsPixel / 8) * width) & ~3) * height;
	std::memset(bitmap.bmBits, back_color, memSize);

	HFONT hFont = ::CreateFontA(
		fontSize, 0, 0, 0, FW_DONTCARE, FALSE, FALSE, FALSE,
		SHIFTJIS_CHARSET, OUT_DEFAULT_PRECIS,
		CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY,
		VARIABLE_PITCH | FF_ROMAN, fontname);
	::SelectObject(hdc, hFont);

	::TextOutA(hdc, 0, height / 3 * 1, text.c_str(), (int)text.length());

	int posX = org.x;
	int posY = org.y - fontSize;

	unsigned char* _tmp;
	unsigned char* _img;
	for (int y = 0; y < bitmap.bmHeight; y++) {
		if (posY + y >= 0 && posY + y < img.rows) {
			_img = img.data + (int)(3 * posX + (posY + y) * (((bitmap.bmBitsPixel / 8) * img.cols) & ~3));
			_tmp = (unsigned char*)(bitmap.bmBits) + (int)((bitmap.bmHeight - y - 1) * (((bitmap.bmBitsPixel / 8) * bitmap.bmWidth) & ~3));
			for (int x = 0; x < bitmap.bmWidth; x++) {
				if (x + posX >= img.cols) {
					break;
				}
				if (_tmp[0] == 0 && _tmp[1] == 0 && _tmp[2] == 0) {
					_img[0] = (unsigned char)color.val[0];
					_img[1] = (unsigned char)color.val[1];
					_img[2] = (unsigned char)color.val[2];
				}
				_img += 3;
				_tmp += 3;
			}
		}
	}

	::DeleteObject(hFont);
	::DeleteObject(hbmp);
	::DeleteDC(hdc);
}

std::vector<matrix<rgb_pixel>> jitter_image(
	const matrix<rgb_pixel>& img
);

inline int get_imagelist(std::vector<std::string>& imagelist)
{
	FILE* fp = fopen("imagelist.txt", "r");
	if (!fp)
	{
		return -1;
	}

	char buf[256];
	while (fgets(buf, 256, fp) != NULL)
	{
		char* p = strchr(buf, '\n');
		if (p) *p = '\0';
		imagelist.push_back("images\\" + std::string(buf));
	}
	fclose(fp);

	return 0;
}

inline int get_shapelist(std::vector<std::string>& shapelist)
{
	FILE* fp = fopen("shapelist.txt", "r");
	if (!fp)
	{
		return 0;
	}

	char buf[256];
	while (fgets(buf, 256, fp) != NULL)
	{
		char* p = strchr(buf, '\n');
		if (p) *p = '\0';
		shapelist.push_back("user_shape\\" + std::string(buf));
	}
	fclose(fp);

	return 1;
}

inline std::vector<std::vector<float>> get_shapevalue_list(const std::vector<std::string>& shapelist)
{
	std::vector<std::vector<float>> shapevalue_list;

	int n = shapelist.size() / 80;
	for (int i = 0; i < shapelist.size(); i++)
	{
		printf("                                                             \r");
		printf("%d/%d", i, shapelist.size() - 1);
		FILE* fp = fopen(shapelist[i].c_str(), "r");
		if (fp)
		{
			char buf[128];
			std::vector<float> v;
			for (int j = 0; j < 128; j++)
			{
				fgets(buf, 128, fp);
				v.push_back(atof(buf));
			}
			shapevalue_list.push_back(v);
			fclose(fp);
		}
	}
	printf("                                                             \r");
	printf("done.\n");
	return shapevalue_list;
}

inline std::string face_recognition(cv::Mat& face_image, frontal_face_detector detector, shape_predictor sp, anet_type net, std::vector<std::string>& shapelist, std::vector<std::vector<float>>& shapevalue_list)
{
	try
	{
		if (shapelist.size() == 0)
		{
			FILE* fp = fopen("shapelist.txt", "r");
			if (!fp)
			{
				printf("shapelist.txt open error");
				return "unknown";
			}

			if (!get_shapelist(shapelist)) return "unknown";

			printf("target users=%d\n", shapelist.size());
		}

		if (shapevalue_list.size() == 0)
		{
			printf("loading[shape features]...");
			shapevalue_list = get_shapevalue_list(shapelist);
			printf("number of users=%d\n", shapevalue_list.size());
		}

		if (face_image.empty() == true) {
			cout << "No image!" << endl;
			return "";
		}

		dlib::array2d<bgr_pixel> img;
		assign_image(img, cv_image<bgr_pixel>(face_image));

		std::vector<matrix<rgb_pixel>> faces;
		for (auto face : detector(img))
		{
			auto shape = sp(img, face);
			matrix<rgb_pixel> face_chip;
			extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
			faces.push_back(move(face_chip));
			break;
		}

		if (faces.size() == 0)
		{
			cout << "No faces found in image!" << endl;
			return "";
		}
		cv::Mat x = dlib::toMat(img).clone();
		//cv::cvtColor(x, x, CV_RGB2BGR);
		cv::imshow("", x);
		cv::waitKey(1);

		std::vector<matrix<float, 0, 1>> face_descriptors = net(faces);
		matrix<float, 0, 1> face_descriptor = mean(mat(net(jitter_image(faces[0]))));
		//cout << "jittered face descriptor for one face: " << trans(face_descriptor) << endl;

		std::vector<float> v(face_descriptor.begin(), face_descriptor.end());
		//{
		//	auto t = face_descriptor;
		//	for (int i = 0; i < t.nc(); i++)
		//	{
		//		for (int j = 0; j < t.nr(); j++)
		//		{
		//			v.push_back(t(i, j));
		//		}
		//	}
		//}
		//{
		//	auto& t = face_descriptor;
		//	for (int i = 0; i < t.size(); i++)
		//	{
		//		v.push_back(t(i));
		//	}
		//}

		printf(" shapevalue_list.size()=%d\n", shapevalue_list.size());

		float mindist = 9999999999.0;
		int id = -1;
		for (int i = 0; i < shapevalue_list.size(); i++)
		{
			float s = 0.0;
			for (int j = 0; j < 128; j++)
			{
				s += (v[j] - shapevalue_list[i][j])*(v[j] - shapevalue_list[i][j]);
			}
			//printf("  (%d)%f\n", i, s);
			if (s < mindist)
			{
				mindist = s;
				id = i;
			}
		}
		printf("%f\n", mindist);

		if (id >= 0)
		{
			return shapelist[id];
		}
		else
		{
			return "unknown";
		}
	}
	catch (...)
	{
		return "unknown";
	}
}

inline std::string webcam_face_recognition(frontal_face_detector detector, shape_predictor sp, anet_type net, int camID = 0)
{
	try
	{
		cv::VideoCapture cap(camID);
		if (!cap.isOpened())
		{
			cerr << "Unable to connect to camera" << endl;
			return "unknown";
		}

		FILE* fp = fopen("shapelist.txt", "r");
		if (!fp)
		{
			cerr << "shapelist.txt open error" << endl;
			return "unknown";
		}

		std::vector<std::string> shapelist;
		if (!get_shapelist(shapelist)) return "unknown";

		printf("target users=%d\n", shapelist.size());

		printf("loading[shape features]...");
		std::vector<std::vector<float>>& shapevalue_list = get_shapevalue_list(shapelist);
		printf("number of users%d\n", shapevalue_list.size());

		int wait_time = 20;
		int count = 0;
		char time_count[32];
		while (true)
		{
			cv::Mat temp;
			cap >> temp;

			if (temp.empty() == true) {
				cout << "No image!" << endl;
				continue;
			}

			sprintf(time_count, "count %d", wait_time - count);
			cv::Mat temp2 = temp.clone();
			cv::putText(temp2, time_count, cv::Point(50, 90), cv::FONT_HERSHEY_COMPLEX, 3, cv::Scalar(1, 1, 1), 3);
			cv::imshow("", temp2);
			cv::waitKey(1);
			if (count < wait_time)
			{
				count++;
				continue;
			}

			std::string user = face_recognition(temp, detector, sp, net, shapelist, shapevalue_list);

			if (user == "")
			{
				continue;
			}
			return user;
		}
	}
	catch (...)
	{
		return "unknown";
	}
}

inline std::string webcam_face_recognition(int camID = 0)
{
	try
	{
		frontal_face_detector detector = get_frontal_face_detector();
		// We will also use a face landmarking model to align faces to a standard pose:  (see face_landmark_detection_ex.cpp for an introduction)
		shape_predictor sp;
		deserialize("db\\shape_predictor_5_face_landmarks.dat") >> sp;
		// And finally we load the DNN responsible for face recognition.
		anet_type net;
		deserialize("db\\dlib_face_recognition_resnet_model_v1.dat") >> net;

		webcam_face_recognition(detector, sp, net, camID);
	}
	catch (...)
	{
		return "unknown";
	}
}

inline int make_shape()
{
	std::vector<std::string> imagelist;
	if (get_imagelist(imagelist) < 0) return -1;

	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor sp;
	deserialize("db\\shape_predictor_5_face_landmarks.dat") >> sp;
	anet_type net;
	deserialize("db\\dlib_face_recognition_resnet_model_v1.dat") >> net;

	for (int id = 0; id < imagelist.size(); id++)
	{
		matrix<rgb_pixel> img;
		load_image(img, imagelist[id]);

		std::vector<matrix<rgb_pixel>> faces;
		for (auto face : detector(img))
		{
			auto shape = sp(img, face);
			matrix<rgb_pixel> face_chip;
			extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
			faces.push_back(move(face_chip));
			break;
		}

		if (faces.size() == 0)
		{
			cout << "No faces found in image!" << endl;
			continue;
		}

		std::vector<matrix<float, 0, 1>> face_descriptors = net(faces);
		std::vector<sample_pair> edges;
		for (size_t i = 0; i < face_descriptors.size(); ++i)
		{
			for (size_t j = i; j < face_descriptors.size(); ++j)
			{
				if (length(face_descriptors[i] - face_descriptors[j]) < 0.6)
					edges.push_back(sample_pair(i, j));
			}
		}
		std::vector<unsigned long> labels;
		const auto num_clusters = chinese_whispers(edges, labels);
		// This will correctly indicate that there are 4 people in the image.
		cout << "number of people found in the image: " << num_clusters << endl;

		if (num_clusters <= 0)
		{
			continue;
		}

		std::string pathname;
		std::string extname;
		std::string filename = getFilename(imagelist[id], pathname, extname);

		// Now let's display the face clustering results on the screen.  You will see that it
		// correctly grouped all the faces. 
		std::vector<image_window> win_clusters(num_clusters);
		for (size_t cluster_id = 0; cluster_id < num_clusters; ++cluster_id)
		{
			std::vector<matrix<rgb_pixel>> temp;
			for (size_t j = 0; j < labels.size(); ++j)
			{
				if (cluster_id == labels[j])
					temp.push_back(faces[j]);
			}
			win_clusters[cluster_id].set_title("face cluster " + cast_to_string(cluster_id));
			win_clusters[cluster_id].set_image(tile_images(temp));

			char clipped[256];
			sprintf(clipped, "user_images\\%s.png", filename.c_str());
			cv::Mat x = dlib::toMat(tile_images(temp)).clone();
			cv::cvtColor(x, x, CV_RGB2BGR);
			cv::imwrite(clipped, x);
			break;
		}
		//cout << "face descriptor for one face: " << trans(face_descriptors[0]) << endl;

		matrix<float, 0, 1> face_descriptor = mean(mat(net(jitter_image(faces[0]))));
		cout << "jittered face descriptor for one face: " << trans(face_descriptor) << endl;

		char user_shape[256];
		sprintf(user_shape, "user_shape\\%s.txt", filename.c_str());
		printf("%s\n", user_shape);

		FILE* fp = fopen(user_shape, "w");
		if (fp)
		{
			auto& t = face_descriptor;
			for (int i = 0; i < t.size(); i++)
			{
				fprintf(fp, "%f\n", t(i));
			}
			//for (int i = 0; i < t.nc(); i++)
			//{
			//	for (int j = 0; j < t.nr(); j++)
			//	{
			//		fprintf(fp, "%f\n", t(i, j));
			//	}
			//}
			fclose(fp);
		}
	}
	return 0;
}


