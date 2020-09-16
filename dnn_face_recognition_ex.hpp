#pragma ones

#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
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

namespace dnn_face_recognition_
{
	int face_chk = 0;
};

inline cv::Mat hconcat_ex(cv::Mat& im1, cv::Mat& im2)
{
	cv::Mat cat;
	int h1 = im1.rows;
	int w1 = im1.cols;
	int h2 = im2.rows;
	int w2 = im2.cols;
	if (h1 < h2)
	{
		h1 = h2;
		w1 = int(((float)h2 / (float)h1) * w2);
		cv::resize(im1, im1, cv::Size(w1, h1));
	}
	else {
		h2 = h1;
		w2 = int(((float)h1 / (float)h2) * w1);
		cv::resize(im2, im2, cv::Size(w2, h2));
	}
	cv::hconcat(im1, im2, cat);

	return cat.clone();
}

inline cv::Mat vconcat_ex(cv::Mat& im1, cv::Mat& im2)
{
	cv::Mat cat;
	int h1 = im1.rows;
	int w1 = im1.cols;
	int h2 = im2.rows;
	int w2 = im2.cols;
	if (h1 < h2)
	{
		w1 = w2;
		h1 = int((w2 / w1) * h2);
		cv::resize(im1, im1, cv::Size(w1, h1));
	}
	else {
		w2 = w1;
		h2 = int((w1 / w2) * h1);
		cv::resize(im2, im2, cv::Size(w2, h2));
	}
	cv::vconcat(im1, im2, cat);

	return cat.clone();
}

inline  void draw_face(cv::Mat& temp, rectangle& rect, std::vector<image_window::overlay_line>& line, std::vector<image_window::overlay_circle>& circle)
{
	cv::rectangle(temp,
		cv::Point(rect.bl_corner().x(), rect.bl_corner().y()),
		cv::Point(rect.tr_corner().x(), rect.tr_corner().y()),
		cv::Scalar(0, 0, 200));

	for (int i = 0; i < line.size(); i++)
	{
		cv::line(temp,
			cv::Point(line[i].p1.x(), line[i].p1.y()),
			cv::Point(line[i].p2.x(), line[i].p2.y()),
			cv::Scalar(line[i].color.red, line[i].color.green, line[i].color.blue));
	}
	for (int i = 0; i < circle.size(); i++)
	{
		cv::circle(temp,
			cv::Point(circle[i].center.x(), circle[i].center.y()),
			circle[i].radius,
			cv::Scalar(circle[i].color.red, circle[i].color.green, circle[i].color.blue));
	}
}


inline std::string getFilename(const char* name, std::string& pathname, std::string& extname)
{
	std::string fullpath = std::string(name);
	int path_i = 0;
	if (strchr(fullpath.c_str(), '\\')) path_i = fullpath.find_last_of("\\") + 1;//7
	else path_i = fullpath.find_last_of("/") + 1;//7

	int ext_i = fullpath.find_last_of(".");//10
	pathname = fullpath.substr(0, path_i + 1);//0•¶Žš–Ú‚©‚ç‚V•¶ŽšØ‚èo‚· "C:\\aaa\\"
	extname = fullpath.substr(ext_i, fullpath.size() - ext_i); // 10•¶Žš–Ú‚©‚ç‚S•¶ŽšØ‚èo‚· ".txt"
	std::string filename = fullpath.substr(path_i, ext_i - path_i);// ‚V•¶Žš–Ú‚©‚ç‚R•¶ŽšØ‚èo‚·@"bbb"

	return filename;
}

inline std::string getFilename(std::string& name, std::string& pathname, std::string& extname)
{
	return getFilename(name.c_str(), pathname, extname);
}

inline std::string getUserName(const char* name)
{
	std::string pathname;
	std::string extname;
	std::string& fname = getFilename(name, pathname, extname);

	char _name[256];
	strcpy(_name, fname.c_str());

	char* p = _name;
	int n = strlen(p);
	if ( n <= 1)return std::string(_name);

	for (int i = n - 1; i >= 0; i--)
	{
		if (p[i] == '_')
		{
			p[i] = '\0';
			break;
		}
	}
	return std::string(_name);
}

inline void _putText(cv::Mat& img, const cv::String& text, const cv::Point& org, const char* fontname, double fontScale, cv::Scalar color)
{
	int fontSize = (int)(10 * fontScale); // 10 is suitable
	int width = img.cols;
	int height = fontSize * 3 / 2; // Height is 1.5 times the font size

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
		imagelist.push_back("images/" + std::string(buf));
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
		if (buf[0] == '\0') continue;
		shapelist.push_back("user_shape/" + std::string(buf));
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

inline int load_shapelist(std::vector<std::string>& shapelist, std::vector<std::vector<float>>& shapevalue_list)
{
	if (shapelist.size() == 0)
	{
		if (!get_shapelist(shapelist)) return -1;

		printf("target users=%d\n", shapelist.size());
	}

	if (shapevalue_list.size() == 0)
	{
		printf("loading[shape features]...");
		shapevalue_list = get_shapevalue_list(shapelist);
		printf("number of users=%d\n", shapevalue_list.size());
	}
	return 0;
}

inline std::vector<full_object_detection> face_shape_predictor(dlib::array2d<bgr_pixel>& img, std::vector<rectangle>& dets, shape_predictor& sp);
inline std::vector<image_window::overlay_circle> render_face_detections2(
	const std::vector<full_object_detection>& dets,
	int& image_error,
	const rgb_pixel color = rgb_pixel(255, 255, 0)
)
{
	image_error = 0;
	std::vector<image_window::overlay_circle> circles;
	for (unsigned long i = 0; i < dets.size(); ++i)
	{
		if (dets[i].num_parts() != 68)
		{
			image_error = -1;
			return std::vector<image_window::overlay_circle>();
		}

		const full_object_detection& d = dets[i];

		{
			//// Around Chin. Ear to Ear
			circles.push_back(image_window::overlay_circle(d.part(0), 2, color));
			circles.push_back(image_window::overlay_circle(d.part(16), 2, color));
			circles.push_back(image_window::overlay_circle(d.part(8), 2, color));

			circles.push_back(image_window::overlay_circle(d.part(28), 2, color));
			circles.push_back(image_window::overlay_circle(d.part(19), 4, color));
			circles.push_back(image_window::overlay_circle(d.part(24), 4, color));
			circles.push_back(image_window::overlay_circle(d.part(30), 4, color));

			int xv = d.part(30).x() - d.part(27).x();
			if (std::abs(xv) > 14)
			{
				printf("Please straighten your face\n");
				image_error = -1;
				if (std::abs(xv) > 20)
				{
					image_error = -2;
				}
			}

			int xv1 = d.part(0).x() - d.part(36).x();
			int xv2 = d.part(45).x() - d.part(16).x();
			if (std::abs(xv1 - xv2) > 25)
			{
				printf("Please straighten your face\n");
				image_error = -3;
			}
			xv1 = d.part(3).x() - d.part(48).x();
			xv2 = d.part(54).x() - d.part(13).x();
			if (std::abs(xv1 - xv2) > 25)
			{
				printf("Please straighten your face\n");
				image_error = -4;
			}
			xv1 = d.part(62).y() - d.part(66).y();
			if (std::abs(xv1) > 10)
			{
				printf("Please close your mouth\n");
				image_error = -5;
			}


			dlib::point p = d.part(36);
			for (unsigned long i = 37; i <= 41; ++i)
			{
				p += d.part(i);
			}
			p = p / 6.0;
			circles.push_back(image_window::overlay_circle(p, 4, color));

			float d1 = std::abs(p.x() - d.part(0).x());

			p = d.part(42);
			for (unsigned long i = 43; i <= 47; ++i)
			{
				p += d.part(i);
			}
			p = p / 6.0;
			circles.push_back(image_window::overlay_circle(p, 4, color));
			float d2 = std::abs(p.x() - d.part(16).x());

			if (fabs(d1 - d2) > 50)
			{
				printf("Please make your face front\n");
				image_error = -6;
			}
			circles.push_back(image_window::overlay_circle(d.part(48), 4, color));
			circles.push_back(image_window::overlay_circle(d.part(54), 4, color));
			p = d.part(61);
			for (unsigned long i = 62; i <= 67; ++i)
			{
				p += d.part(i);
			}
			p = p / 7.0;
			circles.push_back(image_window::overlay_circle(p, 4, color));
		}
	}
	return circles;
}
inline bool face_dir_check(cv::Mat& face, frontal_face_detector detector, shape_predictor sp68)
{
	if (!dnn_face_recognition_::face_chk) return true;

	dlib::array2d<bgr_pixel> img;
	assign_image(img, cv_image<bgr_pixel>(face));

	std::vector<rectangle> dets = detector(img);
	if (dets.size() == 0) return false;

	std::vector<full_object_detection> shapes = face_shape_predictor(img, dets, sp68);
	cout << shapes.size() << endl;
	if (shapes.size() != 1) return false;

	int error_code;
	render_face_detections2(shapes, error_code);

	return (error_code == 0);
}

inline float distance(std::vector<float>& v1, std::vector<float>& v2)
{
	float s = 0.0;
	for (int j = 0; j < 128; j++)
	{
		s += (v1[j] - v2[j])*(v1[j] - v2[j]);
	}
	return s;
}

inline float cos_distance(std::vector<float>& v1, std::vector<float>& v2)
{
	float cos_dist = 0.0;
	for (int j = 0; j < 128; j++)
	{
		cos_dist += v1[j] * v2[j];
	}
	return cos_dist;
}

inline float distance(std::vector<float>& v, std::vector<std::vector<float>>& shapevalue_list, int& id, float& cos_dist)
{
	float mindist = 9999999999.0;
	id = -1;

	const size_t sz = shapevalue_list.size();

	std::vector<float> dist(sz);
	std::vector<float> cos_dst(sz);
#pragma omp parallel for
	for (int i = 0; i < sz; i++)
	{
		dist[i] = distance(v, shapevalue_list[i]);
	}

	for (int i = 0; i < sz; i++)
	{
		if (dist[i] < mindist)
		{
			mindist = dist[i];
			id = i;
		}
	}

	float d1 = cos_distance(v, v);
	float d2 = cos_distance(shapevalue_list[id], shapevalue_list[id]);
	cos_dist = cos_distance(v, shapevalue_list[id]) / sqrt(d1*d2);

	return mindist;
}

inline std::string face_recognition(cv::Mat& face_image, frontal_face_detector detector, shape_predictor sp, anet_type net, std::vector<std::string>& shapelist, std::vector<std::vector<float>>& shapevalue_list, float& dist, float& cos_dist)
{
	try
	{
		if (shapelist.size() == 0)
		{
			if (load_shapelist(shapelist, shapevalue_list) != 0)
			{
				return "unknown";
			}
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

		int id = -1;
		float mindist = distance(v, shapevalue_list, id, cos_dist);
		//for (int i = 0; i < shapevalue_list.size(); i++)
		//{
		//	float s = 0.0;
		//	for (int j = 0; j < 128; j++)
		//	{
		//		s += (v[j] - shapevalue_list[i][j])*(v[j] - shapevalue_list[i][j]);
		//	}
		//	//printf("  (%d)%f\n", i, s);
		//	if (s < mindist)
		//	{
		//		mindist = s;
		//		id = i;
		//	}
		//}
		printf("%f\n", mindist);
		dist = mindist;

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

inline std::string webcam_face_recognition(frontal_face_detector detector, shape_predictor sp, shape_predictor sp68, anet_type net, std::vector<std::string>& shapelist, std::vector<std::vector<float>>& shapevalue_list, int camID = 0)
{
	try
	{
		cv::VideoCapture cap(camID);
		if (!cap.isOpened())
		{
			cerr << "Unable to connect to camera" << endl;
			return "unknown";
		}
		if (shapelist.size() == 0)
		{
			if (load_shapelist(shapelist, shapevalue_list) != 0)
			{
				return "unknown";
			}
		}


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
			if (!face_dir_check(temp, detector, sp68))
			{
				printf("You are not facing the front or you can see multiple people.\n");
				count++;
				continue;
			}

			float dist;
			float cos_dist;
			std::string user = face_recognition(temp, detector, sp, net, shapelist, shapevalue_list, dist, cos_dist);

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
		deserialize("db/shape_predictor_5_face_landmarks.dat") >> sp;
		// And finally we load the DNN responsible for face recognition.
		anet_type net;
		deserialize("db/dlib_face_recognition_resnet_model_v1.dat") >> net;

		shape_predictor sp68;
		deserialize("db/shape_predictor_68_face_landmarks.dat") >> sp68;

		std::vector<std::string> shapelist;
		std::vector<std::vector<float>> shapevalue_list;
		if (shapelist.size() == 0)
		{
			if (load_shapelist(shapelist, shapevalue_list) != 0)
			{
				return "unknown";
			}
		}

		webcam_face_recognition(detector, sp, sp68, net, shapelist, shapevalue_list, camID);
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
	deserialize("db/shape_predictor_5_face_landmarks.dat") >> sp;
	anet_type net;
	deserialize("db/dlib_face_recognition_resnet_model_v1.dat") >> net;

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
			sprintf(clipped, "user_images/%s.png", filename.c_str());
			cv::Mat x = dlib::toMat(tile_images(temp)).clone();
			cv::cvtColor(x, x, CV_RGB2BGR);
			cv::imwrite(clipped, x);
			break;
		}
		//cout << "face descriptor for one face: " << trans(face_descriptors[0]) << endl;

		matrix<float, 0, 1> face_descriptor = mean(mat(net(jitter_image(faces[0]))));
		cout << "jittered face descriptor for one face: " << trans(face_descriptor) << endl;

		char user_shape[256];
		sprintf(user_shape, "user_shape/%s.txt", filename.c_str());
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

inline std::vector<full_object_detection> face_shape_predictor(dlib::array2d<bgr_pixel>& img, std::vector<rectangle>& dets, shape_predictor& sp)
{
	frontal_face_detector detector = get_frontal_face_detector();


	// Now we will go ask the shape_predictor to tell us the pose of
	// each face we detected.
	std::vector<full_object_detection> shapes;
	//cout << "number of dets: " << dets.size() << endl;
	for (unsigned long j = 0; j < dets.size(); ++j)
	{
		full_object_detection shape = sp(img, dets[j]);
		//cout << "number of parts: " << shape.num_parts() << endl;
		//cout << "pixel position of first part:  " << shape.part(0) << endl;
		//cout << "pixel position of second part: " << shape.part(1) << endl;

		//for (int k = 2; k < shape.num_parts(); k++)
		//{
		//	cout << "pixel position of (" << k << "): " << shape.part(k) << endl;
		//}
		// You get the idea, you can get all the face part locations if
		// you want them.  Here we just store them in shapes so we can
		// put them on the screen.
		shapes.push_back(shape);
	}
	return shapes;
}




inline int cam2face_shape(char* user, int camID = 0)
{
	int error_code = 0;
	int count = 0;
	try
	{
		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor sp68;
		deserialize("db/shape_predictor_68_face_landmarks.dat") >> sp68;


		cv::VideoCapture cap(camID);
		if (!cap.isOpened())
		{
			cerr << "Unable to connect to camera" << endl;
			return 1;
		}

		while (true)
		{
			error_code = 0;
			cv::Mat temp;
			cap >> temp;

			if (temp.empty() == true) {
				break;
			}
			if (temp.size().width <= 640 || temp.size().height <= 640)
			{
				float a = 640.0 / temp.size().width;
				float b = 640.0 / temp.size().height;

				if (b > a) a = b;
				cv::resize(temp, temp, cv::Size(temp.size().width*a, temp.size().height*a), 0, 0, cv::INTER_CUBIC);
			}

			dlib::array2d<bgr_pixel> img;
			assign_image(img, cv_image<bgr_pixel>(temp));

			std::vector<rectangle> dets = detector(img);
			cout << "Number of faces detected: " << dets.size() << endl;
			if (dets.size() != 1) continue;

			std::vector<full_object_detection> shapes = face_shape_predictor(img, dets, sp68);
			cout << shapes.size() << endl;
			if (shapes.size() == 0) continue;
			//for (int i = 0; i < shapes.size(); i++)
			//{
			//	for (int k = 0; k < shapes[i].num_parts(); k++)
			//	{
			//		cout << "pixel position of (" << k << "): " << shapes[i].part(k) << endl;
			//	}
			//}

			std::vector<image_window::overlay_circle>& circle = render_face_detections2(shapes, error_code);

			// Now let's view our face poses on the screen.
			draw_face(temp, dets[0], render_face_detections(shapes), circle);
			cout << "render_face_detections(shapes): " << render_face_detections(shapes).size() << endl;

			cv::imshow("render_face_detections", temp);
			//cv::waitKey(1);

			temp = dlib::toMat(img);
			try
			{
				cv::Mat roi_img(temp,
					cv::Rect(
						dets[0].tl_corner().x(), dets[0].tl_corner().y(),
						dets[0].width(), dets[0].height()
					)
				);
				cv::imshow("cap", roi_img);
				if (error_code == 0)
				{
					char capimage_file[256];
					sprintf(capimage_file, "capture/%s_%03d.png", user, count);
					count++;
					cv::imwrite(capimage_file, roi_img);
					if ( count == 10 ) return error_code;
				}
			}
			catch (...)
			{
				//
			}
			cv::waitKey(1);
		}
	}
	catch (exception& e)
	{
		cout << "\nexception thrown!" << endl;
		cout << e.what() << endl;
	}
	return error_code;
}


