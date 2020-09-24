#pragma once

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
#include <opencv2/dnn.hpp>

#include "config.h"
#include "opencv_util.h"
#include "dlib_util.h"
#include "putText_Jpn/putText_Jpn.h"
#include <filesystem>

using namespace dlib;
using namespace std;

#define NUM_JITTERS			100
#define FEATURE_VECTOR_DIM	128
#define UNKNOWON_FACE_ID	-1
#define UNKNOWON_FACE_NAME "unknowon"

#define CHK_FACE_IMAGE_SIZE	300
#define GAP_DELTA_SIZE		8
// ----------------------------------------------------------------------------------------

template <long num_filters, typename SUBNET> using con5d = con<num_filters, 5, 5, 2, 2, SUBNET>;
template <long num_filters, typename SUBNET> using con5 = con<num_filters, 5, 5, 1, 1, SUBNET>;

template <typename SUBNET> using downsampler = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16, SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5 = relu<affine<con5<45, SUBNET>>>;

using net_type = loss_mmod<con<1, 9, 9, 1, 1, rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

// ----------------------------------------------------------------------------------------


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
	int no_show = 0;
	int face_chk = 0;
	bool one_person = false;
	bool tracking = true;
	float collation_judgmentthreshold = 0.2;
	std::string video_file = "";

	int dnn_face_detection = 0;
	int num_jitters = NUM_JITTERS;
};

inline std::vector<std::string> getFiles(std::string& dir, std::string ext1 = std::string(""), std::string ext2 = std::string(""))
{
	std::vector<std::string> file_paths;

	for (const std::filesystem::directory_entry& de : std::filesystem::recursive_directory_iterator(dir)) 
	{	
		if (ext1 == "" && ext2 == "")
		{
			file_paths.emplace_back(de.path().string());
			continue;
		}else
		if (ext1 != "" && ext2 == "")
		{
			if (strstr(de.path().string().c_str(), ext1.c_str()))
			{
				file_paths.emplace_back(de.path().string());
			}
			continue;
		}
		if (ext1 != "" && ext2 != "")
		{
			if (strstr(de.path().string().c_str(), ext1.c_str()) || strstr(de.path().string().c_str(), ext2.c_str()))
			{
				file_paths.emplace_back(de.path().string());
			}
			continue;
		}
	}

	return file_paths;
}

inline void draw_face_rects(cv::Mat& image, const rectangle& rect, cv::Scalar& bgr, const std::string& name = std::string(""))
{
	long dy = (float)(rect.tr_corner().y() - rect.bl_corner().y()) / 6.0;

	try
	{
		cv::rectangle(image,
			cv::Point(rect.bl_corner().x(), rect.bl_corner().y()),
			cv::Point(rect.tr_corner().x(), rect.tr_corner().y() + dy),
			bgr, 2);

		//printf("[%s]\n", name.c_str());
		if (name != "")
		{
			cv::rectangle(image,
				cv::Point(rect.bl_corner().x(), rect.tr_corner().y() + dy + 15),
				cv::Point(rect.tr_corner().x(), rect.tr_corner().y() + dy),
				cv::Scalar(100, 0, 0), -1, CV_AA);

			if (name == UNKNOWON_FACE_NAME)
			{
				sc::myCV::putText_Jpn(image, name.c_str(), cv::Point(rect.bl_corner().x(), rect.tr_corner().y() + dy + 10), std::string("ÇlÇr ÉSÉVÉbÉN").c_str(), 0.3, cv::Scalar(255, 255, 255), 3);
			}
			else
			{
				sc::myCV::putText_Jpn(image, name.c_str(), cv::Point(rect.bl_corner().x(), rect.tr_corner().y() + dy + 10), std::string("ÇlÇr ÉSÉVÉbÉN").c_str(), 0.3, cv::Scalar(0, 255, 255), 3);
			}
		}
	}
	catch (std::exception& e)
	{
		cout << e.what() << endl;
	}
	catch (...)
	{

	}
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
	if (strchr(name, '\\') == NULL && strchr(name, '/') == NULL)
	{
		return std::string(name);
	}
	std::string fullpath = std::string(name);
	int path_i = 0;
	if (strchr(fullpath.c_str(), '\\')) path_i = fullpath.find_last_of("\\") + 1;
	else path_i = fullpath.find_last_of("/") + 1;

	int ext_i = fullpath.find_last_of(".");
	pathname = fullpath.substr(0, path_i + 1);
	extname = fullpath.substr(ext_i, fullpath.size() - ext_i);
	std::string filename = fullpath.substr(path_i, ext_i - path_i);

	return filename;
}

inline std::string getFilename(std::string& name, std::string& pathname, std::string& extname)
{
	return getFilename(name.c_str(), pathname, extname);
}

inline std::string getUserName(const char* name)
{
	printf("%s\n", name);
	if (std::string(name) == std::string(UNKNOWON_FACE_NAME))
	{
		return std::string(name);
	}

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

inline void running_break_clear()
{
	FILE* fp = fopen("break.run", "r");
	if (fp)
	{
		fclose(fp);
		remove("break.run");
	}
}
inline bool running_break()
{
	FILE* fp = fopen("break.run", "r");
	if (fp)
	{
		fclose(fp);
		remove("break.run");
		return true;
	}
	return false;
}

int image_count = 0;
inline void imgwrite_(const cv::Mat& img)
{
	char buf[256];
	sprintf(buf, "tmp/tmp_%05d.png", image_count++);
	cv::imwrite(buf, img);
}
inline void clear_tmp_img()
{
	image_count = 0;
	int count = 0;

	char buf[256];
	do {
#ifdef _WIN64
		sprintf(buf, "tmp\\tmp_%05d.png", count);
#else
		sprintf(buf, "tmp/tmp_%05d.png", image_count);
#endif
		FILE* fp = fopen(buf, "r");
		if (fp)
		{
			fclose(fp);
			remove(buf);
		}
		else break;

		count++;
	} while (1);

#ifdef _WIN64
	FILE* fp = fopen("tmp\\end_image", "r");
#else
	FILE* fp = fopen("tmp/end_image", "r");
#endif
	if (fp)
	{
		fclose(fp);
#ifdef _WIN64
		sprintf(buf, "tmp\\end_image", image_count);
#else
		sprintf(buf, "tmp/end_image", image_count);
#endif
	}
}
inline void end_tmp_img()
{
#ifdef _WIN64
	FILE* fp = fopen("tmp\\end_image", "r");
#else
	FILE* fp = fopen("tmp/end_image", "r");
#endif
	if (fp)
	{
		fclose(fp);
		remove("tmp/end_image");
	}
}


class face_recognition_str
{
public:
	cv::Mat					face_image;
	std::string				video_file;
	frontal_face_detector	detector;
	shape_predictor			sp;
	shape_predictor			sp68;
	anet_type				net;
	int						num_jitters = NUM_JITTERS;
	std::vector<std::string> shapelist;
	std::vector<std::vector<float>> shapevalue_list;
	std::vector<float> dist;
	std::vector<float> cos_dist;
	std::vector<rectangle> rects;

	std::vector<int> result_id;

	///////////////////////
	net_type detect_net;

	///////////////////////
	cv::dnn::Net res10net;

	void reset()
	{
		result_id.clear();
		dist.clear();
		cos_dist.clear();
		rects.clear();
	}
	face_recognition_str()
	{
		detector = get_frontal_face_detector();
		deserialize("model/shape_predictor_5_face_landmarks.dat") >> sp;
		deserialize("model/dlib_face_recognition_resnet_model_v1.dat") >> net;
		deserialize("model/shape_predictor_68_face_landmarks.dat") >> sp68;

		if (dnn_face_recognition_::dnn_face_detection == 1)
		{
			deserialize("model/mmod_human_face_detector.dat") >> detect_net;
		}
		if (dnn_face_recognition_::dnn_face_detection == 2)
		{
			cv::String modelConfiguration = "model/res10/deploy.prototxt";
			cv::String modelBinary = "model/res10/res10_300x300_ssd_iter_140000.caffemodel";

			res10net = cv::dnn::readNetFromCaffe(modelConfiguration, modelBinary);
			if (res10net.empty())
			{
				cerr << "Can't load network by using the following files: " << endl;
				cerr << "prototxt:   " << modelConfiguration << endl;
				cerr << "caffemodel: " << modelBinary << endl;
				exit(-1);
			}
#ifdef USE_GPU
			res10net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
#else
			res10net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
#endif
		}
	}

	inline int init()
	{
		if (shapelist.size() == 0)
		{
			if (load_shapelist() != 0)
			{
				return -1;
			}
		}
		return 0;
	}

	inline int result(const std::string filename) const
	{
		FILE* fp = fopen(filename.c_str(), "w");
		if (fp == NULL)
		{
			printf("file[%s]open error.\n", filename.c_str());
		}
		fprintf(fp, "%d\n", result_id.size());
		for (int i = 0; i < result_id.size(); i++)
		{
			std::string user_name = "unknown";
			if (result_id[i] >= 0)
			{
				user_name = shapelist[result_id[i]];
				std::string pathname;
				std::string extname;
				user_name = getFilename(shapelist[result_id[i]].c_str(), pathname, extname);
			}

			if (rects.size() > 0)
			{
				fprintf(fp, "%s	dist=	%.4f	cos=	%.4f	rect=	%d	%d	%d	%d\n",
					user_name.c_str(), dist[i], cos_dist[i],
					rects[i].bl_corner().x(), rects[i].bl_corner().y(),
					rects[i].tr_corner().x(), rects[i].tr_corner().y());
			}
			else
			{
				fprintf(fp, "%s	dist=	%.4f	cos=	%.4f	rect=	NA	NA	NA	NA\n",
					user_name.c_str(), dist[i], cos_dist[i]);
			}
			printf("%s	dist=	%.4f	cos=	%.4f\n",
				user_name.c_str(), dist[i], cos_dist[i]);
		}
		fclose(fp);
		return 0;
	}
	inline int get_imagelist(std::vector<std::string>& imagelist)
	{
#ifdef _HAS_CXX17
		imagelist = getFiles(std::string("./images"), ".png", ".jpg");
#else
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
#endif
		return 0;
	}

	inline int get_shapelist()
	{
#ifdef _HAS_CXX17
		shapelist = getFiles(std::string("./user_shape"), ".txt");
#else
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
#endif
		return 1;
	}

	inline std::vector<float> get_shapevalue(std::string& vector_file)
	{
		std::vector<float> v;

		FILE* fp = fopen(vector_file.c_str(), "r");
		if (fp)
		{
			char buf[64];
			for (int j = 0; j < FEATURE_VECTOR_DIM; j++)
			{
				fgets(buf, 64, fp);
				v.push_back(atof(buf));
			}
			fclose(fp);
		}

		return v;
	}

	inline std::vector<std::vector<float>> get_shapevalue_list()
	{
		int n = shapelist.size() / 80;
		for (int i = 0; i < shapelist.size(); i++)
		{
			printf("                                                             \r");
			printf("%d/%d", i, shapelist.size() - 1);
			shapevalue_list.push_back(get_shapevalue(shapelist[i]));
		}
		printf("                                                             \r");
		printf("\ndone.\n");
		return shapevalue_list;
	}

	inline int load_shapelist()
	{
		if (shapelist.size() == 0)
		{
			if (!get_shapelist()) return -1;

			printf("target users=%d\n", shapelist.size());
		}

		if (shapevalue_list.size() == 0)
		{
			printf("loading[shape features]...");
			shapevalue_list = get_shapevalue_list();
			printf("number of users=%d\n", shapevalue_list.size());
		}
		return 0;
	}
};

inline std::vector<matrix<rgb_pixel>> jitter_image(
	const matrix<rgb_pixel>& img,  int num_jitters = 100
)
{
	// All this function does is make 100 copies of img, all slightly jittered by being
	// zoomed, rotated, and translated a little bit differently. They are also randomly
	// mirrored left to right.
	thread_local dlib::rand rnd;

	std::vector<matrix<rgb_pixel>> crops;
	if (num_jitters == 0)
	{
		crops.push_back(img);
		return crops;
	}
	for (int i = 1; i < 100; ++i)
	{
		crops.push_back(jitter_image(img, rnd));
	}
	return crops;
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
			if (std::abs(xv) > 4* GAP_DELTA_SIZE)
			{
				printf("Please straighten your face\n");
				image_error = -1;
				if (std::abs(xv) > 6* GAP_DELTA_SIZE)
				{
					image_error = -2;
				}
			}

			int xv1 = d.part(0).x() - d.part(36).x();
			int xv2 = d.part(45).x() - d.part(16).x();
			if (std::abs(xv1 - xv2) > 4* GAP_DELTA_SIZE)
			{
				printf("Please straighten your face\n");
				image_error = -3;
			}
			xv1 = d.part(3).x() - d.part(48).x();
			xv2 = d.part(54).x() - d.part(13).x();
			if (std::abs(xv1 - xv2) > 4* GAP_DELTA_SIZE)
			{
				printf("Please straighten your face\n");
				image_error = -4;
			}
			xv1 = d.part(62).y() - d.part(66).y();
			if (std::abs(xv1) > 2* GAP_DELTA_SIZE)
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

			if (fabs(d1 - d2) > 5* GAP_DELTA_SIZE)
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
inline bool face_dir_check(cv::Mat& face, frontal_face_detector detector, shape_predictor sp68, cv::Mat& out)
{
	if (!dnn_face_recognition_::one_person) return true;
	if (!dnn_face_recognition_::face_chk) return true;

	cv::Mat face_tmp = face.clone();
	opencv_util::resize_padd(face_tmp, CHK_FACE_IMAGE_SIZE);

	dlib::array2d<bgr_pixel> img;
	assign_image(img, cv_image<bgr_pixel>(face_tmp));

	std::vector<rectangle> dets = detector(img);
	if (dets.size() == 0) return false;

	std::vector<full_object_detection> shapes = face_shape_predictor(img, dets, sp68);
	cout << shapes.size() << endl;
	if (shapes.size() != 1) return false;

	int error_code;
	std::vector<image_window::overlay_circle>& circle = render_face_detections2(shapes, error_code);

#if 10
	{
		draw_face(face_tmp, dets[0], render_face_detections(shapes), circle);
		if (error_code != 0)
		{
			cv::cvtColor(face_tmp, face_tmp, CV_RGB2BGR);
			out = face_tmp.clone();
			//cv::imshow("", out);
			//cv::waitKey(10000);
		}
	}
#endif

	return (error_code == 0);
}
inline bool face_dir_check(matrix<rgb_pixel>& img, frontal_face_detector detector, shape_predictor sp68, cv::Mat& out)
{
	cv::Mat face = toMat(img).clone();
	return face_dir_check(face, detector, sp68, out);
}

inline float distance(const std::vector<float>& v1, const std::vector<float>& v2)
{
	float s = 0.0;
	for (int j = 0; j < FEATURE_VECTOR_DIM; j++)
	{
		s += (v1[j] - v2[j])*(v1[j] - v2[j]);
	}
	return s;
}

inline float cos_distance(const std::vector<float>& v1, const std::vector<float>& v2)
{
	float cos_dist = 0.0;
	for (int j = 0; j < FEATURE_VECTOR_DIM; j++)
	{
		cos_dist += v1[j] * v2[j];
	}
	return cos_dist;
}

inline float distance(const std::vector<float>& v, const std::vector<std::vector<float>>& shapevalue_list, int& id, float& cos_dist)
{
	float mindist = 9999999999.0;
	id = -1;

	const size_t sz = shapevalue_list.size();

	std::vector<float> dist(sz);
	std::vector<float> cos_dst(sz);
	float d1, d2;
#pragma omp parallel
	{
#pragma omp parallel for
		for (int i = 0; i < sz; i++)
		{
			dist[i] = distance(v, shapevalue_list[i]);
		}
#pragma omp barrier

#pragma omp parallel for num_threads(4)
		for (int i = 0; i < sz; i++)
		{
			if (dist[i] < mindist)
			{
#pragma omp critical
				{
					mindist = dist[i];
					id = i;
				}
			}
		}
#pragma omp barrier

#pragma omp sections
		{
			#pragma omp section
			{
				d1 = cos_distance(v, v);
			}
			#pragma omp section
			{
				d2 = cos_distance(shapevalue_list[id], shapevalue_list[id]);
			}
		}
	}
	cos_dist = cos_distance(v, shapevalue_list[id]) / sqrt(d1*d2);

	return mindist;
}

inline void draw_recgnition(cv::Mat& face_image, const std::vector<int>& user_id, const face_recognition_str& fr)
{
	for (int i = 0; i < user_id.size(); i++)
	{
		std::string user_name = UNKNOWON_FACE_NAME;
		if (user_id[i] >= 0) user_name = fr.shapelist[user_id[i]];
		//printf("user %s\n", user_name.c_str());

		cv::Scalar bgr(50, 255, 0);
		//if (fr.dist[i] > dnn_face_recognition_::collation_judgmentthreshold*0.5)
		//{
		//	bgr = cv::Scalar(12, 215, 243);
		//}
		//if (fr.dist[i] > dnn_face_recognition_::collation_judgmentthreshold*0.7)
		//{
		//	bgr = cv::Scalar(0, 128, 255);
		//}
		if (fr.dist[i] > dnn_face_recognition_::collation_judgmentthreshold*0.95)
		{
			bgr = cv::Scalar(0, 0, 255);
		}
		if (fr.dist[i] < 0.001)
		{
			bgr = cv::Scalar(255, 115, 0);
		}

		std::string name = UNKNOWON_FACE_NAME;
		if (user_name == UNKNOWON_FACE_NAME) bgr = cv::Scalar(128, 128, 128);
		else
		{
			std::string pathname;
			std::string extname;
			name = getFilename(user_name, pathname, extname);
			//printf("name:%s\n", name.c_str());
			name = getUserName(name.c_str());
		}
		draw_face_rects(face_image, fr.rects[i], bgr, name);
	}
}

inline std::vector<int> face_compare(face_recognition_str& fr, std::vector < std::vector<float>>& recog_faces_fvector)
{
	std::vector<int>& recog_faces = fr.result_id;
	for (int i = 0; i < recog_faces_fvector.size(); i++)
	{
		printf(" shapevalue_list.size()=%d\n", fr.shapevalue_list.size());
		printf(" recog_faces_fvector.size()=%d\n", recog_faces_fvector.size());
		int id = -1;
		float mindist = distance(recog_faces_fvector[i], fr.shapevalue_list, id, fr.cos_dist[i]);
		printf("id:%d %f\n", id, mindist);
		fr.dist[i] = mindist;

		printf("%f\n", dnn_face_recognition_::collation_judgmentthreshold);
		if (id >= 0 && fr.dist[i] < dnn_face_recognition_::collation_judgmentthreshold)
		{
			recog_faces.push_back(id);
		}
		else
		{
			recog_faces.push_back(UNKNOWON_FACE_ID);
		}
	}
	return recog_faces;
}

inline std::vector<int> face_recognition(face_recognition_str& face_recog_image)
{
	face_recog_image.reset();

	printf("Verifying the same person...\n"); fflush(stdout);
	std::vector<int>& recog_faces = face_recog_image.result_id;
	recog_faces.clear();
	try
	{
		if (face_recog_image.init() != 0)
		{
			recog_faces.push_back(UNKNOWON_FACE_ID);
			return recog_faces;
		}


		if (face_recog_image.face_image.empty() == true) {
			cout << "No image!" << endl;
			return recog_faces;
		}
		if (face_recog_image.face_image.channels() == 4)
			cvtColor(face_recog_image.face_image, face_recog_image.face_image, cv::COLOR_BGRA2BGR);

		dlib::array2d<bgr_pixel> img;
		assign_image(img, cv_image<bgr_pixel>(face_recog_image.face_image));

		face_recog_image.rects.clear();
		std::vector<matrix<rgb_pixel>> faces;

		if (dnn_face_recognition_::dnn_face_detection == 2)
		{
			//res10_300x300_ssd
			const size_t inWidth = 300;
			const size_t inHeight = 300;
			const double inScaleFactor = 1.0;
			const cv::Scalar meanVal(104.0, 177.0, 123.0);

			cv::Mat inputBlob = cv::dnn::blobFromImage(face_recog_image.face_image, inScaleFactor,
				cv::Size(inWidth, inHeight), meanVal, false, false); 
			face_recog_image.res10net.setInput(inputBlob, "data");
			
			cv::Mat detection = face_recog_image.res10net.forward("detection_out");
			
			cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
			const float confidenceThreshold = 0.5;

			for (int i = 0; i < detectionMat.rows; i++)
			{
				const float confidence = detectionMat.at<float>(i, 2);

				if (confidence > confidenceThreshold)
				{
					const int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * face_recog_image.face_image.cols);
					const int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * face_recog_image.face_image.rows);
					const int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * face_recog_image.face_image.cols);
					const int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * face_recog_image.face_image.rows);
					
					cv::Rect cvrect((int)xLeftBottom, (int)yLeftBottom,
						(int)(xRightTop - xLeftBottom),
						(int)(yRightTop - yLeftBottom));

					dlib::rectangle face = dlib_util::openCVRectToDlib(cvrect);
#if 0
					face_recog_image.rects.push_back(face);
					cv::rectangle(face_recog_image.face_image, cvrect, cv::Scalar(0, 255, 0));

					cv::Mat cutimg(face_recog_image.face_image, cvrect);
					opencv_util::resize_padd(cutimg, 150);
					matrix<rgb_pixel> img2;
					assign_image(img2, cv_image<bgr_pixel>(cutimg));
					faces.push_back(img2);
#else
					face_recog_image.rects.push_back(face);
					auto shape = face_recog_image.sp(img, face);
					matrix<rgb_pixel> face_chip;
					extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
					faces.push_back(move(face_chip));
#endif
				}
			}
		}else
		if (dnn_face_recognition_::dnn_face_detection == 1)
		{
			matrix<rgb_pixel> img2;
			assign_image(img2, cv_image<bgr_pixel>(face_recog_image.face_image));

			auto dets = face_recog_image.detect_net(img2);
			for (auto&& face : dets)
			{
#if 0
				face_recog_image.rects.push_back(face.rect);
				cv::Rect cvrect = dlibRectangleToOpenCV(face.rect);
				cv::rectangle(face_recog_image.face_image, cvrect, cv::Scalar(0, 255, 0));

				cv::Mat cutimg(face_recog_image.face_image, cvrect);
				opencv_util::resize_padd(cutimg, 150);
				matrix<rgb_pixel> img2;
				assign_image(img2, cv_image<bgr_pixel>(cutimg));
				faces.push_back(img2);
#else
				face_recog_image.rects.push_back(face.rect);
				auto shape = face_recog_image.sp(img, face.rect);
				matrix<rgb_pixel> face_chip;
				extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
				faces.push_back(move(face_chip));
#endif
			}
		}
		else
		{
			for (auto face : face_recog_image.detector(img))
			{
#if 0
				face_recog_image.rects.push_back(face);
				cv::Rect cvrect = dlibRectangleToOpenCV(face);
				cv::rectangle(face_recog_image.face_image, cvrect, cv::Scalar(0, 255, 0));

				cv::Mat cutimg(face_recog_image.face_image, cvrect);
				opencv_util::resize_padd(cutimg, 150);
				matrix<rgb_pixel> img2;
				assign_image(img2, cv_image<bgr_pixel>(cutimg));
				faces.push_back(img2);
#else
				face_recog_image.rects.push_back(face);
				auto shape = face_recog_image.sp(img, face);
				matrix<rgb_pixel> face_chip;
				extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
				faces.push_back(move(face_chip));
#endif
			}
		}

		if (faces.size() == 0)
		{
			cout << "No faces found in image!" << endl;
			return recog_faces;
		}
		cv::Mat x = dlib::toMat(img).clone();
		//cv::cvtColor(x, x, CV_RGB2BGR);
		if (!dnn_face_recognition_::tracking)
		{
			if (!dnn_face_recognition_::no_show)
			{
				cv::imshow("", x);
				cv::waitKey(1);
			}
		}


		std::vector<matrix<float, 0, 1>> face_descriptors = face_recog_image.net(faces);
		std::vector < std::vector<float>> recog_faces_fvector;
		for (auto &face : faces)
		{
			matrix<float, 0, 1> face_descriptor = mat(face_recog_image.net(face));
			//matrix<float, 0, 1> face_descriptor = mean(mat(face_recog_image.net(jitter_image(face))));
			//cout << "jittered face descriptor for one face: " << trans(face_descriptor) << endl;
			recog_faces_fvector.push_back(std::vector<float>(face_descriptor.begin(), face_descriptor.end()));
		}

		face_recog_image.dist.resize(recog_faces_fvector.size());
		face_recog_image.cos_dist.resize(recog_faces_fvector.size());
		recog_faces = face_compare(face_recog_image, recog_faces_fvector);

	}
	catch (std::exception& e)
	{
		recog_faces.push_back(UNKNOWON_FACE_ID);
		cout << e.what() << endl;
	}
	catch (...)
	{
		recog_faces.push_back(UNKNOWON_FACE_ID);
	}
	printf("Verifying the same person done.\n"); fflush(stdout);

	return recog_faces;
}

inline std::vector<int> webcam_face_recognition(face_recognition_str& face_recog_image, int camID = 0)
{
	std::vector<int> users;
	try
	{
		cv::VideoCapture cap;

		if (dnn_face_recognition_::video_file != "")
		{
			cap.open(dnn_face_recognition_::video_file);
		}
		else
		{
			cap = cv::VideoCapture(camID);
		}
		if (!cap.isOpened())
		{
			cerr << "Unable to connect to camera" << endl;
			return users;
		}
		if (face_recog_image.init() != 0)
		{
			return users;
		}
///////////////////////////
		cv::VideoWriter writer;
		int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
		int    width = 0, height = 0;
		double fps = cap.get(cv::CAP_PROP_FPS);
		if (fps <= 0) fps = 5;
///////////////////////////

		int wait_time = 20;
		int count = 0;
		char time_count[32];
		while (true)
		{
			cv::Mat temp;
			cap >> temp;

			if (temp.empty() == true) {
				cout << "No image!" << endl;
				break;
			}

			temp = opencv_util::resize_ex(temp);
			if (width == 0)
			{
				std::string pathname;
				std::string extname;
				std::string& filename = getFilename(dnn_face_recognition_::video_file, pathname, extname);

				
				width = (int)temp.size().width;
				height = (int)temp.size().height;
				writer.open(filename+"_output.mp4", fourcc, fps, cv::Size(width, height));
			}

			if (dnn_face_recognition_::one_person)
			{
				sprintf(time_count, "count %d", wait_time - count);
				cv::Mat temp2 = temp.clone();
				cv::putText(temp2, time_count, cv::Point(50, 90), cv::FONT_HERSHEY_COMPLEX, 3, cv::Scalar(1, 1, 1), 3);
				if (!dnn_face_recognition_::no_show)
				{
					cv::imshow("", temp2);
					if (cv::waitKey(1) == 27) break;
				}else
				{
					imgwrite_(temp2);
					if (running_break()) break;
				}
				if (count < wait_time)
				{
					count++;
					continue;
				}
				cv::Mat errorImg;
				if (!face_dir_check(temp, face_recog_image.detector, face_recog_image.sp68, errorImg))
				{
					printf("You are not facing the front or you can see multiple people.\n");
					count++;
					writer << temp;
					if (dnn_face_recognition_::no_show)imgwrite_(temp);
					continue;
				}
			}

			face_recog_image.face_image = temp.clone();
			float dist;
			float cos_dist;
			std::vector<int>& user_id = face_recognition(face_recog_image);
			//face_recog_image.result("result.txt");

			if (user_id.size() == 0)
			{
				writer << temp;
				if (!dnn_face_recognition_::no_show)
				{
					cv::imshow("", temp);
					if (cv::waitKey(1) == 27) break;
				}
				else
				{
					imgwrite_(temp);
					if (running_break()) break;
				}
				continue;
			}
			if (dnn_face_recognition_::tracking)
			{
				draw_recgnition(temp, user_id, face_recog_image);
				writer << temp;
				if (!dnn_face_recognition_::no_show)
				{
					cv::imshow("", temp);
					if (cv::waitKey(1) == 27) break;
				}else
				{
					imgwrite_(temp);
					if (running_break()) break;
				}
				continue;
			}
			return users;
		}
		writer.release();
		cap.release();
	}
	catch (std::exception& e)
	{
		cout << e.what() << endl;
	}
	catch (...)
	{
	}
	return users;
}

inline std::string webcam_face_recognition(int camID = 0)
{
	try
	{
		face_recognition_str fr;

		if (fr.shapelist.size() == 0)
		{
			if (fr.load_shapelist() != 0)
			{
				return "unknown";
			}
		}

		webcam_face_recognition(fr, camID);
	}
	catch (std::exception& e)
	{
		cout << e.what() << endl;
	}
	catch (...)
	{
		return UNKNOWON_FACE_NAME;
	}
}

inline int make_shape(face_recognition_str& fr)
{
	std::vector<std::string> imagelist;
	if (fr.get_imagelist(imagelist) < 0) return -1;

	printf("imagelist:%d\n", imagelist.size());
	const size_t sz = imagelist.size();

//#pragma omp parallel for
	for (int id = 0; id < sz; id++)
	{
		matrix<rgb_pixel> img;
		load_image(img, imagelist[id]);

		cv::Mat error_img;
		if (!face_dir_check(img, fr.detector, fr.sp68, error_img))
		{
			printf("not facing the front or you can see multiple people\n");
			if (error_img.empty())
			{
				dlib::save_png(img, std::string("tmp/error_face_image_") + to_string(id) + ".png");
			}
			else
			{
				cv::imwrite(std::string("tmp/error_face_image_") + to_string(id) + ".png", error_img);
			}
			continue;
		}

		std::vector<matrix<rgb_pixel>> faces;
		for (auto face : fr.detector(img))
		{
			auto shape = fr.sp(img, face);
			matrix<rgb_pixel> face_chip;
			extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
			faces.push_back(move(face_chip));
		}

		if (faces.size() == 0)
		{
			cout << "No faces found in image!" << endl;
			continue;
		}

		std::vector<matrix<float, 0, 1>> face_descriptors = fr.net(faces);
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
		//std::vector<image_window> win_clusters(num_clusters);
		for (size_t cluster_id = 0; cluster_id < num_clusters; ++cluster_id)
		{
			std::vector<matrix<rgb_pixel>> temp;
			for (size_t j = 0; j < labels.size(); ++j)
			{
				if (cluster_id == labels[j])
					temp.push_back(faces[j]);
			}
			//win_clusters[cluster_id].set_title("face cluster " + cast_to_string(cluster_id));
			//win_clusters[cluster_id].set_image(tile_images(temp));

			char clipped[256];
			if (cluster_id == 0)
			{
				sprintf(clipped, "user_images/%s.png", filename.c_str());
			}
			else
			{
				sprintf(clipped, "user_images/%s_%04d.png", filename.c_str(), cluster_id);
			}
			cv::Mat x = dlib::toMat(tile_images(temp)).clone();
			cv::cvtColor(x, x, CV_RGB2BGR);
			cv::imwrite(clipped, x);
		}
		//cout << "face descriptor for one face: " << trans(face_descriptors[0]) << endl;

		matrix<float, 0, 1> face_descriptor = mean(mat(fr.net(jitter_image(faces[0]))));
		//cout << "jittered face descriptor for one face: " << trans(face_descriptor) << endl;

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
		deserialize("model/shape_predictor_68_face_landmarks.dat") >> sp68;

		cv::VideoCapture cap;

		if (dnn_face_recognition_::video_file != "")
		{
			cap.open(dnn_face_recognition_::video_file);
		}
		else
		{
			cap = cv::VideoCapture(camID);
		}


		if (!cap.isOpened())
		{
			cerr << "Unable to connect to camera" << endl;
			return 1;
		}

		int empty_coun = 0;
		while (true)
		{
			error_code = 0;
			cv::Mat temp;
			cap >> temp;

			if (temp.empty() == true) {
				break;
			}
			temp = opencv_util::resize_ex(temp);

			//if (temp.size().width <= 640 || temp.size().height <= 640)
			//{
			//	float a = 640.0 / temp.size().width;
			//	float b = 640.0 / temp.size().height;

			//	if (b > a) a = b;
			//	cv::resize(temp, temp, cv::Size(temp.size().width*a, temp.size().height*a), 0, 0, cv::INTER_CUBIC);
			//}

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

			if (!dnn_face_recognition_::no_show)
			{
				cv::imshow("render_face_detections", temp);
				if (cv::waitKey(1) == 27) break;
			}
			else
			{
				imgwrite_(temp);
				if (running_break()) break;
			}
			temp = dlib::toMat(img);
			try
			{
				cv::Mat roi_img(temp,
					cv::Rect(
						dets[0].tl_corner().x(), dets[0].tl_corner().y(),
						dets[0].width(), dets[0].height()
					)
				);
				if (!dnn_face_recognition_::no_show)
				{
					cv::imshow("cap", roi_img);
				}
				if (error_code == 0)
				{
					char capimage_file[256];
					sprintf(capimage_file, "capture/%s_%03d.png", user, count);
					count++;
					cv::imwrite(capimage_file, roi_img);
					if (cv::waitKey(1) == 27) break;
					//if ( count == 10 ) return error_code;

					if (dnn_face_recognition_::no_show)
					{
						imgwrite_(temp);
						if (running_break()) break;
					}
				}
			}
			catch (std::exception& e)
			{
				cout << e.what() << endl;
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


