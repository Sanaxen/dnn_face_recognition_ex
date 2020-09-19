#ifndef PUT_TEXT_JPN_H_
#define PUT_TEXT_JPN_H_


// opencv
#pragma warning(push)
#pragma warning ( disable : 4819 )
#include <opencv2/opencv.hpp>
#pragma warning(pop)

#ifdef _WIN64

//
#include <tchar.h>

#define USES_CONVERSION
#include "atlstr.h"

#endif

//
namespace sc
{
	namespace myCV
	{
#ifdef _WIN64
		void putText_Jpn(cv::Mat& a_r_img_dst, const TCHAR *a_p_text, cv::Point a_pos_org, const TCHAR *a_p_fontname, double a_font_scale, cv::Scalar a_font_color, int a_thickness = 1, int lineType = 8);
		//void putText_Jpn(cv::Mat& a_r_img_dst, const char *a_p_text, cv::Point a_pos_org, const char *a_p_fontname, double a_font_scale, cv::Scalar a_font_color, int a_thickness = 1, int lineType = 8);
		//void putText_Jpn(cv::Mat& a_r_img_dst, std::string& a_p_text, cv::Point a_pos_org, std::string& a_p_fontname, double a_font_scale, cv::Scalar a_font_color, int a_thickness = 1, int a_lineType = 8);
#else
		void putText_Jpn(cv::Mat& a_r_img_dst, const char *a_p_text, cv::Point a_pos_org, const int a_p_fontname, double a_font_scale, cv::Scalar a_font_color, int a_thickness = 1, int lineType = 8);
#endif
	}
}

#endif	// PUT_TEXT_JPN_H_
