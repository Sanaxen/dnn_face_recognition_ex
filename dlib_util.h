#pragma once
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <dlib/opencv.h>

namespace dlib_util
{
	inline dlib::rectangle openCVRectToDlib(cv::Rect r)
	{
		return dlib::rectangle((long)r.tl().x, (long)r.tl().y, (long)r.br().x - 1, (long)r.br().y - 1);
	}
	inline cv::Rect dlibRectangleToOpenCV(dlib::rectangle r)
	{
		return cv::Rect(cv::Point2i(r.left(), r.top()), cv::Point2i(r.right() + 1, r.bottom() + 1));
	}
};
