#pragma once
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace opencv_util
{
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

	inline cv::Mat hresize_ex(cv::Mat& image, int hmax = 832)
	{
		cv::Mat temp = image.clone();
		float a = 1.0;
		if (hmax < image.size().height)
		{
			a = (float)hmax / (float)image.size().height;
		}
		else
		{
			a = (float)image.size().height / (float)hmax;
		}
		cv::resize(temp, temp, cv::Size(temp.size().width*a, temp.size().height*a), 0, 0, cv::INTER_CUBIC);
		return temp;
	}

	inline cv::Mat wresize_ex(cv::Mat& image, int wmax = 832)
	{
		cv::Mat temp = image.clone();
		float a = 1.0;
		if (wmax < image.size().width)
		{
			a = (float)wmax / (float)image.size().width;
		}
		else
		{
			a = (float)image.size().width / (float)wmax;
		}
		cv::resize(temp, temp, cv::Size(temp.size().width*a, temp.size().height*a), 0, 0, cv::INTER_CUBIC);
		return temp;
	}

	inline cv::Mat resize_ex(cv::Mat& image, int max = 832)
	{
		cv::Mat temp = image.clone();
		if (image.size().width > image.size().height)
		{
			temp = wresize_ex(image, max);
		}
		else
		{
			temp = hresize_ex(image, max);
		}
		return temp;
	}

	inline void resize_padd(cv::Mat& target_mat, int width)
	{
		cv::Mat work_mat = cv::Mat::zeros(cv::Size(width, width), CV_8UC3);

		int big_width = target_mat.cols > target_mat.rows ? target_mat.cols : target_mat.rows;
		double ratio = ((double)width / (double)big_width);

		cv::Mat convert_mat;
		cv::resize(target_mat, convert_mat, cv::Size(), ratio, ratio, cv::INTER_NEAREST);

		cv::Mat Roi1(work_mat, cv::Rect((width - convert_mat.cols) / 2, (width - convert_mat.rows) / 2,
			convert_mat.cols, convert_mat.rows));
		convert_mat.copyTo(Roi1);

		target_mat = work_mat.clone();
	}
};
