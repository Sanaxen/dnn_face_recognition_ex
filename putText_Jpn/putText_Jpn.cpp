//
#include "../config.h"
#include "putText_Jpn.h"

#ifdef _WIN64
#include <windows.h>
#endif

#include <cstring>
//#include <tchar.h>

// opencv
#pragma warning(push)
#pragma warning ( disable : 4819 )
#include <opencv2/opencv.hpp>
#pragma warning(pop)

//
#include <string>

//
#ifdef USE_JAPANESE_CHAR
namespace
{
	const TCHAR* P_STR_DEF_FONTNAME = TEXT("ＭＳ ゴシック");
	const unsigned char THRESH_MOJI = 0xFF;
	const unsigned char BACK_COLOR = 0xFF;

	//
	enum
	{
		COL_ID_B = 0,
		COL_ID_G = 1,
		COL_ID_R = 2,
	};
}

//---------------------------------------------------------------
//
// static関数
//
//---------------------------------------------------------------

//
static HBITMAP sttc_CreateBackbuffer(int nWidth, int nHeight)
{
	HBITMAP hBmp;
	LPVOID           lp;
	BITMAPINFO       bmi;
	BITMAPINFOHEADER bmiHeader;

	::ZeroMemory(&bmiHeader, sizeof(BITMAPINFOHEADER));
	bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	bmiHeader.biWidth = nWidth;
	bmiHeader.biHeight = nHeight;
	bmiHeader.biPlanes = 1;
	bmiHeader.biBitCount = 24;

	bmi.bmiHeader = bmiHeader;

	hBmp = ::CreateDIBSection(NULL, (LPBITMAPINFO)&bmi, DIB_RGB_COLORS, &lp, NULL, 0);
	return hBmp;
}

// 横書き
static int sttc_DrawText_Horizontal(cv::Mat& a_r_img_dst, const BITMAP* a_p_bmp, int a_pos_x, int a_pos_y, cv::Scalar a_font_color)
{
	int result = 0;
	int attach_width, attach_height, attach_bit, attach_linesize;
	int dst_width, dst_height, dst_linesize;

	attach_width = a_p_bmp->bmWidth;
	attach_height = a_p_bmp->bmHeight;
	attach_bit = a_p_bmp->bmBitsPixel;
	attach_linesize = ((attach_bit / 8) * a_p_bmp->bmWidth + 3) & ~3;

	dst_width = a_r_img_dst.cols;
	dst_height = a_r_img_dst.rows;
	dst_linesize = static_cast<int>(a_r_img_dst.step);

	unsigned char r08, g08, b08;
	r08 = static_cast<unsigned char>(a_font_color.val[COL_ID_R]);
	g08 = static_cast<unsigned char>(a_font_color.val[COL_ID_G]);
	b08 = static_cast<unsigned char>(a_font_color.val[COL_ID_B]);

	unsigned char* p_attach_img, * p_dst_img;
	int x, y;

	for (y = 0; y < attach_height; y++)
	{
		if (a_pos_y + y >= 0 && a_pos_y + y < dst_height)
		{
			p_dst_img = a_r_img_dst.data + 3 * a_pos_x + (a_pos_y + y) * dst_linesize;
			p_attach_img = static_cast<unsigned char*>(a_p_bmp->bmBits) + (attach_height - y - 1) * attach_linesize;
			for (x = 0; x < attach_width; x++)
			{
				if (x + a_pos_x >= dst_width)
				{
					break;
				}

				// chは何でも良い。
				const int SELECTED_ID = COL_ID_B;
				unsigned char val_u08 = p_attach_img[COL_ID_B];
				if (val_u08 < THRESH_MOJI)
				{
					// アンチエイリアス処理
					// 速度優先のため、計算は厳密ではない。(画素値が 255/256 になる)
					int coef_a = BACK_COLOR - val_u08;
					p_dst_img[COL_ID_B] = static_cast<unsigned char>((coef_a * b08 + val_u08 * p_dst_img[COL_ID_B]) >> 8);
					p_dst_img[COL_ID_G] = static_cast<unsigned char>((coef_a * g08 + val_u08 * p_dst_img[COL_ID_G]) >> 8);
					p_dst_img[COL_ID_R] = static_cast<unsigned char>((coef_a * r08 + val_u08 * p_dst_img[COL_ID_R]) >> 8);
				}

				p_dst_img += 3;
				p_attach_img += 3;
			}
		}

	}

	return result;
}



//
static void sttc_putTextCore(cv::Mat& a_r_img_dst, const TCHAR* a_p_text, cv::Point a_pos_org, const TCHAR* a_p_fontname, double a_font_scale, cv::Scalar a_font_color, int a_thickness, int a_lineType)
{
	// 未使用
	UNREFERENCED_PARAMETER(a_lineType);
	UNREFERENCED_PARAMETER(a_thickness);

	//
	const double FONT_SCALE_COEF = 50.0;
	HDC hdc_Compatible;
	HBITMAP hbmp, hbmpPrev;
	BITMAP  bmp;
	HFONT hFont, hFont_old;
	int width, height, linesize;
	int size_y = 100;
	int str_size;

	str_size = ::lstrlen(a_p_text);

	size_y = static_cast<int>(FONT_SCALE_COEF * a_font_scale + 0.5);
	width = a_r_img_dst.cols;
	height = (size_y * 3) / 2;
	hdc_Compatible = ::CreateCompatibleDC(NULL);
	hbmp = sttc_CreateBackbuffer(width, height);
	hbmpPrev = reinterpret_cast<HBITMAP>(::SelectObject(hdc_Compatible, hbmp));

	::GetObject(hbmp, sizeof(BITMAP), &bmp);

	// 背景色 = 白
	linesize = (((bmp.bmBitsPixel / 8) * width) + 3) & ~3;	// 4バイト境界
	std::memset(bmp.bmBits, BACK_COLOR, linesize * height);

	// 黒文字
	hFont = ::CreateFont(
		size_y, 0, 0, 0, FW_DONTCARE, FALSE, FALSE, FALSE,
		SHIFTJIS_CHARSET, OUT_DEFAULT_PRECIS,
		CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY,
		VARIABLE_PITCH | FF_ROMAN, a_p_fontname
	);
	hFont_old = reinterpret_cast<HFONT>(::SelectObject(hdc_Compatible, hFont));
	::SetTextColor(hdc_Compatible, RGB(0, 0, 0));

	// 基準座標
	volatile int v_flag_org_left_bottom = 0;
	int y_org = size_y;
	if (v_flag_org_left_bottom)
	{
		y_org = size_y;	// left bottom
	}
	else
	{
		y_org = size_y / 3;	// left center(適当なパラメータ、要最適化)
	}

	// 白いウィンドウにグレー背景の黒文字を描画する。
	::TextOut(hdc_Compatible, 0, y_org, a_p_text, str_size);

	// hdc_Compatibleの文字領域と同じ位置の画素を、a_font_colorで指定されたBGR値に置き換える。
	sttc_DrawText_Horizontal(a_r_img_dst, &bmp, a_pos_org.x, a_pos_org.y - size_y, a_font_color);

	::SelectObject(hdc_Compatible, hFont_old);
	::SelectObject(hdc_Compatible, hbmpPrev);
	::DeleteObject(hbmp);
	::DeleteObject(hFont);
	::DeleteDC(hdc_Compatible);

	return;
}

/////////////////////////////////////////////////////////////////////////////////////
//
// extern関数
//
/////////////////////////////////////////////////////////////////////////////////////
//void sc::myCV::putText_Jpn(cv::Mat& a_r_img_dst, const char* a_p_text, cv::Point a_pos_org, const char* a_p_fontname, double a_font_scale, cv::Scalar a_font_color, int a_thickness, int a_lineType)
//{
//	TCHAR fontname[512];
//	TCHAR text[512];
//
//	_tcscpy(fontname, CA2T(a_p_fontname));
//	_tcscpy(text, CA2T(a_p_text));
//	
//	sc::myCV::putText_Jpn(a_r_img_dst, text, a_pos_org, fontname, a_font_scale, a_font_color, a_thickness, a_lineType);
//}

//void sc::myCV::putText_Jpn(cv::Mat& a_r_img_dst, std::string& a_p_text, cv::Point a_pos_org, std::string& a_p_fontname, double a_font_scale, cv::Scalar a_font_color, int a_thickness, int a_lineType)
//{
//	sc::myCV::putText_Jpn(a_r_img_dst, a_p_text.c_str(), a_pos_org, a_p_fontname.c_str(), a_font_scale, a_font_color, a_thickness, a_lineType);
//}

//
void sc::myCV::putText_Jpn(cv::Mat& a_r_img_dst, const TCHAR* a_p_text, cv::Point a_pos_org, const TCHAR* a_p_fontname, double a_font_scale, cv::Scalar a_font_color, int a_thickness, int a_lineType)
{
	// a_r_img_dstが空ではない
	if (!a_r_img_dst.empty())
	{
		// チャンネル数 = 3、チャンネルごとのビット深度が8(unsigned char)
		if (a_r_img_dst.channels() == 3 && a_r_img_dst.depth() == CV_8U)
		{
			int str_size;

			str_size = ::lstrlen(a_p_text);

			// 入力文字列の長さが0より大きい
			if (str_size > 0)
			{
				const TCHAR* p_current_fontname = P_STR_DEF_FONTNAME;
				if (a_p_fontname)
				{
					p_current_fontname = a_p_fontname;
				}
				sttc_putTextCore(a_r_img_dst, a_p_text, a_pos_org, p_current_fontname, a_font_scale, a_font_color, a_thickness, a_lineType);
			}
		}
	}

	return;
}
#else
void sc::myCV::putText_Jpn(cv::Mat& a_r_img_dst, const char *a_p_text, cv::Point a_pos_org, const int a_p_fontname, double a_font_scale, cv::Scalar a_font_color, int a_thickness, int lineType)
{
	cv::putText(a_r_img_dst, a_p_text, a_pos_org, cv::FONT_HERSHEY_PLAIN, a_font_scale, a_font_color, a_thickness, lineType);
}
#endif


