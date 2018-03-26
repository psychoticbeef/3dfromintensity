#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/filesystem/convenience.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include "uraster.hpp"

const float L_H = 86.0;
const float L_V = 69.0;
// [mm/px]	{ ein pixel an sich ist 6,7 µm }
#define RES_H (L_H * 100.0 / 1280.0)
#define RES_V (L_V * 100.0 / 1024.0)

typedef std::vector<cv::Point> CvBlob;
typedef std::vector<CvBlob> CvBlobs;

namespace po = boost::program_options;

std::string _gradient;
std::string _watermark_file;
std::string _base;
std::string _base_name;
float _border_factor, _font_scale, _histogram_scale,
			_rot_x, _rot_y, _rot_z;
uint32_t _threshold, _mm, image_width;
cv::Mat _lut;
cv::Mat _watermark;

bool _draw_miniature = false;

void print(cv::Point& s) {
	std::cout << "x: " << s.x << " y: " << s.y << std::endl;
}
void print(cv::Mat& m) {
	std::cout << "w: " << m.cols << " h: " << m.rows << std::endl;
}
void print(cv::Size& s) {
	std::cout << "w: " << s.width << " h: " << s.height << std::endl;
}
void print(cv::Rect& r) {
	std::cout << "x: " << r.x << " y: " << r.y << " w: " << r.width << " h: " << r.height << std::endl;
}

cv::Size get_miniature_size() {
	const float f = (float)image_width / (float)_mm;
	return cv::Size(f * L_H, f * L_V);
}

void overlay_image(cv::Mat& src, cv::Mat& overlay, const cv::Point& location) {
	for (int y = std::max(location.y, 0); y < src.rows; ++y) {
		int fY = y - location.y;
		if (fY >= overlay.rows) break;
		for (int x = std::max(location.x, 0); x < src.cols; ++x) {
			int fX = x - location.x;
			if (fX >= overlay.cols) break;
			double opacity = ((double)overlay.data[fY * overlay.step + fX * overlay.channels() + 3]) / 255.0;
			for (int c = 0; opacity > 0 && c < src.channels(); ++c) {
				unsigned char overlayPx = overlay.data[fY * overlay.step + fX * overlay.channels() + c];
				unsigned char srcPx = src.data[y * src.step + x * src.channels() + c];
				src.data[y * src.step + src.channels() * x + c] = srcPx * (1. - opacity) + overlayPx * opacity;
			}
		}
	}
}

void load_watermark() {
	_watermark = cv::imread(_watermark_file, cv::IMREAD_UNCHANGED);
	if (_watermark.empty()) {
		std::cerr << "Error: Watermark file \"" << _watermark_file << "\" not readable" << std::endl;
		exit(1);
	}
}

void draw_watermark(cv::Mat& in, int pos) {
	if (pos == 0) {
		overlay_image(in, _watermark, cv::Point(10, in.rows - _watermark.rows - 10));
	} else {
		overlay_image(in, _watermark, cv::Point(in.cols - _watermark.cols - 10, 10));
	}
}

void draw_legend(cv::Mat& in, int pos) {
	const int w = 10, h = 100, border = 1, offset_x = 10, offset_y = 10, o_t_x = 10;
	cv::Mat legend, lut_transposed, tmp;
	cv::transpose(_lut, lut_transposed);
	cv::flip(lut_transposed, lut_transposed, 0);
	cv::resize(lut_transposed, legend, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
	cv::line(legend, cv::Point(0, legend.rows*3.0/4), cv::Point(legend.cols, legend.rows*3.0/4), cv::Scalar(255, 255, 255));
	cv::line(legend, cv::Point(0, legend.rows/2), cv::Point(legend.cols, legend.rows/2), cv::Scalar(255, 255, 255));
	cv::line(legend, cv::Point(0, legend.rows/4), cv::Point(legend.cols, legend.rows/4), cv::Scalar(255, 255, 255));
	cv::copyMakeBorder(legend, tmp, border, border, border, border, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

	auto text = "10257%";
	int baseline = 0;
	cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, _font_scale, 1, &baseline);
	int text_x = offset_x + w + o_t_x;
	int text_y = offset_y + text_size.height / 3;
	cv::putText(in, "100%", cv::Point_<int> (text_x, text_y), cv::FONT_HERSHEY_SIMPLEX, _font_scale/2, cv::Scalar(255, 255, 255));
	cv::putText(in, " 75%", cv::Point_<int> (text_x, text_y + legend.rows/4), cv::FONT_HERSHEY_SIMPLEX, _font_scale/2, cv::Scalar(255, 255, 255));
	cv::putText(in, " 50%", cv::Point_<int> (text_x, text_y + legend.rows/2), cv::FONT_HERSHEY_SIMPLEX, _font_scale/2, cv::Scalar(255, 255, 255));
	cv::putText(in, " 25%", cv::Point_<int>(text_x, text_y + legend.rows*3.0/4), cv::FONT_HERSHEY_SIMPLEX, _font_scale/2, cv::Scalar(255, 255, 255));
	cv::putText(in, "  0%", cv::Point_<int>(text_x, text_y + legend.rows-1), cv::FONT_HERSHEY_SIMPLEX, _font_scale/2, cv::Scalar(255, 255, 255));

	if (pos == 0) {
		//overlay_image(in, legend, cv::Point(10, in.rows - legend.rows - 10));
		tmp.copyTo(in(cv::Rect(offset_x, offset_y, w+2*border, h+2*border)));
	} else {
		legend.copyTo(in(cv::Rect(in.cols - legend.cols - 10, 10, w, h)));
	}
}

void load_lut() {
	_lut = cv::imread(_gradient, cv::IMREAD_UNCHANGED);
	if (_lut.empty()) {
		std::cerr << "Error: Gradient file \"" << _gradient << "\" not readable" << std::endl;
		exit(1);
	}
	if (_lut.cols != 256 || _lut.rows != 1) {
		std::cerr << "Error: Gradient file \"" << _gradient << "\" needs to be 256x1 px" << std::endl;
		exit(1);
	}
}

int adjust(double color, double factor) {
	const int max_intensity = 255;
	const float gamma = 0.8f;
	if (color == 0.0)
		return 0;
	return round(max_intensity * pow(color * factor, gamma));
}

cv::Scalar wavelength_to_rgb(int wave_length) {
	float r, g, b;

	switch(wave_length) {
		case 380 ... 439:
			r = -(wave_length - 440) / (440 - 380);
			g = 0.0;
			b = 1.0;
			break;
		case 440 ... 489:
			r = 0.0;
			g = (wave_length - 440) / (490 - 440);
			b = 1.0;
			break;
		case 490 ... 509:
			r = 0.0;
			g = 1.0;
			b = -(wave_length - 510) / (510 - 490);
			break;
		case 510 ... 579:
			r = (wave_length - 510) / (580 - 510);
			g = 1.0;
			b = 0.0;
			break;
		case 580 ... 644:
			r = 1.0;
			g = -(wave_length - 645) / (645 - 580);
			b = 0.0;
			break;
		case 645 ... 780:
			r = 1.0;
			g = 0.0;
			b = 0.0;
			break;
		default:
			r = 1.0;
			g = 1.0;
			b = 1.0;
			break;
	}

	float factor;

	switch(wave_length) {
		case 380 ... 419:
			factor = 0.3 + 0.7 * (wave_length - 380) / (420 - 380);
			break;
		case 420 ... 700:
			factor = 1.0;
			break;
		case 701 ... 780:
			factor = 0.3 + 0.7 * (780 - wave_length) / (780 - 700);
			break;
		default:
			factor = 0.0;
			break;
	}

	auto result = cv::Scalar(adjust(b, factor), adjust(g, factor), adjust(r, factor));
	return result;
}

cv::Mat preprocess(cv::Mat& in) {
	cv::Mat blurred, result, thresholded;
	cv::medianBlur(in, blurred, 3);
	cv::threshold(blurred, thresholded, _threshold, 255, CV_THRESH_BINARY);
	cv::morphologyEx(thresholded, result, cv::MORPH_CLOSE, cv::Mat(), cv::Point(-1,-1), 3);
	return result;
}

CvBlobs find_blobs(cv::Mat& image) {
	CvBlobs result;
	auto preprocessed = preprocess(image);   // irgendwie frisst der opencv das nicht wenn ich das direkt findContours uebergebe
	//imwrite("pre.png", preprocessed);
	findContours(preprocessed, result, CV_RETR_LIST, CV_CHAIN_APPROX_NONE/*, offset*/);
	return result;
}

CvBlob get_biggest_blob(CvBlobs blobs) {
	if (blobs.size() == 1) return blobs[0];
	return *(std::max_element(begin(blobs), end(blobs),[](std::vector<cv::Point> a, std::vector<cv::Point> b) {
				return cv::contourArea(a) < cv::contourArea(b);
				}));
}

float get_scaling(cv::Mat& input, bool log = false) {
	cv::Mat out(input.size(), CV_8U);
	double min, max;
	cv::minMaxLoc(input, &min, &max);
	if (log) {
		std::cout << "min:max " << min << ":" << max << std::endl;
	}
	return 255.0/(max-min);
}

cv::Rect get_square_rect(const cv::Rect& r, const cv::Size& max_size, const cv::Point_<float> centroid) {
	// ziel: quadratischer bereich zum ausschneiden
	// so gross wie die groesste seite...
	auto length = std::max(r.width, r.height);
	// ... aber nicht groesser als die bilddimensionen hergeben
	length = std::min(length, max_size.width);
	length = std::min(length, max_size.height);
	cv::Rect result(r);

	// wenn eine seite zu kurz ist, kasten groesser machen
	if (r.width < length) {
		result.x -= (length-result.width)/2;
		result.x = std::max(result.x, 0);
	}
	if (r.height < length) {
		result.y -= (length-result.height)/2;
		result.y = std::max(result.y, 0);
	}

	// wenn eine seite zu lang ist, bildausschnitt um den zentrumspunkt verschieben
	if (r.width > length) {
		result.x = centroid.x - length / 2;
	}
	if (r.height > length) {
		result.y = centroid.y - length / 2;
	}

	result.width = length;
	result.height = length;
	result.x = std::max(result.x, 0);
	result.y = std::max(result.y, 0);

	// kasten geht ueber den rechten bildrand -> auf x nach links schieben
	if (result.x + result.width > max_size.width) {
		result.x = max_size.width - result.width;
	}
	// kasten geht ueber den unteren bildrand -> auf y nach oben schieben
	if (result.y + result.height > max_size.height) {
		result.y = max_size.height - result.height;
	}
	return result;
}

cv::Mat get_bw_image(cv::Mat& src, int target_width, int color) {
	cv::Mat result;
	cv::Mat tmp;
	cv::cvtColor(src, tmp, CV_GRAY2RGB);
	auto rgb = wavelength_to_rgb(color);
	for(int y = 0; y < src.rows; y++) {
		for(int x = 0; x < src.cols; x++) {
			const auto c = src.at<uchar>(y, x);
			if (c == 0) {
				tmp.at<cv::Vec3b>(y, x)[0] = 255;
				tmp.at<cv::Vec3b>(y, x)[1] = 255;
				tmp.at<cv::Vec3b>(y, x)[2] = 255;
				continue;
			}
			float k = c/255.0;
			int r = round((1 - k) * 255.0);
			tmp.at<cv::Vec3b>(y, x)[0] = k*rgb[0] + r;	// blend channels with white
			tmp.at<cv::Vec3b>(y, x)[1] = k*rgb[1] + r;
			tmp.at<cv::Vec3b>(y, x)[2] = k*rgb[2] + r;
		}
	}
	cv::resize(tmp, result, cv::Size(target_width, target_width), 0, 0, cv::INTER_LANCZOS4);
	return result;
}

cv::Mat get_color_image(cv::Mat src, int target_width, int target_height) {
	cv::Mat result, tmp, color;
	cv::cvtColor(src, tmp, CV_GRAY2BGR);
	cv::LUT(tmp, _lut, color);
	cv::resize(color, result, cv::Size(target_width, target_height), 0, 0, cv::INTER_LANCZOS4);
	return result;
}

float round_to_(const float x, const float factor) {
	return floor(x/factor + 0.5) * factor;
}

float draw_scale(cv::Mat& src, cv::Point_<int> position, float scale, int orig_size, cv::Scalar color_scale) {
	auto total_width_mum = src.rows * RES_H / scale;
	int scale_length = 500;
	int scale_size_mum = 0;
	while (scale_size_mum == 0) {
		scale_size_mum = (int)round_to_(total_width_mum/5, scale_length);
		scale_length /= 10;
	}
	auto scale_size_px = scale_size_mum / RES_H * scale;
	const int handle_bar_length = 4;
	cv::line(src, cv::Point_<int>(scale_size_px, position.y-handle_bar_length), cv::Point_<int>(scale_size_px, position.y+handle_bar_length), color_scale, 1);
	cv::line(src, cv::Point_<int>(scale_size_px, position.y), cv::Point_<int>(scale_size_px+ scale_size_px, position.y), color_scale);
	cv::line(src, cv::Point_<int>(scale_size_px + scale_size_px, position.y-handle_bar_length), cv::Point_<int>(scale_size_px + scale_size_px, position.y+handle_bar_length), color_scale, 1);

	int baseline = 0;
	auto text = std::to_string(scale_size_mum) + " micron";
	cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, _font_scale, 1, &baseline);
	cv::putText(src, text, cv::Point_<int>(scale_size_px + (scale_size_px - text_size.width) / 2, position.y + text_size.height + 10), cv::FONT_HERSHEY_SIMPLEX, _font_scale, color_scale);
	return scale_size_px;
}

void write_image(std::string add, std::string type, cv::Mat src) {
	auto filename = _base + _base_name + "_" + add + "." + type;
	std::cout << "writing: " << filename << " " << std::endl;
	if (add != "hi") {
		draw_watermark(src, 0);
	} else  {
		draw_watermark(src, 1);
	}
	if (add == "3d" || add == "bp" || add == "wi") {
		draw_legend(src, 0);
	}
	cv::imwrite(filename, src);
}

cv::Point_<float> get_blob_center(CvBlob b) {
	auto mu = cv::moments(b, true);
	return cv::Point_<float>(mu.m10/mu.m00, mu.m01/mu.m00);
}

void draw_cross(cv::Mat& src, cv::Point_<float> px, cv::Scalar color) {
	const int thickness = 1;
	cv::Point pt1(px.x - 100, px.y - 100);
	cv::Point pt2(px.x + 100, px.y + 100);
	if (cv::clipLine(src.size(), pt1, pt2)) {   // line is visible (not entirely outside the image) => draw the clipped version
		cv::line(src, pt1, pt2, color, thickness);
	}
	cv::Point pt3(px.x + 100, px.y - 100);
	cv::Point pt4(px.x - 100, px.y + 100);
	if (cv::clipLine(src.size(), pt3, pt4)) {
		cv::line(src, pt3, pt4, color, thickness);
	}
}

cv::Point_<float> find_center(cv::Mat& src, cv::Rect r) {
	cv::Mat tmp;
	cv::threshold(src, tmp, 50, 255, CV_THRESH_BINARY);
	auto blobs = find_blobs(tmp);
	if (blobs.size() == 0) {
		return cv::Point_<float> (r.x + r.width/2, r.y + r.height/2);
	}
	auto blob = get_biggest_blob(blobs);
	return get_blob_center(blob);
}

cv::Mat add_miniature(cv::Mat& reference_grey, cv::Mat& reference_color, cv::Mat& full_color, cv::Scalar text_color) {
	cv::Mat result, full_resize, grey_resize;
	auto s = get_miniature_size();
	cv::resize(full_color, full_resize, s);
	cv::resize(reference_grey, grey_resize, cv::Size(reference_color.cols, reference_color.rows), 0, 0, cv::INTER_LANCZOS4);
	float smallest = 256;
	int corner;
	cv::Rect r;
	for (int i = 0; i < 4; i++) {
		cv::Rect ri(0, 0, s.width, s.height);
		if (!(i & 2)) ri.x = grey_resize.cols - s.width; // spiegelung an y achse
		if (i & 1) ri.y = grey_resize.rows - s.height; // spiegelung an x achse
		float mean = cv::mean(grey_resize(ri))[0];
		if (mean < smallest) {
			smallest = mean;
			corner = i;
			r = cv::Rect(ri.x, ri.y, s.width, s.height);
		}
	}
	result = reference_color;
	cv::rectangle(full_resize, cv::Rect(0, 0, full_resize.cols, full_resize.rows), text_color, 3);
	full_resize.copyTo(result(r));
	//if (corner == 0) return result;
	if (corner == 1) cv::flip(reference_color, result,  0);
	if (corner == 2) cv::flip(reference_color, result,  1);	// spiegelung an y
	if (corner == 3) cv::flip(reference_color, result, -1);
	int baseline = 0;
	auto text = "1:1";
	auto text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, _font_scale, 1, &baseline);
	cv::putText(result, text, cv::Point_<int>(result.rows - text_size.width, full_resize.cols), cv::FONT_HERSHEY_SIMPLEX, _font_scale, text_color);
	return result;
}

void fft_shift_mask(cv::Mat mag_i) {
	// crop if it has an odd number of rows or columns
	mag_i = mag_i(cv::Rect(0, 0, mag_i.cols & -2, mag_i.rows & -2));

	int cx = mag_i.cols/2;
	int cy = mag_i.rows/2;

	cv::Mat q0(mag_i, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	cv::Mat q1(mag_i, cv::Rect(cx, 0, cx, cy));  // Top-Right
	cv::Mat q2(mag_i, cv::Rect(0, cy, cx, cy));  // Bottom-Left
	cv::Mat q3(mag_i, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

	cv::Mat tmp;                            // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);                     // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);
}

/** src needs to be padded already */
cv::Mat fft_compute_dft(cv::Mat src) {
	// copy the source image, on the border add zero values
	cv::Mat planes[] = { cv::Mat_< float> (src), cv::Mat::zeros(src.size(), CV_32F) };
	// create a complex matrix
	cv::Mat complex;
	cv::merge(planes, 2, complex);
	cv::dft(complex, complex, cv::DFT_COMPLEX_OUTPUT);  // fourier transform
	return complex;
}

void fft_update_magnitude(cv::Mat complex) {
	cv::Mat mag_i;
	cv::Mat planes[] = {
		cv::Mat::zeros(complex.size(), CV_32F),
		cv::Mat::zeros(complex.size(), CV_32F)
	};
	cv::split(complex, planes); // planes[0] = Re(DFT(I)), planes[1] = Im(DFT(I))
	cv::magnitude(planes[0], planes[1], mag_i); // sqrt(Re(DFT(I))^2 + Im(DFT(I))^2)
	// switch to logarithmic scale: log(1 + magnitude)
	mag_i += cv::Scalar::all(1);
	cv::log(mag_i, mag_i);
	fft_shift_mask(mag_i); // rearrage quadrants
	// Transform the magnitude matrix into a viewable image (float values 0-1)
	cv::normalize(mag_i, mag_i, 1, 0, cv::NORM_INF);
}

cv::Mat fft_update_result(cv::Mat complex) {
	cv::Mat work;
	cv::idft(complex, work);
	// equivalent to:
	// dft(complex, result, DFT_INVERSE + DFT_SCALE);
	cv::Mat planes[] = {
		cv::Mat::zeros(complex.size(), CV_32F),
		cv::Mat::zeros(complex.size(), CV_32F)
	};
	cv::split(work, planes); // planes[0] = Re(DFT(I)), planes[1] = Im(DFT(I))
	cv::magnitude(planes[0], planes[1], work); // sqrt(Re(DFT(I))^2 + Im(DFT(I))^2)
	cv::normalize(work, work, 0, 255, cv::NORM_MINMAX);
	cv::Mat result;
	work.convertTo(result, CV_8U);
	return result;
}

cv::Mat get_fft_mask(cv::Size s, cv::Rect& size_dot) {
	const int w = s.width, h = s.height;
	const int d = s.width <= 120 ? h/4 : h/15;
	const int radius_mask_dot = std::max(std::max(size_dot.width, size_dot.height) >> 1, w/3);
	const int f = std::max(1, h/120);
	cv::Mat result = cv::Mat::zeros(s, CV_32F);;
	cv::circle(result, cv::Point2i(w/2, h/2), radius_mask_dot, cv::Scalar(255), CV_FILLED);
	cv::rectangle(result, cv::Point_<int>(0, h/2-f), cv::Point_<int>(w/2-d, h/2+f), cv::Scalar(0), CV_FILLED); // left
	cv::rectangle(result, cv::Point_<int>(w/2+d, h/2-f), cv::Point_<int>(w, h/2+f), cv::Scalar(0), CV_FILLED); // right
	cv::rectangle(result, cv::Point_<int>(w/2-f, 0), cv::Point_<int>(w/2+f, h/2-d), cv::Scalar(0), CV_FILLED); // top
	cv::rectangle(result, cv::Point_<int>(w/2-f, h/2+d), cv::Point_<int>(w/2+f, h), cv::Scalar(0), CV_FILLED); // bottom
	//cv::imwrite("mask.png", result);
	return result;
}


cv::Mat fft_apply_mask(cv::Mat& src, cv::Mat mask) {
	cv::Mat complex = fft_compute_dft(src);
	fft_update_magnitude(complex);
	fft_update_result(complex);
	fft_shift_mask(mask);
	cv::Mat planes[] = {cv::Mat::zeros(complex.size(), CV_32F), cv::Mat::zeros(complex.size(), CV_32F)};
	cv::Mat kernel_spec;
	planes[0] = mask; // real
	planes[1] = mask; // imaginar
	cv::merge(planes, 2, kernel_spec);
	cv::mulSpectrums(complex, kernel_spec, complex, cv::DFT_ROWS);
	//cv::mulSpectrums(complex, kernel_spec, complex, cv::DFT_ROWS); // only DFT_ROWS accepted
	fft_update_magnitude(complex);     // show spectrum
	return fft_update_result(complex);
}

cv::Mat fft_denoise(cv::Mat& src, cv::Rect& size_dot) {
	auto complex = fft_compute_dft(src);
	fft_update_magnitude(complex);
	fft_update_result(complex);
	return fft_apply_mask(src, get_fft_mask(complex.size(), size_dot));
}

/** to not scale tiny images too large, increase border area on small images */
cv::Rect add_border(cv::Rect in, int image_width) {
	cv::Rect result(in);
	if (result.width / (float)image_width < _border_factor) {
		auto border_v = (std::ceil(image_width * _border_factor) - result.width);
		result.width += border_v;
		result.x -= border_v/2;
	}
	if (result.height / (float)image_width < _border_factor) {
		auto border_h = (std::ceil(image_width * _border_factor) - result.height);
		result.height += border_h;
		result.y -= border_h/2;
	}
	return result;
}

cv::Rect adjust_border_for_fft(const cv::Rect in) {
	cv::Rect result;
	int q = cv::getOptimalDFTSize(in.width);
	int r = cv::getOptimalDFTSize(in.height);
	result.x = in.x - (q-in.width)/2;
	result.y = in.y - (r-in.height)/2;
	result.width = q;
	result.height = r;
	return result;
}

cv::Mat get_denoised_image(cv::Mat& src, int target_sidelength = -1, bool prevent_large_scaling = true) {
	cv::Mat src_8u, src_normalized_8u, src_normalized_32f;

	/* rescale from min_value in image,max_value to 0..1023 */
	auto scaling = get_scaling(src);
	cv::normalize(src, src_normalized_32f, 0, 1023, cv::NORM_MINMAX);
	src.convertTo(src_8u, CV_8U, scaling);
	src_normalized_32f.convertTo(src_normalized_8u, CV_8U, scaling);

	auto blobs = find_blobs(src_normalized_8u);
	if (blobs.size() == 0) {
		std::cerr << "No blobs found.";
		exit(1);
	}
	auto blob = get_biggest_blob(blobs);
	auto size_dot = boundingRect(blob);
	cv::Rect roi;
	if (prevent_large_scaling) {
		roi = target_sidelength == -1 ? cv::Rect(0, 0, src.cols, src.rows) : add_border(size_dot, target_sidelength);
	} else {
		roi = size_dot;
	}
	roi = adjust_border_for_fft(roi);
	if (target_sidelength != -1) {
		roi = get_square_rect(roi, src_8u.size(), find_center(src_8u, roi));
	}
	cv::Mat src_rect_32f;
	src(roi).copyTo(src_rect_32f);
	return fft_denoise(src_rect_32f, size_dot);
}

cv::Mat generate_2d_beam_profiles(cv::Mat& src, int target_sidelength, int wavelength, cv::Mat& denoised_full) {
	auto result_denoised_bw = get_bw_image(src, target_sidelength, wavelength);
	auto result_denoised_colorized = get_color_image(src, target_sidelength, target_sidelength);
	cv::Mat result = result_denoised_bw.clone();

	if (_draw_miniature) {
		auto s = get_miniature_size();
		auto miniature_colorized = get_color_image(denoised_full, s.width, s.height);
		auto miniature_bw = get_bw_image(denoised_full, s.width, wavelength);
		//write_image(base, "ref", "png", denoised_full);
		result_denoised_colorized = add_miniature(src, result_denoised_colorized, miniature_colorized, cv::Scalar(255, 255, 255));
		result_denoised_bw = add_miniature(src, result_denoised_bw, miniature_bw, cv::Scalar(0, 0, 0));
	}

	draw_scale(result_denoised_bw, cv::Point_<int>(10, 10), (float)target_sidelength/(float)src.cols, src.rows, cv::Scalar(0, 0, 0));
	draw_scale(result_denoised_colorized, cv::Point_<int>(10, 10), (float)target_sidelength/(float)src.cols, src.rows, cv::Scalar(255, 255, 255));
	write_image("bp", "png", result_denoised_colorized);
	//write_image("c", "png", result_denoised_colorized);
	write_image("bw", "png", result_denoised_bw);
	return result;
}

void draw_histogram(cv::Mat& src, cv::Mat& dst, const int hist_size, cv::Scalar color, const int target_sidelength, cv::Rect roi, int wavelength) {
	auto laser_color = wavelength_to_rgb(wavelength);
	cv::Mat hist_i_h, tmp;
	bool transposed = false;
	if (roi.width > roi.height) {
		transposed = true;
		transpose(src(roi), tmp);
	} else {
		tmp = src(roi);
	}
	cv::normalize(tmp, hist_i_h, 0, 255, cv::NORM_MINMAX);
	int bin_w = std::round((double)hist_i_h.rows / hist_size);
	//std::cout << bin_w << std::endl;
	if (!transposed) {
		for( int i = 1; i < hist_size; i++) {   // vertical
			//std::cout << +hist_i_h.at<uchar>(i-1) << " ";
			cv::line(dst,
					cv::Point(0, bin_w*(i)),
					cv::Point(hist_i_h.at<uchar>(i) * _histogram_scale - 1, bin_w*(i)),
					laser_color, 1
					);
			cv::line(dst,
					cv::Point(hist_i_h.at<uchar>(i-1) * _histogram_scale, bin_w*(i-1)),
					cv::Point(hist_i_h.at<uchar>(i) * _histogram_scale, bin_w*(i)),
					color, 2
					);
		}
	} else {
		for( int i = 1; i < hist_size; i++) {   // horizontal
			//std::cout << +hist_i_h.at<uchar>(i-1) << " ";
			cv::line(dst,
					cv::Point(bin_w*(i), target_sidelength),
					cv::Point(bin_w*(i), target_sidelength - hist_i_h.at<uchar>(i) * _histogram_scale + 1),
					laser_color, 1
					);
			cv::line(dst,
					cv::Point(bin_w*(i-1), target_sidelength - hist_i_h.at<uchar>(i-1) * _histogram_scale),
					cv::Point(bin_w*(i), target_sidelength - hist_i_h.at<uchar>(i) * _histogram_scale),
					color, 2
					);
		}
	}
}

void generate_histogram_images(cv::Mat& src, int target_sidelength, std::string name, int wavelength) {
	auto result_denoised_colorized = get_color_image(src, target_sidelength, target_sidelength);
	auto result_denoised_bw = get_bw_image(src, target_sidelength, wavelength);
	cv::Mat resized_bw;
	cv::resize(src, resized_bw, cv::Size(target_sidelength, target_sidelength), 0, 0, cv::INTER_LANCZOS4);
	auto scale_size = draw_scale(result_denoised_bw, cv::Point_<int>(10, 10), (float)target_sidelength/(float)src.cols, src.rows, cv::Scalar(0, 0, 0));
	for (float i = 0; i < result_denoised_bw.cols; i+= scale_size/5) {
		auto color = cv::Scalar(222, 222, 222);
		cv::line(result_denoised_bw, cv::Point(i, 0), cv::Point(i, result_denoised_bw.cols), color, 1);
		cv::line(result_denoised_bw, cv::Point(0, result_denoised_bw.cols - i - 1), cv::Point(result_denoised_bw.cols, result_denoised_bw.cols - i - 1), color, 1);
	}
	for (float i = 0; i < result_denoised_bw.cols; i+= scale_size) {
		auto color = cv::Scalar(128, 128, 128);
		cv::line(result_denoised_bw, cv::Point(i, 0), cv::Point(i, result_denoised_bw.cols), color, 1);
		cv::line(result_denoised_bw, cv::Point(0, result_denoised_bw.cols - i - 1), cv::Point(result_denoised_bw.cols, result_denoised_bw.cols - i - 1), color, 1);
	}
	draw_scale(result_denoised_bw, cv::Point_<int>(10, 10), (float)target_sidelength/(float)src.cols, src.rows, cv::Scalar(0, 0, 0));

	draw_histogram(resized_bw, result_denoised_bw, resized_bw.cols, cv::Scalar(0, 0, 0), target_sidelength, cv::Rect(0, resized_bw.rows/2-1, resized_bw.cols, 1), wavelength);
	draw_histogram(resized_bw, result_denoised_bw, resized_bw.cols, cv::Scalar(0, 0, 0), target_sidelength, cv::Rect(resized_bw.cols/2-1, 0, 1, resized_bw.rows), wavelength);
	draw_scale(result_denoised_colorized, cv::Point_<int>(10, 10), (float)target_sidelength/(float)src.cols, src.rows, cv::Scalar(255, 255, 255));
	write_image("hi", "png", result_denoised_bw);
}

uraster::VertVsOut vertex_shader(const Eigen::Vector3f& vin, const Eigen::Matrix4f& mvp, float t) {
	uraster::VertVsOut vout;
	vout.p = mvp * Eigen::Vector4f(vin[0], vin[1], vin[2], 1.0f);
	float index = vin[1];
	vout.color = Eigen::Vector3f(index, index, index);
	return vout;
}

uraster::VertVsOut vertex_shader_wireframe(const Eigen::Vector3f& vin, const Eigen::Matrix4f& mvp, float t) {
	uraster::VertVsOut vout;
	vout.p = mvp * Eigen::Vector4f(vin[0], vin[1], vin[2], 1.0f);
	float index = vin[1];
	vout.color = Eigen::Vector3f(index, index, index);
	return vout;
}

uraster::Pixel fragment_shader(const uraster::VertVsOut& fsin) {
	uraster::Pixel p;
	int index = std::round(fsin.color[0] * 255.0);
	if (index < 10) index = 10;
	auto at_index = _lut.at<cv::Vec3b>(0, index);
	p.color = Eigen::Vector4f(at_index[0], at_index[1], at_index[2], -1e10f);
	return p;
}

uraster::Pixel fragment_shader_wireframe(const uraster::VertVsOut& fsin) {
	uraster::Pixel p;

	if (!(fsin.position()[3] < 0.04)) {
		p.color = Eigen::Vector4f(0, 0, 0, 0);
		return p;
	}

	int index = std::round(fsin.color[0] * 255.0);
	auto at_index = _lut.at<cv::Vec3b>(0, index);
	p.color = Eigen::Vector4f(at_index[0], at_index[1], at_index[2], fsin.position()[3]);
	p.color = Eigen::Vector4f(at_index[0], at_index[1], at_index[2], 0);
	return p;
}

uraster::Pixel fragment_shader_wireframe_bw(const uraster::VertVsOut& fsin) {
	uraster::Pixel p;

	if (!(fsin.position()[3] < 0.04)) {
		auto col = std::min(fsin.color[0]/2.0 * 255.0 + 127, 255.0);
		p.color = Eigen::Vector4f(col, col, col, 0);
		//p.color = Eigen::Vector4f(255, 255, 255, 0);
		return p;
	}
	p.color = Eigen::Vector4f(0, 0, 0, 0);
	return p;
}

uraster::Pixel fragment_shader_wireframe_bw_shaded(const uraster::VertVsOut& fsin) {
	uraster::Pixel p;

	if (!(fsin.position()[3] < 0.04)) {
		float theta=M_PI;
		Eigen::Vector3f ld(1.0,sin(theta),cos(theta));
		//(fsin.n == normalenvektor dreieck)
		Eigen::Vector3f po;
		po[0] = fsin.p[0];
		po[1] = fsin.p[1];
		po[2] = fsin.p[2];
		float intensity=ld.normalized().dot(po);

		//auto col = std::min(fsin.color()[0]/2.0 * 255.0 + 127, 255.0);
		auto col = (intensity+1)/2.0;
		//col = col * 200 + 55;
		//
		col = 150 + intensity*200;

		col = col < 0 ? 0 : col; col = col > 255 ? 255 : col;
		p.color = Eigen::Vector4f(col, col, col, 0);
		//p.color = Eigen::Vector4f(255, 255, 255, 0);
		return p;
	}
	p.color = Eigen::Vector4f(0, 0, 0, 0);
	return p;
}

void generate_triangulation(std::vector<Eigen::Vector3f>& vectors, std::vector<std::size_t>& tid, cv::Mat& src) {
	int i = 0;
	const int dist = 1;
	for(int y = 0; y < src.rows-2*dist; y+=dist) {
		for(int x = 0; x < src.cols-2*dist; x+=dist) {
			vectors.push_back(Eigen::Vector3f(2.0*x/src.cols - 1, src.at<uchar>(y, x)/255.0, 2.0*y/src.rows - 1));
			tid.push_back(i++);
			vectors.push_back(Eigen::Vector3f(2.0*(x)/src.cols - 1, src.at<uchar>(y+dist, x)/255.0, 2.0*(y+dist)/src.rows - 1));
			tid.push_back(i++);
			vectors.push_back(Eigen::Vector3f(2.0*(x+dist)/src.cols - 1, src.at<uchar>(y, x+dist)/255.0, 2.0*(y)/src.rows - 1));
			tid.push_back(i++);
			vectors.push_back(Eigen::Vector3f(2.0*(x+dist)/src.cols - 1, src.at<uchar>(y+dist, x+dist)/255.0, 2.0*(y+dist)/src.rows - 1));
			tid.push_back(i-1); // reuse previous two vertices
			tid.push_back(i-2);
			tid.push_back(i++);
		}
	}
}

void generate_triangulation_wireframe(std::vector<Eigen::Vector3f>& vectors, std::vector<std::size_t>& tid, cv::Mat& src) {
	int i = 0;
	const int dist = 20;
	for(int y = 0; y < src.rows-2*dist; y+=dist) {
		for(int x = 0; x < src.cols-2*dist; x+=dist) {
			vectors.push_back(Eigen::Vector3f(2.0*x/src.cols - 1, src.at<uchar>(y, x)/255.0, 2.0*y/src.rows - 1));
			tid.push_back(i++);
			vectors.push_back(Eigen::Vector3f(2.0*(x)/src.cols - 1, src.at<uchar>(y+dist, x)/255.0, 2.0*(y+dist)/src.rows - 1));
			tid.push_back(i++);
			vectors.push_back(Eigen::Vector3f(2.0*(x+dist)/src.cols - 1, src.at<uchar>(y, x+dist)/255.0, 2.0*(y)/src.rows - 1));
			tid.push_back(i++);
		}
	}
	for(int y = 0; y < src.rows-2*dist; y+=dist) {
		for(int x = 0; x < src.cols-2*dist; x+=dist) {
			vectors.push_back(Eigen::Vector3f(2.0*(x+dist)/src.cols - 1, src.at<uchar>(y, x+dist)/255.0, 2.0*(y)/src.rows - 1));
			tid.push_back(i++);
			vectors.push_back(Eigen::Vector3f(2.0*(x)/src.cols - 1, src.at<uchar>(y+dist, x)/255.0, 2.0*(y+dist)/src.rows - 1));
			tid.push_back(i++);
			vectors.push_back(Eigen::Vector3f(2.0*(x+dist)/src.cols - 1, src.at<uchar>(y+dist, x+dist)/255.0, 2.0*(y+dist)/src.rows - 1));
			tid.push_back(i++);
		}
	}
}

cv::Mat mo_close(cv::Mat src) {
	cv::Mat dst;
	int morph_size = 1; // 21
	cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size( 2*morph_size + 1, 2*morph_size+1 ), cv::Point( morph_size, morph_size ) );
	cv::morphologyEx( src, dst, cv::MORPH_CLOSE, element, cv::Point(-1, -1), 1);
	cv::morphologyEx( dst, src, cv::MORPH_OPEN, element, cv::Point(-1, -1), 1);
	return src;
}


//cv::Mat mo_thinning(cv::Mat src) {
//	cv::threshold(img, img, 127, 255, cv::THRESH_BINARY);
//	cv::Mat skel(img.size(), CV_8UC1, cv::Scalar(0));
//	cv::Mat temp;
//	cv::Mat eroded;
//
//	cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
//
//	bool done;
//	do
//	{
//		cv::erode(img, eroded, element);
//		cv::dilate(eroded, temp, element); // temp = open(img)
//		cv::subtract(img, temp, temp);
//		cv::bitwise_or(skel, temp, skel);
//		eroded.copyTo(img);
//
//		done = (cv::countNonZero(img) == 0);
//	} while (!done);
//}

void save_3d_image(std::string name, uraster::Framebuffer<uraster::Pixel>& tp, int target_sidelength, bool close = false) {
	cv::Mat result = cv::Mat(target_sidelength, target_sidelength, CV_8UC3);
	for(int y = 0; y < target_sidelength; y++) {
		for(int x = 0; x < target_sidelength; x++) {
			auto px = tp(x, y);
			result.at<cv::Vec3b>(y, x)[0] = px.color[0];
			result.at<cv::Vec3b>(y, x)[1] = px.color[1];
			result.at<cv::Vec3b>(y, x)[2] = px.color[2];
		}
	}
	//if (close) result = mo_close(result);
	write_image(name, "png", result);
}

Eigen::Matrix4f get_camera_matrix() {
	Eigen::Matrix4f camera_matrix = Eigen::Matrix4f::Identity();
	Eigen::Matrix3f m;
	m = Eigen::AngleAxisf(_rot_x*M_PI, Eigen::Vector3f::UnitX()) * Eigen::AngleAxisf(_rot_y*M_PI, Eigen::Vector3f::UnitY()) * Eigen::AngleAxisf(_rot_z*M_PI, Eigen::Vector3f::UnitZ());
	Eigen::Matrix4f camera_transform;
	camera_transform.setIdentity();
	camera_transform.block<3, 3>(0, 0) = m;	// rotation
	camera_matrix *= camera_transform;
	return camera_matrix;
}

void generate_3d_image_triangulation(cv::Mat& src, int target_sidelength) {
	std::vector<Eigen::Vector3f> vectors;
	std::vector<std::size_t> tid;
	generate_triangulation(vectors, tid, src);
	uraster::Framebuffer<uraster::Pixel> tp(target_sidelength, target_sidelength);
	uraster::draw(tp, &*vectors.begin(), &*vectors.end(), &*tid.begin(), &*tid.end(),
			(uraster::VertVsOut*)nullptr, (uraster::VertVsOut*)nullptr,
			std::bind(vertex_shader, std::placeholders::_1, get_camera_matrix(), 0.0),
			fragment_shader
			);
	save_3d_image("3d", tp, target_sidelength);
	vectors.clear();
	tid.clear();
	generate_triangulation_wireframe(vectors, tid, src);
	tp.clear();
	uraster::draw(tp, &*vectors.begin(), &*vectors.end(), &*tid.begin(), &*tid.end(),
			(uraster::VertVsOut*)nullptr, (uraster::VertVsOut*)nullptr,
			std::bind(vertex_shader, std::placeholders::_1, get_camera_matrix(), 0.0),
			fragment_shader_wireframe, true
			);
	save_3d_image("wi", tp, target_sidelength, true);
	uraster::Pixel p;
	p.color = Eigen::Vector4f(255, 255, 255, -1);
	tp.clear(p);
	uraster::draw(tp, &*vectors.begin(), &*vectors.end(), &*tid.begin(), &*tid.end(),
			(uraster::VertVsOut*)nullptr, (uraster::VertVsOut*)nullptr,
			std::bind(vertex_shader, std::placeholders::_1, get_camera_matrix(), 0.0),
			fragment_shader_wireframe_bw_shaded, true
			);
	save_3d_image("wb", tp, target_sidelength, true);
}

void generate_3d_image_wireframe(cv::Mat& src, int target_sidelength) {
	std::vector<Eigen::Vector3f> vectors;
	std::vector<std::size_t> tid;
}


void handle_image(cv::Mat& src, int target_sidelength, int wavelength) {
	auto denoised_square = get_denoised_image(src, target_sidelength);
	cv::Mat denoised_full, denoised_noborder, denoised_noborder_tmp;
	if (_draw_miniature) denoised_full = get_denoised_image(src);
	denoised_noborder_tmp = get_denoised_image(src, target_sidelength, false);
	cv::resize(denoised_noborder_tmp, denoised_noborder, cv::Size(target_sidelength, target_sidelength), 0, 0, cv::INTER_LANCZOS4);
	auto bw_scaled = generate_2d_beam_profiles(denoised_square, target_sidelength, wavelength, denoised_full);
	generate_histogram_images(denoised_square, target_sidelength, "bp", wavelength);
	generate_3d_image_triangulation(denoised_noborder, target_sidelength);
	generate_3d_image_wireframe(denoised_noborder, target_sidelength);
}


int main(int argc, char** argv) {
	std::string infile;
	uint32_t wavelength;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h", "print usage message")
			("input-file,i", po::value<std::string>(&infile)->required(), "input file")
			("output-folder,o", po::value<std::string>(&_base)->default_value("out/"), "output folder")
			("sidelength,s", po::value<uint32_t>(&image_width)->default_value(550), "length of each side, output is square")
			("wavelength,w", po::value<uint32_t>(&wavelength)->default_value(650), "wavelength in nm")
			("gradient,g", po::value<std::string>(&_gradient)->default_value("gradient.png"), "gradient file")
			("watermark,w", po::value<std::string>(&_watermark_file)->default_value("emboss.png"), "watermark")
			("border-factor,b", po::value<float>(&_border_factor)->default_value(0.2f, "0.2"), "ratio to add a border before upscaling")
			("font-scale,f", po::value<float>(&_font_scale)->default_value(0.5f, "0.5"), "font size")
			("histogram-scale,r", po::value<float>(&_histogram_scale)->default_value(0.5f, "0.5"), "histogram scaling")
			("threshold,t", po::value<uint32_t>(&_threshold)->default_value(40), "lower threshold to discard")
			("size-mm,mm", po::value<uint32_t>(&_mm), "size in mm, if embedded in pdf / printed")
			("rotate-x,x", po::value<float>(&_rot_x)->default_value(-0.25f, "-0.25"), "x-axis rotation")
			("rotate-y,y", po::value<float>(&_rot_y)->default_value(0.25f, "0.25"), "y-axis rotation")
			("rotate-z,z", po::value<float>(&_rot_z)->default_value(1.0f, "1"), "z-axis rotation")
			;

		po::positional_options_description p;
		p.add("input-file", -1);
		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).
				options(desc).positional(p).run(), vm);

		if (vm.count("help")) {
			std::cout << desc << "\n";
			return 0;
		}

		po::notify(vm);

		if (vm.count("size-mm")) {
			_draw_miniature = true;
		}
	} catch (std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}

	load_lut();
	load_watermark();

	if (!boost::filesystem::is_directory(_base)) {
		std::cerr << "Error: \"" << _base << "\" is not a directory" << std::endl;
		return 1;
	}
	if (boost::filesystem::is_directory(infile)) {
		std::cerr << "Error: \"" << infile << "\" is a directory" << std::endl;
		return 1;
	}
	if (!boost::filesystem::exists(infile)) {
		std::cerr << "Error: File \"" << infile << "\" does not exist" << std::endl;
		return 1;
	}

	cv::Mat image = cv::imread(infile, CV_LOAD_IMAGE_ANYDEPTH);
	if (image.empty()) {
		std::cerr << "Error: File \"" << infile << "\" not readable" << std::endl;
		return 1;
	}

	std::cout << "Processing: " << infile << std::endl;
	boost::filesystem::path p(infile);
	_base_name = p.stem().string();
	handle_image(image, image_width, wavelength);
	return 0;
}

