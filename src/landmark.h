#pragma once
#ifndef LANDMARKNET_H_
#define LANDMARKNET_H_
#include <string>
#include "net.h"
#include "opencv2/opencv.hpp"


class Landmark {
public:
	Landmark(const std::string &model_path);
	~Landmark();
	void start(const cv::Mat& img, ncnn::Mat &output, int landmark_size_width, int landmark_size_height);
	float ear(int landmark_size_width, int landmark_size_height, float x1, float y1);
	int head_deflection(int landmark_size_width, int landmark_size_height, float x1, float y1);
private:
	void LandmarkNet(ncnn::Mat& img_);
	float eye_ear(int eye_top1 , int eye_top2, int eye_bottom1, int eye_bottom2, int eye_left, int eye_right);
	ncnn::Net Landmark106;
	ncnn::Mat ncnn_img;
	ncnn::Mat out;
	float sw, sh;
	float C1 = 0;
	float C2 = 0;
	float C3 = 0;
	
};
#endif // !MOBILEFACENET_H_
