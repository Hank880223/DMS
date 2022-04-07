#include "landmark.h"



Landmark::Landmark(const std::string &model_path) {
    std::string param_files = model_path + "/landmark106.param";
    std::string bin_files = model_path + "/landmark106.bin";
    Landmark106.load_param(param_files.c_str());
    Landmark106.load_model(bin_files.c_str());
}

Landmark::~Landmark() {
    Landmark106.clear();
}

void Landmark::LandmarkNet(ncnn::Mat& img_) {
    ncnn::Extractor ex = Landmark106.create_extractor();
    ex.set_num_threads(4);
	ex.set_light_mode(true);
    ex.input("data", img_);
    ex.extract("bn6_3_bn6_3_scale", out);
}

void Landmark::start(const cv::Mat& img, ncnn::Mat &output, int landmark_size_width, int landmark_size_height) {
	ncnn::Mat ncnn_img = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR, img.cols, img.rows, landmark_size_width, landmark_size_height);
    //数据预处理
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1/127.5f, 1/127.5f, 1/127.5f};
    ncnn_img.substract_mean_normalize(mean_vals, norm_vals);
    LandmarkNet(ncnn_img);
	output = out;
	
	sw = (float)img.cols/(float)landmark_size_width;
	sh = (float)img.rows/(float)landmark_size_height;

}

float Landmark::eye_ear(int eye_top1 , int eye_top2, int eye_bottom1, int eye_bottom2, int eye_left, int eye_right){
	
	float A = abs(eye_top1 - eye_bottom1);
	float B = abs(eye_top2 - eye_bottom2);
	float C = 2.0 * abs(eye_left - eye_right);
	float ear = (A+B)/C;
	return ear;

}

float Landmark::ear(int landmark_size_width, int landmark_size_height, float x1, float y1) {
	
	float eye_left = eye_ear(out[107]*landmark_size_width*sh+y1, out[109]*landmark_size_width*sh+y1,
							 out[115]*landmark_size_width*sh+y1, out[113]*landmark_size_width*sh+y1,
							 out[104]*landmark_size_width*sw+x1, out[110]*landmark_size_width*sw+x1);

    float eye_right = eye_ear(out[119]*landmark_size_width*sh+y1, out[121]*landmark_size_width*sh+y1,
							  out[127]*landmark_size_width*sh+y1, out[125]*landmark_size_width*sh+y1,
							  out[116]*landmark_size_width*sw+x1, out[122]*landmark_size_width*sw+x1);

	float ear = (eye_left+eye_right)/2.f;

	return ear;
}

int Landmark::head_deflection(int landmark_size_width, int landmark_size_height, float x1, float y1) {
	
	int turn_ret = 0;
	float NC0_x = out[92]*landmark_size_width*sw+x1;
	float NR1_x = out[2]*landmark_size_width*sw+x1;
	float NR2_x = out[4]*landmark_size_width*sw+x1;
	float NR3_x = out[6]*landmark_size_width*sw+x1;

	float NL1_x = out[58]*landmark_size_width*sw+x1;
	float NL2_x = out[60]*landmark_size_width*sw+x1;
	float NL3_x = out[62]*landmark_size_width*sw+x1;   

    if(abs(abs(NC0_x - NR1_x)-abs(NL1_x - NC0_x))<5.f)
		C1 = abs(abs(NC0_x - NR1_x)+abs(NL1_x - NC0_x))/2.f;
	if(abs(abs(NC0_x - NR2_x)-abs(NL2_x - NC0_x))<5.f)
		C2 = abs(abs(NC0_x - NR2_x)+abs(NL2_x - NC0_x))/2.f;
	if(abs(abs(NC0_x - NR3_x)-abs(NL3_x - NC0_x))<5.f)
		C3 = abs(abs(NC0_x - NR3_x)+abs(NL3_x - NC0_x))/2.f;


    float R_D1 = abs(NC0_x - NR1_x)/C1;
	float R_D2 = abs(NC0_x - NR2_x)/C2;
	float R_D3 = abs(NC0_x - NR3_x)/C3;
	float R_D = (R_D1 + R_D2 + R_D3) / 3.f;

	float L_D1 = abs(NL1_x - NC0_x)/C1;
	float L_D2 = abs(NL2_x - NC0_x)/C2;
	float L_D3 = abs(NL3_x - NC0_x)/C3;
	float L_D = (L_D1 + L_D2 + L_D3) / 3.f;

	if (R_D1 > 0.f && R_D2 > 0.f && R_D3 > 0.f && R_D < 0.5){
        //printf("turn\n");
		printf("R_D = %f\n",R_D);
        turn_ret = 1;
    }
    else if (L_D1 > 0.f && L_D2 > 0.f && L_D3 > 0.f && L_D < 0.5){
        //printf("turn\n");
		printf("L_D = %f\n",L_D);
        turn_ret = 1;
    }	

	return turn_ret;
}

