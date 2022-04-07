#include "benchmark.h"
#include "cpu.h"
#include "datareader.h"
#include "net.h"
#include "gpu.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <vector>
#include <algorithm>

#include "mtcnn.h"
#define MAXFACEOPEN 1

#include "landmark.h"
#include "mobilefacenet.h"
#include "yolov3.h"

char name_text[3][256];
char name_path[256];
int eye_flag = 0;
double start_t;
cv::Mat icon_head0;
cv::Mat icon_head1;
cv::Mat icon_smoke0;
cv::Mat icon_smoke1;
cv::Mat icon_phone0;
cv::Mat icon_phone1;
cv::Mat icon_sleep0;
cv::Mat icon_sleep1;
cv::Mat icon_roi;

int smoke_c = 0;
int smoke_nc = 0;
int phone_c = 0;
int phone_nc = 0;

cv::Mat extractCircularMask(cv::Mat img, int col, int row, int r) {
    cv::Mat cirMask = img.clone();
    cirMask.setTo(cv::Scalar::all(0));
    cv::circle(cirMask, cv::Point(col, row), r, cv::Scalar(255, 255, 255), -1, 8, 0);
    return cirMask;
}

cv::Mat icon_frame(cv::Mat frame, cv::Mat icon, cv::Mat icon_roi, int x, int y, int width, int height) {
   cv::Mat new_frame = frame(cv::Rect(x, y, width, height)).clone();

    for (int j = 0; j< icon.rows; j++)
    {
	    for (int i = 0; i< icon.cols; i++)
	    {
            
            if(icon_roi.at<cv::Vec3b>(j, i)[0] == 0)
		        icon.at<cv::Vec3b>(j, i)[0] = new_frame.at<cv::Vec3b>(j, i)[0];
            else
                icon.at<cv::Vec3b>(j, i)[0] = icon.at<cv::Vec3b>(j, i)[0];

            if(icon_roi.at<cv::Vec3b>(j, i)[1] == 0)
		        icon.at<cv::Vec3b>(j, i)[1] = new_frame.at<cv::Vec3b>(j, i)[1];
            else
                icon.at<cv::Vec3b>(j, i)[1] = icon.at<cv::Vec3b>(j, i)[1];

            if(icon_roi.at<cv::Vec3b>(j, i)[2] == 0)
		        icon.at<cv::Vec3b>(j, i)[2] = new_frame.at<cv::Vec3b>(j, i)[2];
            else
                icon.at<cv::Vec3b>(j, i)[2] = icon.at<cv::Vec3b>(j, i)[2];
	    }                     
    }

    return icon;
}

static cv::Mat draw_objects(const cv::Mat &bgr, const std::vector<Object> &objects)
{
	static const char* labels[] = {"background", "smoke", "phone"};
	unsigned char color_val[3]={0,0,0};
	cv::Mat image = bgr.clone();
	Object obj;
	float temp = 0;
	float smoke_prob=0;
	float phone_prob=0;
	cv::Rect smoke_rect;
	cv::Rect phone_rect;
	for (size_t i = 0; i < objects.size(); i++)
	{
		obj = objects[i];

		switch (obj.label)
		{
			case 1:
				color_val[0] = 255;
				color_val[1] = 0;
				color_val[2] = 255; 
				if(obj.prob > smoke_prob)
				{
					smoke_prob = obj.prob;
					smoke_rect = obj.rect;
				}
			break;
			case 2:
				color_val[0] = 255;
				color_val[1] = 255;
				color_val[2] = 0; 
				if(obj.prob > phone_prob)
				{
				phone_prob = obj.prob;
				phone_rect = obj.rect;
				}
			break;
		}
	}

  	if(smoke_prob>0.7)
  	{
		smoke_c++;
		if(smoke_c > 3)
  		{
			cv::rectangle(image, smoke_rect, cv::Scalar(255, 0, 255),2);		
			icon_smoke1.copyTo(image(cv::Rect(400, 0, 60, 60)));
			cv::circle(image, cv::Point(400+30, 0+30), 30, cv::Scalar(73, 80, 207), 1);
		}
	}
	else
	{
		smoke_nc++;
		if(smoke_nc > 3) smoke_c = 0;
	}
    
 	
  	if(phone_prob>0.7)
  	{
		phone_c++;
		if(phone_c > 3)
  		{
			cv::rectangle(image, phone_rect, cv::Scalar(255, 255, 0),2);		
			icon_phone1.copyTo(image(cv::Rect(480, 0, 60, 60))); 
			cv::circle(image, cv::Point(480+30, 0+30), 30, cv::Scalar(73, 80, 207), 1);   
		}	
	}
    else
	{
		phone_nc++;
		if(phone_nc > 3) phone_c = 0;
	}

	return image;
}

int runlandmark(cv::Mat& roi, cv::Mat& image, Landmark &landmark, Yolov3 &yolov3, float x1, float y1)
{
    ncnn::Mat landmark_out;
	landmark.start(roi, landmark_out, 112, 112);
	float ear = landmark.ear(112, 112, x1, y1);
	printf("ear =%f\n",ear);
	int turn = landmark.head_deflection(112, 112, x1, y1);
	printf("turn =%d\n",turn);

	if(turn == 1){
		icon_head1.copyTo(image(cv::Rect(240, 0, 60, 60)));
		cv::circle(image, cv::Point(240+30, 0+30), 30, cv::Scalar(73, 80, 207), 1);
		eye_flag=0;
	}
	else{

		if(ear < 0.15f  ){
			
			if(eye_flag == 0) 
			{
				start_t = ncnn::get_current_time();
				eye_flag=1;
			}
		    printf("ear_time=%f\n",(double)(ncnn::get_current_time() - start_t)/1000.f);
        
			if((double)(ncnn::get_current_time() - start_t)/1000.f > 1.0f){
			
            	icon_sleep1.copyTo(image(cv::Rect(320, 0, 60, 60)));
				cv::circle(image, cv::Point(320+30, 0+30), 30, cv::Scalar(73, 80, 207), 1);
			}
        }
		else
		{
			eye_flag=0;
		}

		std::vector<Object> objects;
		yolov3.start(image, 128, 128, objects);
		image = draw_objects(image, objects);
	}


	float sw, sh;
	sw = (float)roi.cols/(float)112;
	sh = (float)roi.rows/(float)112;
    
    for (int i = 0; i < 106; i++)
    {
        float px,py;
        px = landmark_out[i*2]*112*sw+x1;
        py = landmark_out[i*2+1]*112*sh+y1;
	    cv::circle(image, cv::Point(px, py), 1, cv::Scalar(255,255,255),-1);
    }
	return 0;
}

int face_datection_mtcnn(cv::Mat& bgr, cv::Mat& image, MTCNN &mtcnn, Landmark &landmark, Yolov3 &yolov3)
{

    //cv::Mat bgr = data.clone();
    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows);
    std::vector<Bbox> finalBbox;
#if(MAXFACEOPEN==1)
    mtcnn.detectMaxFace(ncnn_img, finalBbox);
#else
    mtcnn.detect(ncnn_img, finalBbox);
#endif

    const int num_box = finalBbox.size();
    std::vector<cv::Rect> bbox;
    bbox.resize(num_box);

    // for(int i = 0; i < num_box; i++){
    //     bbox[i] = cv::Rect(finalBbox[i].x1, finalBbox[i].y1, finalBbox[i].x2 - finalBbox[i].x1 + 1, finalBbox[i].y2 - finalBbox[i].y1 + 1);

    //     for (int j = 0; j<5; j = j + 1)
    //     {
    //         cv::circle(frame, cvPoint(finalBbox[i].ppoint[j], finalBbox[i].ppoint[j + 5]), 2, CV_RGB(0, 255, 0), CV_FILLED);
    //     }
    // }
    //clock_t start_time = clock();
    for(int i = 0; i < num_box; i++)
    {
            
        int x1 = finalBbox[i].x1;
		int y1 = finalBbox[i].y1;
		int x2 = finalBbox[i].x2;
		int y2 = finalBbox[i].y2;
		if (x1 < 0) x1 = 0;
		if (y1 < 0) y1 = 0;
        if (x2 < 0) x2 = 0;
		if (y2 < 0) y2 = 0;
		if (x1 > bgr.cols) x1 = bgr.cols;
		if (y1 > bgr.rows) y1 = bgr.rows;
		if (x2 > bgr.cols) x2 = bgr.cols;
		if (y2 > bgr.rows) y2 = bgr.rows;
        

        cv::Rect r = cv::Rect(x1, y1, x2-x1, y2-y1);
		cv::rectangle(image, r, cv::Scalar(0, 255, 0), 2, 8, 0);
		cv::Mat roi;
		roi = bgr(cv::Rect(x1, y1, x2-x1, y2-y1)).clone();
		runlandmark(roi, image, landmark, yolov3, x1, y1);
		float a = finalBbox[i].x2 - finalBbox[i].x1 + 1;
        //printf("a = %f\n",a);
        float f = 736.66; //distance * object (width or height) pixel size / real object size
        float dis = (f * 13.9)/(a);
		char dis_text[256];
        sprintf(dis_text, "Distance : %.1fcm%", dis);
        cv::putText(image, dis_text, cv::Point(10, 60), cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0, 0, 255),2);
        //printf("dis = %f\n",dis);
    }

	if(num_box==0){
		cv::putText(image, "No face detected", cv::Point(10, 60), cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0, 0, 255),2);
	}	
    return 0;
}


void Euclidean_distance_averag(Recognize &recognize, const std::string &Driver_Information, std::vector<float>&samplefea0,
                                                                        std::vector<float>&samplefea1){
    std::string sample0_files = Driver_Information + "/sample0.jpg";
    std::string sample1_files = Driver_Information + "/sample1.jpg";

    //const char *model_path = "model";
    //Recognize recognize(model_path);
    cv::Mat sampleimg0 = cv::imread(sample0_files);
    cv::Mat sampleimg1 = cv::imread(sample0_files);

    //std::vector<float>samplefea0, sampleimg1;

    recognize.start(sampleimg0, samplefea0);
    recognize.start(sampleimg1, samplefea1);

    printf("Euclidean_distance_averag\n");
}

double calculSimilar_avg(std::vector<float>&croppedfea, std::vector<float>&samplefea0, std::vector<float>&samplefea1){

    double similar0 = calculSimilar(samplefea0, croppedfea);
    double similar1 = calculSimilar(samplefea1, croppedfea);
    
    return (similar0+similar1)/2;
}

int face_recongition(cv::VideoCapture &mVideoCapture, cv::Mat &frame, MTCNN &mtcnn, Recognize &recognize, int &name_count){

    char verification_text[256]="Driver is being authenticated";
    std::vector<float> samplefea0, samplefea1;
    char name_path[256];
    while(1){
        
        mVideoCapture >> frame;
		cv::resize(frame,frame, cv::Size(640,480),0,0,cv::INTER_LINEAR);
        ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);
        std::vector<Bbox> finalBbox;
        
#if(MAXFACEOPEN==1)
        mtcnn.detectMaxFace(ncnn_img, finalBbox);
#else
        mtcnn.detect(ncnn_img, finalBbox);
#endif
        
        const int num_box = finalBbox.size();
        std::vector<cv::Rect> bbox;
        bbox.resize(num_box);
        
        for(int i = 0; i < num_box; i++)
        {
            int x1 = finalBbox[i].x1;
			int y1 = finalBbox[i].y1;
			int x2 = finalBbox[i].x2;
			int y2 = finalBbox[i].y2;
			if (x1 < 0) x1 = 0;
			if (y1 < 0) y1 = 0;
            if (x2 < 0) x2 = 0;
			if (y2 < 0) y2 = 0;
			if (x1 > frame.cols) x1 = frame.cols;
			if (y1 > frame.rows) y1 = frame.rows;
			if (x2 > frame.cols) x2 = frame.cols;
			if (y2 > frame.rows) y2 = frame.rows;
            
            printf("x1=%d\n",x1);
            printf("y2=%d\n",y1);
            printf("x2=%d\n",x2);
            printf("y2=%d\n",y2);
            cv::Rect r = cv::Rect(x1, y1, x2-x1, y2-y1);
            for (int i = 0; i < num_box; i++)
            {
                //cv::rectangle(frame, r, cv::Scalar(0, 255, 0), 2, 8, 0);
                /*
                for (int j = 0; j < 5; j = j + 1)
                {
                    cv::circle(frame, Point(finalBbox[i].ppoint[j], finalBbox[i].ppoint[j + 5]), 2, CV_RGB(0, 255, 0), FILLED);
                }
                */
            }
            cv::Rect face_roi = cv::Rect(100, 40, 420, 380);
            
            if(x1 > 100 && y1 > 40 && x2 < 520 && y2 < 420){

                
                cv::Mat ROI(frame, r);
                cv::Mat croppedImage;
                std::vector<float> croppedfea;
                ROI.copyTo(croppedImage);
                recognize.start(croppedImage, croppedfea);
                //double similar = calculSimilar(samplefea1, croppedfea);
                cv::rectangle(frame, face_roi, cv::Scalar(0, 255, 0), 2, 8, 0);
                
                for (int i = 0; i < name_count ;i++){
                    sprintf(name_path, "User-information/%s", name_text[i]);
                    Euclidean_distance_averag(recognize, name_path, samplefea0, samplefea1);
                    double similar = calculSimilar_avg(croppedfea, samplefea0, samplefea1);
                    printf("similar = %f \n", similar);

                    
                    if(similar>0.8)
                    {
                        return i;
                        break;
                    }
                }

            }
            else{
                cv::rectangle(frame, face_roi, cv::Scalar(0, 0, 255), 2, 8, 0);
            }

        }
        
        //cv::putText(frame, verification_text, cv::Point(10, 240), cv::FONT_HERSHEY_COMPLEX, 1.2, cv::Scalar(0, 0, 255),2);
        imshow("demo",frame);
        cv::waitKey(1);
    }
}

int main()
{
    
	int name_count = 0;
    int name_id = 0;
	char szTest[256] = {0};
    FILE *fp = fopen("User-information/name.txt", "r");
    while(!feof(fp))
	{
		fscanf(fp, "%s\n", &szTest);
        //printf("%s\n",szTest);
        strcpy(name_text[name_count],szTest);
        name_count++;
	}
    printf("%s\n",name_text[0]);
    printf("%s\n",name_text[1]);
    printf("%s\n",name_text[2]);
    sprintf(name_path, "User-information/%s", name_text[0]);
    fclose(fp);

	icon_head0 = cv::imread("icon/head0.jpg");
    cv::resize(icon_head0, icon_head0, cv::Size(60, 60), 0, 0, cv::INTER_LINEAR);
    icon_head1 = cv::imread("icon/head1.jpg");
    cv::resize(icon_head1, icon_head1, cv::Size(60, 60), 0, 0, cv::INTER_LINEAR);

    icon_smoke0 = cv::imread("icon/smoke0.jpg");
    cv::resize(icon_smoke0, icon_smoke0, cv::Size(60, 60), 0, 0, cv::INTER_LINEAR);
    icon_smoke1 = cv::imread("icon/smoke1.jpg");
    cv::resize(icon_smoke1, icon_smoke1, cv::Size(60, 60), 0, 0, cv::INTER_LINEAR);

    icon_phone0 = cv::imread("icon/phone0.jpg");
    cv::resize(icon_phone0, icon_phone0, cv::Size(60, 60), 0, 0, cv::INTER_LINEAR);
    icon_phone1 = cv::imread("icon/phone1.jpg");
    cv::resize(icon_phone1, icon_phone1, cv::Size(60, 60), 0, 0, cv::INTER_LINEAR);

    icon_sleep0 = cv::imread("icon/sleep0.jpg");
    cv::resize(icon_sleep0 ,icon_sleep0, cv::Size(60, 60), 0, 0, cv::INTER_LINEAR);
    icon_sleep1 = cv::imread("icon/sleep1.jpg");
    cv::resize(icon_sleep1, icon_sleep1, cv::Size(60, 60), 0, 0, cv::INTER_LINEAR);

	icon_roi = extractCircularMask(icon_head0, 30, 30, 30);


	Landmark landmark("model/");

	MTCNN mtcnn("model/");
    mtcnn.SetMinFace(40);
	
	Recognize recognize("model/");

	Yolov3 yolov3("model/");

    cv::Mat frame;
    cv::VideoCapture cap(0);
	cap.set(cv::CAP_PROP_FRAME_WIDTH , 1280);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT , 960);
	
	name_id = face_recongition(cap, frame, mtcnn, recognize, name_count);

    while (true)
    {
        cap >> frame;
		cv::resize(frame,frame, cv::Size(640,480),0,0,cv::INTER_LINEAR);
		cv::Mat data = frame.clone();

		icon_frame(frame, icon_head0, icon_roi, 240, 0, 60, 60);
        icon_frame(frame, icon_head1, icon_roi, 240, 0, 60, 60);
        
        icon_frame(frame, icon_sleep0, icon_roi, 320, 0, 60, 60);
        icon_frame(frame, icon_sleep1, icon_roi, 320, 0, 60, 60);

        icon_frame(frame, icon_smoke0, icon_roi, 400, 0, 60, 60);
        icon_frame(frame, icon_smoke1, icon_roi, 400, 0, 60, 60);

        icon_frame(frame, icon_phone0, icon_roi, 480, 0, 60, 60);
        icon_frame(frame, icon_phone1, icon_roi, 480, 0, 60, 60);
        
        
        
        icon_head0.copyTo(frame(cv::Rect(240, 0, 60, 60)));
		cv::circle(frame, cv::Point(240+30, 0+30), 30, cv::Scalar(176, 127, 71), 1);
		icon_sleep0.copyTo(frame(cv::Rect(320, 0, 60, 60)));
		cv::circle(frame, cv::Point(320+30, 0+30), 30, cv::Scalar(176, 127, 71), 1);
		icon_smoke0.copyTo(frame(cv::Rect(400, 0, 60, 60)));
		cv::circle(frame, cv::Point(400+30, 0+30), 30, cv::Scalar(176, 127, 71), 1);
		icon_phone0.copyTo(frame(cv::Rect(480, 0, 60, 60)));
		cv::circle(frame, cv::Point(480+30, 0+30), 30, cv::Scalar(176, 127, 71), 1);


        double start = ncnn::get_current_time();
      	
		face_datection_mtcnn(data, frame, mtcnn, landmark, yolov3);

        double end = ncnn::get_current_time();
        double time = end - start;
        printf("Time:%7.2f \n",time);
		cv::putText(frame, name_text[name_id], cv::Point(10, 30), cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(255, 0, 0),2);
        cv::imshow("demo", frame);
        cv::waitKey(1);
    }
    return 0;
}
