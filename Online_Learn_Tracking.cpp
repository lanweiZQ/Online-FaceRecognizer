/*
 * DetectionBasedTracker_example.cpp
 *
 * Copyright 2012 RITESH <ritesh@ritsz>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * Do anything You want with the code. :)
 */

//------------------------------ INCLUDES ------------------------------------------------

#include "iostream"
#include "opencv/cv.h"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/contrib/detection_based_tracker.hpp"


//------------------------------ NAMESPACES -----------------------------------------------

using namespace cv;
using namespace std;


//------------------------------- GLOBALS -------------------------------------------------

//vector< Rect_<int> > faces;
vector< DetectionBasedTracker::Object > faces;
vector<cv::Mat> learnt_face;
vector<int> learnt_label;
cv::Rect_<int> face_i;

bool pred = false;

Ptr<cv::FaceRecognizer> model = cv::createLBPHFaceRecognizer();

void learn(cv::Mat image,int label)
{
    learnt_face.push_back(image);
    learnt_label.push_back(label);

    if(learnt_face.size() >= 20)
    {
        learnt_face.erase(learnt_face.begin());
        learnt_label.erase(learnt_label.begin());
    }
}

void train()
{
    model->train(learnt_face,learnt_label);
}


//------------------------------- MAIN -----------------------------------------------------

int main( int argc,char* argv[])
{
    // Parameter Strucrture for DetectionBasedTracker
    DetectionBasedTracker::Parameters param;
    param.maxObjectSize = 400;
    param.maxTrackLifetime = 10;
    param.minDetectionPeriod = 1;
    param.minNeighbors = 4;
    param.minObjectSize = 25;
    param.scaleFactor = 1.1;

    // The constructer is called with the cascade of choice and the Parameter structure.Then run it.

    DetectionBasedTracker obj = DetectionBasedTracker("haarcascade_frontalface_alt.xml",param);
    obj.run();

    // Create the FaceRecognizer model

     /*  Also supported FaceRecognizer models are Fisherface(LDA) and Local Binary Pattern Histogram(LBPH)

        Ptr<cv::FaceRecognizer> model = cv::createFisherFaceRecognizer();
        Ptr<cv::FaceRecognizer> model = cv::createLBPHFaceRecognizer();

    */

    // Start VideoCapture
    VideoCapture cap(-1);

    Mat img,gray_img,crop_face,crop_face_res,img_ld;

    int once = 1;
    int largest_label = 0;

    cv::namedWindow("Detection Based Tracker",cv::WINDOW_AUTOSIZE);
    //cv::setWindowProperty("Detection Based Tracker", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);

    for(;;)
    {
        cap>>img;

        // EigenFaces works only on GrayScaleImages
        cv::cvtColor(img,gray_img,CV_RGB2GRAY);

        // Detect faces
        obj.process(gray_img);
        obj.getObjects(faces);

        int i = 0;
        for(vector<DetectionBasedTracker::Object>::iterator it = faces.begin() ; it != faces.end() ; it++ ,i ++)
        {

            if(i>=largest_label) largest_label = i;
            //faces is a pair<Rect,int> hence faces.first and faces.second
            face_i = it->first;
            int n_face = it->second;
            std::cout<<endl<<endl<<n_face<<std::endl<<endl;

            rectangle(img, face_i, CV_RGB(0, 255,0), 2);
            crop_face = gray_img(face_i);
            cv::resize(crop_face, crop_face_res, Size(100,100), 1.0, 1.0, INTER_CUBIC);
			cv::equalizeHist(crop_face_res,crop_face_res);

			if((!crop_face.empty()) && pred == false  )
			{
				cout<<"Learning"<<endl;
				cv::rectangle(img, face_i, CV_RGB(255*i, 255*(i-1),255*(i+1)), 1);

                learn(crop_face_res.clone(),i);
                once = 1;
                string box_text = format("Learning Model = %d", i);
				int pos_x = std::max(face_i.tl().x - 10, 0);
				int pos_y = std::max(face_i.tl().y - 10, 0);
				putText(img, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);

            }


				else if((!crop_face.empty()) && pred == true)
			{
					     // train the model

					     if(once == 1)
					     {
					        train();
                            once++;
					     }

				cout<<"Predicting"<<endl;
				int prediction = -1;
				double predicted_confidence = 0.0;

                // Predict the label and confidence for each image
				model->predict(crop_face_res,prediction,predicted_confidence);
				if(predicted_confidence >= 40 && predicted_confidence <= 55 && prediction<largest_label)
				{
				    learn(crop_face_res.clone(),prediction);
				    train();

				}

				if(predicted_confidence>75 || prediction > largest_label)
				{
				    learn(crop_face_res.clone(),-1);
				    train();
				    continue;
				}
				cv::rectangle(img, face_i, CV_RGB(0, 0,255), 1);

				string box_text = format("Prediction = %d  Confidence = %f", prediction,predicted_confidence);

				int pos_x = std::max(face_i.tl().x - 10, 0);
				int pos_y = std::max(face_i.tl().y - 10, 0);

				putText(img, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
			}

        }

    cv::imshow("Detection Based Tracker",img);      // Show the results.
    char esc = cv::waitKey(33);

    if(esc == 27) break;
    if(esc == 32) pred = !pred;

    }

    cout<<learnt_face.size();
    obj.stop();

	return 0;
}
