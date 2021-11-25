#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

#include <opencv2/imgproc.hpp>
#include <math.h>
using namespace cv;
using namespace std;


void GammaCorrection(cv::Mat& src, cv::Mat& dst, float fGamma)
{
	unsigned char lut[256];
	for (int i = 0; i < 256; i++)
	{
		lut[i] = cv::saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f);
	}
	dst = src.clone();

	const int channels = dst.channels();
	switch (channels)
	{
	case 1:
	{
		cv::MatIterator_<uchar> it, end;
		for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++)
			*it = lut[(*it)];
		break;
	}
	case 3:
	{
		cv::MatIterator_<cv::Vec3b> it, end;
		for (it = dst.begin<cv::Vec3b>(), end = dst.end<cv::Vec3b>(); it != end; it++)
		{
			(*it)[0] = lut[((*it)[0])];
			(*it)[1] = lut[((*it)[1])];
			(*it)[2] = lut[((*it)[2])];
		}
		break;
	}
	}
}

Mat center_pixel(Mat img_mat)
{
	double mid_row, mid_col, n_channels = img_mat.channels();
	Mat split_image[3], fundu, out = Mat::zeros(3, 1, CV_32F);
	
	split(img_mat, split_image);

	for (size_t i = 0; i < n_channels; i++)
	{
		
		
		mid_row = ceil(split_image[i].rows/2.0);
		mid_col = ceil(split_image[i].cols / 2.0);
		fundu = split_image[i].row(mid_row);
		Mat my_center = fundu.col(mid_col).clone();
		out.at<float>(i) += sum(my_center)[0];

	}	
	return out;
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        cout << " Usage: " << argv[0] << " ImageToLoadAndDisplay" << endl;
        return -1;
    }
    Mat image, gcimage, normimage, binimage,totimage;
    image = imread(argv[1], IMREAD_COLOR); // Read the file
	GammaCorrection(image, gcimage, 0.25);
	normalize(image, normimage, 100, 235, NORM_INF);
    Scalar mn, gcmn, find;
	Mat channels[3], fundu;
	split(image, channels);
	//int find;
    mn =  mean(image);
	gcmn = mean(gcimage);
	cvtColor(image, totimage, COLOR_BGR2GRAY);
	threshold(totimage, binimage, 0, 245,THRESH_OTSU);
		
	Mat mc2;
	mc2 = center_pixel(binimage);
	cout << "CENTER : " << mc2 << endl;

    
    
    if (image.empty()) // Check for invalid input
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("Display window", binimage); // Show our image inside it.
    waitKey(0); // Wait for a keystroke in the window
    return 0;
}