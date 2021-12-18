#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <math.h>
using namespace cv;
using namespace std;

#define CHANNEL_NUM 3

void show(string window_name, cv::Mat img)
{
	cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
	cv::moveWindow(window_name, 700, 150);
	cv::imshow(window_name, img);
	cv::waitKey(0);
};


////*** Edge detection ***
Mat cannyWrap(cv::Mat& img, double low_thresh, double high_thresh)
{

	// Canny example
	cv::Mat cannyMat;
	double hilo_diff = high_thresh - low_thresh, eps = 10;
	vector<vector<Point> > contours0;
	while (((contours0.size() > 4500) || (contours0.size() < 1000)) && (hilo_diff > 0.1) && (eps > 9)) {
		cv::Canny(img, cannyMat, low_thresh, high_thresh);
		vector<Vec4i> hierarchy;
		double oldcontours = contours0.size();
		findContours(cannyMat, contours0, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
		cout << "I found " << contours0.size() << " contours. FOOL! " << endl;
		eps = pow((oldcontours - contours0.size()), 2);
		if (contours0.size() > 4500) {
			cout << "I found  too many contours. Making lower threshold higher. " << endl;
			low_thresh += 0.5 * hilo_diff;
		}
		else if (contours0.size() < 1000) {
			cout << "I found too few contours. Bringing down the high threshold. " << endl;
			high_thresh -= 0.5 * hilo_diff;
		}
		hilo_diff = high_thresh - low_thresh;

		cout << "high-low diff: " << hilo_diff << " . " << endl;
	}
	cout << "I found " << contours0.size() << " contours. COOL! " << endl;
	return cannyMat;
}



////*** Circle detection ***
void houghCircleWrap(cv::Mat& img, cv::Mat& refimg, string outtext, string outimage, double low_thresh = 100, double high_thresh = 300, double min_dist = 50, double min_rad = 75, double max_rad = 300)
{
	vector<cv::Vec3f> circles;
	ofstream outputtext{ outtext };
	cout << "WOrking on circles for " << outimage << endl;
	cv::HoughCircles(img, circles, cv::HOUGH_GRADIENT, 1.5, min_rad, low_thresh, high_thresh); //, min_rad, max_rad);

	outputtext << "#Image file: " << outimage << endl;
	outputtext << "#Circle; Center" << endl;
	for (int i = 0; i < circles.size(); i++)
	{

		cv::Vec3i c = circles[i];
		cv::Point center = cv::Point(c[0], c[1]);
		int radius = c[2];
		circle(refimg, center, radius, cv::Scalar(0, 255, 0), 3);
		outputtext << center.x << "," << center.y << endl;

	}
	outputtext << "#Characteristics: " << min_dist << " " << low_thresh << " " << high_thresh << " " << min_rad << " " << max_rad << endl;
	show("eqimage", refimg);
	imwrite(outimage, refimg);


}


Vec2f CircleTexttoVec(string circletext) {
	Vec2f circleVec;
	// Split string 

	// atof each part of the tuple
	// assign to x, y in Vector
	return circleVec;
}


vector<string> printCircle(string myline, string indelim = ",")
{
	/// <summary>
	/// Convert line of format "double,double" to a  Vector of Strings
	/// If the first character in the line is a "#", ignore the line
	/// </summary>
	/// <param name="myline"></param>
	/// <param name="indelim"></param>
	/// <returns></returns>
	vector<string> circle;
	int pos;
	if (myline.find("#") == 0)
	{
		return circle;
	}
	else
	{
		while ((pos = myline.find(indelim)) != string::npos)
		{
			// cout << myline.substr(0, pos) << endl;
			circle.push_back(myline.substr(0, pos));
			circle.push_back(myline.substr(pos + indelim.length(), string::npos));
			// cout << myline.substr(pos + indelim.length(), string::npos) << endl;
			myline.erase(0, pos + indelim.length());
		}
	}
	return circle;
}

vector<Vec2f> readF1File(string filename) {

	vector<Vec2f> myCircles;
	vector<string> mycirgle;
	string line;
	ifstream infile{ filename };

	if (!infile.bad())
	{
		while (!infile.eof())
		{

			getline(infile, line);

			mycirgle = printCircle(line);

			if (!mycirgle.empty())
			{
				Vec2f singleCircle(stod(mycirgle.front()), stod(mycirgle.back()));
				myCircles.push_back(singleCircle);
			}
		}

	}
	return myCircles;
}


double F1Finder(string predFile, string trueFile, double threshold) {
	/// <summary>
	/// F1Finder: calculates
	/// </summary>
	/// <param name="pred"></param>
	/// <param name="argv"></param>
	/// <returns> F1 score</returns>

	vector<Vec2f> predCircles = readF1File(predFile), trueCircles = readF1File(trueFile);
	double fitness = 0.0, precision, recall;
	float truepos = 0, fpos = 0;
	float pos = predCircles.size();


	for (Vec2f predpt : predCircles) {
		for (Vec2f truept : trueCircles) {

			double dist = sqrt(cv::norm(truept, predpt));
			if (dist < threshold) {
				truepos++;
			}
			else {
				fpos++;
			}
		}

	}
	precision = truepos / pos;
	recall = truepos / trueFile.length();
	fitness = 2 * (precision * recall) / (precision + recall);
	return fitness;
}

Mat gammaCorrection(const Mat& img, const double gamma_)
{
	CV_Assert(gamma_ >= 0);
	//! [changing-contrast-brightness-gamma-correction]
	Mat lookUpTable(1, 256, CV_8U);
	uchar* p = lookUpTable.ptr();
	for (int i = 0; i < 256; ++i)
		p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma_) * 255.0);

	Mat res = img.clone();
	LUT(img, lookUpTable, res);
	//! [changing-contrast-brightness-gamma-correction]


	imshow("Gamma correction", res);
	return res;
}


Mat eqimage;
void drawCircle(int action, int x, int y, int flags, void* userdata)
{
	Mat eqimage2 = eqimage.clone();
	Vec2f centerVec, circleVec;
	Point center;
	// Mark the top left corner when left mouse button is pressed
	if (action == EVENT_LBUTTONDOWN)
	{
		centerVec = Vec2f(x, y);
		center = Point(x, y);
	}
	// When left mouse button is released, mark bottom right corner
	else if (action == EVENT_LBUTTONUP)
	{
		circleVec = Vec2f(x, y);
		double calc_rad = sqrt(cv::norm(centerVec, circleVec));
		circle(eqimage2, center, calc_rad, Scalar(0, 255, 0), 2, 8);
		imshow("Window", eqimage2);
	}
}

int main(int argc, char** argv)
{
	/*
	Main function:
		Usage: circlefinder [image_file_list.txt] [minradius] [maxradius]
		- Check arguments [min radius, max radius], help string
	*/


	if (argc < 2) {
		cout << "Please supply command line arguments" << endl;
		cout << "Enter `CoinDetector0 help` into terminal for help" << endl;
	}
	else if (strcmp(argv[1], "help") == 0) {
		cout << "USAGE: CoinDetector0 [imagefile.txt] [lower_thresh] [higher_thresh] [min_dist] [min rad] [max rad]" << endl;
		cout << "Where " << endl;
		cout << "[imagefile.txt] is a plain text file giving the file path for each image, one per line \n[lower thresh] is the lower threshold for non-max suppression\n[higher thresh] is the upper threshold for non-max suppression\n[min dist] is minimum distance between circle centers\n[min rad] is the minimum radus of a circle\n[max rad] is the maximum radius of a circle" << endl;


	}
	else {
		Mat image, smallimage, greyimage, eqimage, edgeimage, circleimage;
		string filename = argv[1], line, strInput, greyout, eqout,
			edgeout, textout, circleout;
		ifstream infile{ filename };
		Point2i  circ_center;
		vector<cv::Point2i> circle_centers;

		double lower_thresh = atof(argv[2]), upper_thresh = atof(argv[3]), min_dist = atof(argv[4]), min_rad = atof(argv[5]), max_rad = atof(argv[6]);

		if (!infile.bad())
		{
			while (!infile.eof()) {

				getline(infile, line);
				if (line.length() < 4) {
					break;
				}
				image = imread(line, IMREAD_COLOR); // Read the file

				if ((image.cols > 1200) || (image.rows > 1200)) {
					resize(image, smallimage, Size(), 0.1725, 0.1725, INTER_AREA);
				}
				else {
					resize(image, smallimage, Size(), 1.0, 1.0, INTER_AREA);
				}

				// assume file extension is last four characters 
				int pos = line.length() - 4;
				string short_line = line.substr(0, pos);

				greyout = short_line + "greyout.jpg";
				eqout = short_line + "eqout.jpg";
				edgeout = short_line + "edgeout.jpg";
				circleout = short_line + "circleout.jpg";
				textout = short_line + "report.txt";

				if (smallimage.empty()) // Check for invalid input
				{
					cout << "Could not open or find the image" << std::endl;
					return -1;
				}

				cout << "Starting " << line << endl;
				bilateralFilter(smallimage, greyimage, 90, 180, 45);
				cvtColor(greyimage, greyimage, cv::COLOR_BGR2GRAY);
				show("greyimage", greyimage);
				imwrite(greyout, greyimage);

				eqimage = gammaCorrection(greyimage, 1.1);
				show("eqimage", eqimage);
				imwrite(eqout, eqimage);


				//	medianBlur(eqimage, eqimage, 5);
					/*namedWindow("Window", WINDOW_AUTOSIZE);
					setMouseCallback("Window", drawCircle, NULL);
					imshow("eqimage", eqimage);
					waitKey(0);*/




				edgeimage = cannyWrap(eqimage, lower_thresh, upper_thresh);
				imwrite(edgeout, edgeimage);
				show("edgeimage", edgeimage);

				//annotateCircle(eqimage);

				houghCircleWrap(edgeimage, smallimage, textout, circleout,
					lower_thresh, upper_thresh, min_dist, min_rad, max_rad);


				cout << "Done with " << line << endl;
			}
		}

		return 0;
	}
}