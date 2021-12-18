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


// Define our callback which we will install for
// mouse events
//
void my_mouse_callback(
	int event, int x, int y, int flags, void* param
);

// Global variables we _have_ to have for the annotation mousecallback
Rect box;
vector<Point2f> circlecenters;
bool drawing_box = false;

// A little subroutine to draw a box onto an image
//
void draw_box(cv::Mat& img, cv::Rect box) {
	cv::rectangle(
		img,
		box.tl(),
		box.br(),
		cv::Scalar(0x00, 0x00, 0xff)    /* red */
	);
}


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
		//cout << "Commented line" << endl;
		return circle;
	}
	else if (myline.empty()) {
		//cout << "Empty Line" << endl;
		return circle;
	}
	else
	{
		while ((pos = myline.find(indelim)) != string::npos)
		{
			//	cout << myline.substr(0, pos) << endl;
			circle.push_back(myline.substr(0, pos));
			circle.push_back(myline.substr(pos + indelim.length(), string::npos));
			//	cout << myline.substr(pos + indelim.length(), string::npos) << endl;
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
	cout << "Reading file " << filename << endl;

	while (getline(infile, line))
	{
		// strip whitespace
		std::string::iterator end_pos = std::remove(line.begin(), line.end(), ' ');
		line.erase(end_pos, line.end());

		mycirgle = printCircle(line);

		if (!mycirgle.empty())
		{
			Vec2f singleCircle(stod(mycirgle.front()), stod(mycirgle.back()));
			myCircles.push_back(singleCircle);
		}


	}
	cout << "Done with file " << filename << endl;
	return myCircles;
}


double F1Finder(string predFile, string trueFile, double threshold) {
	/// <summary>
	/// F1Finder: calculates F1 score from two "circle report" files 
	/// Where each line that starts with a `#` is ignored 
	/// and each line of the format `double,double` is read into an OpenCV Vec2f 
	/// </summary>
	/// <param name="pred"></param>
	/// <param name="argv"></param>
	/// <returns> F1 score</returns>

	vector<Vec2f> predCircles = readF1File(predFile), trueCircles = readF1File(trueFile);
	double fitness = 0.0, precision = 0.0, recall = 0.0;
	float truepos = 0, fpos = 0;
	float pos = predCircles.size();


	for (Vec2f predpt : predCircles) {
		float start_true = truepos;
		for (Vec2f truept : trueCircles) {

			double dist = cv::norm(truept, predpt, NORM_L2);
			if (dist < threshold) {
				truepos++;

				cout << "True positive: " << predpt << " vs " << truept << "at " << dist << endl;
				remove_copy(trueCircles.begin(), trueCircles.end(), trueCircles.begin(), truept);
				break;
			}

		}
		// found no matches? 
		if (truepos == start_true) {
			fpos++;
			cout << "False positive: " << predpt << endl;
		}

	}
	precision = truepos / pos;
	recall = truepos / trueCircles.size();
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
		Mat image, smallimage, greyimage, eqimage, edgeimage, circleimage, temp;
		string filename = argv[1], line, strInput, greyout, eqout,
			edgeout, textout, truetextin, circleout;
		ifstream infile{ filename };
		Point2i  circ_center;
		vector<cv::Point2i> circle_centers;
		vector <double> f1s;

		double lower_thresh = atof(argv[2]), upper_thresh = atof(argv[3]), min_dist = atof(argv[4]), min_rad = atof(argv[5]), max_rad = atof(argv[6]);

		if (!infile.bad())
		{
			while (getline(infile, line)) {

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
				truetextin = short_line + "true_circles.txt";

				if (smallimage.empty()) // Check for invalid input
				{
					cout << "Could not open or find the image" << std::endl;
					return -1;
				}

				cout << "Starting " << line << endl;
				bilateralFilter(smallimage, greyimage, 36, 72, 72);
				cvtColor(greyimage, greyimage, cv::COLOR_BGR2GRAY);
				//show("greyimage", greyimage);
				imwrite(greyout, greyimage);

				eqimage = gammaCorrection(greyimage, 1.1);
				//show("eqimage", eqimage);
				imwrite(eqout, eqimage);


				medianBlur(eqimage, eqimage, 5);

				// CHECK if [Name]true_circles.txt exists!
				ifstream truecirclecheck;
				truecirclecheck.open("b.txt");
				if (truecirclecheck) {
					cout << truetextin + " file exists";
				}
				else {
					cout << truetextin + " file doesn't exist. Dropping into ANNOTATION MODE.";
					box = cv::Rect(-1, -1, 0, 0);


					cv::namedWindow("Drag Boxes to Select Coins; press <ESC> to exit");

					cv::setMouseCallback(
						"Drag Boxes to Select Coins; press <ESC> to exit",
						my_mouse_callback,
						(void*)&eqimage
					);

					for (;;) {

						eqimage.copyTo(temp);
						if (drawing_box) draw_box(temp, box);
						cv::imshow("Drag Boxes to Select Coins; press <ESC> to exit", temp);

						if (cv::waitKey(15) == 27) break;
					}
					ofstream outputtext{ truetextin };
					outputtext << "#Image file: " << truetextin << endl;
					outputtext << "#Circle; Center" << endl;
					for (Point2d elem : circlecenters) {
						cout << "Annotated Circle is: " << elem << endl;

						outputtext << elem.x << "," << elem.y << endl;

					}
					cout << "ANNOTATION IS DONE." << endl;

				}



				edgeimage = cannyWrap(eqimage, lower_thresh, upper_thresh);
				imwrite(edgeout, edgeimage);
				//show("edgeimage", edgeimage);

				//annotateCircle(eqimage);

				houghCircleWrap(edgeimage, smallimage, textout, circleout,
					lower_thresh, upper_thresh, min_dist, min_rad, max_rad);

				double f1 = F1Finder(textout, truetextin, 0.33 * min_rad);
				f1s.push_back(f1);

				string f1string = "# best f1: " + to_string(f1);
				cout << "F1  for " << line << ": " << f1 << endl;
				ofstream appendout;

				//append f1 score to outputfile
				appendout.open(textout, std::ios_base::app);
				appendout << f1string << endl;

				cout << "F1  for " << line << ": " << f1 << endl;
				cout << "Done with " << line << endl;

			}
		}

		// reporting
		double avg_f1 = 0;
		for (double elem : f1s) {
			avg_f1 += elem / f1s.size();
		}
		cout << "Average F1 from coin images: " << avg_f1 << endl;
		return 0;
	}
}


void my_mouse_callback(
	int event, int x, int y, int flags, void* param
) {

	cv::Mat& image = *(cv::Mat*)param;
	switch (event) {

	case cv::EVENT_MOUSEMOVE: {
		if (drawing_box) {
			box.width = x - box.x;
			box.height = y - box.y;
		}
	}
							break;

	case cv::EVENT_LBUTTONDOWN: {
		drawing_box = true;
		box = cv::Rect(x, y, 0, 0);
	}
							  break;

	case cv::EVENT_LBUTTONUP: {
		drawing_box = false;
		if (box.width < 0) {
			box.x += box.width;
			box.width *= -1;
		}
		if (box.height < 0) {
			box.y += box.height;
			box.height *= -1;
		}
		draw_box(image, box);

		// b = a + (b-a)
		// a + (b-a)/2 = a/2 + b/2 -> midpoint 
		Point2f difference = Point2f(box.br() - box.tl());
		Point2f center = 0.5 * difference + Point2f(box.tl());

		int rad = (int)round(0.5 * sqrt(box.area()));
		circle(image, center, rad, cv::Scalar(0xf0, 0x03, 0xff));
		circlecenters.push_back(center);
	}
							break;
	}

}