#include <stdio.h>
#include <iostream>
#include <opencv2\core\core.hpp> //basic data type
#include <opencv2\highgui\highgui.hpp> //input/output
#include <opencv2\imgproc\imgproc.hpp> //process images

using namespace cv;
using namespace std;
Mat img, kernel;
string kernelType = "k_identical";

void multiplyValue(double val)
{
	for (int i = 0; i < kernel.rows; i++)
	{
		for (int j = 0; j < kernel.cols; j++)
		{
			kernel.ptr<double>(i)[j] *= val;
		}
	}
}
void loadKernel()
{
	kernel.create(3, 3, CV_64FC1);

	if (kernelType == "k_identical")
	{
		kernel.ptr<double>(0)[0] = 0; //1st row
		kernel.ptr<double>(0)[1] = 0;
		kernel.ptr<double>(0)[2] = 0;
		kernel.ptr<double>(1)[0] = 0; //2nd row
		kernel.ptr<double>(1)[1] = 1;
		kernel.ptr<double>(1)[2] = 0;
		kernel.ptr<double>(2)[0] = 0;//3rd row
		kernel.ptr<double>(2)[1] = 0;
		kernel.ptr<double>(2)[2] = 0;

		multiplyValue((double)1);
	}
	else if (kernelType == "k_gaussian")
	{
		kernel.create(5, 5, CV_64FC1);

		kernel.ptr<double>(0)[0] = 1; //1st row
		kernel.ptr<double>(0)[1] = 4;
		kernel.ptr<double>(0)[2] = 7;
		kernel.ptr<double>(0)[3] = 4;
		kernel.ptr<double>(0)[4] = 1;
		kernel.ptr<double>(1)[0] = 4; //2nd row
		kernel.ptr<double>(1)[1] = 16;
		kernel.ptr<double>(1)[2] = 26;
		kernel.ptr<double>(1)[3] = 16;
		kernel.ptr<double>(1)[4] = 4;
		kernel.ptr<double>(2)[0] = 7;//3rd row
		kernel.ptr<double>(2)[1] = 26;
		kernel.ptr<double>(2)[2] = 41;
		kernel.ptr<double>(2)[3] = 26;
		kernel.ptr<double>(2)[4] = 7;
		kernel.ptr<double>(3)[0] = 4; //4th row
		kernel.ptr<double>(3)[1] = 16;
		kernel.ptr<double>(3)[2] = 26;
		kernel.ptr<double>(3)[3] = 16;
		kernel.ptr<double>(3)[4] = 4;
		kernel.ptr<double>(4)[0] = 1; //5th row
		kernel.ptr<double>(4)[1] = 4;
		kernel.ptr<double>(4)[2] = 7;
		kernel.ptr<double>(4)[3] = 4;
		kernel.ptr<double>(4)[4] = 1;

		multiplyValue((double)1 / 273);
	}
	else if (kernelType == "k_mean")
	{
		kernel.create(5, 5, CV_64FC1);

		kernel.ptr<double>(0)[0] = 1; //1st row
		kernel.ptr<double>(0)[1] = 1;
		kernel.ptr<double>(0)[2] = 1;
		kernel.ptr<double>(0)[3] = 1;
		kernel.ptr<double>(0)[4] = 1;
		kernel.ptr<double>(1)[0] = 1; //2nd row
		kernel.ptr<double>(1)[1] = 1;
		kernel.ptr<double>(1)[2] = 1;
		kernel.ptr<double>(1)[3] = 1;
		kernel.ptr<double>(1)[4] = 1;
		kernel.ptr<double>(2)[0] = 1;//3rd row
		kernel.ptr<double>(2)[1] = 1;
		kernel.ptr<double>(2)[2] = 1;
		kernel.ptr<double>(2)[3] = 1;
		kernel.ptr<double>(2)[4] = 1;
		kernel.ptr<double>(3)[0] = 1; //4th row
		kernel.ptr<double>(3)[1] = 1;
		kernel.ptr<double>(3)[2] = 1;
		kernel.ptr<double>(3)[3] = 1;
		kernel.ptr<double>(3)[4] = 1;
		kernel.ptr<double>(4)[0] = 1; //5th row
		kernel.ptr<double>(4)[1] = 1;
		kernel.ptr<double>(4)[2] = 1;
		kernel.ptr<double>(4)[3] = 1;
		kernel.ptr<double>(4)[4] = 1;

		multiplyValue((double)1 / 25);
	}
	else if (kernelType == "k_edge")
	{
		kernel.ptr<double>(0)[0] = -1; //1st row
		kernel.ptr<double>(0)[1] = -1;
		kernel.ptr<double>(0)[2] = -1;
		kernel.ptr<double>(1)[0] = -1; //2nd row
		kernel.ptr<double>(1)[1] = 8;
		kernel.ptr<double>(1)[2] = -1;
		kernel.ptr<double>(2)[0] = -1;//3rd row
		kernel.ptr<double>(2)[1] = -1;
		kernel.ptr<double>(2)[2] = -1;

		multiplyValue((double)1);
	}
	else if (kernelType == "k_vertical")
	{
		kernel.ptr<double>(0)[0] = -1; //1st row
		kernel.ptr<double>(0)[1] = 0;
		kernel.ptr<double>(0)[2] = 1;
		kernel.ptr<double>(1)[0] = -2; //2nd row
		kernel.ptr<double>(1)[1] = 0;
		kernel.ptr<double>(1)[2] = 2;
		kernel.ptr<double>(2)[0] = -1;//3rd row
		kernel.ptr<double>(2)[1] = 0;
		kernel.ptr<double>(2)[2] = 1;

		multiplyValue((double)1);
	}
	else if (kernelType == "k_horizontal")
	{
		kernel.ptr<double>(0)[0] = -1; //1st row
		kernel.ptr<double>(0)[1] = -2;
		kernel.ptr<double>(0)[2] = -1;
		kernel.ptr<double>(1)[0] = 0; //2nd row
		kernel.ptr<double>(1)[1] = 0;
		kernel.ptr<double>(1)[2] = 0;
		kernel.ptr<double>(2)[0] = 1;//3rd row
		kernel.ptr<double>(2)[1] = 2;
		kernel.ptr<double>(2)[2] = 1;

		multiplyValue((double)1);
	}
	else if (kernelType == "k_sharpen")
	{
		kernel.ptr<double>(0)[0] = 0; //1st row
		kernel.ptr<double>(0)[1] = -1;
		kernel.ptr<double>(0)[2] = 0;
		kernel.ptr<double>(1)[0] = -1; //2nd row
		kernel.ptr<double>(1)[1] = 5;
		kernel.ptr<double>(1)[2] = -1;
		kernel.ptr<double>(2)[0] = 0;//3rd row
		kernel.ptr<double>(2)[1] = -1;
		kernel.ptr<double>(2)[2] = 0;

		multiplyValue((double)1);
	}
	//cout << kernel << endl;

	filter2D(img, img, -1, kernel);

}
int main(int argc, char** argv)
{
	img = imread("test.jpg");
	VideoCapture camera(0);

	while (1)
	{
		camera >> img;

		char c = waitKey(1); //listen to your keyboard

		loadKernel();

		if (c == 'g')
			kernelType = "k_gaussian";
		if (c == 'i')
			kernelType = "k_identical";
		if (c == 'm')
			kernelType = "k_mean";
		if (c == 'e')
			kernelType = "k_edge";
		if (c == 'v')
			kernelType = "k_vertical";
		if (c == 'h')
			kernelType = "k_horizontal";
		if (c == 's')
			kernelType = "k_sharpen";

		imshow("camera", img);

		if (c == 27) //27 - "Esc" ASCII
			break;
	}
}