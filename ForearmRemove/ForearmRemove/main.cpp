#include<iostream>
#include<fstream>
#include<opencv.hpp>
#include<string>
#include<vector>
#include<chrono>
#include<algorithm>
using std::cin;
using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::string;
using std::vector;
using cv::Mat;
using cv::imshow;
using cv::imread;
using cv::waitKey;
using cv::Point;

void MaxCircleInContour(const vector<cv::Point>& contour, const Mat&  _imgBin, cv::Point& P, double& R)//Ѱ���������Բ
{
	P = cv::Point(-1, -1);
	R = -1;
	for (int i = 0; i < _imgBin.rows - 4; i += 4)
		for (int j = 0; j < _imgBin.cols - 4; j += 4)
			if (_imgBin.at<uchar>(i, j) == 255)
			{
				double minDist = DBL_MAX;
				for (int k = 0; k < (int)contour.size() - 4; k += 4)
				{
					double thisDist = (contour[k].x - j) * (contour[k].x - j) + (contour[k].y - i) * (contour[k].y - i);
					if (thisDist < minDist)
					{
						minDist = thisDist;
					}
				}
				if (R < minDist)
				{
					R = minDist;
					P = cv::Point(j, i);
				}
			}
	if (R > 0) R = sqrt(R);
}

//meanΪ��ֵ��pc_orinetΪ���ɷֵ�ָ��sc_orientΪ�γɷݵ�ָ��
void apply_pca(const vector<cv::Point>& pointset, Point& mean, Point& pc_orinet, Point& sc_orient)
{
	int sz = static_cast<int>(pointset.size());
	Mat pts_mat = Mat(sz, 2, CV_64FC1);         /*�������vectorת��Ϊһ��Mat��Mat��ÿ��Ϊһ����*/
	for (int i = 0; i < pts_mat.rows; ++i)
	{
		*((double*)(pts_mat.data + i*pts_mat.step[0] + 0 * pts_mat.step[1])) = pointset[i].x;
		*((double*)(pts_mat.data + i*pts_mat.step[0] + 1 * pts_mat.step[1])) = pointset[i].y;
	}

	//��2�������� InputArray mean��ƽ��ֵ����������ǿյģ�noArray()����������ݼ��㣻
	//CV_PCA_DATA_AS_ROW��ʾ��������ÿһ�б�ʾһ������
	cv::PCA pca(pts_mat, Mat(), CV_PCA_DATA_AS_ROW);

	//�������ݵľ�ֵ����ά����ƽ��
	cv::Point2d mean_point = cv::Point2d(pca.mean.at<double>(0, 0), pca.mean.at<double>(0, 1));

	vector<cv::Point2d> eigen_vecs(2);   //��������������Ϊ1
	vector<double> eigen_val(2);         //����ֵ
	for (int i = 0; i < 2; ++i)
	{
		eigen_vecs[i] = cv::Point2d(pca.eigenvectors.at<double>(i, 0), pca.eigenvectors.at<double>(i, 1));
		eigen_val[i] = pca.eigenvalues.at<double>(i);
	}

	//cout << eigen_vecs[0].x << "," << eigen_vecs[0].y << endl;  
	//cout << eigen_vecs[1].x << "," << eigen_vecs[1].y << endl;
	//cout << eigen_val[0] <<"   " << eigen_val[1]<< endl;

	cv::Point2d principle;
	cv::Point2d secondary;
	if (eigen_val[0] > eigen_val[1])
	{
		principle = 0.02*cv::Point2d(eigen_vecs[0].x*eigen_val[0], eigen_vecs[0].y*eigen_val[0]);
		secondary = 0.02*cv::Point2d(eigen_vecs[1].x*eigen_val[1], eigen_vecs[1].y*eigen_val[1]);
	}
	else
	{
		principle = 0.02*cv::Point2d(eigen_vecs[1].x*eigen_val[1], eigen_vecs[1].y*eigen_val[1]);
		secondary = 0.02*cv::Point2d(eigen_vecs[0].x*eigen_val[0], eigen_vecs[0].y*eigen_val[0]);
	}

	mean = Point((int)mean_point.x, (int)mean_point.y);
	pc_orinet = Point((int)principle.x, (int)principle.y);
	sc_orient = Point((int)secondary.x, (int)secondary.y);
}

//��㼯��ֵ
Point get_points_mean(const vector<Point>& point_set)
{
	int len = static_cast<int>(point_set.size());
	if (len == 0)
	{
		return Point(-1, -1);    /*(-1,-1)��ʾ�쳣�����ʵ��vector<Point>Ϊ��<<endl;*/
	}
	int sum_x = 0;
	int sum_y = 0;
	for (int i = 0; i < len; i++)
	{
		sum_x += point_set[i].x;
		sum_y += point_set[i].y;
	}
	return Point(sum_x / len, sum_y / len);
}

//����ͷ����ʼ��δstart��ָ��end���ܳ���Ϊscale���ģ�end-start������
void drawAxis(Mat& canvas, Point start, Point end, cv::Scalar color, int thickness = 1, const double scale = 1)
{
	double angle;
	double len;
	angle = atan2((double)start.y - end.y, (double)start.x - end.x); // angle in radians
	len = sqrt((double)(start.y - end.y) * (start.y - end.y) + (start.x - end.x) * (start.x - end.x));

	end.x = (int)(start.x - scale * len * cos(angle));
	end.y = (int)(start.y - scale * len * sin(angle));
	line(canvas, start, end, color, thickness, CV_AA);

	start.x = (int)(end.x + 9 * cos(angle + CV_PI / 4));
	start.y = (int)(end.y + 9 * sin(angle + CV_PI / 4));
	line(canvas, start, end, color, thickness, CV_AA);

	start.x = (int)(end.x + 9 * cos(angle - CV_PI / 4));
	start.y = (int)(end.y + 9 * sin(angle - CV_PI / 4));
	line(canvas, start, end, color, thickness, CV_AA);
}

//��canvas�ϻ�pca���ɷֵ�ָ��
void draw_pc_orient(Mat& canvas, const vector<cv::Point>& pointset,
	cv::Scalar color, int thickness = 1, const double scale = 1.0)
{
	Point mean_point;
	Point principle_orient;
	Point secondary_orinet;
	apply_pca(pointset, mean_point, principle_orient, secondary_orinet);
	Point end((int)(mean_point.x + principle_orient.x), (int)(mean_point.y + principle_orient.y));
	drawAxis(canvas, Point((int)mean_point.x, (int)mean_point.y), end, color, thickness, scale);
}

void remove_arm()
{
	const double dist_coefficient = 1.7;
	const double arm_area_thresh = 400;

	const int max_order = 21;
	for (int i = 0; i <= max_order; i++)
	{
		cout << "src_mask_" << i << ": " << endl;
		string path = "image data/maskimg_witharm_" + std::to_string(i) + ".jpg";
		Mat mask = imread(path, 0);
		if (mask.empty()) { cout << "not read" << path << endl; continue; }
		cv::threshold(mask, mask, 128, 255, cv::THRESH_BINARY);
		imshow("src_mask", mask); waitKey(0);

		vector<vector<cv::Point> > temp_conts;
		cv::findContours(mask, temp_conts, CV_RETR_TREE, cv::CHAIN_APPROX_NONE);
		//cout << "temp_conts.size():"<< temp_conts.size() << endl;
		assert(temp_conts.size() == 1);
		const vector<cv::Point>& src_hand_cont = temp_conts[0];

		Mat canvas;//����
		cv::cvtColor(mask, canvas, cv::COLOR_GRAY2BGR);

		//�������Բ
		cv::Point palm_center;
		double palm_radius;
		MaxCircleInContour(src_hand_cont, mask, palm_center, palm_radius);
		//�����������Բ�뾶��Բ��Ϊ��ɫ
		cv::circle(canvas, palm_center, (int)(palm_radius*dist_coefficient), cv::Scalar(0, 0, 0), -1);
		//�����������Բ
		cv::circle(canvas, palm_center, (int)palm_radius, cv::Scalar(255, 0, 0));
		//imshow("canvas", canvas); waitKey(0);

		//�ٴμ������
		Mat mask_without_incircle;
		mask.copyTo(mask_without_incircle);
		cv::circle(mask_without_incircle, palm_center, (int)(palm_radius*dist_coefficient), cv::Scalar(0), -1);//�ڳ�����Բ
		vector<vector<cv::Point> > contours;
		cv::findContours(mask_without_incircle, contours, CV_RETR_TREE, cv::CHAIN_APPROX_NONE);
		if (contours.empty())
		{
			cout << "src_mask_" << i << "��ǰ��" << endl;
			continue;
		}

		//��������
		for (unsigned int i = 0; i < contours.size(); i++)
		{
			cv::drawContours(canvas, contours, i, cv::Scalar(0, 0, 255), 1);
		}
		imshow("�������", canvas); waitKey(0);

		//����pcaָ���ж�ǰ������
		Point src_hand_cont_mean;
		Point principle_orient;
		Point secondary_orinet;
		apply_pca(src_hand_cont, src_hand_cont_mean, principle_orient, secondary_orinet);
		//src_hand_cont_mean--->cont_center[i]����principle_orient�ļн�
		vector<Point> cont_center(contours.size());
		for (unsigned int i = 0; i<cont_center.size(); i++)
		{
			cont_center[i] = get_points_mean(contours[i]);
		}
		vector<double> angles(contours.size());
		for (unsigned int i = 0; i < cont_center.size(); i++)
		{
			Point orient = cont_center[i] - src_hand_cont_mean;
			double dx1 = orient.x;
			double dy1 = orient.y;
			double dx2 = principle_orient.x;
			double dy2 = principle_orient.y;
			double cosval = (dx1*dx2 + dy1*dy2) / sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
			angles[i] = acos(cosval) * 180 / 3.14159265;
		}
		int arm_cont_index = (int)(std::min_element(angles.begin(), angles.end()) - angles.begin());
		if (cv::contourArea(contours[arm_cont_index]) < arm_area_thresh)
		{
			cout << "src_mask_" << i << "���ֱ�" << endl;
			continue;
		}

		//��pca
		draw_pc_orient(canvas, src_hand_cont, cv::Scalar(0, 255, 0), 2, 1.5);
		imshow("canvas", canvas); waitKey(0);

		//����ǰ������contours[arm_index]
		cv::drawContours(canvas, contours, arm_cont_index, cv::Scalar(0, 255, 0), 2);
		imshow("canvas", canvas); waitKey(0);

		//ȥ��ǰ��
		cv::drawContours(mask, contours, arm_cont_index, cv::Scalar(0), -1);
		imshow("ȥ��ǰ��", mask);  waitKey(0);
	}
}

int main()
{
	remove_arm();
}