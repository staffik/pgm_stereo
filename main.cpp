/*
MIT License

Copyright (c) 2019 Adam Chudaś

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

// Based on http://nghiaho.com/?page_id=1366

#include <iostream>
#include <opencv2/highgui/highgui.hpp>

const int ITER_NUM = 42;
const int LABELS_NUM = 16;
const int WINDOW_RADIUS = 2;

const int INF = 1e9;
const int BORDER = LABELS_NUM;
const int WINDOW_SIZE = (2 * WINDOW_RADIUS + 1) * (2 * WINDOW_RADIUS + 1);

int directions[][2] = {{-1, 0},	// left
				  	   {1, 0},	// right
				  	   {0, -1},	// up
				       {0, 1}};	// down

cv::Mat img_L, img_R;
int width, height, size;

struct Node {
	int potential[LABELS_NUM];
	int msg[4][LABELS_NUM];
	int MAP_assignment;
};

std::vector<Node> Graph;

void read_imgs() {
	img_L = cv::imread("img_L.png", 0);
	img_R = cv::imread("img_R.png", 0);

	width = img_L.cols;
	height = img_L.rows;
	size = width * height;
}

int unary_cost(int x, int y, int disparity) {
	int sum_diff = 0;

	for (int yy = y - WINDOW_RADIUS; yy <= y + WINDOW_RADIUS; ++yy) {
		for (int xx = x - WINDOW_RADIUS; xx <= x + WINDOW_RADIUS; ++xx) {
			int pix_L = img_L.at<uchar>(yy, xx);
			int pix_R = img_R.at<uchar>(yy, xx - disparity);
			sum_diff += abs(pix_L - pix_R);
		}
	}
	
	int avg_diff = sum_diff / WINDOW_SIZE;
	
	return avg_diff;
}

void init_msgs() {
	Graph.resize(size);

	for (auto& n : Graph) {
		memset(n.potential, 0, sizeof n.potential);
		memset(n.msg, 0, sizeof n.msg);
	}

	for (int y = BORDER; y < height - BORDER; ++y) {
		for (int x = BORDER; x < width - BORDER; ++x) {
			for (int i = 0; i < LABELS_NUM; ++i) {
				Graph[y * width + x].potential[i] = unary_cost(x, y, i);
			}
		}
	}
}

int pairwise_cost(int i, int j) {
	return 16 * std::min(abs(i - j), 4);
}

int belief(const Node& n, int label) {
	int result = n.potential[label];

	for (int i = 0; i < 4; ++i) {
		result += n.msg[i][label];
	}

	return result;
}

bool valid_coordinates(int x, int y) {
	return (x >= 0) && (x < width) && (y >= 0) && (y < height);
}

void update_msg(int x, int y, int direction, int label) {
	const Node& sender = Graph[y * width + x];
	int min_cost = INF;

	for (int i = 0; i < LABELS_NUM; ++i) {
		int cost = pairwise_cost(label, i);
		cost += belief(sender, i);
		cost -= sender.msg[direction][i];
		
		min_cost = std::min(min_cost, cost);
	}
	
	int nx = x + directions[direction][0];
	int ny = y + directions[direction][1];
	int opposite_direction = direction ^ 1;

	if (!valid_coordinates(nx, ny)) {
		return;
	}

	Graph[ny * width + nx].msg[opposite_direction][label] = min_cost;
}

void update_msgs() {
	for (int d = 0; d < 4; ++d) {
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				for (int i = 0; i < LABELS_NUM; ++i) {
					update_msg(x, y, d, i);
				}
			}
		}
	}
}

void calc_MAP() {
	for (auto& n : Graph) {
		int min_cost = belief(n, 0);
		n.MAP_assignment = 0;

		for (int i = 1; i < LABELS_NUM; ++i) {
			int cost = belief(n, i);

			if (cost < min_cost) {
				min_cost = cost;
				n.MAP_assignment = i;
			}
		}
	}
}

void write_results() {
	cv::Mat reconstruction(cv::Mat::zeros(height, width, CV_8U));
	
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			reconstruction.at<uchar>(y, x) = Graph[y * width + x].MAP_assignment * (256 / LABELS_NUM);
		}
	}

	cv::imwrite("img_out.png", reconstruction);
}

int main() {

	read_imgs();
	init_msgs();

	for (int i = 0; i < ITER_NUM; ++i) {
		std::cout << "Iter #" << i << std::endl;
		update_msgs();
	}

	calc_MAP();
	write_results();

	return 0;
}
