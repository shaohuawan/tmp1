#include "main.h"

void defeatRepair(const Mat img, const float th, Mat& out)
{
    out = img.clone();
    for (int i = 1; i < img.rows - 1; i++)
    {
        const float* prev = img.ptr<float>(i - 1);
        const float* curr = img.ptr<float>(i);
        const float* next = img.ptr<float>(i + 1);
        for (int j = 1; j < img.cols - 1; j++)
        {
            if (curr[j] > curr[j - 1] * th && curr[j] > curr[j + 1] * th && curr[j] > prev[j] * th && curr[j] > next[j] * th)
            {
                out.at<float>(i, j) = 0.25*(curr[j - 1] + curr[j + 1] + prev[j] + next[j]);
            }
            else if (curr[j] < curr[j - 1] * th && curr[j] < curr[j + 1] * th && curr[j] < prev[j] * th && curr[j] < next[j] * th)
            {
                out.at<float>(i, j) = 0.25*(curr[j - 1] + curr[j + 1] + prev[j] + next[j]);
            }
        }
    }
    return;
}

void findSimilarPatch(const Mat img, const int patchSz, const int searchSz, vector<Point>& locs)
{
    Mat intergralImg;
    cv::integral(img, intergralImg);
}