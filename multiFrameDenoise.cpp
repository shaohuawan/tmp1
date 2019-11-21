#include"main.h"

int intraGroupDenoise(vector<Mat> groupImgs, Mat& out)//input: CV_8UC3, output: CV_8UC3
{
    if (groupImgs.size() == 1)
    {
        groupImgs[0].copyTo(out);
        return 0;
    }
    else if (groupImgs.size() > 1)
    {
        Mat reference;
        vector<Mat> imgs;
        vector<float> weights;
        float refWeight;
        int refIdx = refSelection(groupImgs, 0.f, reference, imgs, refWeight, weights);
        stDenoise(reference, imgs, out, refWeight, weights);
        return refIdx;
    }
    else
        return -1;
    
}

int interGroupDenoise(vector<Mat> denoisedImgs, vector<int> refIdxs, int groupSz, Mat& finalOut)//input: CV_8UC3, output: CV_32FC3
{
    if (denoisedImgs.size() == 1)
    {
        denoisedImgs[0].copyTo(finalOut);
        return refIdxs[0];
    }
    else
    {
        Mat reference;
        vector<Mat> imgs;
        vector<float> weights;
        float refWeight;
        int refIdx = refSelection(denoisedImgs, 1, reference, imgs, refWeight, weights);
        refIdx = groupSz * refIdx + refIdxs[refIdx];
        stDenoise(reference, imgs, finalOut, refWeight, weights);
        return refIdx;
    }   
}

void stDenoise(const Mat ref, const vector<Mat>& imgs, Mat& out, float refWeight, vector<float> weights)
{
    out = ref.clone();
    vector<Mat> alignedImgs, masks, noisePatchs;
    int tileSz = 16;
    float c = 26;
    //float c = 70; // for 20 images
    //float c = 80; //for 10 images
    int level = align(ref, imgs, alignedImgs, masks);
    Mat refLab;
    ref.convertTo(refLab, CV_8UC3, 255);
    cv::cvtColor(refLab, refLab, cv::COLOR_BGR2YUV);
    cv::Vec3f sigmas = multiChannelNLE(refLab, 5, 100, 0.9327, 1.4758, noisePatchs);
    refLab.convertTo(refLab, CV_32FC3, 1. / 255);
    for (int i = 0; i < alignedImgs.size(); i++)
    {
        alignedImgs[i].convertTo(alignedImgs[i], CV_8UC3, 255);
        cv::cvtColor(alignedImgs[i], alignedImgs[i], cv::COLOR_BGR2YUV);
        alignedImgs[i].convertTo(alignedImgs[i], CV_32FC3, 1. / 255);
    }
    //std::cout << "|  sigma: " << sigmas << endl;
    std::cout << "|  |-  merge!" << std::endl;
    multiChannleFrequenceMerge(refLab, alignedImgs, masks, noisePatchs, out, refWeight, weights,level, c, 10.f, tileSz);
    out.convertTo(out, CV_8UC3);
	cv::cvtColor(out, out, cv::COLOR_YUV2BGR);
	return;
}

static void defeatRepair(Mat& img, const float th)
{
    Mat out = img.clone();
    for (int i = 1; i < img.rows - 1; i++)
    {
        const unsigned char* prev = img.ptr<unsigned char>(i - 1);
        const unsigned char* curr = img.ptr<unsigned char>(i);
        const unsigned char* next = img.ptr<unsigned char>(i + 1);
        unsigned char* pout = out.ptr<unsigned char>(i);
        for (int j = 1; j < img.cols - 1; j++)
        {
            if (curr[j] > th * curr[j - 1] && curr[j] > th * curr[j + 1] && curr[j] > th * prev[j] && curr[j] > th* next[j])
            {
                pout[j] = 0.25*(curr[j - 1] + curr[j + 1] + prev[j] + next[j]);
            }
            else if (curr[j] < th * curr[j - 1] && curr[j] < th * curr[j + 1] && curr[j] < th * prev[j] && curr[j] < th * next[j])
            {
                pout[j] = 0.25*(curr[j - 1] + curr[j + 1] + prev[j] + next[j]);
            }
        }
    }
    img = out.clone();
    return;
}

void imgDefeatRepair(Mat& img, const float th)
{
    vector<Mat> imgs;
    cv::split(img, imgs);
    for (int i = 0; i < imgs.size(); i++)
    {
        defeatRepair(imgs[i], th);
    }
    cv::merge(imgs, img);
}
