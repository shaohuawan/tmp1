#include"main.h"
#include<omp.h>

static int matSqrSum(Mat img, int tileSz, Mat& dst)
{
    Mat src;
    if (img.isContinuous())
        src = img;
    else
        src = img.clone();
    //src.convertTo(src, CV_32F);
    dst = Mat::zeros(img.size(), src.type());
    Mat rst = Mat::zeros(img.size(), src.type());
    CV_Assert(img.cols >= tileSz && img.rows >= tileSz);
    //float* pSrc;
    //float* pRst;
    //float* pDst;

    //ROW
//#pragma omp parallel for
    for (int i = 0; i < img.rows; i++)
    {
        float* pSrc = src.ptr<float>(i);
        float* pRst = rst.ptr<float>(i);
        for (int k = 0; k < tileSz && k < img.cols; k++)
        {
            pRst[0] += pSrc[k] * pSrc[k];
        }
    }
//#pragma omp parallel for
    for (int i = 0; i < img.rows; i++)
    {
        float* pSrc = src.ptr<float>(i);
        float* pRst = rst.ptr<float>(i);
        for (int j = 1; j < img.cols - tileSz; j++)
        {
            pRst[j] = pRst[j - 1] - pSrc[j - 1] * pSrc[j - 1] + pSrc[j + tileSz - 1] * pSrc[j + tileSz - 1];
        }
        for (int j = img.cols - tileSz; j < img.cols; j++)
        {
            pRst[j] = pRst[j - 1] - pSrc[j - 1] * pSrc[j - 1];
        }
    }
    //COL
//#pragma omp parallel for
    for (int j = 0; j < img.cols; j++)
    {
        for (int k = 0; k < tileSz && k < img.rows; k++)
        {
            dst.ptr<float>(0)[j] += rst.ptr<float>(k)[j];
        }
    }
//#pragma omp parallel for
    for (int j = 0; j < img.cols; j++)
    {
        float* pRst = rst.ptr<float>(0) + j;
        float* pDst = dst.ptr<float>(0) + j;
        int i = 1;
        for (i = 1; i < img.rows - tileSz; i++)
        {
            dst.ptr<float>(i)[j] = dst.ptr<float>(i - 1)[j] - rst.ptr<float>(i - 1)[j] + rst.ptr<float>(i + tileSz - 1)[j];
        }
        for (; i < img.rows; i++)
        {
            dst.ptr<float>(i)[j] = dst.ptr<float>(i - 1)[j] - rst.ptr<float>(i - 1)[j];
        }
    }

    return 0;
}

static int matSum(Mat img, int tileSz, Mat& dst)
{
    Mat src;
    if (img.isContinuous())
        src = img;
    else
        src = img.clone();
    //src.convertTo(src, CV_32F);
    dst = Mat::zeros(img.size(), src.type());
    Mat rst = Mat::zeros(img.size(), src.type());
    CV_Assert(img.cols >= tileSz && img.rows >= tileSz);
    //float* pSrc;
    //float* pRst;
    //float* pDst;

    //ROW
#pragma omp parallel for
    for (int i = 0; i < img.rows; i++)
    {
        float* pSrc = src.ptr<float>(i);
        float* pRst = rst.ptr<float>(i);
        for (int k = 0; k < tileSz && k < img.cols; k++)
        {
            pRst[0] += pSrc[k];
        }
    }
#pragma omp parallel for
    for (int i = 0; i < img.rows; i++)
    {
        float* pSrc = src.ptr<float>(i);
        float* pRst = rst.ptr<float>(i);
        int j = 1;
        for (; j < img.cols - tileSz; j++)
        {
            pRst[j] = pRst[j - 1] - pSrc[j - 1] + pSrc[j + tileSz - 1];
        }
        for (; j < img.cols; j++)
        {
            pRst[j] = pRst[j - 1] - pSrc[j - 1];
        }
    }
    //COL
#pragma omp parallel for
    for (int j = 0; j < img.cols; j++)
    {
        for (int k = 0; k < tileSz && k < img.rows; k++)
        {
            dst.ptr<float>(0)[j] += rst.ptr<float>(k)[j];
        }
    }
#pragma omp parallel for
    for (int j = 0; j < img.cols; j++)
    {
        float* pRst = rst.ptr<float>(0) + j;
        float* pDst = dst.ptr<float>(0) + j;
        int i = 1;
        for (i = 1; i < img.rows - tileSz; i++)
        {
            dst.ptr<float>(i)[j] = dst.ptr<float>(i - 1)[j] - rst.ptr<float>(i - 1)[j] + rst.ptr<float>(i + tileSz - 1)[j];
        }
        for (; i < img.rows; i++)
        {
            dst.ptr<float>(i)[j] = dst.ptr<float>(i - 1)[j] - rst.ptr<float>(i - 1)[j];
        }
    }

    return 0;
}

static void matVariance(Mat img, int tileSz, Mat& dst)
{
    Mat sqrSum, avg, variance;
    matSqrSum(img, tileSz, sqrSum);
    matSum(img, tileSz, avg);
    avg = avg / (tileSz*tileSz);
    dst = sqrSum / (tileSz*tileSz) - avg.mul(avg);
    dst.setTo(0, dst < 0);
}

static void patchSelection(Mat img, Mat variance, int tileSz, int maxStep, float lambda, Mat& flg)
{
    flg = Mat::zeros(img.size(), CV_8UC1);
    Mat isSat = Mat::zeros(int(0.8*img.rows), int(0.8*img.cols), CV_8UC1);
    cv::copyMakeBorder(isSat, isSat, int((img.rows - int(0.8*img.rows))*0.5), img.rows - (int((img.rows - int(0.8*img.rows))*0.5) + int(0.8*img.rows)),
        int((img.cols - int(0.8*img.cols))*0.5), img.cols - (int((img.cols - int(0.8*img.cols))*0.5) + int(0.8*img.cols)), cv::BORDER_CONSTANT, cv::Scalar::all(255));
    //assert img is in [0,255]
    isSat.setTo(255, img < 3);
    isSat.setTo(255, img > 252);
    //for (int i = 1; i < img.rows-1; i++)
    //{
    //    unsigned char *ps = isSat.ptr<unsigned char>(i);
    //    float* pimg = img.ptr<float>(i);
    //    for (int j = 1; j < img.cols-1; j++)
    //    {
    //        if (fabs(pimg[j] - pimg[j + 1]) < 1 && fabs(pimg[j] - pimg[j - 1]) < 1 && 
    //            fabs(pimg[j] - pimg[j + img.cols]) < 1 && fabs(pimg[j] - pimg[j - img.cols]) < 1)
    //        {
    //            ps[j] = 255;
    //        }
    //    }
    //}
    isSat.setTo(255, variance < 1. / tileSz / tileSz);
    cv::dilate(isSat, isSat, cv::getStructuringElement(cv::MORPH_RECT, Size(2 * tileSz - 1, 2 * tileSz - 1)));
    vector<float> vv;
    for (int i = 0; i < variance.rows - tileSz; i++)
    {
        float* pv = variance.ptr<float>(i);
        unsigned char *ps = isSat.ptr<unsigned char>(i);
        for (int j = 0; j < variance.cols - tileSz; j++)
        {
            if (!ps[j])
                vv.push_back(pv[j]);
        }
    }
    int initialSz = 10;
    std::partial_sort(vv.begin(), vv.begin() + initialSz, vv.end());
    //std::sort(vv.begin(), vv.end());
    float initialSigma = 0;
    for (auto iter = vv.begin(); iter != vv.begin() + initialSz; iter++)
        initialSigma += *iter;
    initialSigma /= float(initialSz);
    initialSigma = std::max(initialSigma, float(1. / tileSz / tileSz));
    float endSigma = std::accumulate(vv.begin(), vv.end(), 0.) / vv.size();
    float deltaSigma = (endSigma - initialSigma) / maxStep;
    float diff = endSigma - initialSigma;
    int idx = 0;
    float sumSigma = 0;
    for (float sigma = initialSigma; sigma <= endSigma; sigma += deltaSigma)
    {
        for (int i = tileSz - 1; i < variance.rows - tileSz; i++)
        {
            float *pv = variance.ptr<float>(i);
            unsigned char *pflg = flg.ptr<unsigned char>(i);
            unsigned char *ps = isSat.ptr<unsigned char>(i);
            for (int j = tileSz - 1; j < variance.cols - tileSz; j++)
            {
                if (pv[j] < lambda*sigma && !ps[j] && !pflg[j])
                {
                    sumSigma += pv[j];
                    idx++;
                    pflg[j] = 255;
                    isSat(Range(i - tileSz + 1, i + tileSz), Range(j - tileSz + 1, j + tileSz)).setTo(255);
                }
            }
        }
        diff = sumSigma / idx - sigma;
        if (diff <= 0)
            break;
    }
    return;
}

static float dimSelctNLE(float* eigenvalues, int sz)
{
    float sigma = sqrt(eigenvalues[sz - 1]);
    for (int i = 0; i < sz; i++)
    {
        float t = 0;
        for (int j = i; j < sz; j++)
            t += eigenvalues[j];
        t = t / (sz - i + 1);
        int idx1 = std::max((sz + i) / 2 - 1, 0);
        int idx2 = std::min((sz + i) / 2 + 1, sz - 1);
        if (idx1 <= idx2 && t <= eigenvalues[idx1] && t >= eigenvalues[idx2])
        {
            sigma = sqrt(t);
            break;
        }
    }
    return sigma;
}

static float noiseEstimation(Mat img, Mat variance, Mat flg, int tileSz, float rou)
{
#define NOISE_PCA_EST
    int cnt = cv::countNonZero(flg);
    float sigma = 1;
    if (cnt > 0)
    {
#ifdef NOISE_PCA_EST
        Mat texture = Mat::zeros(cnt, tileSz*tileSz, CV_32FC1);
#endif
        int idx = 0;
        float avgSigma = 0;
        for (int i = 0; i < img.rows - tileSz; i++)
        {
            unsigned char* pflg = flg.ptr<unsigned char>(i);
            float *pv = variance.ptr<float>(i);
            for (int j = 0; j < img.cols - tileSz; j++)
            {
                if (pflg[j])
                {
#ifdef NOISE_PCA_EST
                    Mat roi = img(Range(i, i + tileSz), Range(j, j + tileSz)).clone();
                    roi = roi.reshape(1, 1);
                    roi.copyTo(texture.row(idx));
                    idx++;
#endif
                    avgSigma += pv[j];
                }
            }
        }
        avgSigma /= cnt;
        sigma = sqrt(avgSigma / rou);
#ifdef NOISE_PCA_EST
        if (cnt > 10 * tileSz * tileSz)
        {
            cv::PCA pca(texture, cv::noArray(), cv::PCA::DATA_AS_ROW);
            float* eigval = new float[tileSz*tileSz];
#pragma omp parallel for
            for (int i = 0; i < tileSz*tileSz; i++)
                eigval[i] = std::max(pca.eigenvalues.at<float>(i, 0), 0.f);
            //sigma = pca.eigenvalues.at<float>(1, 0);
            sigma = dimSelctNLE(eigval, tileSz*tileSz);
            //sigma = pca.eigenvalues.at<float>(1, 0);
            //sigma = pca.eigenvalues.at<float>(tileSz*tileSz-1,0);
            sigma = sigma / sqrt(rou);
            delete[]eigval;
        }
#endif
#undef NOISE_PCA_EST
    }

    return sigma;
}

float singleChannleNLE(Mat img, int tileSz, int maxStep, float rou, float lambda, Mat& flg)
{
    //assert single channel, RGB
    Mat variance;
    if (img.type() != CV_32FC1)
        img.convertTo(img, CV_32FC1);
    matVariance(img, tileSz, variance);
    patchSelection(img, variance, tileSz, maxStep, lambda, flg);
    float sigma = noiseEstimation(img, variance, flg, tileSz, rou);
    return sigma;
}

float singleChannleNLE2(Mat img, int tileSz, int maxStep, float rou, float lambda, const Mat flg)
{
    //assert single channel, RGB
    Mat variance;
    if (img.type() != CV_32FC1)
        img.convertTo(img, CV_32FC1);
    matVariance(img, tileSz, variance);
    Mat flg2 = flg.clone();
    float sigma = noiseEstimation(img, variance, flg2, tileSz, rou);
    return sigma;
}

cv::Vec3f multiChannelNLE(Mat img, int tileSz, int maxStep, float rou, float lambda, vector<Mat>& flgs)
{
    vector<Mat> channels;
    cv::split(img, channels);
    cv::Vec3f sigmas;
    flgs.resize(3);
    for (int i = 0; i < 3; i++)
    {
        float sigma = singleChannleNLE(channels[i], tileSz, maxStep, rou, lambda, flgs[i]);
        sigmas[i] = sigma;
    }
    return sigmas;
}