//#include "main.h"
#include "guidedFilter.h"

static inline void mydft(Mat img, Mat& frequence)
{
    cv::dft(img, frequence, cv::DFT_COMPLEX_OUTPUT);
    ////fftshift
    //int cx = img.cols / 2;
    //int cy = img.rows / 2;
    //Mat q0(frequence, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    //Mat q1(frequence, cv::Rect(cx, 0, cx, cy));  // Top-Right
    //Mat q2(frequence, cv::Rect(0, cy, cx, cy));  // Bottom-Left
    //Mat q3(frequence, cv::Rect(cx, cy, cx, cy)); // Bottom-Right
    //Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    //q0.copyTo(tmp);
    //q3.copyTo(q0);
    //tmp.copyTo(q3);
    //q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    //q2.copyTo(q1);
    //tmp.copyTo(q2);
    return;
}

static inline void myidft(Mat frequence, Mat& img)
{
    ////fftshift
    //int cx = frequence.cols / 2;
    //int cy = frequence.rows / 2;
    //Mat q0(frequence, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    //Mat q1(frequence, cv::Rect(cx, 0, cx, cy));  // Top-Right
    //Mat q2(frequence, cv::Rect(0, cy, cx, cy));  // Bottom-Left
    //Mat q3(frequence, cv::Rect(cx, cy, cx, cy)); // Bottom-Right
    //Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    //q0.copyTo(tmp);
    //q3.copyTo(q0);
    //tmp.copyTo(q3);
    //q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    //q2.copyTo(q1);
    //tmp.copyTo(q2);
    cv::dft(frequence, img, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE | cv::DFT_INVERSE);
    return;
}

static Mat noiseEstimation(const Mat ref, const vector<Mat>& alignedImgs, const vector<Mat>& masks, int tileSz)
{
    Mat avgImg = ref.clone();
    for (int i = 0; i < alignedImgs.size(); i++)
    {
        avgImg += alignedImgs[i];
    }
    avgImg /= (1. + alignedImgs.size());

    Mat mask = Mat::zeros(ref.size(), CV_32FC1);
    for (int i = 0; i < masks.size(); i++)
    {
        Mat tmp;
        masks[i].convertTo(tmp, CV_32FC1);
        mask += tmp;
    }
    mask /= (masks.size());
    mask.setTo(0, mask < 0.5);

    Rect roi(0.1*ref.cols, 0.1*ref.rows, 0.8*ref.cols, 0.8*ref.rows);
    int rowCnt = roi.height / tileSz;
    int colCnt = roi.width / tileSz;
    Mat difImg = (ref - avgImg)(roi);
    mask = mask(roi);
    vector<Mat> noises;
    noises.resize(rowCnt*colCnt); 
    vector<float> weigths;
    weigths.resize(rowCnt*colCnt);
#pragma omp parallel for
    for (int i = 0; i < rowCnt; i++)
    {
        for (int j = 0; j < colCnt; j++)
        {
            Mat patch = difImg(Rect(j*tileSz, i*tileSz, tileSz, tileSz));
            Mat freq;
            mydft(patch, freq);
            //cv::dct(patch, freq);
            vector<Mat> complex;
            cv::split(freq, complex);
            Mat coef;
            cv::magnitude(complex[0], complex[1], coef);
            coef = coef.mul(coef) / (float)(tileSz*tileSz);
            //Mat coef = freq.mul(freq) / (float)(tileSz*tileSz);
            noises[i*colCnt + j] = coef.clone();
            Mat wpatch = mask(Rect(j*tileSz, i*tileSz, tileSz, tileSz));
            weigths[i*colCnt + j] = cv::sum(wpatch)[0] / (tileSz*tileSz);
        }
    }
    Mat noise = Mat::zeros(tileSz, tileSz, CV_32FC1);
    float w = 0;
    for (int i = 0; i < rowCnt*colCnt; i++)
    {
        noise += noises[i];
        w += weigths[i];
    }
        
    noise /= w;
    return noise;
}

static float noiseEstimation2(const Mat ref, const vector<Mat>& alignedImgs, const vector<Mat>& masks, int tileSz)
{
    Mat avgImg = ref.clone();
    for (int i = 0; i < alignedImgs.size(); i++)
    {
        avgImg += alignedImgs[i];
    }
    avgImg /= (1. + alignedImgs.size());

    Mat mask = Mat::zeros(ref.size(), CV_32FC1);
    for (int i = 0; i < masks.size(); i++)
    {
        Mat tmp;
        masks[i].convertTo(tmp, CV_32FC1);
        mask += tmp;
    }
    mask /= (masks.size());
    mask.setTo(0, mask < 0.5);
    cv::boxFilter(mask, mask, CV_32FC1, Size(tileSz, tileSz));


    float sigma = 0;
    float weight = 0;
    for (int i = 0.1*ref.rows; i < 0.9*ref.rows; i++)
    {
        float* pmask = mask.ptr<float>(i);
        for (int j = 0.1*ref.cols; j < 0.9*ref.cols; j++)
        {
            float var = 0;
            for (int k = 0; k < alignedImgs.size(); k++)
            {
                float dif = alignedImgs[k].ptr<float>(i)[j] - avgImg.ptr<float>(i)[j];
                var += dif * dif;
            }
            var = var / alignedImgs.size();
            sigma += var * pmask[j];
            weight += pmask[j];
        }
    }
    sigma /= weight;
    sigma = sqrt(sigma);
    return sigma;
}

static void decompose(const Mat img, int nscale, vector<Mat>& pyramid)
{
    pyramid.resize(nscale);
    pyramid[0] = img.clone();
    for (int i = 1; i < nscale; i++)
    {
        cv::pyrDown(pyramid[i - 1], pyramid[i]);
    }
    return;
}

static void recompose(vector<Mat>& pyramid, Mat& out)
{
#pragma omp parallel for
    for (int i = 0; i < pyramid.size() - 1; i++)
    {
        Mat up, down;
        cv::pyrDown(pyramid[i], down);
        cv::pyrUp(down, up, pyramid[i].size());
        pyramid[i] = pyramid[i] - up;
    }
    for (int i = pyramid.size()-2; i >= 0; i--)
    {
        Mat up;
        cv::pyrUp(pyramid[i + 1], up, pyramid[i].size());
        pyramid[i] = pyramid[i] + up;
    }
    out = pyramid[0].clone();
    return;
}

static void fbaMerge(const vector<Mat>& imgs_, int tileSz, float p, Mat& out)
{
    //out = Mat::zeros(imgs_[0].size(), imgs_[0].type());
    //for (int i = 0; i < imgs_.size(); i++)
    //{
    //    out += imgs_[i];
    //}
    //out = out / imgs_.size();
    //return;
    vector<Mat> imgs(imgs_.size());
    Mat totalFreq = Mat::zeros(imgs_[0].size(), CV_32FC2);
    Mat totalWeight = Mat::zeros(imgs_[0].size(), CV_32FC1);
    int w = imgs_[0].cols;
    int h = imgs_[0].rows;
    int pw, ph;
    if (w / tileSz * tileSz == w)
        pw = w + tileSz;
    else
        pw = w / tileSz * tileSz + 2 * tileSz;
    if (h / tileSz * tileSz == h)
        ph = h + tileSz;
    else
        ph = h / tileSz * tileSz + 2 * tileSz;
#pragma omp parallel for
    for (int idx = 0; idx < imgs_.size(); idx++)
    {
        cv::copyMakeBorder(imgs_[idx], imgs[idx], tileSz / 2, ph - h - tileSz / 2, tileSz / 2, pw - w - tileSz / 2, cv::BorderTypes::BORDER_DEFAULT);
        imgs[idx].convertTo(imgs[idx], CV_64F);
    }
    out = Mat::zeros(ph, pw, CV_64FC1);
    for (int i = 0; i < ph / (tileSz / 2) - 1; i++)
    {
        for (int j = 0; j < pw / (tileSz / 2) - 1; j++)
        {
            Rect rect(j*(tileSz / 2), i*(tileSz / 2), tileSz, tileSz);
            vector<Mat> freqs(imgs.size()), weights(imgs.size());
#pragma omp parallel for
            for (int idx = 0; idx < imgs.size(); idx++)
            {
                Mat patch = imgs[idx](rect);
                Mat freq, nf, coef;
                cv::dft(patch, freq, cv::DFT_COMPLEX_OUTPUT);
                freq = freq / (tileSz*tileSz);
                vector<Mat> complex;
                cv::split(freq, complex);
                cv::magnitude(complex[0], complex[1], nf);
                //cv::GaussianBlur(nf, nf, Size(ksize, ksize), sigma, sigma);
                cv::pow(nf, p, coef);
                coef = 1e5 * coef + FLT_EPSILON;
                //coef = 100 * coef + FLT_EPSILON;
                complex[0] = coef;
                complex[1] = coef;
                cv::merge(complex, coef);  
                weights[idx] = coef.clone();
                freqs[idx] = coef.mul(freq)*(tileSz*tileSz);
            }
            Mat tfreq = freqs[0], tweight = weights[0];
            for (int idx = 1; idx < imgs.size(); idx++)
            {
                tfreq += freqs[idx];
                tweight += weights[idx];
            }
            tfreq = tfreq / (tweight);
            Mat tpatch;
            cv::dft(tfreq, tpatch, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE | cv::DFT_INVERSE);
            out(rect) = out(rect) + 0.25*tpatch;
        }
    }
    out = out(Rect(tileSz / 2, tileSz / 2, w, h)).clone();
    out.convertTo(out, CV_64F);
    return;
}

static void singleChannelWGDCT(const Mat noisy, const Mat guide,
    const float sigma, const int dctSz, Mat& rst)
{
    int rpad = dctSz - noisy.rows%dctSz;
    int cpad = dctSz - noisy.cols%dctSz;
    Mat padNoisy, padGuide;
    cv::copyMakeBorder(noisy, padNoisy, 0, rpad, 0, cpad, cv::BORDER_REPLICATE);
    cv::copyMakeBorder(guide, padGuide, 0, rpad, 0, cpad, cv::BORDER_REPLICATE);
    Mat padRst = Mat::zeros(padNoisy.size(), CV_32FC1);
    Mat padWeight = Mat::zeros(padNoisy.size(), CV_32FC1);

    for (int i = 0; i < noisy.rows; i++)
    {
        for (int j = 0; j < noisy.cols; j++)
        {
            Mat npatch = padNoisy(Rect(j, i, dctSz, dctSz)).clone();
            Mat gpatch = padGuide(Rect(j, i, dctSz, dctSz)).clone();
            Mat nfreq, gfreq;
            cv::dct(npatch, nfreq);
            cv::dct(gpatch, gfreq);
            gfreq = gfreq.mul(gfreq);
            Mat shrinkage = gfreq / (gfreq + sigma * sigma);
            shrinkage.at<float>(0, 0) = 1;
            nfreq = nfreq.mul(shrinkage);
            cv::dct(nfreq, npatch, cv::DCT_INVERSE);
            float weight = cv::sum(shrinkage.mul(shrinkage))[0];
            weight = 1. / weight;
            npatch = weight * npatch;
            padRst(Rect(j, i, dctSz, dctSz)) += npatch.clone();
            padWeight += weight;
        }
    }

    padRst = padRst / padWeight;
    rst = padRst(Rect(0, 0, noisy.cols, noisy.rows)).clone();
    return;
}

static void guidedMerge(vector<Mat>& imgs, Mat guide, const Mat noisePatch, const vector<Mat>& masks, 
    const int scale,  const int level, Mat& rst, bool useDct)
{
    if (imgs.empty())
    {
        rst = imgs[0].clone();
        return;
    }
    rst = guide.clone();
    float isigma = 0.1f;
    Mat noiseflg = noisePatch.clone();
    if (scale == 0)
        isigma = singleChannleNLE2(rst, 5, 100, 0.9327, 1.4758, noiseflg);
    else
        isigma = singleChannleNLE(rst, 5, 100, 0.9327, 1.4758, noiseflg);

    if (useDct)
    {
        Mat coef = Mat::ones(rst.size(), CV_32FC1);
        for (int i = 0; i < masks.size(); i++)
        {
            Mat tmp;
            cv::boxFilter(1 - masks[i], tmp, imgs[0].depth(), Size(8, 8));
            tmp = tmp * 2;
            tmp.setTo(1, tmp < 1);
            coef += tmp;
        }
           
        coef = coef / (1 + masks.size());

        RedundantDXTDenoise dctDenoise;
        dctDenoise(rst, rst, coef);

    }
    else
    {
        vector<Mat> outs;
        outs.resize(imgs.size());
        int itrNum = 8;
        int sz = 3;
        float th = std::max(0.25f / (1 << scale), 0.1f);
        int cmp = 2;
        //float th = 0.4f;
        if (level == 0)
            itrNum = 0;
        else if (level == 1)
        {
            //sz = 2;
            itrNum = std::max(6 / (1 << scale), 2);
        }
        else
        {
            cmp = 4;
        }
        
        int itr = 1;
        Mat iguide = guide;
        while (isigma > th && itr < sz)
        {
            //cv::ximgproc::guidedFilter(iguide, imgs[0], outs[0], itr, isigma);
            for (int i = 1; i < imgs.size(); i++)
            {
                Mat coef;
                cv::boxFilter(1 - masks[i - 1], coef, imgs[0].depth(), Size(8, 8));
                coef = coef * cmp;
                coef.setTo(1, coef < 1);
                guidedFilter(iguide, imgs[i], outs[i], itr, isigma*coef, imgs[i].depth());
            }
            rst = Mat::zeros(guide.size(), CV_32FC1);
            for (int i = 0; i < imgs.size(); i++)
                rst += outs[i];
            rst = rst / (imgs.size());
            iguide = rst;
            isigma = singleChannleNLE2(rst, 5, 100, 0.9327, 1.4758, noiseflg);
            itr++;
        }
        //itr = 1;
        //while (isigma > th && itr < 3)
        //{
        //    int sz = 3;
        //    cv::ximgproc::guidedFilter(iguide, imgs[0], outs[0], sz, sigma);
        //    for (int i = 1; i < imgs.size(); i++)
        //    {
        //        Mat coef;
        //        cv::boxFilter(1 - masks[i - 1], coef, imgs[0].depth(), Size(8, 8));
        //        coef = coef * 2;
        //        coef.setTo(1, coef < 1);
        //        guidedFilter(iguide, imgs[i], outs[i], sz, sigma*coef, imgs[i].depth());
        //    }
        //    rst = Mat::zeros(guide.size(), CV_32FC1);
        //    for (int i = 0; i < imgs.size(); i++)
        //        rst += outs[i];
        //    rst = rst / (imgs.size());
        //    iguide = rst;
        //    isigma = singleChannleNLE2(rst, 5, 100, 0.9327, 1.4758, noiseflg);
        //    itr++;
        //}
        //itr = 1;
        //itr = 1;
        while (isigma > th && itr < itrNum + sz)
        {
            //cv::ximgproc::guidedFilter(iguide, imgs[0], outs[0], sz, isigma*(itr - sz + 1));
            for (int i = 1; i < imgs.size(); i++)
            {
                Mat coef;
                cv::boxFilter(1 - masks[i - 1], coef, imgs[0].depth(), Size(8, 8));
                coef = coef * cmp;
                coef.setTo(1, coef < 1);
                guidedFilter(iguide, imgs[i], outs[i], sz, isigma*(itr - sz + 1)*coef, imgs[i].depth());
            }
            rst = Mat::zeros(guide.size(), CV_32FC1);
            for (int i = 0; i < imgs.size(); i++)
                rst += outs[i];
            rst = rst / (imgs.size());
            iguide = rst;
            isigma = singleChannleNLE2(rst, 5, 100, 0.9327, 1.4758, noiseflg);
            itr++;
        }

        if (scale == 0)
        {
            Mat coef = Mat::ones(rst.size(), CV_32FC1);
            for (int i = 0; i < masks.size(); i++)
            {
                Mat tmp;
                cv::boxFilter(1 - masks[i], tmp, imgs[0].depth(), Size(8, 8));
                tmp = tmp * cmp;
                tmp.setTo(1, tmp < 1);
                coef += tmp;
            }
            coef = coef / (1 + masks.size());
            RedundantDXTDenoise dctDenoise;
            dctDenoise(rst, rst, coef);
        }
        
        cout << ". itr: " << itr << ". sigma: " << isigma << endl;
        
    }

    return;
}

static void guidedMergeUV(vector<Mat>& imgs, Mat guide, const Mat noisePatch, 
    const vector<Mat>& masks, const int scale, const int level, Mat& rst, bool useDct)
{
    if (imgs.empty())
    {
        rst = imgs[0].clone();
        return;
    }
    //rst = Mat::zeros(guide.size(), CV_32FC1);
    //for (int i = 0; i < imgs.size(); i++)
    //    rst += imgs[i];
    //rst = rst / imgs.size();
    fbaMerge(imgs, 64, 11.f, rst);
    float isigma = 0.1f;
    Mat noiseflg = noisePatch.clone();
    if (scale == 0)
        isigma = singleChannleNLE2(rst, 5, 100, 0.9327, 1.4758, noiseflg);
    else
        isigma = singleChannleNLE(rst, 5, 100, 0.9327, 1.4758, noiseflg);
    if (useDct)
    {
        Mat coef = Mat::ones(rst.size(), CV_32FC1);
        for (int i = 0; i < masks.size(); i++)
        {
            Mat tmp;
            cv::boxFilter(1 - masks[i], tmp, imgs[0].depth(), Size(8, 8));
            tmp = tmp * 2;
            tmp.setTo(1, tmp < 1);
            coef += tmp;
        }
        coef = coef / (1 + masks.size());
        RedundantDXTDenoise dctDenoise;
        float th = std::max(3.f, isigma);
        dctDenoise(rst, rst, th*coef, Size(16, 16));
    }
    else
    {
        vector<Mat> outs;
        outs.resize(imgs.size());        
        vector<Mat> guides;
        guides.resize(2);
        guides[0] = guide;
        guides[1] = rst;
        Mat iguide;
        cv::merge(guides, iguide);
        float th = std::max(0.1f / (1 << scale), 0.02f);
        int sz = 5;
        int itr = 1;
        int itrNum = 10;
        int cmp = 2;
        if (level == 0)
            itrNum = 0;
        else if (level == 1)
        {
            itrNum = 5;
            sz = 6;
            //th = th * 2;
        }            
        else
        {
            itrNum = 10;
            sz = 3;
            cmp = 4;
        }
            
        //float sigma = sqrt(imgs.size())*isigma;
        //while (isigma > th && itr < 5)
        //{
        //    int sz = 3;
        //    cv::ximgproc::guidedFilter(iguide, imgs[0], outs[0], sz, sigma);
        //    for (int i = 1; i < imgs.size(); i++)
        //    {
        //        Mat coef;
        //        cv::boxFilter(1 - masks[i - 1], coef, imgs[0].depth(), Size(8, 8));
        //        coef = coef * 2;
        //        coef.setTo(1, coef < 1);
        //        guidedFilter(iguide, imgs[i], outs[i], sz, sigma*coef, imgs[i].depth());
        //    }

        //    rst = Mat::zeros(guide.size(), CV_32FC1);
        //    for (int i = 0; i < imgs.size(); i++)
        //        rst += outs[i];
        //    rst = rst / (imgs.size());
        //    guides[1] = rst;
        //    cv::merge(guides, iguide);
        //    isigma = singleChannleNLE2(rst, 5, 100, 0.9327, 1.4758, noiseflg);
        //    itr++;
        //}
        //itr = 4;
        while (isigma > th && itr < itrNum)
        {
            //cv::ximgproc::guidedFilter(iguide, imgs[0], outs[0], sz, isigma*itr*sqrt(imgs.size()));
            for (int i = 1; i < imgs.size(); i++)
            {
                Mat coef;
                cv::boxFilter(1 - masks[i - 1], coef, imgs[0].depth(), Size(8, 8));
                coef = coef * cmp;
                coef.setTo(1, coef < 1);
                guidedFilter(iguide, imgs[i], outs[i], sz, isigma*itr*coef*sqrt(imgs.size()), imgs[i].depth());
            }

            rst = Mat::zeros(guide.size(), CV_32FC1);
            for (int i = 0; i < imgs.size(); i++)
                rst += outs[i];
            rst = rst / (imgs.size());
            guides[1] = rst;
            cv::merge(guides, iguide);
            isigma = singleChannleNLE2(rst, 5, 100, 0.9327, 1.4758, noiseflg);
            itr++;
        }

        if (scale == 0)
        {
            Mat coef = Mat::ones(rst.size(), CV_32FC1);
            for (int i = 0; i < masks.size(); i++)
            {
                Mat tmp;
                cv::boxFilter(1 - masks[i], tmp, imgs[0].depth(), Size(8, 8));
                tmp = tmp * cmp;
                tmp.setTo(1, tmp < 1);
                coef += tmp;
            }
            coef = coef / (1 + masks.size());
            RedundantDXTDenoise dctDenoise;
            dctDenoise(rst, rst, coef);
        }
        cout << ". itr: " << itr << ". sigma: " << isigma << endl;
    }
    
    return;
}

void singleChannleFrequenceMerge(const Mat reference, const vector<Mat>& imgs, const vector<Mat>& masks,
    const Mat noisePatch, Mat& out, Mat& noise, bool isNoise, float refWeight, vector<float>& weights, 
    int tileSz, float c, float divTh, int scale, Mat& guide, int level, bool useDct, bool useUV)
{
    int searchSz = 4;
    float gamma = 11.f;

    if (imgs.empty())
        return;
    out = reference.clone();

    Mat cosWin2D(tileSz, tileSz, CV_32FC1);
#pragma omp parallel for
    for (int m = 0; m < cosWin2D.rows; m++)
    {
        float* pwin = cosWin2D.ptr<float>(m);
        for (int n = 0; n < cosWin2D.cols; n++)
        {
            pwin[n] = (0.5f - 0.5f * cos(2 * CV_PI * (0.5f + n) / tileSz))*(0.5f - 0.5f * cos(2 * CV_PI * (0.5f + m) / tileSz));
        }
    }

    // for each image
    vector<Mat> outImgs;
    outImgs.resize(imgs.size() + 1);
    outImgs[0] = reference.clone();
    //noise = noiseEstimation(reference, imgs, masks, tileSz);
    float var = noiseEstimation2(reference, imgs, masks, tileSz);
    var = var * var;
#pragma omp parallel for
    for (int imgIdx = 0; imgIdx < imgs.size(); imgIdx++)
    {
        outImgs[imgIdx + 1] = Mat::zeros(reference.size(), reference.type());
        vector<float> patchWeights(imgs.size());
        for (int i = 0; i < reference.rows; i += tileSz / 2)
        {
            int beginy = i;
            int endy = std::min(i + tileSz, reference.rows);
            for (int j = 0; j < reference.cols; j += tileSz / 2)
            {
                int beginx = j;
                int endx = std::min(j + tileSz, reference.cols);
                Mat rpatch = reference(Range(beginy, endy), Range(beginx, endx));
                Mat rfreq;
                mydft(rpatch, rfreq);
                Mat tpatch = imgs[imgIdx](Range(beginy, endy), Range(beginx, endx));
                Mat tfreq;
                mydft(tpatch, tfreq);
                Mat dfreq;
                dfreq = rfreq - tfreq;
                vector<Mat> complex;
                cv::split(dfreq, complex);
                Mat coef;
                cv::magnitude(complex[0], complex[1], coef);
                coef = coef / (float)(tileSz);
                coef = coef.mul(coef);
                //Mat noise2 = noise(Rect(0, 0, coef.cols, coef.rows));
                Mat patchMask = masks[imgIdx](Range(beginy, endy), Range(beginx, endx));
                patchMask.convertTo(patchMask, CV_32FC1);
                float msum = cv::sum(patchMask)[0] / (float)(patchMask.cols*patchMask.rows);
                //msum = msum > 0.7 ? msum : 1 / (1 + std::exp(-15 * (msum - 0.7)));
                float c_ = std::max(std::min(c, std::pow(c, msum)), 0.8f);
                //coef = coef / (coef + c_ * noise2);
                coef = coef / (coef + c_ * var);
                complex.clear();
                complex.push_back(coef);
                complex.push_back(coef);
                cv::merge(complex, coef);
                Mat mfreq = tfreq + coef.mul(dfreq);
                Mat mpatch;
                myidft(mfreq, mpatch);
                outImgs[imgIdx + 1](Range(beginy, endy), Range(beginx, endx)) += mpatch.mul(cosWin2D(cv::Rect(0, 0, mpatch.cols, mpatch.rows)));
                if (i < tileSz / 2 && j < tileSz / 2)
                {
                    mpatch(Range(0, tileSz / 2 - beginy), Range(0, tileSz / 2 - beginx)).copyTo(outImgs[imgIdx + 1](Range(beginy, tileSz / 2), Range(beginx, tileSz / 2)));
                }
                if (i < tileSz / 2 && j >= tileSz / 2)
                {
                    mpatch(Range(0, tileSz / 2 - beginy), Range(0, endx - beginx)).copyTo(outImgs[imgIdx + 1](Range(beginy, tileSz / 2), Range(beginx, endx)));
                }
                if (i >= tileSz / 2 && j < tileSz / 2)
                {
                    mpatch(Range(0, endy - beginy), Range(0, tileSz / 2 - beginx)).copyTo(outImgs[imgIdx + 1](Range(beginy, endy), Range(beginx, tileSz / 2)));
                }
            }
        }
    }

    if (scale == 0)
    {
        if (!useUV)
        {
            //guide = Mat::zeros(outImgs[0].size(), outImgs[0].type());
            //for (int i = 0; i < outImgs.size(); i++)
            //    guide += outImgs[i];
            //guide = guide / (float)(outImgs.size());
            fbaMerge(outImgs, 64, 11.f, guide);
        }       
    }
    else
        cv::pyrDown(guide, guide, outImgs[0].size());
    if (useUV)
    {
        guidedMergeUV(outImgs, guide, noisePatch, masks, scale, level, out, useDct);
    }
    else
    {
        guidedMerge(outImgs, guide, noisePatch, masks, scale, level, out, useDct);
    }
    
    return;
}

static void multiScaleSingleChannleFrequenceMerge(const Mat reference, const vector<Mat>& imgs, 
    const vector<Mat>& masks, const Mat noisePatch, Mat& out, float refWeight, vector<float>& weights,
    int nscale, int tileSz, int level, float c, float divTh, bool useDct, bool useUV, Mat& guide)
{
    vector<Mat> refPyramid;
    vector<vector<Mat>> imgPyramids;//[i][j] means the ith image jth scale
    imgPyramids.resize(imgs.size());
    decompose(reference, nscale, refPyramid);
#pragma omp parallel for
    for (int i = 0; i < imgs.size(); i++)
    {
        decompose(imgs[i], nscale, imgPyramids[i]);
    }

    vector<Mat> rstPyramid;
    rstPyramid.resize(refPyramid.size());
    Mat noise;
    for (int i = 0; i < nscale; i++)
    {
        Mat sRef = refPyramid[i];
        Mat sOut;
        vector<Mat> sImgs, sMasks;
        sImgs.resize(imgs.size());
        sMasks.resize(imgs.size());
#pragma omp parallel for
        for (int j = 0; j < imgs.size(); j++)
        {
            sImgs[j] = imgPyramids[j][i];
            cv::resize(masks[j], sMasks[j], sImgs[j].size());
        }
        Mat sNoisePatch;
        if (i == 0)
            singleChannleFrequenceMerge(sRef, sImgs, sMasks, noisePatch, sOut, noise, true, refWeight, weights, tileSz, c, divTh, i, guide, level, useDct, useUV);
        else
            singleChannleFrequenceMerge(sRef, sImgs, sMasks, noisePatch, sOut, noise, true, refWeight, weights, tileSz, c, divTh, i, guide, level, useDct, useUV);
        rstPyramid[i] = sOut;
    }

    recompose(rstPyramid, out);
    return;
}

void multiChannleFrequenceMerge(const Mat reference, const vector<Mat>& imgs, 
    const vector<Mat>& masks, const vector<Mat>& noisePatchs, Mat& out, float refWeight, 
    vector<float>& weights, int level, float c, float divTh, int tileSz)
{
    vector<Mat> rfVec, outVec;
    cv::split(reference, rfVec);
    outVec.resize(3);
    cv::Vec3f noises;
    Mat guide;
    for (int i = 0; i < 3; i++)
    {
        Mat rf = rfVec[i];
		rf.convertTo(rf, CV_32FC1, 255);
        vector<Mat> ivec;
        ivec.resize(imgs.size());
#pragma omp parallel for
        for (int j = 0; j < imgs.size(); j++)
        {
            vector<Mat> nimgs;
            cv::split(imgs[j], nimgs);
            nimgs[i].convertTo(ivec[j], CV_32FC1, 255);
        }
        cout << "|  |  " << "channel: " << i << ". tileSize: " << tileSz;
        Mat noisePatch;
        cv::resize(noisePatchs[i], noisePatch, rf.size());
        if (level==2)
        {
            if (i > 0)
            {
                guide = outVec[0].clone();
                multiScaleSingleChannleFrequenceMerge(rf, ivec, masks, noisePatch, outVec[i], refWeight, weights, 4, tileSz, level, c, divTh, false, true, guide);
                
            }
            else
            {
                guide = Mat();
                multiScaleSingleChannleFrequenceMerge(rf, ivec, masks, noisePatch, outVec[i], refWeight, weights, 5, tileSz, level, c, divTh, false, false, guide);
            }
        }
        else if (level==1)
        {
            if (i > 0)
            {
                guide = outVec[0].clone();
                multiScaleSingleChannleFrequenceMerge(rf, ivec, masks, noisePatch, outVec[i], refWeight, weights, 1, tileSz, level, c, divTh, false, true, guide);

            }
            else
            {
                guide = Mat();
                multiScaleSingleChannleFrequenceMerge(rf, ivec, masks, noisePatch, outVec[i], refWeight, weights, 3, tileSz, level, c, divTh, false, false, guide);
            }
        }
        else
        {
            if (i > 0)
            {
                guide = outVec[0].clone();
                multiScaleSingleChannleFrequenceMerge(rf, ivec, masks, noisePatch, outVec[i], refWeight, weights, 1, tileSz, level, c, divTh, true, true, guide);
               
            }
            else
            {
                guide = Mat();
                multiScaleSingleChannleFrequenceMerge(rf, ivec, masks, noisePatch, outVec[i], refWeight, weights, 1, tileSz, level, c, divTh, true, false, guide);
            }
        }
    }
    cv::merge(outVec, out);
    return;
}
