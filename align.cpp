#include "main.h"

using cv::KeyPoint;

class CBlock
{
private:
public:
    int m_criterionX;
    int m_criterionY;
    int m_corresX;
    int m_corresY;
    bool symmetry_match;
    float m_dis;
    float m_compareFactor;
    bool operator<(const CBlock &m) const
    {
        return m_compareFactor < m.m_compareFactor;
    }
};

static void get_raw_corner_points(Mat &src, float cp_thre, int ref_resize_factor, std::vector<KeyPoint> &keypoints)
{
    Mat src_rsz;
    cv::resize(src, src_rsz, Size(), 1./ref_resize_factor, 1./ref_resize_factor);
    blur(src_rsz, src_rsz, Size(3, 3));
    FAST(src_rsz, keypoints, cp_thre);
    return;
}

static void select_corner_points(vector<KeyPoint> &keypoints,
    vector<CBlock> &l_corner_points,
    const int max_fp_num_per_area,
    const int area_num_in_width,
    const int area_num_in_height,
    const float cp_min_distance,
    const int width, const int height)
{

    float dis_square = cp_min_distance * cp_min_distance;

    vector<CBlock> vBlk;

    for (int m = 0; m < area_num_in_height; ++m) {

        int y_start = height * m / area_num_in_height;
        int y_end = height * (m + 1) / area_num_in_height;

        for (int n = 0; n < area_num_in_width; ++n) {

            int x_start = width * n / area_num_in_width;
            int x_end = width * (n + 1) / area_num_in_width;

            vector<CBlock> vTempBlks;
            CBlock tempCB;

            for (int i = 0; i < (int)keypoints.size(); ++i) {

                if (keypoints[i].pt.x >= x_start && keypoints[i].pt.x < x_end && keypoints[i].pt.y >= y_start && keypoints[i].pt.y < y_end) {

                    tempCB.m_criterionX = (int)keypoints[i].pt.x;
                    tempCB.m_criterionY = (int)keypoints[i].pt.y;
                    tempCB.m_compareFactor = keypoints[i].response;
                    vTempBlks.push_back(tempCB);
                }
            }

            int kpts = vTempBlks.size();
            for (int i = 0; i < kpts; ++i) {
                for (int j = 0; j < i; ++j) {
                    if (vTempBlks[j].m_compareFactor > 0 && pow(vTempBlks[i].m_criterionX - vTempBlks[j].m_criterionX, 2) + pow(vTempBlks[i].m_criterionY - vTempBlks[j].m_criterionY, 2) < dis_square) {
                        vTempBlks[i].m_compareFactor < vTempBlks[j].m_compareFactor ? vTempBlks[i].m_compareFactor = 0 : vTempBlks[j].m_compareFactor = 0;
                    }
                }
            }

            list<CBlock> tempBlkList;
            for (int i = 0; i < kpts; ++i) {
                if (vTempBlks[i].m_compareFactor > 0) {
                    tempBlkList.push_back(vTempBlks[i]);
                }
            }

            int numInCurArea = (int)tempBlkList.size();

            if (numInCurArea > max_fp_num_per_area) {
                tempBlkList.sort();
                tempBlkList.reverse();
                numInCurArea = max_fp_num_per_area;
            }

            list<CBlock>::iterator iter = tempBlkList.begin();
            for (int i = 0; i < numInCurArea; ++i) {
                vBlk.push_back(*iter++);
            }
        }
    }

    for (int i = 0; i < (int)vBlk.size(); ++i) {
        for (int j = 0; j < i; ++j) {
            if (vBlk[j].m_compareFactor > 0 &&
                pow(vBlk[i].m_criterionX - vBlk[j].m_criterionX, 2) + pow(vBlk[i].m_criterionY - vBlk[j].m_criterionY, 2) < dis_square) {
                vBlk[i].m_compareFactor < vBlk[j].m_compareFactor ? vBlk[i].m_compareFactor = 0 : vBlk[j].m_compareFactor = 0;
            }
        }
    }

    for (int i = 0; i < (int)vBlk.size(); ++i) {
        if (vBlk[i].m_compareFactor > 0) {
            l_corner_points.push_back(vBlk[i]);
        }
    }

    return;
}

bool get_corner_points_lists(Mat &src, float cp_thre, vector<CBlock> &l_corner_points)
{
    int image_width = src.cols;
    int image_height = src.rows;
    int max_fp_num_per_area = 15;
    int area_num_in_width=8;
    int area_num_in_height=8;
    float cp_min_distance=10.f;
    int ref_resize_factor=4;
    int blk_win_size=28;

    std::vector<KeyPoint> keypoints;
    get_raw_corner_points(src, cp_thre, ref_resize_factor, keypoints);

    //    ALOGI("corner points raw = %d\n", (int)keypoints.size());

        //对求出的特征点进行筛选
    select_corner_points(keypoints, l_corner_points, max_fp_num_per_area,
        area_num_in_width, area_num_in_height,
        cp_min_distance, image_width / ref_resize_factor, image_height / ref_resize_factor);

    //由于前面对图像进行了resize，因此求出的特征点坐标需要进行修正
    for (auto iter = l_corner_points.begin(); iter != l_corner_points.end(); iter++) {

        iter->m_criterionX = iter->m_criterionX * ref_resize_factor + ref_resize_factor / 2;
        iter->m_criterionY = iter->m_criterionY * ref_resize_factor + ref_resize_factor / 2;
    }

    //为了防止在寻找特征点时，出现越界，对特别靠近边缘的特征点进行剔除
    const int fp_border = (blk_win_size  + 1) / 2;
    for (auto iter = l_corner_points.begin(); iter != l_corner_points.end();) {

        if (iter->m_criterionX < fp_border ||
            iter->m_criterionY < fp_border ||
            iter->m_criterionX > image_width - fp_border ||
            iter->m_criterionY > image_height - fp_border) {
            iter = l_corner_points.erase(iter);
        }
        else {
            iter++;
        }
    }
    return true;
}

static void find_pos_last_level(Mat &target, Mat &scope, int &xx, int &yy, int TM_method)
{
    Mat targetcur_level, scopecur_level;
    cv::resize(target, targetcur_level, Size(), 0.5, 0.5);
    cv::resize(scope, scopecur_level, Size(), 0.5, 0.5);
    //////////////////////////////////////////////////////////////////////////
    Mat dis;
    double minVal, maxVal;
    Point minLoc, maxLoc;

    matchTemplate(scopecur_level, targetcur_level, dis, TM_method);
    //blur(dis, dis, Size(3, 3));
    minMaxLoc(dis, &minVal, &maxVal, &minLoc, &maxLoc);

    if (TM_method == cv::TM_SQDIFF || TM_method == cv::TM_SQDIFF_NORMED) {
        xx = minLoc.x;
        yy = minLoc.y;
    }
    else {
        xx = maxLoc.x;
        yy = maxLoc.y;
    }
    //////////////////////////////////////////////////////////////////////////

    return;
}

static void find_pos_interlayer(Mat &target, Mat &scope, int &xx, int &yy, int total_level, int cur_level, int TM_method)
{
    Mat targetcur_level, scopecur_level;
    cv::resize(target, targetcur_level, Size(), 0.5, 0.5);
    cv::resize(scope, scopecur_level, Size(), 0.5, 0.5);
    //////////////////////////////////////////////////////////////////////////
    if (cur_level == total_level - 1) {
        find_pos_last_level(targetcur_level, scopecur_level, xx, yy, TM_method);
    }
    else {
        cur_level++;
        find_pos_interlayer(targetcur_level, scopecur_level, xx, yy, total_level, cur_level, TM_method);
    }
    //////////////////////////////////////////////////////////////////////////
    xx <<= 1;
    yy <<= 1;

    int x_start = std::max(0, xx - 2);
    int x_end = std::min(scopecur_level.cols, xx + 2 + targetcur_level.cols);
    int y_start = std::max(0, yy - 2);
    int y_end = std::min(scopecur_level.rows, yy + 2 + targetcur_level.rows);

    Mat searchMat = scopecur_level(Range(y_start, y_end), Range(x_start, x_end));

    Mat dis;
    double minVal, maxVal;
    Point minLoc, maxLoc;

    matchTemplate(searchMat, targetcur_level, dis, TM_method);
    minMaxLoc(dis, &minVal, &maxVal, &minLoc, &maxLoc);

    if (TM_method == cv::TM_SQDIFF || TM_method == cv::TM_SQDIFF_NORMED) {
        xx = x_start + minLoc.x;
        yy = y_start + minLoc.y;
    }
    else {
        xx = x_start + maxLoc.x;
        yy = y_start + maxLoc.y;
    }

    return;
}

static void find_pos(Mat &img2M_y, Mat &img_1_y, CBlock &curBlk, int total_level, int TM_method)
{
    int cur_level = 1;
    int blk_win_size = 28;
    int search_range = 198;
    int center2Border = blk_win_size / 2;

    Mat target_L1;
    target_L1 = img_1_y(Range(curBlk.m_criterionY - center2Border, curBlk.m_criterionY + center2Border),
        Range(curBlk.m_criterionX - center2Border, curBlk.m_criterionX + center2Border));

    int scope_x_start = std::max(0, curBlk.m_criterionX - center2Border - search_range);
    int scope_x_end = std::min(img2M_y.cols, curBlk.m_criterionX + center2Border + search_range);
    int scope_y_start = std::max(0, curBlk.m_criterionY - center2Border - search_range);
    int scope_y_end = std::min(img2M_y.rows, curBlk.m_criterionY + center2Border + search_range);

    Mat scope_L1;
    scope_L1 = img2M_y(Range(scope_y_start, scope_y_end), Range(scope_x_start, scope_x_end));

    //////////////////////////////////////////////////////////////////////////
    int xx(0), yy(0);

    if (cur_level == total_level - 1) {
        find_pos_last_level(target_L1, scope_L1, xx, yy, TM_method);
    }
    else {
        cur_level++;
        find_pos_interlayer(target_L1, scope_L1, xx, yy, total_level, cur_level, TM_method);
    }
    //////////////////////////////////////////////////////////////////////////
    xx <<= 1;
    yy <<= 1;

    int x_start = std::max(0, xx - 2);
    int x_end = std::min(scope_L1.cols, xx + 2 + target_L1.cols);
    int y_start = std::max(0, yy - 2);
    int y_end = std::min(scope_L1.rows, yy + 2 + target_L1.rows);

    Mat searchMat = scope_L1(Range(y_start, y_end), Range(x_start, x_end));

    Mat dis;
    double minVal, maxVal;
    Point minLoc, maxLoc;
    matchTemplate(searchMat, target_L1, dis, TM_method);
    minMaxLoc(dis, &minVal, &maxVal, &minLoc, &maxLoc);

    if (TM_method == cv::TM_SQDIFF || TM_method == cv::TM_SQDIFF_NORMED) {
        xx = x_start + minLoc.x;
        yy = y_start + minLoc.y;
    }
    else {
        xx = x_start + maxLoc.x;
        yy = y_start + maxLoc.y;
    }

    curBlk.m_corresX = scope_x_start + center2Border + xx;
    curBlk.m_corresY = scope_y_start + center2Border + yy;
    ///////////////////////////////////////////////////////////////////////////
    // double match
    target_L1 = img2M_y(Range(curBlk.m_corresY - center2Border, curBlk.m_corresY + center2Border),
        Range(curBlk.m_corresX - center2Border, curBlk.m_corresX + center2Border));

    scope_x_start = std::max(0, curBlk.m_criterionX - center2Border - 1);
    scope_x_end = std::min(img2M_y.cols, curBlk.m_criterionX + center2Border + 1);
    scope_y_start = std::max(0, curBlk.m_criterionY - center2Border - 1);
    scope_y_end = std::min(img2M_y.rows, curBlk.m_criterionY + center2Border + 1);

    scope_L1 = img_1_y(Range(scope_y_start, scope_y_end), Range(scope_x_start, scope_x_end));

    matchTemplate(scope_L1, target_L1, dis, TM_method);
    minMaxLoc(dis, &minVal, &maxVal, &minLoc, &maxLoc);

    if (TM_method == cv::TM_SQDIFF || TM_method == cv::TM_SQDIFF_NORMED) {
        curBlk.symmetry_match = (minLoc.x == 1) && (minLoc.y == 1);
    }
    else {
        curBlk.symmetry_match = (maxLoc.x == 1) && (maxLoc.y == 1);
    }

    return;
}

static Mat solve_persp_matrix(vector<CBlock> &vBlk)
{

    Mat srcPts((int)vBlk.size(), 1, CV_32FC2);
    Mat dstPts((int)vBlk.size(), 1, CV_32FC2);
    int index = 0;
    for (vector<CBlock>::iterator iter = vBlk.begin(); iter != vBlk.end(); iter++) {
        dstPts.at<Point2f>(index, 0) = Point2f(iter->m_criterionX, iter->m_criterionY);
        srcPts.at<Point2f>(index, 0) = Point2f(iter->m_corresX, iter->m_corresY);

        index++;
    }
    Mat PerspectiveMat = findHomography(dstPts, srcPts, cv::RANSAC, 1.5);

    return PerspectiveMat;
}

bool is_persp_matrix_valid(Mat &persMat, int border, int width, int height)
{

    //向外扩几个像素，增加代码的鲁棒性，同时可以抵挡掉透视变换矩阵的计算误差
    //例如本来（64，48）是理论上允许的边缘，但是实际上偏移到了（64，64），但是计算的时候只是取最小值，
    //因此计算出来的最优解仍然是（64，48），这样结果显然是错误的
    //向外扩展几个像素，可以保证不会出现因为超出边缘而使得边缘成为最优的解的情况。
    float x_shrink = (float)border - 2;
    float y_shrink = (float)border - 2;

    Mat points(4, 1, CV_32FC2);
    points.at<Point2f>(0, 0) = Point2f(x_shrink, y_shrink);
    points.at<Point2f>(1, 0) = Point2f(width - 1 - x_shrink, y_shrink);
    points.at<Point2f>(2, 0) = Point2f(x_shrink, height - 1 - y_shrink);
    points.at<Point2f>(3, 0) = Point2f(width - 1 - x_shrink, height - 1 - y_shrink);

    Mat points_trans(4, 1, CV_32FC2);
    perspectiveTransform(points, points_trans, persMat);

    for (int i = 0; i < 4; ++i) {

        if (points_trans.at<Point2f>(i, 0).x < 0 || points_trans.at<Point2f>(i, 0).x > width - 1 || points_trans.at<Point2f>(i, 0).y < 0 || points_trans.at<Point2f>(i, 0).y > height - 1) {
            return false;
        }
    }
    if ((fabsf(persMat.at<double>(0, 0) - 1.0) > 0.05f) || (fabsf(persMat.at<double>(1, 1) - 1.0) > 0.05f)) {
        return false;
    }

    return true;
}

static Mat get_trans_matrix(vector<CBlock> &l_corner_points, Mat reference, Mat target)
{
    int total_level = 2;
    const int TM_METHOD = cv::TM_CCORR_NORMED;
    int search_range = 198;
    vector<CBlock> v_symmetry_match_blk;
    for (auto iter = l_corner_points.begin(); iter != l_corner_points.end(); iter++)
    {
        find_pos(target, reference, *iter, total_level, TM_METHOD);
        if ((*iter).symmetry_match)
        {
            v_symmetry_match_blk.push_back(*iter);
        }
    }
    
    Mat h;
    if (v_symmetry_match_blk.size() > 15) 
    {
        h = solve_persp_matrix(v_symmetry_match_blk);
    }
    else 
    {
        h = solve_persp_matrix(l_corner_points);
    }
    h.convertTo(h, CV_32FC1);
    return h;
    //if (is_persp_matrix_valid(h, search_range, reference.cols, reference.rows))
    //{
    //    return h;
    //}
    //else
    //{
    //    return Mat::eye(Size(3, 3), CV_32FC1);
    //}
}

//static int keyPointExtraction(Mat src, int gridWidth, int gridHei, int maxPtNum, float minDistance, vector<Point2f>& pts)
//{
//    Mat img = src(cv::Rect(0.15*src.cols, 0.15*src.rows, 0.7*src.cols, 0.7*src.rows));
//    for (int i = 0; i < gridHei; i++)
//    {
//        int yBegin = img.rows / gridHei * i;
//        int yEnd = img.rows / gridHei * (i + 1);
//        for (int j = 0; j < gridWidth; j++)
//        {
//            int xBegin = img.cols / gridWidth * j;
//            int xEnd = img.cols / gridWidth * (j + 1);
//            Mat roi = img(cv::Range(yBegin, yEnd), cv::Range(xBegin, xEnd));
//            vector<Point2f> rPts;
//            cv::goodFeaturesToTrack(roi, rPts, maxPtNum, 0.01, minDistance);
//            for (int k = 0; k < rPts.size(); k++)
//            {
//                pts.push_back(rPts[k] + Point2f(xBegin, yBegin)+Point2f(0.15*src.cols, 0.15*src.rows));
//            }
//        }
//    }
//    return 0;
//}
//
static int pixelShift(Size sz, Mat h, Mat& xTrans, Mat& yTrans)
{
    xTrans.create(sz, CV_32FC1);
    yTrans.create(sz, CV_32FC1);
    vector<Point2f> input, output;
    for (int i = 0; i < sz.height; i++)
        for (int j = 0; j < sz.width; j++)
        {
            input.push_back(Point2f(1.0*j, 1.0*i));
        }
    cv::perspectiveTransform(input, output, h);
    for (int i = 0; i < sz.height; i++)
    {
        float* px = xTrans.ptr<float>(i);
        float* py = yTrans.ptr<float>(i);
        for (int j = 0; j < sz.width; j++)
        {
            px[j] = output[i*sz.width + j].x - input[i*sz.width + j].x;
            py[j] = output[i*sz.width + j].y - input[i*sz.width + j].y;
        }
    }
    return 0;
}

//static int homographyAlign(Mat reference, Mat target, vector<Point2f>& basePoints, int ptTh, Mat& flow)
//{
//    vector<Point2f> src, dst, src_, dst_;
//    vector<unsigned char> status0, status1;
//    vector<float> err;
//    cv::calcOpticalFlowPyrLK(reference, target, basePoints, dst_, status0, err);
//    cv::calcOpticalFlowPyrLK(target, reference, dst_, src_, status1, err);
//    for (int i = 0; i < src_.size(); i++)
//    {
//        if (status0[i] && status1[i])
//        {
//            if (abs(basePoints[i].x - src_[i].x) < 1 && abs(basePoints[i].y - src_[i].y) < 1)
//            {
//                src.push_back(0.5*(basePoints[i] + src_[i]));
//                dst.push_back(dst_[i]);
//            }
//        }
//    }
//    //showPtImg(src, reference);
//    //showPtImg(dst, target);
//    Mat inxTrans = Mat::zeros(reference.size(), CV_32FC1), inyTrans = Mat::zeros(reference.size(), CV_32FC1);
//    if (src.size() > ptTh)
//    {
//        Mat h = cv::findHomography(src, dst, cv::RANSAC);
//        if (abs(h.at<double>(2, 0)) > 0.0001 && abs(h.at<double>(2, 1)) > 0.0001)
//            flow = Mat::zeros(reference.size(), CV_32FC1);
//        else
//        {
//            pixelShift(reference.size(), h, inxTrans, inyTrans);
//            vector<Mat> flows = { inxTrans,inyTrans };
//            cv::merge(flows, flow);
//        }       
//    }
//    else
//        flow = Mat::zeros(reference.size(), CV_32FC1);
//    return 0;
//}
//
static void checkConsistence(Mat ref, Mat& rst)
{
    for (int i = 0; i < ref.rows; i++)
    {
        cv::Vec3f *pref = ref.ptr<cv::Vec3f>(i);
        cv::Vec3f *prst = rst.ptr<cv::Vec3f>(i);
        for (int j = 0; j < ref.cols; j++)
        {
            if (prst[j][0] < 0)
            {
                prst[j][0] = pref[j][0];
                prst[j][1] = pref[j][1];
                prst[j][2] = pref[j][2];
            }
        }
    }
}

static void homographyAlign(Mat reference, Mat target, vector<CBlock>& basePoints, Mat& flow)
{
    Mat h = get_trans_matrix(basePoints, reference, target);
    Mat inxTrans = Mat::zeros(reference.size(), CV_32FC1), inyTrans = Mat::zeros(reference.size(), CV_32FC1);
    pixelShift(reference.size(), h, inxTrans, inyTrans);
    vector<Mat> flows = { inxTrans,inyTrans };
    cv::merge(flows, flow);       
    return ;
}

//static void matTileSum(Mat img, int tsz, Mat& dst)
//{
//    //assert CV_32FC1
//    Mat rst = Mat::zeros(img.rows / tsz, img.cols / tsz, CV_32FC1);
//#pragma omp parallel for
//    for (int i = 0; i < rst.rows; i++)
//    {
//        float* prst = rst.ptr<float>(i);
//        for (int j = 0; j < rst.cols; j++)
//        {
//            int beginx = std::max(0, j*tsz);
//            int endx = std::min((j + 1)*tsz, img.cols);
//            int beginy = std::max(0, i*tsz);
//            int endy = std::min((i + 1)*tsz, img.rows);
//            prst[j] = cv::sum(img(Range(beginy, endy), Range(beginx, endx)))[0];
//        }
//    }
//    cv::boxFilter(rst, dst, CV_32FC1, Size(2, 2), cv::Point(0, 0), false);
//    return;
//}

static void divCheck(const Mat u, const Mat v, float th, Mat& mask)
{
    Mat dx, dy;
    cv::Scharr(u, dx, CV_32FC1, 1, 0);
    cv::Scharr(v, dy, CV_32FC1, 0, 1);
    mask = Mat::ones(u.size(), CV_8U);
    mask.setTo(0, cv::abs(dx + dy) > th);
    //cv::erode(mask, mask, cv::getStructuringElement(cv::MORPH_RECT, Size(3, 3)));
    //mask = cv::abs(dx + dy);
    //mask.setTo(0, mask < th);
    return;
}

static void smoothCheck(const Mat flow, float th, Mat& mask)
{
    mask = Mat::ones(flow.size(), CV_8UC1);
    Mat maxflow, minflow, dif, mag;
    cv::erode(flow, minflow, cv::getStructuringElement(cv::MORPH_RECT, Size(3, 3)));
    cv::dilate(flow, maxflow, cv::getStructuringElement(cv::MORPH_RECT, Size(3, 3)));
    dif = maxflow - minflow;
    vector<Mat> uv;
    cv::split(dif, uv);
    cv::magnitude(uv[0], uv[1], mag);
    cv::dilate(mag, mag, cv::getStructuringElement(cv::MORPH_RECT, Size(5, 5)));
    mask.setTo(0, mag > th);
    return;

}

int align_for_input(const Mat ref, const vector<Mat>& imgs, vector<Mat>& alignedImgs, vector<Mat>& masks)
{

    std::cout << "|  |-  " << "align! ";
    int level = 0;
    alignedImgs.assign(imgs.begin(),imgs.end());
    for(int i = 0; i < imgs.size(); i++){
      masks.push_back(Mat::ones(ref.size(),CV_8U));
    }   
    return level;

}
int align(const Mat ref, const vector<Mat>& imgs, vector<Mat>& alignedImgs, vector<Mat>& masks)
{
    std::cout << "|  |-  " << "align! ";

    Mat refgray;
    ref.convertTo(refgray, CV_8UC3, 255);
    vector<Mat> noisePatchs;
    //Mat noisePatch;
    //cv::Vec3f sigmas = multiChannelNLE(refgray, 5, 100, 0.9327, 1.4758, noisePatchs);
    //float sigma = std::max(std::max(sigmas[0], sigmas[1]), sigmas[2]);
    //float sigma = std::min(std::min(sigmas[0], sigmas[1]), sigmas[2]);
//#define MEDIAN(x,y,z) \
//((x)<(y)?((y)<(z)?(y):(x)<(z)?(z):(x)):((y)>(z)?(y):(x)>(z)?(z):(x)))
//    float sigma = MEDIAN(sigmas[0], sigmas[1], sigmas[2]);
    //cout << "sigma = " << sigma << endl;
    cv::cvtColor(refgray, refgray, cv::COLOR_BGR2GRAY);
    //float sigma = singleChannleNLE(refgray, 5, 100, 0.9327, 14758, noisePatch);
    //cout << "sigma = " << sigma << endl;
    int level = 0;
    //if (sigma < 2.5f)
    //    level = 0;
    //else if (sigma < 5.f)
    //    level = 1;
    //else
    //    level = 2;

    int optScale = 1;
    //if (sigma > 2.5f)
    //    optScale = 2;
    if (optScale > 1)
    {
        cv::resize(refgray, refgray, Size(), 1.f / optScale, 1.f / optScale);
    }
        
    
    alignedImgs.resize(imgs.size());
    masks.resize(imgs.size());
    vector<CBlock> refPoints;
        
    cv::Ptr<cv::DISOpticalFlow> dis = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_ULTRAFAST);
    //dis->setGradientDescentIterations(10);
    dis->setFinestScale(0);
    dis->setPatchStride(1);
    dis->setUseMeanNormalization(true);
    //if (sigma > 2.5f)
    //{
    //    dis->setFinestScale(0);
    //    dis->setPatchStride(4);
    //    dis->setPatchSize(16);
    //    //dis->setGradientDescentIterations(25);
    //    dis->setVariationalRefinementIterations(2);
    //    dis->setVariationalRefinementAlpha(5.f);
    //    //dctDenoise(refgray, refgray, cv::Vec3f(5.f, 0.1f, 0.1f));
    //    //get_corner_points_lists(refgray, 8, refPoints);
    //    //cout << "sigma>1.9., use another configure!" << endl;
    //}     
   

    vector<Mat> flows;
    flows.resize(imgs.size());
//    Mat guide(refgray.size(), CV_32FC3);
//#pragma omp parallel for
//    for (int i = 0; i < guide.rows; i++)
//    {
//        cv::Vec3f *pguide = guide.ptr<cv::Vec3f>(i);
//        uchar *pgray = refgray.ptr<uchar>(i);
//        for (int j = 0; j < guide.cols; j++)
//        {
//            pguide[j][0] = (float)(j)/(float)(guide.cols);
//            pguide[j][1] = (float)(i)/(float)(guide.rows);
//            pguide[j][2] = (float)(pgray[j]);
//            
//        }
//    }

    //cv::Ptr<cv::ximgproc::GuidedFilter> fgf = cv::ximgproc::createGuidedFilter(guide, 3, 5);

    for (int idx = 0; idx < imgs.size(); idx++)
    {
        std::cout << "|  |   " << idx + 1 << "/" << imgs.size() << std::endl;
        Mat targray;
        imgs[idx].convertTo(targray, CV_8UC3, 255);
        cv::cvtColor(targray, targray, cv::COLOR_BGR2GRAY);
        if (optScale > 1)
            cv::resize(targray, targray, Size(), 1.f / optScale, 1.f / optScale);
        dis->calc(refgray, targray, flows[idx]);
        //Mat flow;
        //fgf->filter(flows[idx], flow);
        //flows[idx] = flow;
        if (optScale > 1)
        {
            cv::resize(flows[idx], flows[idx], ref.size(), 0.0, 0.0, cv::INTER_CUBIC);
            flows[idx] *= optScale;
        }
    }
    
//#pragma omp parallel for
    for(int idx=0;idx<flows.size();idx++)
    {
        vector<Mat> uv;
        //smoothCheck(flows[idx], 2, masks[idx]);
        cv::split(flows[idx], uv);
        divCheck(uv[0], uv[1], 12.f, masks[idx]);
        for (int i = 0; i < flows[idx].rows; i++)
        {
            float* pu = uv[0].ptr<float>(i);
            float* pv = uv[1].ptr<float>(i);
            for (int j = 0; j < flows[idx].cols; j++)
            {
                pu[j] += j;
                pv[j] += i;
            }
        }       
        cv::remap(imgs[idx], alignedImgs[idx], uv[0], uv[1], cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar::all(-1));
        //checkConsistence(ref, rsts[idx]);
    }
    //saveImgs(alignedImgs, "C:\\Users\\menglong\\Desktop\\adaptiveWeight\\5a9e_20141006_213702_813\\align_dis");
    return level;
}

cv::Vec3f align_asign(const Mat ref, const vector<Mat> imgs, vector<Mat>& alignedImgs, vector<Mat>& masks, vector<Mat>& noisePatchs)
{
    std::cout << "|  |-  " << "align!" << std::endl;
    Mat refgray;
    ref.convertTo(refgray, CV_8UC3, 255);
    cv::Vec3f sigmas = multiChannelNLE(refgray, 5, 100, 0.9327, 1.4758, noisePatchs);
    alignedImgs.resize(imgs.size());
    masks.resize(imgs.size());
#pragma omp parallel for
    for (int idx = 0; idx < imgs.size(); idx++)
    {
        masks[idx] = Mat::ones(ref.size(), CV_8U);
        alignedImgs[idx] = imgs[idx].clone();
        //checkConsistence(ref, rsts[idx]);
    }
    //saveImgs(alignedImgs, "C:\\Users\\menglong\\Desktop\\adaptiveWeight\\5a9e_20141006_213702_813\\align_dis");
    return sigmas;
}
