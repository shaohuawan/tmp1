#include "main.h"

static void stringSplit(const std::string& s, std::vector<string>& tokens, char delimiter)
{
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter))
    {
        tokens.push_back(token);
    }
    return;
}

int main()
{
    clock_t start_t, end_t;
    double total_t;
    string fileName = "/home/work/xumenglong/buildGT20191112/imgs/folderNames.txt";
    vector<string> folderNames;
    std::ifstream file(fileName);
    cout<<fileName<<endl;
    while (!file.eof())
    {
        string folderName;
        std::getline(file, folderName);
        if (folderName != string("") && folderName[0] != '#')
            folderNames.push_back(folderName);
    }
    cout<<"folders_num: " << folderNames.size()<<endl;

    RedundantDXTDenoise dctDenoise;
    for (int idx = 0; idx < folderNames.size(); idx++)
    {
        // Load all images
        std::cout << "|--" << folderNames[idx] << std::endl;
        start_t = clock();
        string folder = folderNames[idx] + "/imgNames.txt";
        vector<string> imgNames;
        std::ifstream imgFile(folder);
        
        int totalImgSz = 50;
        int groupSz = 10;
        int groupCnt = std::ceil(double(totalImgSz) / groupSz);
        vector<Mat> intraGroupDenoisedImgs;
        intraGroupDenoisedImgs.resize(groupCnt);
        vector<int> intraGroupRefIdxs;
        intraGroupRefIdxs.resize(groupCnt);
        Mat denoisedImg;
        int ff = 0;
        int gg = 0;
        vector<Mat> groupImgs;
        Mat reference;        
        while (ff < totalImgSz && gg < groupCnt)
        {
		if(!imgFile.eof())
		{	
            	        string imgName;
            	        std::getline(imgFile, imgName);           
            	        if (imgName != string("") && imgName[0] != '#')
            	        {
                        	imgName = imgName;
                                std::cout << imgName << std::endl;
                        	Mat img = cv::imread(imgName, cv::IMREAD_COLOR);       
                        	groupImgs.push_back(img);//CV_8UC3 for memory save
               
                        	ff++;
                        	if ((ff % groupSz == 0) || (gg == groupCnt - 1 && ff == totalImgSz))
                        	{
                            	std::cout << "|  |--" << "group: " << gg << std::endl;
                            	Mat out;
                            	int refIdx = intraGroupDenoise(groupImgs, out);
                            	intraGroupDenoisedImgs[gg] = out.clone();
                            	intraGroupRefIdxs[gg] = refIdx;
                            	gg++;
                            	if (groupCnt == 1)
                                	reference = groupImgs[refIdx].clone();
                            	groupImgs.clear();
                            	vector<Mat>().swap(groupImgs);
                        	}
            	        }
		}
		else
		{
				int actualGroupCnt = 0;
				for (int i=0; i<groupCnt; i++)
				{
					if(intraGroupDenoisedImgs[i].empty())
					{
						actualGroupCnt = i;
						break;
					}
				}
				intraGroupDenoisedImgs.resize(actualGroupCnt);
				if(ff>actualGroupCnt*groupSz && !groupImgs.empty())
				{
					std::cout<<"| |--"<<"group: "<<actualGroupCnt<<std::endl;
					std::cout<<"     "<<"actual frames: "<<groupImgs.size()<<std::endl;
					Mat out;
					int refIdx = intraGroupDenoise(groupImgs, out);
					intraGroupDenoisedImgs.push_back(out.clone());
					intraGroupRefIdxs.push_back(refIdx);
					gg++;
					if(groupCnt == 1)
						reference = groupImgs[refIdx].clone();
					groupImgs.clear();
					vector<Mat>().swap(groupImgs);
				}
				break;
			}		
        }
        if(ff == 0){
           cout << "No image in the txt" <<endl;
           continue;
        }
        std::cout << "|  |--" << "final: " << std::endl;
        int refIdx = interGroupDenoise(intraGroupDenoisedImgs, intraGroupRefIdxs, groupSz, denoisedImg);
        Mat rst = denoisedImg;
        string name = folderNames[idx] + "/ziyan_merged_muchDetail_20.png";
        cv::imwrite(name, rst);
        //计时
        end_t = clock();
        total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
        double total_t_min = total_t / 60.0;
        cout << "one merge Time: " << total_t_min << endl;

        std::cout << "|--  refIdx: " << refIdx << std::endl;
        reference.convertTo(rst, CV_8UC3, 255);
        if (groupCnt == 1)
        {
            stringstream ss;
            ss << refIdx;
            ss >> name;
            name = folderNames[idx] + "/ziyan_ref_"+name+".png";
            //cv::imwrite(name, reference);
        }        
    }
    return 0;
}
