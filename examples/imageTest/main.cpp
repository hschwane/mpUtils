/*
 * mpUtils
 * main.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail: hendrik.schwanekamp@gmx.net
 *
 * mpUtils = my personal Utillities
 * A utility library for my personal c++ projects
 *
 * Copyright 2016 Hendrik Schwanekamp
 *
 */

/*
 * This project is used by the developers to play with and test new features
 */

#include <mpUtils/mpUtils.h>

using namespace mpu;
using namespace std;
using namespace std::chrono;

int main()
{
//    Log myLog( LogLvl::ALL, ConsoleSink());
//    myLog.printHeader("devTest", MPU_VERSION_STRING, MPU_VERSION_COMMIT, "");


    Image<unsigned char> img(MPU_LIB_RESOURCE_PATH "missingTexture.png");
    Image<unsigned char> img2 = img;

    img.setZero(0,0,img.height()/2,img.width()/2,0);
    img.storeJPG("/home/hendrik/test/test.jpg",50);

    img2[20][19][1] = 255;
    img2[20][20][1] = 255;
    img2[20][21][1] = 255;
    img2.storePNG("/home/hendrik/test/test.png");

    Image<unsigned char> sub = img2.cloneSubregion(img2.height()/2,img2.width()/2,img2.height(),img2.width());

    sub.storeTGA("/home/hendrik/test/test.tga");


    Image<float> imgf(MPU_LIB_RESOURCE_PATH "missingTexture.png");
    Image<float> imgf2 = imgf;

    imgf.setZero(0,0,imgf.height()/2,imgf.width()/2,0);
    imgf.storeJPG("/home/hendrik/test/testf.jpg",50);

    imgf2[20][19][1] = 255;
    imgf2[20][20][1] = 255;
    imgf2[20][21][1] = 255;
    imgf2.storePNG("/home/hendrik/test/testf.png");

    Image<float> subf = imgf2.cloneSubregion(img2.height()/2,img2.width()/2,img2.height(),img2.width());

    subf.storeTGA("/home/hendrik/test/testf.tga");

    return 0;
}